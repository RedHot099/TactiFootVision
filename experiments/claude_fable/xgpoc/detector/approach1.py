"""Approach 1 of the xg-detector study (ticket 03): G-combo with detector inputs, A1 (geometry in the head) and A2 (a
raster CNN branch). The frozen rules and hyperparameters are in the ticket; this module implements them.

Both variants are G-combo (`FinetuneViTXG`: DINOv2-L, last 4 blocks, attention-pool head with the xG and on-target
logits) with one extra input branch whose 64-wide output `e` enters the head through its auxiliary path: the head MLP
reads concat(z, LayerNorm(e)), z the 5120-wide pooled embedding. Everything else is G-combo's recipe.

    A1  `GeometryBranch`: Linear(76, 64), GELU, Linear(64, 64) on the A1 vector (`A1_COLUMNS` per offset, t0 and -400 ms:
        the sanitised geometry standardised and imputed with the training-fold mean, a missing indicator per geometry
        feature, the sanitised flags, the ball source one-hot, the sanitation indicators, n_valid and n_masked). The
        standardisation (`Standardiser`) is fitted on the training folds only and stored in the bundle's spec.
    A2  `RasterBranch`: a small CNN trained from scratch on `RASTER_CHANNELS` x 72 x 64 (`rasterise`), with a learned
        missing vector in place of its output when neither offset has a raster.

The raster is the pitch window x in [44, 108] m, y in [-2, 70] m (aligned, attacked goal at x = 105) at 1 m per cell.
Per offset (t0, then -400 ms): the five `sanitation.CHANNELS` (a Gaussian of sigma 1 m and peak 1 per point, summed),
a constant plane that is 1 when the offset has raster points, and a constant plane that is 1 when it has a homography;
then two coordinate planes (x and y scaled to [-1, 1]). The image's random horizontal flip mirrors the pitch across
its long axis in aligned coordinates (y -> 68 - y), so a flipped image gets a flipped raster (`DetectorDataset`); the
A1 vector holds no signed lateral quantity and needs no flip.
"""

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from xgpoc.bundle import LoadedBundle, load_bundle, sha256_file
from xgpoc.data.datasets import ImageDataset, load_image, random_crop_box, to_tensor
from xgpoc.detector.conventions import WIDTH
from xgpoc.detector.features import FLAGS, GEOMETRY
from xgpoc.detector.sanitation import CHANNELS, COUNTS, INDICATORS
from xgpoc.models.common import (
    TrainConfig,
    _predict_logits,
    n_trainable,
    seed_all,
    train_torch,
)
from xgpoc.models.d_foundation import BACKBONES, FinetuneViTXG
from xgpoc.models.g_bundle import pretrained_weight_file

VARIANTS = ("A1", "A2")
OFFSETS = (0, -400)
BRANCH_DIM = 64
KIND = "xgd_approach1"

# ------------------------------------------------------------------ A1 vector
BALL_SOURCES = (0, 1, 2)          # own frame, first earlier frame, second earlier frame (fallback_offsets order)
CONTINUOUS = [*GEOMETRY, *COUNTS]
A1_COLUMNS = [*GEOMETRY, *[f"{c}_missing" for c in GEOMETRY], *FLAGS, *[f"ball_source_{k}" for k in BALL_SOURCES],
              *INDICATORS, *COUNTS]


def a1_raw(features: pd.DataFrame, offset: int) -> pd.DataFrame:
    """The A1 columns of one offset before standardisation, indexed by shot_id (geometry NaN where missing)."""
    f = features[features.offset_ms == offset].set_index("shot_id")
    out = pd.DataFrame(index=f.index)
    for c in GEOMETRY:
        out[c] = f[c].astype(float)
        out[f"{c}_missing"] = f[c].isna().astype(float)
    for c in FLAGS:
        out[c] = f[c].astype(float)
    for k in BALL_SOURCES:
        out[f"ball_source_{k}"] = (f.ball_source == k).astype(float)
    for c in [*INDICATORS, *COUNTS]:
        out[c] = f[c].astype(float)
    return out[A1_COLUMNS]


def a1_table(features: pd.DataFrame) -> pd.DataFrame:
    """Both offsets side by side (columns `<column>@<offset>`), one row per shot."""
    parts = [a1_raw(features, o).add_suffix(f"@{o}") for o in OFFSETS]
    return parts[0].join(parts[1], how="outer")


def continuous_columns() -> list[str]:
    return [f"{c}@{o}" for o in OFFSETS for c in CONTINUOUS]


@dataclass(frozen=True)
class Standardiser:
    """Mean and standard deviation of the continuous A1 columns over the training-fold shots; a missing value becomes
    the mean (0 after standardisation), its indicator says so."""

    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, table: pd.DataFrame) -> "Standardiser":
        cols = continuous_columns()
        mean, std = table[cols].mean(skipna=True), table[cols].std(skipna=True, ddof=0)
        std = std.where(std > 1e-6, 1.0).fillna(1.0)
        return cls({c: float(mean[c]) if np.isfinite(mean[c]) else 0.0 for c in cols}, {c: float(std[c]) for c in cols})

    def __call__(self, table: pd.DataFrame) -> np.ndarray:
        out = table.copy()
        for c, m in self.mean.items():
            out[c] = ((out[c] - m) / self.std[c]).fillna(0.0)
        values = out.to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("A1 vector has non-finite values after standardisation")
        return values

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


# ------------------------------------------------------------------ A2 raster
X0, X1, Y0, Y1, CELL, SIGMA = 44.0, 108.0, -2.0, 70.0, 1.0, 1.0
H, W = int((Y1 - Y0) / CELL), int((X1 - X0) / CELL)
PER_OFFSET = len(CHANNELS) + 2
RASTER_CHANNELS = PER_OFFSET * len(OFFSETS) + 2
_XC = X0 + CELL * (np.arange(W) + 0.5)
_YC = Y0 + CELL * (np.arange(H) + 0.5)


@dataclass(frozen=True)
class FramePoints:
    """One offset's raster input: aligned points (K, 2), their channels (K,), and the two presence flags."""

    xy: np.ndarray
    channel: np.ndarray
    aligned: bool
    homography: bool


def rasterise(frames: list[FramePoints], mirrored: bool = False) -> np.ndarray:
    """`RASTER_CHANNELS` x H x W float32 of a shot's offsets (in `OFFSETS` order); `mirrored` flips y -> 68 - y."""
    out = np.zeros((RASTER_CHANNELS, H, W), np.float32)
    for k, fp in enumerate(frames):
        base = k * PER_OFFSET
        if fp.aligned:
            out[base + len(CHANNELS)] = 1.0
        if fp.homography:
            out[base + len(CHANNELS) + 1] = 1.0
        for (x, y), c in zip(fp.xy, fp.channel, strict=True):
            if mirrored:
                y = WIDTH - y
            if x < X0:
                continue                      # own half beyond the window
            y = min(max(y, Y0), Y1)
            gx = np.exp(-((_XC - x) ** 2) / (2 * SIGMA**2))
            gy = np.exp(-((_YC - y) ** 2) / (2 * SIGMA**2))
            out[base + c] += np.outer(gy, gx).astype(np.float32)
    out[-2] = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    out[-1] = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    return out


def frame_points(table: pd.DataFrame) -> dict[str, list[FramePoints]]:
    """shot_id -> the `FramePoints` of `OFFSETS` from the sanitised table (columns pts_x, pts_y, pts_channel,
    aligned, homography_ok)."""
    rows = {(r["shot_id"], int(r["offset_ms"])): r for r in table.to_dict("records")}
    out = {}
    for shot_id in table.shot_id.unique():
        frames = []
        for o in OFFSETS:
            r = rows[(shot_id, o)]
            xy = np.column_stack([np.asarray(r["pts_x"], float), np.asarray(r["pts_y"], float)]).reshape(-1, 2)
            frames.append(FramePoints(xy, np.asarray(r["pts_channel"], int), bool(r["aligned"]), bool(r["homography_ok"])))
        out[shot_id] = frames
    return out


# ------------------------------------------------------------------ dataset
class DetectorDataset(ImageDataset):
    """G-combo's `ImageDataset` (t0 frame, overlay mask, random resized crop, flip, photometric jitter, on-target target)
    plus the variant's input: "geometry" (A1 vector) or "raster" and "has_raster" (A2). `__getitem__` makes the same
    random draws in the same order as `ImageDataset.__getitem__`, so a seed gives the same image augmentations
    (tests/test_xgd_approach1.py); it also knows whether the image was flipped, which the raster needs."""

    def __init__(self, df: pd.DataFrame, *, variant: str, geometry: dict[str, np.ndarray] | None = None,
                 points: dict[str, list[FramePoints]] | None = None, **kwargs):
        if kwargs.get("path_col", "frame_path") != "frame_path" or kwargs.get("path_cols") or kwargs.get("flip"):
            raise ValueError("DetectorDataset reads the t0 frame only, without forced flips or temporal sampling")
        super().__init__(df, **kwargs)
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        source = geometry if variant == "A1" else points
        if source is None or (missing := set(self.df.shot_id) - set(source)):
            raise ValueError(f"{variant} input missing for {len(missing) if source is not None else 'all'} shots")
        self.variant, self.geometry, self.points = variant, geometry, points

    def __getitem__(self, i):
        r = self.df.iloc[i]
        crop = None
        if self.train and self.rrc_scale is not None:
            raw = cv2.imread(r.frame_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if raw is not None:
                crop = random_crop_box(raw.shape[1] * 8, raw.shape[0] * 8, self.rrc_scale)
        img = load_image(r.frame_path, self.size, self.mask, crop=crop)
        mirrored = False
        if self.train:
            if np.random.rand() < 0.5:
                img, mirrored = img[:, ::-1], True
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        out = {"image": to_tensor(img), "y": torch.tensor(float(r.goal))}
        if self.aux_col is not None:
            out["y_aux"] = torch.tensor([float(r[self.aux_col])])
        if self.variant == "A1":
            out["geometry"] = torch.from_numpy(self.geometry[r.shot_id])
        else:
            frames = self.points[r.shot_id]
            out["raster"] = torch.from_numpy(rasterise(frames, mirrored))
            out["has_raster"] = torch.tensor(float(any(f.aligned for f in frames)))
        return out


# ------------------------------------------------------------------ branches and model
class GeometryBranch(nn.Module):
    def __init__(self, n_in: int = len(A1_COLUMNS) * len(OFFSETS), dim: int = BRANCH_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.net(geometry.float())


def _block(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(), nn.MaxPool2d(2)]


class RasterBranch(nn.Module):
    """Three conv blocks (3x3 conv, GroupNorm(8), GELU, 2x2 max-pool: 72 x 64 -> 9 x 8), a 1x1 conv to 16 channels, then
    Flatten, Dropout(0.2), Linear(1152, 64). A shot without any raster gets the learned `missing` vector instead."""

    def __init__(self, c_in: int = RASTER_CHANNELS, dim: int = BRANCH_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(*_block(c_in, 32), *_block(32, 64), *_block(64, 64), nn.Conv2d(64, 16, 1), nn.GELU(),
                                 nn.Flatten(), nn.Dropout(dropout), nn.Linear(16 * (H // 8) * (W // 8), dim))
        self.missing = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, raster: torch.Tensor, has_raster: torch.Tensor) -> torch.Tensor:
        e = self.net(raster.float())
        return torch.where(has_raster.bool()[:, None], e, self.missing.to(e.dtype).expand_as(e))


class DetectorG(FinetuneViTXG):
    """`FinetuneViTXG` whose head also reads the branch output (its auxiliary input, LayerNorm-ed and concatenated with
    the pooled embedding before the head MLP)."""

    def __init__(self, variant: str, backbone: str = "dinov2_l", n_blocks: int = 4, n_out: int = 2, pretrained: bool = True):
        super().__init__(backbone, n_blocks=n_blocks, aux_dim=BRANCH_DIM, n_out=n_out, pretrained=pretrained)
        self.variant = variant
        self.branch = GeometryBranch() if variant == "A1" else RasterBranch()

    def forward(self, image: torch.Tensor, geometry: torch.Tensor | None = None, raster: torch.Tensor | None = None,
                has_raster: torch.Tensor | None = None) -> torch.Tensor:
        e = self.branch(geometry) if self.variant == "A1" else self.branch(raster, has_raster)
        return self.head(self.backbone.forward_features(image), e)


# ------------------------------------------------------------------ spec
def spec(variant: str, backbone: str, n_blocks: int, mask: str, standardiser: Standardiser | None) -> dict:
    """Everything needed to rebuild the model and its inputs: the architecture and, for A1, the training-fold
    standardisation."""
    if (variant == "A1") != (standardiser is not None):
        raise ValueError("A1 needs its training-fold standardiser; A2 has no fitted preprocessing")
    inputs = {"size_wh": list(BACKBONES[backbone][1]), "mask": mask, "frame_col": "frame_path", "offsets": list(OFFSETS)}
    if variant == "A1":
        inputs |= {"a1_columns": A1_COLUMNS, "standardiser": standardiser.as_dict()}
    else:
        inputs |= {"raster": {"x": [X0, X1], "y": [Y0, Y1], "cell": CELL, "sigma": SIGMA, "channels": list(CHANNELS),
                              "n_channels": RASTER_CHANNELS}}
    return {"kind": KIND, "variant": variant, "model": {"backbone": backbone, "n_blocks": n_blocks, "n_out": 2,
                                                        "branch_dim": BRANCH_DIM}, "input": inputs}


def build(spec_: dict) -> DetectorG:
    if spec_["kind"] != KIND:
        raise ValueError(f"not an approach-1 bundle: {spec_['kind']}")
    m = spec_["model"]
    return DetectorG(spec_["variant"], m["backbone"], n_blocks=m["n_blocks"], n_out=m["n_out"])


def standardiser_of(spec_: dict) -> Standardiser:
    s = spec_["input"]["standardiser"]
    return Standardiser(s["mean"], s["std"])


def geometry_inputs(table: pd.DataFrame, standardiser: Standardiser) -> dict[str, np.ndarray]:
    """shot_id -> standardised A1 vector (float32) of the shots of the A1 table."""
    values = standardiser(table)
    return dict(zip(table.index, values, strict=True))


# ------------------------------------------------------------------ development split
EXPERIMENT = Path(__file__).resolve().parents[2]
PLAN_DIR = EXPERIMENT / "results/xgv_split_plan"
FOLDS = EXPERIMENT / "cache/folds_xgv_home5.csv"
PLAN_SHA256 = "a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a"   # the xg-vendor plan (seed 0, 2026-10-11)
PLAN_COUNTS = {"training": 328, "gate": 37, "evaluation": 135}


def verify_plan(index: Path, plan_dir: Path = PLAN_DIR, folds: Path = FOLDS, *, expected: str = PLAN_SHA256) -> pd.DataFrame:
    """The frozen split plan, refused unless plan.csv hashes to `expected` and agrees with its completion record and
    plan.json, the index is the one the plan was built on, the folds file is the plan's training folds, and the role
    counts are the plan's. Hash checks only: the plan's own rebuild reads the StatsBomb match map, which no detector
    model path may read."""
    complete = json.loads((plan_dir / "complete.json").read_text())
    for name, digest in complete["files"].items():
        if sha256_file(plan_dir / name) != digest:
            raise ValueError(f"{plan_dir / name} changed since its completion record")
    record = json.loads((plan_dir / "plan.json").read_text())
    found = {"plan.csv": sha256_file(plan_dir / "plan.csv"), "plan.json": record["plan_csv_sha256"]}
    if wrong := {k: v for k, v in found.items() if v != expected}:
        raise ValueError(f"split plan is not the approved one ({expected}): {wrong}")
    if (digest := sha256_file(index)) != record["inputs"]["index"]["sha256"]:
        raise ValueError(f"index {index} hashes to {digest}, the plan was built on {record['inputs']['index']['sha256']}")
    if (digest := sha256_file(folds)) != record["folds_file"]["sha256"]:
        raise ValueError(f"folds file {folds} hashes to {digest}, the plan wrote {record['folds_file']['sha256']}")
    plan = pd.read_csv(plan_dir / "plan.csv")
    training = plan.loc[plan.role == "training", ["game", "home", "fold"]].reset_index(drop=True)
    if not pd.read_csv(folds).equals(training):
        raise ValueError(f"{folds} is not the fold assignment of the plan's training matches")
    if (counts := plan.role.value_counts().to_dict()) != PLAN_COUNTS:
        raise ValueError(f"role counts {counts} differ from {PLAN_COUNTS}")
    return plan


def training_pool(shots: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """The shots of the plan's training matches with their fold. Raises ValueError if a shot sits in a match the plan
    does not list."""
    role = plan.set_index("game").role
    if unknown := sorted(set(shots.game) - set(role.index)):
        raise ValueError(f"{len(unknown)} index matches are not in the plan, e.g. {unknown[0]}")
    training = plan.loc[plan.role == "training", ["game", "fold"]]
    return shots[shots.game.map(role) == "training"].merge(training, on="game", validate="many_to_one")


# ------------------------------------------------------------------ fit and score
def dataset_args(spec_: dict, geometry: dict | None, points: dict | None) -> dict:
    i = spec_["input"]
    return {"size_wh": tuple(i["size_wh"]), "mask": i["mask"], "variant": spec_["variant"], "geometry": geometry,
            "points": points}


def score(model: nn.Module, df: pd.DataFrame, args: dict, batch_size: int, device: str, num_workers: int = 4) -> np.ndarray:
    """xG logits of `df`'s rows in row order (eval mode, the training-time evaluation path)."""
    loader = DataLoader(DetectorDataset(df, **args), batch_size, shuffle=False, num_workers=num_workers)
    return _predict_logits(model.to(device), loader, device)


def fit(train: pd.DataFrame, select: pd.DataFrame, cfg: TrainConfig, recipe_kw: dict, spec_: dict, *, geometry: dict | None,
        points: dict | None, aux_col: str) -> tuple[nn.Module, dict, dict]:
    """G-combo's fit (`xgpoc.models.g_fit.fit_g`) with the variant's model and dataset: seed, build, train on `train`,
    select the epoch and the temperature on `select`. Returns (model with the selected EMA weights, train_torch's
    result, compute). Raises ValueError if no epoch could be selected."""
    started = time.time()
    seed_all(cfg.seed)
    model = build(spec_)
    try:
        args = dataset_args(spec_, geometry, points)
        result = train_torch(model, DetectorDataset(train, train=True, rrc_scale=tuple(recipe_kw["rrc"]), aux_col=aux_col, **args),
                             DetectorDataset(select, **args), cfg, backbone_lr_mult=recipe_kw["backbone_lr_mult"])
        if result["best_epoch"] < 0:
            raise ValueError("no epoch was selected (non-finite select-set score); refusing to go on")
        compute = {"train_minutes": (time.time() - started) / 60, "epochs_run": len(result["history"]),
                   "trainable_parameters": n_trainable(model), "branch_parameters": n_trainable(model.branch),
                   "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        return model, result, compute
    except BaseException:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        raise


def load(path: Path) -> LoadedBundle:
    """An approach-1 bundle rebuilt on the pretrained backbone (`xgpoc.bundle.load_bundle`)."""
    return load_bundle(path, lambda s: (build(s), pretrained_weight_file(s["model"]["backbone"])))
