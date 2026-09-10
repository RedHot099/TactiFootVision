"""Strict, cohort-preserving inputs for the v1.3 context ablations."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from xgpoc.v13.protocol import digest

ROOT = Path(__file__).resolve().parents[2]
V13_CACHE = ROOT / "cache/v13"
FULL_FEATURES = ROOT / "cache/full_v1/features_dinov2_l_overlay.npz"
PRIMARY_POSITIONAL = ROOT / "cache/full_v1/positional"
OFFSETS = (-1600, -1200, -800, -400, 0)
PATCH_SHAPE = (577, 1024)
CONTROLS = {"time_mean_t0": "time_mean", "time_gru_t0": "time_gru", "geometry_zero": "geometry"}
VARIANTS = ("full", "roi", "control", "time_mean", "time_gru", "geometry", *CONTROLS)
CONTEXT_FILES = {
    "roi": "roi_context.npz",
    "control": "control_context.npz",
    "time_mean": "temporal_context.npz",
    "time_gru": "temporal_context.npz",
    "geometry": "geometry_context.npz",
}


@dataclass
class InputCache:
    patches: dict[str, np.ndarray]
    patch_valid: dict[str, float]
    contexts: dict[str, dict[str, np.ndarray]]
    context_shapes: dict[str, tuple[int, ...]]
    metadata: dict


def validate_offsets(offsets) -> tuple[int, ...]:
    result = tuple(int(x) for x in offsets)
    if any(x > 0 for x in result):
        raise ValueError("positive temporal offsets can expose the shot outcome")
    if len(set(result)) != len(result):
        raise ValueError("temporal offsets must be unique")
    return result


def _bounded_square(box: np.ndarray, minimum: float = 0.28, margin: float = 0.08) -> tuple[float, float, float, float]:
    x0, y0 = box[:, :2].min(0)
    x1, y1 = box[:, 2:4].max(0)
    side = min(1.0, max(minimum, x1 - x0, y1 - y0) + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0.0), 1.0 - side)
    top = min(max(cy - side / 2, 0.0), 1.0 - side)
    return left, top, left + side, top + side


def _control_box(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = roi
    width, height = x1 - x0, y1 - y0
    left = 0.0 if (x0 + x1) / 2 >= 0.5 else 1.0 - width
    top = 0.0 if (y0 + y1) / 2 >= 0.5 else 1.0 - height
    return left, top, left + width, top + height


def roi_and_control_boxes(boxes: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, float]]:
    """Return equal-area crops and `(shooter_valid, detected_gk)` flags.

    The crop spans the detected ball, nearest outfield player and nearest detected
    goalkeeper. Without a ball, both views fall back to the whole frame.
    """
    valid = np.asarray(boxes, np.float32)
    valid = valid[valid[:, 8] > 0]
    ball = valid[valid[:, 4] > 0]
    if not len(ball):
        whole = (0.0, 0.0, 1.0, 1.0)
        return whole, whole, (0.0, 0.0)
    ball = ball[:1]
    def centre(a):
        return np.column_stack(((a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2))
    selected = [ball[0]]
    players = valid[valid[:, 6] > 0]
    shooter_valid = float(len(players) > 0)
    if len(players):
        selected.append(players[np.linalg.norm(centre(players) - centre(ball), axis=1).argmin()])
    keepers = valid[valid[:, 5] > 0]
    gk_valid = float(len(keepers) > 0)
    if len(keepers):
        selected.append(keepers[np.linalg.norm(centre(keepers) - centre(ball), axis=1).argmin()])
    roi = _bounded_square(np.asarray(selected))
    return roi, _control_box(roi), (shooter_valid, gk_valid)


def normalized_crop_pixels(box: tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(0, int(round(x0 * width))), max(0, int(round(y0 * height))),
            min(width, max(1, int(round(x1 * width)))), min(height, max(1, int(round(y1 * height)))))


def normalize_geometry(table: pd.DataFrame, fit_ids: set[str], flag_columns: set[str]) -> tuple[pd.DataFrame, dict]:
    if "shot_id" not in table or not fit_ids:
        raise ValueError("geometry normalization needs shot_id and fit rows")
    out = table.copy()
    columns = [c for c in out if c != "shot_id"]
    continuous = [c for c in columns if c not in flag_columns]
    fit = out[out.shot_id.isin(fit_ids)]
    if fit.empty:
        raise ValueError("no fit rows found for geometry normalization")
    mean = fit[continuous].mean().fillna(0.0)
    scale = fit[continuous].std().replace(0, 1).fillna(1.0)
    out[continuous] = (out[continuous].fillna(mean) - mean) / scale
    out[list(flag_columns)] = out[list(flag_columns)].fillna(0.0)
    stats = {"mean": {k: float(v) for k, v in mean.items()},
             "scale": {k: float(v) for k, v in scale.items()},
             "flags": sorted(flag_columns)}
    return out, stats


def pool_fingerprint(pool: pd.DataFrame) -> str:
    columns = ["shot_id", "role", *["frame_path" if o == 0 else f"frame_path_{o:+d}" for o in OFFSETS]]
    present = [c for c in columns if c in pool]
    payload = pool[present].sort_values("shot_id").fillna("").to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def input_files_fingerprint(pool: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for offset in OFFSETS:
        column = "frame_path" if offset == 0 else f"frame_path_{offset:+d}"
        entries = []
        for value in pool.get(column, pd.Series(dtype=str)).fillna("").astype(str):
            path = Path(value)
            checksum = digest(path) if value and path.is_file() else "missing"
            entries.append(f"{path.resolve() if value else ''}\t{checksum}")
        result[column] = {"count": len(entries), "sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}
    return result


def metadata_fingerprint(metadata: dict) -> str:
    payload = {k: v for k, v in metadata.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_file(record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"stale or missing v1.3 input: {path}")
    if digest(path) != record["sha256"]:
        raise ValueError(f"v1.3 input content changed: {path}")
    return path


def unique_map(ids, values, wanted):
    result = {}
    for sid, value in zip(ids, values, strict=True):
        if sid not in wanted:
            continue
        if sid in result and not np.array_equal(result[sid], value, equal_nan=True):
            raise ValueError(f"Conflicting cached values for shot {sid}")
        result[sid] = value
    return result


def _load_map(path: Path, key: str, wanted: set[str]) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as data:
        ids = data["shot_id"].astype(str)
        values = data[key]
        shape = tuple(values.shape[1:])
        return unique_map(ids, values, wanted), shape


def load_inputs(pool: pd.DataFrame, cache_dir: Path = V13_CACHE) -> InputCache:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"run scripts/prepare_v13_inputs.py first: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("fingerprint") != metadata_fingerprint(metadata):
        raise ValueError("v1.3 input metadata fingerprint changed")
    if metadata.get("mask") != "overlay" or tuple(metadata.get("offsets", ())) != OFFSETS:
        raise ValueError("v1.3 input preprocessing changed")
    if metadata.get("pool_sha256") != pool_fingerprint(pool):
        raise ValueError("v1.3 cache belongs to another development pool")
    if metadata.get("input_files") != input_files_fingerprint(pool):
        raise ValueError("v1.3 source frame paths or sizes changed")
    sources = {name: _check_file(record) for name, record in metadata["sources"].items()}
    outputs = {name: _check_file(record) for name, record in metadata["outputs"].items()}
    wanted = set(pool.shot_id.astype(str))
    patches, patch_shape = _load_map(sources["full_features"], "feats", wanted)
    if patch_shape != PATCH_SHAPE or any(x.dtype != np.float16 for x in patches.values()):
        raise ValueError(f"full patch cache must contain float16 {PATCH_SHAPE} tensors")
    if set(patches) != wanted:
        raise ValueError("Full patch cache is missing development shots")
    patch_valid = {sid: float(sid in patches) for sid in wanted}
    contexts, shapes = {}, {}
    for variant, filename in CONTEXT_FILES.items():
        source = outputs[filename]
        values, shape = _load_map(source, "context", wanted)
        if set(values) != wanted:
            raise ValueError(f"Incomplete context cache for {variant}")
        contexts[variant], shapes[variant] = values, shape
    for variant, base in CONTROLS.items():
        contexts[variant], shapes[variant] = contexts[base], shapes[base]
    return InputCache(patches, patch_valid, contexts, shapes, metadata)


class ContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache: InputCache, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"unknown v1.3 variant {variant!r}")
        self.df = df.copy().reset_index(drop=True)
        self.cache, self.variant = cache, variant

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        patches = self.cache.patches.get(str(row.shot_id))
        if patches is None:
            patches = np.zeros(PATCH_SHAPE, np.float16)
        item = {"patches": torch.from_numpy(np.asarray(patches)), "y": torch.tensor(float(row.goal))}
        if self.variant != "full":
            context = self.cache.contexts[self.variant].get(str(row.shot_id))
            if context is None:
                context = np.zeros(self.cache.context_shapes[self.variant], np.float32)
            if self.variant in ("time_mean_t0", "time_gru_t0"):
                context = np.repeat(context[-1:], len(context), axis=0)
            elif self.variant == "geometry_zero":
                context = np.zeros_like(context)
            item["context"] = torch.from_numpy(np.asarray(context))
        return item
