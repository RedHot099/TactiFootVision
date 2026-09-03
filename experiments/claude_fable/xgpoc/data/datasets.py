"""Torch datasets over the index + caches. Images are loaded with OpenCV, resized to a
fixed 16:9 size and ImageNet-normalised; horizontal flip is a valid augmentation
because the attacked goal can be on either side."""

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FILL = 114                                        # neutral grey for masked-out pixels
CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TOP_BAND = 0.14                                   # broadcasts place scoreboards inside it
FRAME_COL_RE = re.compile(r"^frame_path(_-\d+)?$")


def check_frame_col(col: str) -> str:
    """Only the shot frame and *pre-shot* offsets are legal inputs. Frames after t0 (and
    the second half of the 8 s clips) can show the outcome - ball in the net, the save -
    so feeding them would leak the label. Enforced here, not in a comment."""
    if not FRAME_COL_RE.match(col):
        raise ValueError(f"illegal input frame column {col!r}: only frame_path / frame_path_-<ms> (t <= 0) are allowed")
    return col


@lru_cache(maxsize=8)
def overlay_mask(size_wh: tuple[int, int], mode: str) -> np.ndarray:
    """Boolean [H, W] of pixels to blank out, at the target size.

    `overlay` removes the graphics detected by scripts/find_overlay_region.py plus a
    generous top band, because the detected mask only covers scoreboard positions
    common to most broadcasters.

    Two complementary control probes, which answer different questions:
    `overlay_only` keeps everything the wide mask would remove - graphics *and* the
    thin static strips along the frame edges, which still contain pitch pixels;
    `graphics_only` keeps just the two graphics widgets. Comparing them separates
    "the model reads the scoreboard" from "the model reads camera framing at the
    edges of the frame", which are very different findings.
    """
    w, h = size_wh

    def load(name: str) -> np.ndarray:
        raw = cv2.imread(str(CONFIGS / name), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"{CONFIGS / name} - run scripts/find_overlay_region.py first")
        return cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST) > 127

    if mode == "graphics_only":
        return ~load("graphics_mask.png")
    m = load("overlay_mask.png")
    if mode == "overlay_only":
        return ~m
    band = np.zeros((h, w), bool)
    band[: int(round(TOP_BAND * h))] = True
    return m | band


def load_image(path: str, size_wh: tuple[int, int], mask: str = "none",
               crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """`crop` = (x0, y0, x1, y1) in native pixels, applied before the resize. When
    cropping, the overlay is blanked at native resolution first so the graphics stay
    covered wherever the crop lands; without a crop the mask is applied after the
    resize, exactly as the cached features were computed."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if crop is not None:
        if mask != "none":
            img[overlay_mask((img.shape[1], img.shape[0]), mask)] = FILL
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    img = cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)
    if crop is None and mask != "none":
        img[overlay_mask(size_wh, mask)] = FILL
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def random_crop_box(w: int, h: int, scale: tuple[float, float]) -> tuple[int, int, int, int]:
    """Aspect-preserving random crop covering an area fraction drawn from `scale`.
    Keeping the aspect ratio keeps pitch geometry truthful; the crop only jitters
    framing, which is what varies between broadcasters anyway."""
    s = float(np.sqrt(np.random.uniform(*scale)))
    cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
    x0, y0 = np.random.randint(0, w - cw + 1), np.random.randint(0, h - ch + 1)
    return x0, y0, x0 + cw, y0 + ch


class ImageDataset(Dataset):
    """`mask` blanks the broadcast overlay (see `overlay_mask`); `flip` mirrors every
    frame deterministically, which is how test-time augmentation is requested.

    Training-only options: `path_cols` draws the input frame at random from several
    pre-shot offsets (temporal augmentation against the ~1 s annotation noise),
    `rrc_scale` enables the aspect-preserving random crop, `aux_col` adds a second
    target ("y_aux") for multi-task heads."""

    def __init__(self, df: pd.DataFrame, size_wh=(448, 256), train=False, path_col: str = "frame_path",
                 mask: str = "none", flip: bool = False, path_cols: tuple[str, ...] | None = None,
                 rrc_scale: tuple[float, float] | None = None, aux_col: str | None = None):
        check_frame_col(path_col)
        for c in path_cols or ():
            check_frame_col(c)
        if path_col != "frame_path":
            df = df[df[path_col].astype(str).str.len() > 0]
        self.df, self.size, self.train = df.reset_index(drop=True), size_wh, train
        self.path_col, self.mask, self.flip = path_col, mask, flip
        self.path_cols, self.rrc_scale, self.aux_col = tuple(path_cols or ()), rrc_scale, aux_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        col = self.path_col
        if self.train and self.path_cols:
            opts = [c for c in self.path_cols if str(getattr(r, c, "") or "")]
            if opts:
                col = opts[np.random.randint(len(opts))]
        crop = None
        if self.train and self.rrc_scale is not None:
            raw = cv2.imread(getattr(r, col), cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if raw is not None:
                crop = random_crop_box(raw.shape[1] * 8, raw.shape[0] * 8, self.rrc_scale)
        img = load_image(getattr(r, col), self.size, self.mask, crop=crop)
        if self.flip:
            img = img[:, ::-1]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            # mild photometric jitter; geometry is left alone so player/goal layout stays truthful
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        out = {"image": to_tensor(img), "y": torch.tensor(float(r.goal))}
        if self.aux_col is not None:
            out["y_aux"] = torch.tensor([float(r[self.aux_col])])
        return out


class TokenDataset(Dataset):
    """Positional tokens (PoC B). No flip needed: tokens are already canonical."""

    def __init__(self, df: pd.DataFrame, tokens: dict[str, np.ndarray], train=False):
        self.df = df[df["shot_id"].isin(tokens)].reset_index(drop=True)
        self.tokens, self.train = tokens, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        t = self.tokens[r.shot_id].copy()
        if self.train:
            t[:, :2] += np.random.normal(0, 0.01, t[:, :2].shape).astype(np.float32) * t[:, 7:8]
        return {"tokens": torch.from_numpy(t), "y": torch.tensor(float(r.goal))}


class FeatureDataset(Dataset):
    """Precomputed backbone patch features (PoC D) with optional detection boxes (PoC E)
    and an optional per-shot auxiliary feature vector (engineered features).

    `alt` holds the same shots encoded from the mirrored frame. With `train=True` one
    of the two is drawn per sample, which turns the flip into a real augmentation for
    the head even though the backbone is frozen; with `train=False` it selects the
    mirrored view outright, so a caller can average the two as test-time augmentation.
    `aux_col` adds a second target ("y_aux") for multi-task heads.
    """

    def __init__(self, df: pd.DataFrame, feats: dict[str, np.ndarray], boxes: dict[str, np.ndarray] | None = None,
                 aux: dict[str, np.ndarray] | None = None, alt: dict[str, np.ndarray] | None = None,
                 train: bool = False, aux_col: str | None = None):
        keep = df["shot_id"].isin(feats)
        if boxes is not None:
            keep &= df["shot_id"].isin(boxes)
        if aux is not None:
            keep &= df["shot_id"].isin(aux)
        if alt is not None:
            keep &= df["shot_id"].isin(alt)
        self.df, self.feats, self.boxes, self.aux = df[keep].reset_index(drop=True), feats, boxes, aux
        self.alt, self.train, self.aux_col = alt, train, aux_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        src = self.feats
        if self.alt is not None and (not self.train or np.random.rand() < 0.5):
            src = self.alt
        out = {"patches": torch.from_numpy(src[r.shot_id]), "y": torch.tensor(float(r.goal))}
        if self.boxes is not None:
            out["boxes"] = torch.from_numpy(self.boxes[r.shot_id])
        if self.aux is not None:
            out["aux"] = torch.from_numpy(self.aux[r.shot_id])
        if self.aux_col is not None:
            out["y_aux"] = torch.tensor([float(r[self.aux_col])])
        return out
