"""Torch datasets over the index + caches. Images are loaded with OpenCV, resized to a
fixed 16:9 size and ImageNet-normalised; horizontal flip is a valid augmentation
because the attacked goal can be on either side."""

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
OVERLAY_MASK_PATH = Path(__file__).resolve().parents[2] / "configs" / "overlay_mask.png"
TOP_BAND = 0.14                                   # broadcasts place scoreboards inside it


@lru_cache(maxsize=8)
def overlay_mask(size_wh: tuple[int, int], mode: str) -> np.ndarray:
    """Boolean [H, W] of pixels to blank out, at the target size.

    `overlay` removes the graphics detected by scripts/find_overlay_region.py plus a
    generous top band, because the detected mask only covers scoreboard positions
    common to most broadcasters. `overlay_only` is the complement: it keeps the
    graphics and blanks the pitch, which is the control probe asking how much a model
    can score knowing nothing but the score bug.
    """
    w, h = size_wh
    raw = cv2.imread(str(OVERLAY_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"{OVERLAY_MASK_PATH} - run scripts/find_overlay_region.py first")
    m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    if mode == "overlay_only":
        return ~m
    band = np.zeros((h, w), bool)
    band[: int(round(TOP_BAND * h))] = True
    return m | band


def load_image(path: str, size_wh: tuple[int, int], mask: str = "none") -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)
    if mask != "none":
        img[overlay_mask(size_wh, mask)] = FILL
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ImageDataset(Dataset):
    """`mask` blanks the broadcast overlay (see `overlay_mask`); `flip` mirrors every
    frame deterministically, which is how test-time augmentation is requested."""

    def __init__(self, df: pd.DataFrame, size_wh=(448, 256), train=False, path_col: str = "frame_path",
                 mask: str = "none", flip: bool = False):
        if path_col != "frame_path":
            df = df[df[path_col].astype(str).str.len() > 0]
        self.df, self.size, self.train = df.reset_index(drop=True), size_wh, train
        self.path_col, self.mask, self.flip = path_col, mask, flip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_image(getattr(r, self.path_col), self.size, self.mask)
        if self.flip:
            img = img[:, ::-1]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            # mild photometric jitter; geometry is left alone so player/goal layout stays truthful
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        return {"image": to_tensor(img), "y": torch.tensor(float(r.goal))}


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
    """

    def __init__(self, df: pd.DataFrame, feats: dict[str, np.ndarray], boxes: dict[str, np.ndarray] | None = None,
                 aux: dict[str, np.ndarray] | None = None, alt: dict[str, np.ndarray] | None = None,
                 train: bool = False):
        keep = df["shot_id"].isin(feats)
        if boxes is not None:
            keep &= df["shot_id"].isin(boxes)
        if aux is not None:
            keep &= df["shot_id"].isin(aux)
        if alt is not None:
            keep &= df["shot_id"].isin(alt)
        self.df, self.feats, self.boxes, self.aux = df[keep].reset_index(drop=True), feats, boxes, aux
        self.alt, self.train = alt, train

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
        return out
