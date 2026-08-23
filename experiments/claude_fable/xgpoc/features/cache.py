"""Run the positional extractor once per shot and cache the result.

Outputs under <cache_dir>/:
    tabular.parquet   shot_id + FEATURE_NAMES columns
    tokens.npz        per-shot arrays for the set model (padded, canonical coords)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from xgpoc.features.freeze_frame import CLS_GK, FreezeFrame, FreezeFrameExtractor, PITCH_LENGTH, PITCH_WIDTH
from xgpoc.features.tabular import FEATURE_NAMES, freeze_frame_features

MAX_TOKENS = 24  # 22 players + ball + slack
TOKEN_DIM = 8    # x, y, is_ball, is_gk, is_attacking, is_defending, team_known, valid
MAX_BOXES = 32
BOX_DIM = 9      # x1 y1 x2 y2 (normalised to image), one-hot cls(4), valid


def detection_boxes(ff: FreezeFrame, img_hw: tuple[int, int]) -> np.ndarray:
    """Image-space detections for the hybrid model (PoC E), padded."""
    h, w = img_hw
    out = np.zeros((MAX_BOXES, BOX_DIM), np.float32)
    order = np.argsort(ff.boxes_cls, kind="stable")[:MAX_BOXES]  # ball first (class 0)
    for i, j in enumerate(order):
        x1, y1, x2, y2 = ff.boxes_xyxy[j]
        out[i, :4] = [x1 / w, y1 / h, x2 / w, y2 / h]
        out[i, 4 + int(ff.boxes_cls[j])] = 1
        out[i, 8] = 1
    return out


def freeze_frame_tokens(ff: FreezeFrame) -> np.ndarray:
    """Fixed-size token matrix; coordinates normalised to [0, 1] in canonical orientation."""
    ff = ff.canonical()
    tok = np.zeros((MAX_TOKENS, TOKEN_DIM), np.float32)
    i = 0
    if ff.ball_xy is not None:
        tok[i] = [ff.ball_xy[0] / PITCH_LENGTH, ff.ball_xy[1] / PITCH_WIDTH, 1, 0, 0, 0, 1, 1]
        i += 1
    for xy, c, t in zip(ff.players_xy, ff.players_cls, ff.players_team):
        if i >= MAX_TOKENS:
            break
        known = ff.attacking_team >= 0 and t >= 0
        tok[i] = [xy[0] / PITCH_LENGTH, xy[1] / PITCH_WIDTH, 0, float(c == CLS_GK),
                  float(known and t == ff.attacking_team), float(known and t != ff.attacking_team), float(known), 1]
        i += 1
    tok[:, :2] = np.clip(tok[:, :2], -0.2, 1.2)
    return tok


def build_cache(index: pd.DataFrame, cache_dir: Path, extractor: FreezeFrameExtractor | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor = extractor or FreezeFrameExtractor()
    feats, toks, boxes, ids = [], [], [], []
    for row in tqdm(index.itertuples(), total=len(index), desc="positional"):
        img = cv2.imread(row.frame_path)
        if img is None:
            continue
        ff = extractor(img)
        feats.append(freeze_frame_features(ff))
        toks.append(freeze_frame_tokens(ff))
        boxes.append(detection_boxes(ff, img.shape[:2]))
        ids.append(row.shot_id)
    pd.DataFrame(np.array(feats), columns=FEATURE_NAMES).assign(shot_id=ids).to_parquet(cache_dir / "tabular.parquet", index=False)
    np.savez_compressed(cache_dir / "tokens.npz", shot_id=np.array(ids), tokens=np.array(toks), boxes=np.array(boxes))


def load_cache(cache_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    tab = pd.read_parquet(cache_dir / "tabular.parquet")
    z = np.load(cache_dir / "tokens.npz")
    ids = z["shot_id"].tolist()
    return tab, dict(zip(ids, z["tokens"])), dict(zip(ids, z["boxes"]))
