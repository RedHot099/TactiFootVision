"""Dataset contract shared by all PoCs.

A shot dataset is a parquet file with one row per shot:

    shot_id      str   unique id
    source       str   "soccernet" | "soccertrack" | "dummy" | ...
    game         str   game identifier; splits are always by game
    split        str   "train" | "valid" | "test"
    goal         bool  target
    frame_path   str   absolute path to the JPEG at the shot moment
    is_penalty   bool  optional (default False)
    visibility   str   optional ("visible" | "not shown" | "unknown")
    gt_positions str   optional path to a JSON with ground-truth pitch positions
                       (SoccerTrack v2 GSR) in the same schema as FreezeFrame

Anything else is ignored. The data-preparation agent produces this file; everything
below only reads it or builds a small dummy one for smoke tests.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["shot_id", "source", "game", "split", "goal", "frame_path"]


def load_index(path: Path, require_frames: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"index {path} lacks columns {missing}")
    df["goal"] = df["goal"].astype(bool)
    if "is_penalty" not in df:
        df["is_penalty"] = False
    if require_frames:
        exists = df["frame_path"].map(lambda p: Path(p).exists())
        if not exists.all():
            df = df[exists].reset_index(drop=True)
    return df


def split_by_game(games: pd.Series, seed: int = 0, frac=(0.7, 0.15, 0.15)) -> pd.Series:
    """Deterministic game-level split from a hash of the game id (stable across reruns)."""
    def bucket(g: str) -> str:
        h = int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest(), 16) % 1000 / 1000
        return "train" if h < frac[0] else "valid" if h < frac[0] + frac[1] else "test"
    return games.map(bucket)


def make_dummy_index(frames_root: Path, out: Path, n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Random labels over SoccerNet tracking frames, for end-to-end smoke tests only.

    The label is weakly tied to the frame index inside the sequence so that a model can
    overfit something and we can check the training loop actually learns.
    """
    rng = np.random.default_rng(seed)
    seqs = sorted(p for p in frames_root.glob("*/*/img1") if p.is_dir())
    rows = []
    for i in range(n):
        seq = seqs[i % len(seqs)]
        frames = sorted(seq.glob("*.jpg"))
        j = int(rng.integers(0, len(frames)))
        goal = bool(rng.random() < (0.25 if j > len(frames) / 2 else 0.05))
        rows.append(dict(shot_id=f"dummy_{i:04d}", source="dummy", game=seq.parent.name, split="",
                         goal=goal, frame_path=str(frames[j]), is_penalty=False, visibility="visible"))
    df = pd.DataFrame(rows)
    df["split"] = split_by_game(df["game"], seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
