"""PoC F: temporal variant of D - frozen DINOv2 features from several pre-shot
offsets (-1600..0 ms), shared attention pooling per frame, small transformer over the
sequence. Only offsets <= 0 are used so the input cannot show the shot's outcome."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class TemporalFeatureDataset(Dataset):
    """Stacks per-offset patch features; requires the shot to have all offsets cached."""

    def __init__(self, df: pd.DataFrame, feats_per_offset: dict[int, dict[str, np.ndarray]]):
        if not feats_per_offset or any(o > 0 for o in feats_per_offset):
            raise ValueError("Temporal offsets must be nonempty and all <= 0")
        self.offsets = sorted(feats_per_offset)
        keep = df["shot_id"].map(lambda s: all(s in feats_per_offset[o] for o in self.offsets))
        self.df = df[keep].reset_index(drop=True)
        self.feats = feats_per_offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        stack = np.stack([self.feats[o][r.shot_id] for o in self.offsets])  # [T, 1+P, C]
        return {"patches": torch.from_numpy(stack), "y": torch.tensor(float(r.goal))}


class TemporalXG(nn.Module):
    def __init__(self, dim: int = 384, n_heads: int = 4, dropout: float = 0.2, n_frames: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.time_pos = nn.Parameter(torch.randn(1, n_frames, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 2, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, T, P, C = patches.shape
        p = self.norm(patches.float()).reshape(B * T, P, C)
        q = self.query.expand(B * T, -1, -1)
        pooled, _ = self.pool(q, p[:, 1:], p[:, 1:])
        per_frame = pooled[:, 0].reshape(B, T, C) + p[:, 0].reshape(B, T, C)  # attn + CLS
        h = self.temporal(per_frame + self.time_pos[:, :T])
        return self.head(h.mean(1)).squeeze(-1)
