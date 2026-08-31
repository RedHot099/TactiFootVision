"""PoC E: hybrid object-centric + global image model.

Inputs per shot:
  * frozen DINOv2 patch grid (from PoC D cache)
  * detector boxes (ball / GK / player / referee) from the positional cache
Each detection becomes a token: ROI-averaged patch features inside its box +
box geometry + class. Tokens attend to each other and to a CLS token carrying the
global scene; attention pooling gives the logit. The idea is to give the network
explicit "who is where" structure without trusting the homography.
"""

import torch
from torch import nn

from xgpoc.features.cache import BOX_DIM


class HybridXG(nn.Module):
    def __init__(self, dim: int = 384, grid_hw: tuple[int, int] = (18, 32), d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.grid_hw = grid_hw
        self.norm = nn.LayerNorm(dim)
        self.proj_obj = nn.Linear(dim + BOX_DIM, d_model)
        self.proj_cls = nn.Linear(dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def roi_features(self, grid: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Mean of patch tokens whose centre lies inside each box (at least the nearest patch)."""
        B, P, C = grid.shape
        H, W = self.grid_hw
        ys = (torch.arange(H, device=grid.device).float() + 0.5) / H
        xs = (torch.arange(W, device=grid.device).float() + 0.5) / W
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = cx.reshape(1, 1, P), cy.reshape(1, 1, P)
        x1, y1, x2, y2 = (boxes[..., i:i + 1] for i in range(4))
        inside = ((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)).float()  # [B, K, P]
        # guarantee at least one patch: the one nearest the box centre
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        nearest = ((cx - bx) ** 2 + (cy - by) ** 2).argmin(-1, keepdim=True)
        inside.scatter_(-1, nearest, 1.0)
        return (inside @ grid) / inside.sum(-1, keepdim=True)

    def forward(self, patches: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        p = self.norm(patches.float())
        cls, grid = p[:, 0], p[:, 1:]
        valid = boxes[..., -1] > 0
        obj = self.proj_obj(torch.cat([self.roi_features(grid, boxes), boxes], -1))
        tokens = torch.cat([self.proj_cls(cls)[:, None], obj], 1)
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        h = self.encoder(tokens, src_key_padding_mask=pad)
        pooled, _ = self.pool(self.query.expand(h.size(0), -1, -1), h, h, key_padding_mask=pad)
        return self.head(pooled[:, 0]).squeeze(-1)
