"""PoC B: permutation-invariant transformer over positional tokens (ball + players).

Each token gets its raw canonical features plus ball-relative and goal-relative
geometry computed inside the model, so the network does not have to rediscover
distance/angle from coordinates. Attention pooling with a learned query yields the logit.
"""

import torch
from torch import nn

from xgpoc.features.cache import TOKEN_DIM

GOAL = torch.tensor([1.0, 0.5])  # canonical attacked goal in normalised coordinates


class SetTransformerXG(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM + 6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    @staticmethod
    def geometry(tokens: torch.Tensor) -> torch.Tensor:
        xy, is_ball, valid = tokens[..., :2], tokens[..., 2:3], tokens[..., 7:8]
        ball = (xy * is_ball).sum(1, keepdim=True) / is_ball.sum(1, keepdim=True).clamp(min=1)
        has_ball = (is_ball.sum(1, keepdim=True) > 0).float()
        rel_ball = (xy - ball) * has_ball
        rel_goal = GOAL.to(xy.device) - xy
        d_ball = rel_ball.norm(dim=-1, keepdim=True)
        d_goal = rel_goal.norm(dim=-1, keepdim=True)
        return torch.cat([rel_ball, rel_goal, d_ball, d_goal], -1) * valid

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        valid = tokens[..., 7] > 0
        x = self.embed(torch.cat([tokens, self.geometry(tokens)], -1))
        pad = ~valid
        pad[:, 0] = False  # never mask everything
        h = self.encoder(x, src_key_padding_mask=pad)
        q = self.query.expand(h.size(0), -1, -1)
        pooled, _ = self.pool(q, h, h, key_padding_mask=pad)
        return self.head(pooled[:, 0]).squeeze(-1)
