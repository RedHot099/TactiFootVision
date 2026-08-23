import numpy as np
import pandas as pd
import torch

from xgpoc.features.cache import BOX_DIM, MAX_BOXES, MAX_TOKENS, TOKEN_DIM
from xgpoc.models.b_set_transformer import SetTransformerXG
from xgpoc.models.common import TrainConfig, train_torch
from xgpoc.models.d_foundation import AttentionPoolHead
from xgpoc.models.e_hybrid import HybridXG


def test_forward_shapes():
    b = 3
    assert SetTransformerXG()(torch.rand(b, MAX_TOKENS, TOKEN_DIM)).shape == (b,)
    assert AttentionPoolHead(32)(torch.rand(b, 1 + 6 * 8, 32)).shape == (b,)
    boxes = torch.zeros(b, MAX_BOXES, BOX_DIM)
    boxes[:, :2, :4] = torch.tensor([[0.1, 0.1, 0.3, 0.4], [0.5, 0.5, 0.6, 0.9]])
    boxes[:, :2, -1] = 1
    assert HybridXG(32, grid_hw=(6, 8))(torch.rand(b, 1 + 6 * 8, 32), boxes).shape == (b,)


def _synthetic_tokens(n, rng):
    """Ball + 10 players; goal iff ball within 0.2 (normalised) of the attacked goal."""
    toks = np.zeros((n, MAX_TOKENS, TOKEN_DIM), np.float32)
    y = np.zeros(n, bool)
    for i in range(n):
        ball = rng.uniform([0.5, 0.2], [1.0, 0.8])
        toks[i, 0] = [*ball, 1, 0, 0, 0, 1, 1]
        for j in range(1, 11):
            toks[i, j] = [*rng.uniform(0, 1, 2), 0, 0, j % 2, 1 - j % 2, 1, 1]
        y[i] = np.hypot(1 - ball[0], 0.5 - ball[1]) < 0.2
    return toks, y


def test_set_transformer_learns_distance_rule():
    from xgpoc.data.datasets import TokenDataset
    rng = np.random.default_rng(0)
    toks, y = _synthetic_tokens(600, rng)
    ids = [f"s{i}" for i in range(600)]
    df = pd.DataFrame({"shot_id": ids, "goal": y})
    tokens = dict(zip(ids, toks))
    cfg = TrainConfig(epochs=15, batch_size=64, lr=2e-3, patience=15, num_workers=0, device="cpu", amp=False)
    fit = train_torch(SetTransformerXG(d_model=32), TokenDataset(df.iloc[:480], tokens, train=True), TokenDataset(df.iloc[480:], tokens), cfg)
    assert max(h["valid_auc"] for h in fit["history"]) > 0.85
