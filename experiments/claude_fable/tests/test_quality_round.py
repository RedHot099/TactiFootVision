"""Tests for the post-review additions: frame-column guard, trainer selection / EMA /
multi-task / mixup, calibration helpers, layer-wise LR groups."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from xgpoc.data.datasets import FeatureDataset, ImageDataset, check_frame_col, random_crop_box
from xgpoc.eval.calibration import PlattScaler, penalty_rate, prior_shift, route_penalties
from xgpoc.models.common import TrainConfig, _param_groups, train_torch
from xgpoc.models.d_foundation import AttentionPoolHead


# --- guard against post-shot frames ------------------------------------------------

def test_frame_col_guard_accepts_pre_shot_only():
    for ok in ("frame_path", "frame_path_-400", "frame_path_-1600"):
        assert check_frame_col(ok) == ok
    for bad in ("frame_path_+400", "frame_path_+800", "clip_path", "frame_path_400"):
        with pytest.raises(ValueError):
            check_frame_col(bad)
    df = pd.DataFrame({"shot_id": ["a"], "goal": [True], "frame_path": ["x"], "frame_path_+400": ["y"]})
    with pytest.raises(ValueError):
        ImageDataset(df, path_col="frame_path_+400")
    with pytest.raises(ValueError):
        ImageDataset(df, train=True, path_cols=("frame_path", "frame_path_+400"))


def test_random_crop_box_keeps_aspect_and_bounds():
    np.random.seed(0)
    for _ in range(50):
        x0, y0, x1, y1 = random_crop_box(1280, 720, (0.85, 1.0))
        assert 0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720
        assert abs((x1 - x0) / (y1 - y0) - 1280 / 720) < 0.02
        assert (x1 - x0) * (y1 - y0) >= 0.84 * 1280 * 720


# --- calibration helpers -------------------------------------------------------------

def test_prior_shift_moves_mean_to_target_rate():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 12, 5000)               # mean ~0.143
    q = prior_shift(p, from_rate=p.mean(), to_rate=0.133)
    assert q.mean() < p.mean() and abs(q.mean() - 0.133) < 0.004
    assert np.all(np.diff(q[np.argsort(p)]) >= 0)  # ranking preserved


def test_platt_recovers_scale_and_offset():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 4000)
    y = rng.random(4000) < 1 / (1 + np.exp(-(0.5 * z - 1.0)))
    s = PlattScaler().fit(z, y)
    assert abs(s.a - 0.5) < 0.08 and abs(s.b + 1.0) < 0.12


def test_penalty_rate_and_routing():
    assert abs(penalty_rate(0, 0) - 0.76) < 1e-9
    assert 0.68 < penalty_rate(54, 84) < 0.72
    p = route_penalties(np.array([0.1, 0.2]), np.array([False, True]), 0.7)
    assert p.tolist() == [0.1, 0.7]


# --- trainer -----------------------------------------------------------------------

class _Toy(Dataset):
    """Linear signal + optional second target; `patches` shaped like a tiny token grid."""

    def __init__(self, n, seed, aux=False):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 5, 8)).astype(np.float32)
        s = self.x[:, 1:, 0].mean(1) * 6
        self.y = (rng.random(n) < 1 / (1 + np.exp(-s))).astype(np.float32)
        self.y_aux = (rng.random(n) < 1 / (1 + np.exp(-(s + 1)))).astype(np.float32)
        self.aux = aux

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        out = {"patches": torch.from_numpy(self.x[i]), "y": torch.tensor(self.y[i])}
        if self.aux:
            out["y_aux"] = torch.tensor([self.y_aux[i]])
        return out


def _cfg(**kw):
    return TrainConfig(epochs=8, batch_size=32, lr=3e-3, patience=8, num_workers=0, device="cpu", amp=False,
                       warmup_epochs=2.0, **kw)


def test_trainer_records_calibrated_selection_and_lr_cycle():
    fit = train_torch(AttentionPoolHead(8, n_queries=2, n_heads=2), _Toy(1024, 0), _Toy(512, 1), _cfg())
    h = fit["history"]
    assert {"valid_log_loss_cal", "temperature", "lr"} <= set(h[0])
    assert fit["best_epoch"] == int(np.argmin([e["valid_log_loss_cal"] for e in h]))
    lrs = [e["lr"] for e in h]
    assert int(np.argmax(lrs)) <= 2 and lrs[-1] < 0.2 * max(lrs)  # warm-up peak early, annealed by the end
    assert max(e["valid_auc"] for e in h) > 0.75


def test_trainer_ema_and_multitask_and_mixup_run():
    model = AttentionPoolHead(8, n_queries=2, n_heads=2, n_out=2)
    fit = train_torch(model, _Toy(1024, 0, aux=True), _Toy(512, 1, aux=True), _cfg(ema_decay=0.9, aux_weight=0.5))
    assert len(fit["valid_logits"]) == 512
    assert max(e["valid_auc"] for e in fit["history"]) > 0.7
    # mixup only triggers on an "image" input; make sure the code path is exercised
    class ImgToy(_Toy):
        def __getitem__(self, i):
            d = super().__getitem__(i)
            return {"image": d["patches"], "y": d["y"]}
    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(40, 1)
        def forward(self, image):
            return self.l(image.flatten(1)).squeeze(-1)
    fit = train_torch(Lin(), ImgToy(1024, 0), ImgToy(512, 1), _cfg(mixup_alpha=0.4))
    assert max(e["valid_auc"] for e in fit["history"]) > 0.7


def test_feature_dataset_aux_target():
    feats = {"a": np.zeros((3, 4), np.float32)}
    df = pd.DataFrame({"shot_id": ["a"], "goal": [True], "on_target": [1.0]})
    item = FeatureDataset(df, feats, aux_col="on_target")[0]
    assert item["y_aux"].tolist() == [1.0]


def test_layerwise_lr_groups_decay_from_top():
    from xgpoc.models.d_foundation import FinetuneViTXG
    m = FinetuneViTXG("dinov2_s", n_blocks=3, pretrained=False, grad_checkpointing=False)
    params = [p for p in m.parameters() if p.requires_grad]
    groups, max_lr = _param_groups(m, params, TrainConfig(lr=1e-3, llrd=0.5), backbone_lr_mult=0.1)
    assert len(groups) == 4 and max_lr[:3] == [1e-4, 5e-5, 2.5e-5] and max_lr[3] == 1e-3
    assert sum(len(g["params"]) for g in groups) == len(params)
