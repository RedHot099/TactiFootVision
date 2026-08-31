"""Unit tests for the pre-merge additions: overlay masking, flip TTA, ensembling,
paired significance and the leakage-audit primitives."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xgpoc.data.datasets import FILL, FeatureDataset, ImageDataset, load_image, overlay_mask  # noqa: E402
from xgpoc.models.common import TemperatureScaler, TrainConfig, predict_torch  # noqa: E402


@pytest.fixture
def frame(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), img)
    return str(p)


def _df(paths, n=4):
    return pd.DataFrame({"shot_id": [f"s{i}" for i in range(n)], "goal": [0, 1, 0, 1][:n],
                         "frame_path": [paths] * n})


# --- overlay masking -------------------------------------------------------------

def test_overlay_mask_modes_are_complementary_where_detected():
    m_over = overlay_mask((448, 252), "overlay")
    m_only = overlay_mask((448, 252), "overlay_only")
    assert m_over.shape == m_only.shape == (252, 448)
    # 'overlay_only' keeps exactly the detected graphics; 'overlay' blanks them (plus a band)
    detected = ~m_only
    assert detected.any(), "no overlay was detected - configs/overlay_mask.png looks empty"
    assert m_over[detected].all()
    assert m_over.mean() > m_only.mean() * 0  # sanity: masks are non-degenerate
    assert 0 < m_over.mean() < 0.5


def test_masking_changes_only_masked_pixels(frame):
    plain = load_image(frame, (448, 252), "none")
    masked = load_image(frame, (448, 252), "overlay")
    m = overlay_mask((448, 252), "overlay")
    assert (masked[m] == FILL).all()
    assert (masked[~m] == plain[~m]).all()


def test_top_band_is_blanked(frame):
    masked = load_image(frame, (448, 252), "overlay")
    assert (masked[:20] == FILL).all(), "scoreboard band must be gone"


# --- flip TTA --------------------------------------------------------------------

def test_image_dataset_flip_mirrors(frame):
    df = _df(frame)
    a = ImageDataset(df)[0]["image"]
    b = ImageDataset(df, flip=True)[0]["image"]
    assert torch.allclose(a.flip(-1), b)


def test_feature_dataset_alt_selects_mirrored_view():
    feats = {f"s{i}": np.zeros((3, 4), np.float32) for i in range(4)}
    alt = {f"s{i}": np.ones((3, 4), np.float32) for i in range(4)}
    df = _df("x")
    assert FeatureDataset(df, feats)[0]["patches"].sum() == 0
    assert FeatureDataset(df, feats, alt=alt)[0]["patches"].sum() == 12
    drawn = [FeatureDataset(df, feats, alt=alt, train=True)[0]["patches"].sum().item() for _ in range(60)]
    assert 0 in drawn and 12 in drawn, "training must mix both orientations"


def test_predict_torch_averages_views():
    class Const(torch.nn.Module):
        def forward(self, patches):
            return patches[:, 0, 0]

    feats = {f"s{i}": np.zeros((1, 1), np.float32) for i in range(4)}
    alt = {f"s{i}": np.full((1, 1), 4.0, np.float32) for i in range(4)}
    df = _df("x")
    cfg = TrainConfig(batch_size=2, num_workers=0, device="cpu")
    t = TemperatureScaler()
    both = predict_torch(Const(), [FeatureDataset(df, feats), FeatureDataset(df, feats, alt=alt)], cfg, t)
    # logits averaged before the sigmoid: mean(0, 4) = 2
    assert np.allclose(both, 1 / (1 + np.exp(-2.0)))


# --- fine-tuning model -----------------------------------------------------------

def test_finetune_unfreezes_only_top_blocks():
    from xgpoc.models.d_foundation import FinetuneViTXG

    m = FinetuneViTXG("dinov2_s", n_blocks=2, pretrained=False, grad_checkpointing=False)
    frozen_blocks = [b for b in m.backbone.blocks[:-2] for p in b.parameters() if p.requires_grad]
    assert not frozen_blocks
    assert all(p.requires_grad for b in m.backbone.blocks[-2:] for p in b.parameters())
    out = m(torch.randn(2, 3, 252, 448))
    assert out.shape == (2,)


# --- leakage audit primitives ----------------------------------------------------

def test_dhash_is_stable_and_discriminative(tmp_path):
    from audit_leakage import POPCOUNT, dhash

    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), a)
    cv2.imwrite(str(pb), a[:, ::-1])
    ha, sa = dhash(str(pa))
    hb, _ = dhash(str(pb))
    assert ha.shape == (8,) and sa > 0
    assert POPCOUNT[np.bitwise_xor(ha, ha)].sum() == 0
    assert POPCOUNT[np.bitwise_xor(ha, hb)].sum() > 0
    assert dhash("/nonexistent.jpg")[1] == 0.0


def test_anchored_components_rejects_centre_blobs():
    from find_overlay_region import anchored_components

    m = np.zeros((720, 1280), bool)
    m[350:400, 600:700] = True          # middle of the pitch, must be dropped
    m[10:60, 100:400] = True            # top-left overlay, must be kept
    keep = anchored_components(m)
    assert not keep[350:400, 600:700].any()
    assert keep[10:60, 100:400].all()


# --- ensembling and significance -------------------------------------------------

def test_logit_roundtrip_and_nonnegative_weights():
    from build_ensemble import _fit, logit

    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert np.allclose(1 / (1 + np.exp(-logit(p))), p, atol=1e-6)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] - x[:, 1] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    w, _ = _fit(x, y)          # the second feature is anti-correlated with y
    assert (w >= 0).all()


def test_paired_bootstrap_detects_a_strictly_better_model():
    from significance import paired_bootstrap

    rng = np.random.default_rng(0)
    n = 600
    ids = [f"s{i}" for i in range(n)]
    y = rng.binomial(1, 0.2, n)
    good = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, n), 0.01, 0.99)
    bad = np.full(n, 0.2)
    games = pd.DataFrame({"shot_id": ids, "game": [f"g{i % 20}" for i in range(n)]})
    a = pd.DataFrame({"shot_id": ids, "goal": y, "p": good})
    b = pd.DataFrame({"shot_id": ids, "goal": y, "p": bad})
    r = paired_bootstrap(a, b, games, n_boot=100)
    assert r["p_log_loss_favours_a"] > 0.95
    assert r["d_log_loss"] > 0
