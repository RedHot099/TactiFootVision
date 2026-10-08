"""The inference benchmark times a shared frozen trunk plus per-model tails and its own
preprocessing of decoded frames; these pin both against the code they stand in for. CPU,
random weights, a small DINOv2 so the test stays fast."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def cpu_bench(monkeypatch):
    import bench_inference

    monkeypatch.setattr(bench_inference, "DEV", "cpu")
    return bench_inference


def test_trunk_then_tail_equals_full_forward_and_tails_differ(cpu_bench):
    from xgpoc.models.d_foundation import FinetuneViTXG

    torch.manual_seed(0)
    g = FinetuneViTXG("dinov2_s", n_blocks=cpu_bench.N_TRAINED_BLOCKS, n_out=2, grad_checkpointing=False,
                      pretrained=False).eval()
    x = torch.randn(2, 3, g.size_wh[1], g.size_wh[0])
    with torch.no_grad():
        feats = cpu_bench.trunk(g, x)
        assert torch.allclose(g(x), cpu_bench.Tail(g, seed=None)(feats), atol=1e-5)
        a, b = cpu_bench.Tail(g, seed=0)(feats), cpu_bench.Tail(g, seed=1)(feats)
    assert not torch.allclose(a, b)


def test_preprocess_matches_dataset_loading(cpu_bench, tmp_path):
    from xgpoc.data.datasets import load_image, to_tensor

    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)
    expected = to_tensor(load_image(str(path), (448, 252), cpu_bench.MASK))
    got = cpu_bench.Preprocess((448, 252), cpu_bench.MASK)([frame])
    assert got.shape == (1, *expected.shape) and torch.equal(got[0], expected)


def test_ensemble_adds_positional_mean_to_throughput_and_median_to_latency(cpu_bench, monkeypatch):
    monkeypatch.setattr(cpu_bench, "throughput_ms", lambda fn, frames, reps, amp: {"n": 10.0, "s": 4.0}[fn])
    monkeypatch.setattr(cpu_bench, "latency_ms", lambda fn, frames, amp: {"n": 20.0, "s": 8.0}[fn])
    pipes = {name: {"naive": "n", "shared": "s", "vit_passes": n} for name, n in (("g10", 10), ("ens", 15))}
    out = cpu_bench.configuration_times(pipes, [], batch=16, reps=1, amp=False,
                                        pos={"positional_ms": 50.0, "positional_ms_median": 60.0})
    assert out["configurations"]["g10"] == {"vit_passes": 10, "naive_ms": 10.0, "shared_ms": 4.0}
    assert out["configurations"]["ens"] == {"vit_passes": 15, "naive_ms": 60.0, "shared_ms": 54.0}
    assert out["latency_batch1"]["g10"] == {"vit_passes": 10, "naive_ms": 20.0, "shared_ms": 8.0}
    assert out["latency_batch1"]["ens"] == {"vit_passes": 15, "naive_ms": 80.0, "shared_ms": 68.0}
