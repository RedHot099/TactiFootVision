import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_v13_inputs import freeze_encoder  # noqa: E402
from xgpoc.v13.inputs import (  # noqa: E402
    InputCache,
    CONTROLS,
    VARIANTS,
    ContextDataset,
    normalize_geometry,
    roi_and_control_boxes,
    validate_offsets,
    unique_map,
)
from xgpoc.v13.models import ContextHead, variant_kwargs  # noqa: E402


def _cache() -> InputCache:
    patches = {sid: np.full((577, 1024), i, np.float16) for i, sid in enumerate(("a", "b"), 1)}
    roi = {sid: np.r_[np.full(2048, i, np.float16), [1, 1]].astype(np.float16)
           for i, sid in enumerate(("a", "b"), 1)}
    temporal = {sid: np.c_[np.full((5, 2048), i, np.float16), np.ones(5, np.float16)]
                for i, sid in enumerate(("a", "b"), 1)}
    geometry = {sid: np.array([i, 1, 1], np.float32) for i, sid in enumerate(("a", "b"), 1)}
    cache = InputCache(
        patches=patches,
        patch_valid={"a": 1.0, "b": 1.0},
        contexts={"roi": roi, "control": roi, "time_mean": temporal,
                  "time_gru": temporal, "geometry": geometry},
        context_shapes={"roi": (2050,), "control": (2050,), "time_mean": (5, 2049),
                        "time_gru": (5, 2049), "geometry": (3,)},
        metadata={},
    )
    for variant, base in CONTROLS.items():
        cache.contexts[variant] = cache.contexts[base]
        cache.context_shapes[variant] = cache.context_shapes[base]
    return cache


def test_context_dataset_preserves_order_and_missing_rows():
    df = pd.DataFrame({"shot_id": ["b", "missing", "a"], "goal": [1, 0, 0]})
    ds = ContextDataset(df, _cache(), "roi")
    assert ds.df.shot_id.tolist() == ["b", "missing", "a"]
    assert len(ds) == 3
    missing = ds[1]
    assert missing["patches"].shape == (577, 1024)
    assert missing["patches"].count_nonzero() == 0
    assert missing["context"].shape == (2050,)
    assert missing["context"][-2:].tolist() == [0.0, 0.0]


def test_positive_temporal_offset_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        validate_offsets((-1600, -400, 0, 400))


def test_identical_historical_duplicates_are_allowed_but_conflicts_fail():
    assert unique_map(["a", "a"], np.array([[1.], [1.]]), {"a"})["a"].tolist() == [1.]
    with pytest.raises(ValueError, match="Conflicting"):
        unique_map(["a", "a"], np.array([[1.], [2.]]), {"a"})


def test_roi_and_control_have_identical_area():
    boxes = np.zeros((32, 9), np.float32)
    boxes[0] = [.45, .45, .48, .49, 1, 0, 0, 0, 1]  # ball
    boxes[1] = [.42, .30, .50, .78, 0, 0, 1, 0, 1]  # nearest player
    boxes[2] = [.80, .28, .88, .72, 0, 1, 0, 0, 1]  # detected GK
    roi, control, valid = roi_and_control_boxes(boxes)
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])
    assert area(roi) == pytest.approx(area(control), abs=1e-7)
    assert valid == (1.0, 1.0)
    assert roi[0] <= .45 <= roi[2] and roi[0] <= .84 <= roi[2]


def test_geometry_normalization_uses_fit_rows_only_and_keeps_flags():
    table = pd.DataFrame({"shot_id": ["fit1", "fit2", "eval"], "distance": [2.0, 4.0, 1000.0],
                          "geometry_valid": [1.0, 0.0, 1.0]})
    normalized, stats = normalize_geometry(table, {"fit1", "fit2"}, {"geometry_valid"})
    assert stats["mean"]["distance"] == pytest.approx(3.0)
    assert stats["scale"]["distance"] == pytest.approx(np.sqrt(2.0))
    assert normalized.geometry_valid.tolist() == [1.0, 0.0, 1.0]
    assert normalized.loc[normalized.shot_id == "eval", "distance"].item() > 100


def test_frozen_encoder_keeps_eval_statistics():
    encoder = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.5)).train()
    freeze_encoder(encoder)
    before = encoder[0].running_mean.clone()
    with torch.no_grad():
        encoder(torch.full((8, 2), 10.0))
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    assert torch.equal(before, encoder[0].running_mean)


@pytest.mark.parametrize("variant", VARIANTS)
def test_context_heads_match_runner_contract(variant):
    cache = _cache()
    model = ContextHead(**variant_kwargs(variant, cache)).eval()
    item = ContextDataset(pd.DataFrame({"shot_id": ["a"], "goal": [0]}), cache, variant)[0]
    args = {k: v.unsqueeze(0) for k, v in item.items() if k != "y"}
    with torch.no_grad():
        assert model(**args).shape == (1,)


def test_temporal_control_repeats_t0_and_geometry_control_removes_signal():
    cache = _cache()
    cache.contexts["time_mean_t0"]["a"][:4, :2048] = 99.
    df = pd.DataFrame({"shot_id": ["a"], "goal": [0]})
    context = ContextDataset(df, cache, "time_mean_t0")[0]["context"]
    assert torch.equal(context[0], context[-1])
    assert context[0, 0] == 1.
    assert ContextDataset(df, cache, "geometry_zero")[0]["context"].count_nonzero() == 0
