"""Pieces of the G-combo retraining runner that decide what counts as the same run."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_xgc_g import frame_fingerprint, recipe, verify_finished  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402

ARCHIVED_CFG = {  # cfg of results/v12_cv/g_combo/fold*/seed*/*_metrics.json, minus seed and device
    "epochs": 5, "batch_size": 8, "lr": 0.0003, "weight_decay": 0.01, "patience": 5, "max_minutes": 45, "num_workers": 4,
    "amp": True, "warmup_epochs": 1.0, "select_on": "calibrated", "ema_decay": 0.998, "aux_weight": 0.5,
    "mixup_alpha": 0.0, "llrd": 1.0}


def test_recipe_is_the_archived_g_combo_with_the_auxiliary_weight_as_the_only_knob():
    cfg, kw = recipe(0.5, smoke=False)
    fields = vars(cfg)
    assert len(fields) == 16 and set(fields) == {*ARCHIVED_CFG, "seed", "device"}  # a new field must be looked at here
    assert {k: v for k, v in fields.items() if k not in ("seed", "device")} == ARCHIVED_CFG
    assert (kw["backbone"], kw["n_blocks"], kw["mask"], kw["backbone_lr_mult"], kw["lr"]) == ("dinov2_l", 4, "overlay", 0.02, 3e-4)
    assert kw["mt"] and tuple(kw["rrc"]) == (0.85, 1.0)
    control, _ = recipe(0.0, smoke=False)
    assert control.aux_weight == 0.0 and {**vars(control), "aux_weight": 0.5} == fields
    smoke, _ = recipe(0.5, smoke=True)
    assert smoke.epochs == 2


def test_a_changed_recipe_or_backbone_is_not_resumed_as_the_same_run(tmp_path):
    run = tmp_path / "fold0" / "seed0"
    run.mkdir(parents=True)
    (run / "bundle.pt").write_bytes(b"weights")
    identity = {"fold": 0, "seed": 0, "cfg": {"aux_weight": 0.5, "batch_size": 8}, "backbone": {"sha256": "aa"}}
    record = {"identity": identity, "files": {"bundle.pt": sha256_file(run / "bundle.pt")}}
    (run / "complete.json").write_text(json.dumps(record))
    assert verify_finished(run, identity)
    for changed in ({"cfg": {"aux_weight": 0.0, "batch_size": 8}}, {"cfg": {"aux_weight": 0.5, "batch_size": 16}},
                    {"backbone": {"sha256": "bb"}}):
        with pytest.raises(ValueError, match="different run"):
            verify_finished(run, {**identity, **changed})


def test_finished_run_is_recognised_and_guarded(tmp_path):
    run = tmp_path / "fold0" / "seed0"
    run.mkdir(parents=True)
    (run / "bundle.pt").write_bytes(b"weights")
    identity = {"fold": 0, "seed": 0}
    assert not verify_finished(run, identity)  # no completion record: train
    record = {"identity": identity, "files": {"bundle.pt": sha256_file(run / "bundle.pt")}}
    (run / "complete.json").write_text(json.dumps(record))
    assert verify_finished(run, identity)
    with pytest.raises(ValueError, match="different run"):
        verify_finished(run, {"fold": 0, "seed": 1})
    (run / "bundle.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        verify_finished(run, identity)


def test_frame_fingerprint_follows_pixels_and_paths(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    df = pd.DataFrame({"frame_path": [str(a), str(b), str(a)]})
    first = frame_fingerprint(df)
    assert first["count"] == 2  # a frame shared by several shots counts once
    assert frame_fingerprint(df.iloc[::-1]) == first
    b.write_bytes(b"3")
    assert frame_fingerprint(df) != first
