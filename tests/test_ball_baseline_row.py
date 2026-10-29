"""ball-detection 25 (scripts/ball_baseline_row.py): frame pairing, shares and controls, on synthetic rows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_baseline_row as bb  # noqa: E402


def rows(errors, clips=("A", "A", "B", "B")):
    return pd.DataFrame(
        {
            "clip": list(clips),
            "game": "2",
            "frame": [1, 6, 1, 6],
            "near_goal": False,
            "homography": True,
            "error_m": errors,
        }
    )


def write(tmp_path, top1, track):
    (tmp_path / "read_test").mkdir()
    top1.to_parquet(tmp_path / bb.PIPELINE["test"][0], index=False)
    track.to_parquet(tmp_path / bb.PIPELINE["test"][1], index=False)


def test_read_split_pairs_gains(tmp_path, monkeypatch):
    monkeypatch.setattr(bb, "split_clips", lambda split: ["A", "B"])
    base = rows([9.0, np.nan, 1.0, 9.0])
    write(tmp_path, rows([1.0, 1.0, 1.0, 9.0]), rows([1.0, 1.0, 1.0, 1.0]))
    out = bb.read_split("test", base, tmp_path)
    assert out["rows"]["base"]["well_placed"]["value"] == 0.25
    assert out["base_to_dd2c"]["well_placed"]["difference"] == pytest.approx(0.5)
    assert out["base_to_pipeline"]["well_placed"]["difference"] == pytest.approx(0.75)
    assert out["rows"]["base"]["far_of_placed"]["value"] == pytest.approx(2 / 3)


def test_read_split_rejects_other_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(bb, "split_clips", lambda split: ["A", "B"])
    other = rows([1.0] * 4)
    other["frame"] = [1, 6, 1, 11]
    write(tmp_path, other, other)
    with pytest.raises(SystemExit):
        bb.read_split("test", rows([1.0] * 4), tmp_path)


def test_check_controls_flags_mismatch():
    cell = lambda v: {"well_placed": {"value": v}}  # noqa: E731
    result = {
        "valid": {"rows": {"dd2c_top1": cell(0.6018), "pipeline": cell(0.7630)}},
        "test": {"rows": {"dd2c_top1": cell(0.6896), "pipeline": cell(0.7000)}},
        "selection": cell(0.2475),
    }
    report = bb.check_controls(result)
    assert report["test/pipeline"]["ok"] is False
    assert report["valid/dd2c_top1"]["ok"] is True
