"""The report figures of xg-detector 07 (scripts/xgd_report_figures.py): planted residuals and the record check."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import xgd_report_figures as figs  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402


def planted() -> pd.DataFrame:
    return pd.DataFrame({"shot_id": ["a", "a", "b", "c", "a", "b"],
                         "input": ["stretch", "stretch", "stretch", "stretch", "letterbox", "letterbox"],
                         "vertex": [1, 2, 1, 3, 1, 1], "px": [5.0, 15.0, 25.0, 50.0, 80.0, 90.0]})


def test_keypoint_figures_on_planted_residuals():
    out = figs.keypoint_figures(planted())
    assert out["frames_total"] == 3
    assert out["stretch"] == {"residuals": 4, "frames": 3, "median_px": 20.0, "share_lt10": 0.25}
    assert out["letterbox"] == {"residuals": 2, "frames": 2, "median_px": 85.0, "share_lt10": 0.0}


def write_residuals(folder: Path, recorded: str | None = None) -> Path:
    folder.mkdir()
    path = folder / "keypoint_residuals.csv"
    planted().to_csv(path, index=False)
    (folder / "complete.json").write_text(json.dumps({"files": {path.name: recorded or sha256_file(path)}}))
    return path


def test_main_writes_figures_with_a_record(tmp_path, monkeypatch):
    path = write_residuals(tmp_path / "dev")
    monkeypatch.setattr(figs, "ROOT", tmp_path)          # the source is named relative to the experiment
    figs.main(["--residuals", str(path), "--out", str(tmp_path / "out")])
    written = json.loads((tmp_path / "out/figures.json").read_text())
    assert written["keypoint_residuals"]["stretch"]["median_px"] == 20.0
    assert written["source"] == {"path": "dev/keypoint_residuals.csv", "sha256": sha256_file(path)}
    record = json.loads((tmp_path / "out/complete.json").read_text())
    assert record["files"] == {"figures.json": sha256_file(tmp_path / "out/figures.json")}


def test_main_refuses_residuals_that_differ_from_their_record(tmp_path):
    path = write_residuals(tmp_path / "dev", recorded="0" * 64)
    with pytest.raises(SystemExit, match="completion record"):
        figs.main(["--residuals", str(path), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
