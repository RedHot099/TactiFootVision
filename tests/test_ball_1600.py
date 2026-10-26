"""ball-detection 15 (scripts/ball_1600.py): the 1600 run differs from 14's B only in resolution, the 25 fps dump's
row 1600, the size bins, the small-balls rule and the 45 px filter numbers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_1600 as b16  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402
import detector_train as dt  # noqa: E402


def test_run_is_b_at_1600():
    run, source = dt.RUNS[b16.RUN], dt.RUNS[b16.SOURCE]
    assert b16.changed_settings(run["settings"], source["settings"]) == {
        "imgsz",
        "batch_size",
    }
    assert run["settings"]["imgsz"] == 1600
    assert run["init"] == source["init"]
    assert {
        k: v for k, v in run.items() if k not in ("title", "builder", "settings")
    } == {k: v for k, v in source.items() if k not in ("title", "builder", "settings")}


def test_dump_row_1600_is_the_bench_row(monkeypatch):
    seen = {}

    class Fake:
        class_names = ["ball"]

        def predict(self, image):
            return image

    def load_model(name, weights, **kwargs):
        seen.update(kwargs)
        return Fake()

    monkeypatch.setattr(bt.tv, "load_model", load_model)
    _, _, settings = bt.dump_predictor("1600", Path("w.pt"), None)
    assert (
        settings
        == bench.ROWS["1600"]
        == {"imgsz": 1600, "iou": 0.7, "input_size": None}
    )
    assert seen["imgsz"] == 1600 and seen["iou"] == 0.7 and seen["conf"] == bt.DUMP_CONF


def test_size_bins():
    found = b16.size_bin(np.array([3.0, 7.99, 8.0, 14.0, 14.01, 40.0]))
    assert found.tolist() == ["<8", "<8", "8-14", "8-14", ">14", ">14"]


def test_ball_sizes_follow_the_rows_and_need_every_frame():
    truth = pd.DataFrame(
        {
            "clip": ["a", "a", "b"],
            "frame": [1, 6, 1],
            "tx1": [0.0, 0.0, 0.0],
            "ty1": [0.0, 0.0, 0.0],
            "tx2": [5.0, 10.0, 20.0],
            "ty2": [6.0, 3.0, 2.0],
        }
    )
    rows = pd.DataFrame({"clip": ["b", "a", "a"], "frame": [1, 6, 1]})
    assert b16.ball_sizes(rows, truth).tolist() == [">14", "8-14", "<8"]
    missing = pd.DataFrame({"clip": ["c"], "frame": [1]})
    try:
        b16.ball_sizes(missing, truth)
    except SystemExit:
        pass
    else:
        raise AssertionError("a frame without truth must stop the read")


def _entry(net: int, low: float) -> dict:
    return {"net_frames": net, "gain": {"well_placed": {"ci": [low, low + 0.1]}}}


def test_small_balls_rule():
    pair = "p"
    by_size = {
        "<8": {"frames": 100, pair: _entry(10, 0.01)},
        "8-14": {"frames": 200, pair: _entry(4, -0.01)},
        ">14": {"frames": 300, pair: _entry(1, -0.02)},
    }
    found = b16.small_balls(by_size, pair)
    assert found["sits_in_small_balls"]
    assert np.isclose(found["gain_per_frame_small"], 0.1)
    assert np.isclose(found["gain_per_frame_rest"], 5 / 500)
    assert found["net_frames_total"] == 15
    by_size["<8"][pair] = _entry(10, -0.001)  # lower bound not above zero
    assert not b16.small_balls(by_size, pair)["sits_in_small_balls"]
    by_size["<8"][pair] = _entry(1, 0.01)  # per frame not above the rest's
    by_size["8-14"][pair] = _entry(40, 0.0)
    assert not b16.small_balls(by_size, pair)["sits_in_small_balls"]


def test_filter_numbers_in_1080p_pixels():
    """Boxes are native-frame pixels whatever the imgsz: the 45 px filter reads the same long side for both rows."""
    truth = pd.DataFrame(
        {
            "frame": [1, 2],
            "tx1": [100.0, 500.0],
            "ty1": [100.0, 500.0],
            "tx2": [110.0, 520.0],
            "ty2": [110.0, 520.0],
            "radius": [10.0, 20.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "frame": [1, 1, 2, 2, 2],
            "x1": [100.0, 800.0, 501.0, 900.0, 0.0],
            "y1": [100.0, 800.0, 501.0, 900.0, 0.0],
            "x2": [112.0, 860.0, 521.0, 910.0, 50.0],
            "y2": [110.0, 820.0, 519.0, 910.0, 50.0],
            "conf": [0.9, 0.5, 0.8, 0.1, 0.3],
            "class_id": [0, 0, 0, 0, 2],
        }
    )
    table = b16.matched_balls(candidates, truth)
    assert len(table) == 3  # the 0.1 ball and the person are out
    assert table.on_truth.tolist() == [True, False, True]
    found = b16.filter_numbers(table, 45.0)
    assert found["on_truth"] == 2
    assert np.isclose(found["long_side_ratio_on_truth_median"], (1.2 + 1.0) / 2)
    assert np.isclose(found["dropped_of_all"], 1 / 3)
    assert found["dropped_on_truth"] == 0.0 and found["dropped_off_truth"] == 1.0
    # the tracker's filter drops the same long ball on a 1080p frame
    kept = bt.drop_long_balls(candidates, 45.0, 1080)
    assert 800.0 not in kept.x1.tolist() and len(kept) == 4
