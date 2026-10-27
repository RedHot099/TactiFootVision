"""ball-detection 23 (scripts/ball_filter_scale.py): the filters in 1080p pixels, person heights, the k choice."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_filter_scale as fs  # noqa: E402


def box(w, h, height=1080):
    return {"x1": 0.0, "y1": 0.0, "x2": w, "y2": h, "height": height}


def test_to_1080_scales_boxes_by_frame_height():
    table = pd.DataFrame([box(30, 20, 720), box(30, 20, 1080)])
    out = fs.to_1080(table)
    assert out.x2.tolist() == [45.0, 30.0]
    assert fs.long_side(out).tolist() == [45.0, 30.0]


def test_absolute_filter_is_resolution_independent():
    # the same physical ball: 46 px at 1080p is 30.67 px at 720p
    side = fs.long_side(fs.to_1080(pd.DataFrame([box(30.67, 20, 720)])))
    assert not fs.keep_mask(("abs", 45), side, np.array([100.0]))[0]
    assert fs.keep_mask(("abs", 45), side * 0.9, np.array([100.0]))[0]


def test_relative_filter_follows_people_height():
    side = np.array([50.0, 50.0])
    hmed = np.array([100.0, 200.0])
    keep = fs.keep_mask(("rel", 0.45), side, hmed)
    assert keep.tolist() == [False, True]  # threshold 45 vs 90
    assert fs.keep_mask(("none",), side, hmed).all()


def test_person_heights_and_fallback():
    people = pd.DataFrame(
        {
            "clip": ["a"] * 4 + ["b"] * 2,
            "frame": [1, 1, 1, 1, 1, 1],
            "x1": 0.0,
            "x2": 1.0,
            "y1": 0.0,
            "y2": [100.0, 120.0, 140.0, 5.0, 100.0, 100.0],
            "conf": [0.9, 0.9, 0.9, 0.1, 0.9, 0.9],  # the 0.1 one is under PEOPLE_CONF
        }
    )
    heights = fs.person_heights(people, ["clip", "frame"], None)
    frames = pd.DataFrame({"clip": ["a", "b", "b"], "frame": [1, 1, 2]})
    out = fs.fill_heights(frames, heights, ["clip", "frame"], "clip")
    assert out.hmed.tolist()[0] == 120.0  # median of 100, 120, 140
    assert out.n_people.tolist() == [3, 2, 0]
    assert out.hmed.tolist()[1:] == [
        120.0,
        120.0,
    ]  # < 3 people: median of the other frames


def test_top1_and_choice():
    balls = pd.DataFrame(
        {"clip": ["a", "a", "b"], "frame": [1, 1, 1], "conf": [0.2, 0.5, 0.3]}
    )
    best = fs.top1(balls, ["clip", "frame"])
    assert sorted(best.conf.tolist()) == [0.3, 0.5]
    assert fs.choose_multiple({0.6: 0.5, 0.8: 0.6, 1.0: 0.60001, 1.25: 0.4}) == 0.8
    assert fs.choose_multiple({0.6: 0.7, 0.8: 0.6}) == 0.6
