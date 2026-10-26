"""ball-detection 16 (scripts/ball_xg.py): the arms' ball pick at the shot frames, the ball swap in a cached xG
frame row, the -800 ms need, the t0 agreement and the paired match bootstrap."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_xg as bx  # noqa: E402


def candidates(rows):
    """rows: (shot, offset, box, conf, fused, moved box), 720p frames, in the detector's order."""
    table = pd.DataFrame(
        [
            {"shot_id": s, "offset_ms": o, "height": 720}
            | dict(zip(bx.BOX, b, strict=True))
            | {"conf": c, "fused": f}
            | dict(zip(bx.MOVED, m, strict=True))
            for s, o, b, c, f, m in rows
        ]
    )
    return table


def test_population_keeps_training_non_penalty_shots():
    index = pd.DataFrame(
        {
            "shot_id": ["c", "a", "b", "d"],
            "game": ["g1", "g1", "g2", "g3"],
            "is_penalty": [False, False, True, False],
            "frame_path": ["p"] * 4,
        }
    )
    plan = pd.DataFrame(
        {"game": ["g1", "g2", "g3"], "role": ["training", "training", "evaluation"]}
    )
    shots = bx.population(index, plan)
    assert shots.index.tolist() == ["a", "c"]
    assert "is_penalty" not in shots


def test_choose_arms():
    small, big = (
        (100, 100, 110, 110),
        (300, 300, 340, 340),
    )  # 10 px and 40 px at 720p (15 / 60 px at 1080p)
    rows = [
        ("s", 0, big, 0.9, 0.2, big),  # X0's top-1, filtered out by the 45 px rule
        ("s", 0, small, 0.5, 0.95, (101, 101, 111, 111)),
        (
            "s",
            0,
            (500, 500, 506, 506),
            0.05,
            0.6,
            (500, 500, 506, 506),
        ),  # under 0.15 in dd2c, fused 0.6
        ("t", -400, small, 0.1, 0.1, small),  # nothing reaches 0.15
    ]
    c = candidates(rows)
    x0, p1, s2 = (bx.choose(c, arm) for arm in bx.ARMS)
    assert x0.loc[("s", 0)].x1 == 300 and x0.loc[("s", 0)].conf == 0.9
    assert p1.loc[("s", 0)].x1 == 100 and p1.loc[("s", 0)].conf == 0.5
    assert (
        s2.loc[("s", 0)].x1 == 101 and s2.loc[("s", 0)].conf == 0.95
    )  # the moved box, the fused confidence
    for arm in (x0, p1, s2):
        assert ("t", -400) not in arm.index


def test_choose_keeps_the_first_on_a_tie():
    c = candidates(
        [
            ("s", 0, (10, 10, 20, 20), 0.5, 0.5, (10, 10, 20, 20)),
            ("s", 0, (50, 50, 60, 60), 0.5, 0.5, (50, 50, 60, 60)),
        ]
    )
    assert bx.choose(c, "X0").loc[("s", 0)].x1 == 10


def frame_row(homography=None, readable=True):
    return {
        "shot_id": "s",
        "offset_ms": 0,
        "readable": readable,
        "homography": homography,
        "ball_conf": [0.3],
        "ball_xyxy": [0.0, 0.0, 4.0, 4.0],
        "ball_pitch_xy": [1.0, 1.0],
    }


def scaled(points, h):
    return np.asarray(points, float) @ np.diag([h[0, 0], h[1, 1]])


def test_swap_ball_places_the_centre_with_the_rows_homography():
    ball = pd.Series({"x1": 10.0, "y1": 20.0, "x2": 14.0, "y2": 30.0, "conf": 0.7})
    h = np.diag([2.0, 0.5, 1.0])
    out = bx.swap_ball(frame_row(h.ravel().tolist()), ball, scaled)
    assert out["ball_xyxy"] == [10.0, 20.0, 14.0, 30.0]
    assert out["ball_conf"] == [0.7]
    assert out["ball_pitch_xy"] == [24.0, 12.5]
    no_h = bx.swap_ball(frame_row(None), ball, scaled)
    assert np.isnan(no_h["ball_pitch_xy"]).all()
    none = bx.swap_ball(frame_row(h.ravel().tolist()), None, scaled)
    assert (
        none["ball_conf"] == []
        and none["ball_xyxy"] == []
        and none["ball_pitch_xy"] == []
    )
    unreadable = frame_row(readable=False)
    assert bx.swap_ball(unreadable, ball, scaled) == unreadable


def test_needs_800_when_the_m400_row_has_no_ball_on_the_pitch():
    h = np.eye(3).ravel().tolist()
    frames = pd.DataFrame(
        {
            "shot_id": ["a", "b", "c", "d"],
            "offset_ms": [-400] * 4,
            "readable": [True, True, True, False],
            "homography": [h, h, None, None],
        }
    )
    chosen = pd.DataFrame(
        {"x1": [0.0, 0.0], "conf": [0.5, 0.5]},
        index=pd.MultiIndex.from_tuples(
            [("a", -400), ("c", -400)], names=["shot_id", "offset_ms"]
        ),
    )
    assert bx.needs_800(chosen, frames) == {"b", "c", "d"}


def test_same_ball_radius():
    a = pd.DataFrame(
        {"x1": [0.0, 0.0], "y1": [0.0, 0.0], "x2": [10.0, 2.0], "y2": [10.0, 2.0]}
    )
    b = pd.DataFrame(
        {"x1": [6.0, 3.0], "y1": [0.0, 0.0], "x2": [16.0, 5.0], "y2": [10.0, 2.0]}
    )
    # first: shift 6 px <= half diagonal 7.07; second: shift 3 px > max(3 * 2/3, 1.41) = 2
    assert bx.same_ball(a, b, 2 / 3).tolist() == [True, False]


def test_paired_interval_by_match():
    games = np.array(["g1", "g1", "g2", "g3"])
    lo, hi = bx.paired_interval(np.array([1.0, 1.0, 1.0, np.nan]), games)
    assert lo == hi == 1.0
    lo, hi = bx.paired_interval(np.array([1.0, 1.0, 0.0, 0.0]), games, draws=500)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_shot_measures_and_location_change():
    t0 = pd.DataFrame(
        {
            "ball_ok": [1.0, 1.0, 1.0, 0.0],
            "ball_source": [0.0, 1.0, 2.0, np.nan],
            "shooter_ball_dist": [1.0, 3.0, np.nan, np.nan],
            "shot_ok": [1.0, 1.0, 1.0, 1.0],
            "shot_x": [90.0, 80.0, 70.0, 60.0],
            "shot_y": [30.0, 30.0, 30.0, 30.0],
        },
        index=["a", "b", "c", "d"],
    )
    m = bx.shot_measures(t0)
    assert m.own_t0.tolist() == [1, 0, 0, 0]
    assert m.fallback_m400.tolist() == [0, 1, 0, 0]
    assert m.fallback_m800.tolist() == [0, 0, 1, 0]
    assert m.no_ball.tolist() == [0, 0, 0, 1]
    assert m.shooter_near_ball.tolist()[:2] == [
        1,
        0,
    ] and m.shooter_near_ball.isna().tolist()[2:] == [True, True]
    other = t0.assign(shot_x=[90.0, 80.5, 76.0, 60.0])
    change = bx.location_change(t0, other)
    assert change.moved.tolist() == [0, 1, 1, 0]
    assert change.over_1m.tolist() == [0, 0, 1, 0]
    assert change.over_5m.tolist() == [0, 0, 1, 0]
