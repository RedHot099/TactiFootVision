"""ball-detection 20 (scripts/ball_xg_tracked.py): the causal window, the anchors, the alignment rule, the controls and
how arm T's frames take the tracked ball (only where the track has one, placed by the frame's own homography)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_xg_tracked as bt20  # noqa: E402


def test_window_starts_after_the_last_cut_or_two_seconds_back():
    assert bt20.window_start(-1) == -50
    assert bt20.window_start(12) == -12
    assert bt20.window_start(0) == 0  # the t0 frame is the first after the cut


def test_anchors_every_fifth_frame_without_the_cached_rows():
    assert bt20.anchor_frames(-50, {0, -10}) == [
        -50,
        -45,
        -40,
        -35,
        -30,
        -25,
        -20,
        -15,
        -5,
    ]
    assert bt20.anchor_frames(-12, {0, -10, -20}) == [-5]
    assert bt20.anchor_frames(0, {0}) == []


def test_alignment_needs_shift_zero_best_and_close():
    assert bt20.is_aligned({-2: 3.0, -1: 2.0, 0: 1.5})
    assert bt20.is_aligned(
        {-2: 1.4, -1: 1.6, 0: 1.8}
    )  # within the tolerance (a still camera)
    assert not bt20.is_aligned({-2: 1.0, -1: 4.0, 0: 9.0})
    assert not bt20.is_aligned({-1: 7.0, 0: 7.0})  # equal but far: another image
    assert not bt20.is_aligned({-1: 1.0, 0: np.nan})


def _track(rows):
    return pd.DataFrame(
        rows, columns=["shot_id", "k", "status", "x1", "y1", "x2", "y2", "conf"]
    )


def test_tracked_ball_is_none_without_a_ball():
    track = _track([("a", 0, "none", np.nan, np.nan, np.nan, np.nan, np.nan)])
    assert bt20.tracked_ball(track, 0) is None
    assert bt20.tracked_ball(track, -10) is None


def _row(shot, offset, ball=True, readable=True, homography=True):
    h = np.eye(3).ravel().tolist() if homography else None
    return {
        "shot_id": shot,
        "offset_ms": offset,
        "readable": readable,
        "homography": h,
        "ball_conf": [0.5] if ball else [],
        "ball_xyxy": [10.0, 10.0, 14.0, 14.0] if ball else [],
        "ball_pitch_xy": [12.0, 12.0] if ball else [],
    }


def _project(points, h):
    return (
        np.asarray(points, float) + 1.0
    )  # a planted homography: every point moves by (1, 1)


def test_t_frames_swap_only_where_the_track_has_a_ball():
    frames = pd.DataFrame(
        [_row("a", 0), _row("a", -400), _row("b", 0, ball=False), _row("c", 0)]
    )
    tracks = _track(
        [
            ("a", 0, "measured", 100, 100, 104, 104, 0.3),
            ("a", -10, "none", np.nan, np.nan, np.nan, np.nan, np.nan),
            ("b", 0, "interpolated", 20, 20, 24, 26, np.nan),
        ]
    )
    out, sources = bt20.t_frames(frames, tracks, _project)
    out = out.set_index(["shot_id", "offset_ms"])
    assert list(out.loc[("a", 0), "ball_xyxy"]) == [100, 100, 104, 104]
    assert list(out.loc[("a", 0), "ball_pitch_xy"]) == [103, 103]
    assert list(out.loc[("a", -400), "ball_xyxy"]) == [
        10,
        10,
        14,
        14,
    ]  # no ball tracked: F's stays
    assert list(out.loc[("b", 0), "ball_pitch_xy"]) == [
        23,
        24,
    ]  # a new (interpolated) ball
    assert np.isnan(out.loc[("b", 0), "ball_conf"][0])
    assert list(out.loc[("c", 0), "ball_xyxy"]) == [10, 10, 14, 14]  # no T for c
    assert sources.source.tolist() == [
        "tracked_measured",
        "cached",
        "tracked_interpolated",
        "cached",
    ]


def test_t_frames_leave_unreadable_rows_and_a_missing_homography_gives_nan():
    frames = pd.DataFrame(
        [_row("a", 0, readable=False), _row("b", 0, homography=False)]
    )
    tracks = _track(
        [
            ("a", 0, "measured", 1, 1, 3, 3, 0.9),
            ("b", 0, "measured", 1, 1, 3, 3, 0.9),
        ]
    )
    out, sources = bt20.t_frames(frames, tracks, _project)
    assert sources.source.tolist() == ["cached", "tracked_measured"]
    assert np.isnan(out.iloc[1].ball_pitch_xy).all()


def test_homography_gap_is_zero_for_equal_and_grows_with_a_shift():
    h = np.diag([0.05, 0.05, 1.0])
    assert bt20.homography_gap(h, h) == pytest.approx(0.0)
    moved = np.array([[0.05, 0, 2.0], [0, 0.05, 0], [0, 0, 1.0]])
    assert bt20.homography_gap(moved, h) == pytest.approx(2.0)


def test_top1_matches_cache():
    cand = pd.DataFrame(
        {
            "k": [0, 0, -1],
            "class_id": [0, 0, 0],
            "conf": [0.2, 0.6, 0.9],
            "x1": [0.0, 50.0, 0.0],
            "y1": [0.0, 50.0, 0.0],
            "x2": [4.0, 54.0, 4.0],
            "y2": [4.0, 54.0, 4.0],
        }
    )
    assert bt20.top1_matches(cand, [50.5, 50.5, 54.5, 54.5]) == 1.0
    assert bt20.top1_matches(cand, [0, 0, 4, 4]) == 0.0
    assert bt20.top1_matches(cand, []) == 0.0
    assert bt20.top1_matches(cand[cand.k != 0], []) == 1.0


def test_window_result_filters_long_balls_and_keeps_people_over_the_threshold():
    pytest.importorskip("tactifoot_vision")
    cand = pd.DataFrame(
        {
            "k": [-1, -1, -1, 0, 0],
            "x1": [10.0, 100.0, 300.0, 12.0, 400.0],
            "y1": [10.0, 100.0, 300.0, 12.0, 400.0],
            "x2": [16.0, 160.0, 340.0, 18.0, 440.0],
            "y2": [16.0, 160.0, 420.0, 18.0, 520.0],
            "conf": [0.4, 0.9, 0.8, 0.5, 0.2],
            "class_id": [0, 0, 2, 0, 2],
        }
    )
    result = bt20.window_result(cand, [-1, 0], [np.eye(3), None], 45.0)
    first, last = result.frames
    assert (
        len(first.ball_candidates) == 1
    )  # the 60 px box (90 px at 1080p) is filtered out
    assert (
        len(first.detections) == 1 and len(last.detections) == 0
    )  # 0.2 < people threshold
    assert last.timestamp == pytest.approx(1 / 25)
    assert np.isnan(last.ball_candidates.data["pitch_xy"]).all()


def test_first_cut_after_stops_before_the_cut_and_at_the_last_frame():
    grey = np.full((72, 128, 3), 90, np.uint8)
    other = np.zeros((72, 128, 3), np.uint8)
    other[..., 1] = 255
    frames = {k: grey for k in range(-3, 6)} | {6: other, 7: other}
    assert bt20.first_cut_after(frames) == 5
    assert bt20.first_cut_after({k: grey for k in range(0, 4)}) == 3
    assert (
        bt20.first_cut_after({0: grey, 1: grey, 3: grey}) == 1
    )  # only the run of frames from t0
