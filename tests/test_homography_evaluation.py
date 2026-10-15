"""Homography error in metres of a PipelineResult against a clip's calibration or pitch positions, on hand-computed fixtures."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import supervision as sv

from tactifoot_vision.data import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation import (
    HomographyMetrics,
    evaluate_homography,
    evaluate_homography_clips,
)
from tactifoot_vision.pipeline import FrameResult, PipelineResult
from tactifoot_vision.pitch import SoccerPitch

# A top-down camera at 8 px per pitch unit: the 840 x 544 frame shows the
# whole 105 x 68 pitch, so every landmark and grid point is in view.
FRAME = (840, 544)
TRUE = np.diag([1 / 8, 1 / 8, 1.0])  # frame -> pitch


def _shift(dx, dy=0.0):
    """Pitch -> pitch translation: a frame -> pitch homography followed by it is off by (dx, dy)."""
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def _clip(
    homographies,
    frame_size=FRAME,
    truth=frozenset({Truth.CALIBRATION}),
    people=None,
):
    """A labelled clip of frames 0, 1, ... with these true homographies (``None``: not calibrated).

    ``people`` gives each frame's people as ``sv.Detections`` (none by default).
    """
    empty = sv.Detections.empty()
    people = people or [empty] * len(homographies)
    return LabelledClip(
        name="fixture",
        video_path=Path("clip.mp4"),
        fps=25.0,
        width=frame_size[0],
        height=frame_size[1],
        frames=tuple(
            LabelledFrame(index, p, empty, homography=h)
            for index, (h, p) in enumerate(zip(homographies, people, strict=True))
        ),
        truth=truth,
    )


def _people(points, true=TRUE):
    """People standing at these pitch points: tracked 0, 1, ..., feet on the whole pixel that shows them."""
    points = np.asarray(points, float).reshape(-1, 2)
    feet = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(true).T
    feet = np.round(feet[:, :2] / feet[:, 2:])
    return sv.Detections(
        xyxy=np.column_stack([feet - [5, 40], feet + [5, 0]]).astype(np.float32),
        tracker_id=np.arange(len(points)),
        data={
            "class_name": np.array(["player"] * len(points)),
            "pitch_xy": points,
        },
    )


def _positions_clip(people, homographies=None):
    """A clip whose only truth is these frames' people's pitch positions."""
    homographies = homographies or [None] * len(people)
    return _clip(
        homographies,
        truth=frozenset({Truth.BOXES, Truth.PITCH_POSITIONS}),
        people=people,
    )


def _result(homographies, pitch=None, frame_size=FRAME):
    """A PipelineResult of frames 0, 1, ... with these homographies (``None``: no fit)."""
    empty = sv.Detections.empty()
    frames = [
        FrameResult(
            index, index / 25, empty, empty, ball_candidates=empty, homography=h
        )
        for index, h in enumerate(homographies)
    ]
    return PipelineResult(
        frames, 25.0, frame_size, ["ball", "player"], pitch or SoccerPitch()
    )


def test_the_true_homography_scores_zero():
    metrics = evaluate_homography(_result([TRUE] * 3), _clip([TRUE] * 3))

    assert isinstance(metrics, HomographyMetrics)
    assert metrics.median_error_m == pytest.approx(0, abs=1e-9)
    assert metrics.mean_error_m == pytest.approx(0, abs=1e-9)
    assert metrics.p90_error_m == pytest.approx(0, abs=1e-9)
    assert metrics.jitter_m == pytest.approx(0, abs=1e-9)
    assert (metrics.under_1m, metrics.under_2m, metrics.under_5m) == (1, 1, 1)
    assert metrics.beyond_horizon == 0
    assert metrics.availability == 1
    assert metrics.num_frames == 3
    assert list(metrics.per_points.index) == ["landmarks", "grid"]
    assert np.allclose(metrics.per_points[["median_error_m", "p90_error_m"]], 0)


def test_a_known_offset_scores_exactly_that_offset_everywhere():
    # 3 m along the length and 4 m across: 5 m at every point of the pitch.
    metrics = evaluate_homography(_result([_shift(3, 4) @ TRUE] * 2), _clip([TRUE] * 2))

    for points in ("landmarks", "grid"):
        row = metrics.per_points.loc[points]
        assert row["median_error_m"] == pytest.approx(5)
        assert row["mean_error_m"] == pytest.approx(5)
        assert row["p90_error_m"] == pytest.approx(5)
        assert row["frames"] == 2
    assert metrics.mean_error_m == pytest.approx(5)
    # A constant offset does not move from frame to frame.
    assert metrics.jitter_m == pytest.approx(0, abs=1e-9)
    assert (metrics.under_1m, metrics.under_2m) == (0, 0)


def test_a_known_scale_scores_its_analytic_error():
    # Stretching the length by 2% from the corner puts a point at x 0.02 x
    # off. The landmarks and the grid are both symmetric about the halfway
    # line, so their mean x is 52.5 m and the mean error 1.05 m.
    stretch = np.diag([1.02, 1, 1])
    metrics = evaluate_homography(_result([stretch @ TRUE]), _clip([TRUE]))

    assert metrics.per_points.loc["landmarks", "mean_error_m"] == pytest.approx(1.05)
    assert metrics.mean_error_m == pytest.approx(1.05)
    assert metrics.median_error_m == pytest.approx(1.05)  # one frame
    assert (metrics.under_1m, metrics.under_2m, metrics.under_5m) == (0, 1, 1)


def test_errors_are_in_metres_on_a_pitch_in_other_units():
    # StatsBomb units: 120 x 80 for 105 x 68 m, so 8 units along the length
    # are 7 m and 10 units across are 8.5 m.
    pitch = SoccerPitch(120, 80)
    true = np.diag([120 / 840, 80 / 544, 1.0])
    for shift, metres in [(_shift(8), 7), (_shift(0, 10), 8.5)]:
        metrics = evaluate_homography(
            _result([shift @ true], pitch=pitch), _clip([true])
        )
        assert metrics.mean_error_m == pytest.approx(metres)
        assert metrics.per_points.loc["landmarks", "mean_error_m"] == pytest.approx(
            metres
        )


def test_missing_homographies_count_against_availability_not_as_errors():
    # Frame 3 is not calibrated, so it is not scored at all.
    clip = _clip([TRUE, TRUE, TRUE, None, TRUE])
    off = _shift(1.5) @ TRUE
    metrics = evaluate_homography(_result([off, None, off, off, None]), clip)

    assert metrics.num_frames == 4
    assert metrics.availability == pytest.approx(2 / 4)
    assert metrics.per_points.loc["grid", "frames"] == 2
    assert metrics.median_error_m == pytest.approx(1.5)
    assert metrics.mean_error_m == pytest.approx(1.5)
    assert (metrics.under_1m, metrics.under_2m, metrics.under_5m) == (0, 1, 1)


def test_jitter_is_the_frame_to_frame_change_of_the_error():
    # The error goes 0, 2, 0 m: it changes by 2 m at each step.
    homographies = [TRUE, _shift(2) @ TRUE, TRUE]
    metrics = evaluate_homography(_result(homographies), _clip([TRUE] * 3))

    assert metrics.jitter_m == pytest.approx(2)
    assert metrics.per_points.loc["landmarks", "jitter_m"] == pytest.approx(2)
    assert metrics.median_error_m == pytest.approx(0, abs=1e-9)
    assert metrics.mean_error_m == pytest.approx(2 / 3)
    assert metrics.p90_error_m == pytest.approx(1.6)  # 90th percentile of 0, 0, 2
    assert metrics.under_1m == pytest.approx(2 / 3)


def test_jitter_is_not_measured_across_an_uncalibrated_frame():
    # The error goes 0, 0, (frame 2 not calibrated), 2 m: frames 1 and 3 are
    # not consecutive, so only the step from frame 0 to 1 is measured.
    homographies = [TRUE, TRUE, TRUE, _shift(2) @ TRUE]
    clip = _clip([TRUE, TRUE, None, TRUE])
    metrics = evaluate_homography(_result(homographies), clip)

    assert metrics.jitter_m == pytest.approx(0, abs=1e-9)


def test_a_run_without_homographies_is_unavailable_with_no_error():
    metrics = evaluate_homography(_result([None, None]), _clip([TRUE] * 2))

    assert metrics.availability == 0
    assert np.isnan(metrics.median_error_m)
    assert np.isnan(metrics.jitter_m)
    assert metrics.per_points.loc["grid", "frames"] == 0


def test_only_points_in_view_are_scored():
    # The frame shows the left half (x up to 52.5 m), where the grid's
    # points sit at x = 0, 5, ..., 50: a 2% stretch is off by 0.5 m on average.
    half = (420, 544)
    stretch = np.diag([1.02, 1, 1])
    metrics = evaluate_homography(
        _result([stretch @ TRUE], frame_size=half), _clip([TRUE], frame_size=half)
    )

    assert metrics.mean_error_m == pytest.approx(0.5)


# The predicted view tilts away: frame rows below 300 px fall beyond its
# horizon, so the points seen there have no predicted position.
TILTED = TRUE @ np.array([[1, 0, 0], [0, 1, 0], [0, -1 / 300, 1.0]])


def test_a_frame_beyond_the_predicted_horizon_is_counted_and_left_out_of_the_mean():
    metrics = evaluate_homography(
        _result([TILTED, _shift(2) @ TRUE]), _clip([TRUE] * 2)
    )

    assert metrics.beyond_horizon == pytest.approx(1 / 2)
    assert metrics.per_points.loc["landmarks", "beyond_horizon"] == pytest.approx(1 / 2)
    # The mean is over the finite frame errors, so two runs still compare.
    assert metrics.mean_error_m == pytest.approx(2)
    # The percentiles rank the frame as an infinite error.
    assert metrics.p90_error_m == np.inf
    assert metrics.median_error_m == np.inf  # halfway between 2 and infinity
    assert metrics.under_5m == pytest.approx(1 / 2)


def test_an_infinite_error_does_not_move_a_percentile_that_lands_on_a_finite_one():
    metrics = evaluate_homography(_result([TILTED, TRUE, TRUE]), _clip([TRUE] * 3))

    assert metrics.median_error_m == pytest.approx(0, abs=1e-9)  # of 0, 0, inf
    assert metrics.p90_error_m == np.inf
    assert metrics.mean_error_m == pytest.approx(0, abs=1e-9)
    assert metrics.beyond_horizon == pytest.approx(1 / 3)


def test_the_metrics_name_the_truth_used():
    metrics = evaluate_homography(_result([TRUE]), _clip([TRUE]))

    assert metrics.truth is Truth.CALIBRATION
    assert metrics.as_dict()["truth"] == "calibration"


@pytest.mark.parametrize(
    ("clip", "match"),
    [
        (_clip([TRUE], truth=frozenset({Truth.BOXES})), "no calibration"),
        (_clip([None]), "No processed frame"),
        (_clip([]), "not in labelled clip"),
        (_clip([TRUE], frame_size=(420, 544)), "840 x 544"),
    ],
)
def test_refuses_a_clip_it_cannot_score_the_result_on(clip, match):
    with pytest.raises(ValueError, match=match):
        evaluate_homography(_result([TRUE]), clip)


# The pitch grid's 22 x 15 points, 5 m and 4.9 m apart, edges included.
GRID = np.stack(
    np.meshgrid(np.linspace(0, 105, 22), np.linspace(0, 68, 15)), axis=-1
).reshape(-1, 2)


def test_a_clip_without_calibration_is_scored_on_its_pitch_positions():
    # 3 m along the length and 4 m across: 5 m for everyone.
    people = [_people([(10, 10), (52.5, 34), (90, 60)])] * 2
    metrics = evaluate_homography(
        _result([_shift(3, 4) @ TRUE] * 2), _positions_clip(people)
    )

    assert metrics.truth is Truth.PITCH_POSITIONS
    assert metrics.as_dict()["truth"] == "pitch_positions"
    assert list(metrics.per_points.index) == ["people"]
    assert metrics.per_points.loc["people", "frames"] == 2
    assert metrics.median_error_m == pytest.approx(5)
    assert metrics.mean_error_m == pytest.approx(5)
    assert metrics.jitter_m == pytest.approx(0, abs=1e-9)
    assert metrics.availability == 1


def test_position_truth_scores_only_people_with_a_pitch_position():
    # The second person has no pitch position; a frame where nobody has one
    # has no truth, so it is not scored and does not count against availability.
    known = _people([(10, 10), (60, 30)])
    known.data["pitch_xy"][1] = np.nan
    unknown = _people([(30, 30)])
    unknown.data["pitch_xy"][:] = np.nan
    clip = _positions_clip([known, unknown, known])
    metrics = evaluate_homography(_result([_shift(2) @ TRUE, None, None]), clip)

    assert metrics.num_frames == 2
    assert metrics.availability == pytest.approx(1 / 2)
    assert metrics.mean_error_m == pytest.approx(2)


def test_position_truth_jitter_follows_each_identity():
    # The same two people swap places in the detections between frames: the
    # error at each identity changes by 2 m from one frame to the next.
    people = _people([(20, 20), (70, 40)])
    swapped = people[[1, 0]]
    homographies = [TRUE, _shift(2) @ TRUE]
    metrics = evaluate_homography(
        _result(homographies), _positions_clip([people, swapped])
    )

    assert metrics.jitter_m == pytest.approx(2)


@pytest.mark.parametrize(
    "predicted",
    [
        [_shift(3, 4) @ TRUE] * 3,
        [np.diag([1.02, 1, 1]) @ TRUE, _shift(2) @ TRUE, TRUE],
        [np.diag([0.97, 1.03, 1]) @ TRUE, None, _shift(-1, 1) @ TRUE],
    ],
    ids=["offset", "scale-and-jitter", "anisotropic-scale-and-gap"],
)
def test_position_truth_agrees_with_calibration_truth(predicted):
    # Players standing on the grid points: both truths score the same pitch
    # points, the players' through feet rounded to whole pixels (1/8 m).
    clip = _clip([TRUE] * 3, people=[_people(GRID)] * 3)
    positions = dataclasses.replace(
        clip, truth=frozenset({Truth.BOXES, Truth.PITCH_POSITIONS})
    )

    calibration = evaluate_homography(_result(predicted), clip)
    position = evaluate_homography(_result(predicted), positions)

    assert calibration.truth is Truth.CALIBRATION
    assert position.truth is Truth.PITCH_POSITIONS
    for name, value in calibration.as_dict().items():
        if name != "truth":
            assert position.as_dict()[name] == pytest.approx(
                value, abs=0.1, nan_ok=True
            ), name


def test_refuses_a_clip_with_neither_calibration_nor_pitch_positions():
    with pytest.raises(ValueError, match="no calibration or pitch positions"):
        evaluate_homography(
            _result([TRUE]), _clip([TRUE], truth=frozenset({Truth.BOXES}))
        )


def _pair(errors):
    """A (result, clip) pair whose frames are off by these errors in metres (None: no homography)."""
    predicted = [None if e is None else _shift(e) @ TRUE for e in errors]
    return _result(predicted), _clip([TRUE] * len(errors))


def test_the_aggregate_equals_evaluating_the_concatenated_clips():
    runs = [
        [TRUE, _shift(2) @ TRUE, None],
        [_shift(1) @ TRUE] * 2,
        [TILTED, _shift(3) @ TRUE],
    ]
    pairs = [(_result(run), _clip([TRUE] * len(run))) for run in runs]
    # One clip holding all three, an uncalibrated frame between them so that
    # jitter is not measured from one clip to the next.
    joined, truth = list(runs[0]), [TRUE] * len(runs[0])
    for run in runs[1:]:
        joined += [TRUE, *run]
        truth += [None] + [TRUE] * len(run)
    concatenated = evaluate_homography(_result(joined), _clip(truth))

    combined = evaluate_homography_clips(pairs, draws=50)

    assert combined.num_frames == concatenated.num_frames == 7
    assert combined.as_dict() == pytest.approx(concatenated.as_dict(), nan_ok=True)
    pd.testing.assert_frame_equal(combined.per_points, concatenated.per_points)
    assert concatenated.intervals is None


def test_the_bootstrap_is_deterministic_under_a_seed():
    pairs = [_pair([0.5, 1]), _pair([2, 2.5]), _pair([4, None]), _pair([0, 6])]

    first = evaluate_homography_clips(pairs, draws=200, seed=7)
    again = evaluate_homography_clips(pairs, draws=200, seed=7)
    other = evaluate_homography_clips(pairs, draws=200, seed=8)

    pd.testing.assert_frame_equal(first.intervals, again.intervals)
    assert not first.intervals.equals(other.intervals)
    assert list(first.intervals.columns) == ["low", "high"]
    assert list(first.intervals.index) == [
        name for name in first.as_dict() if name != "truth"
    ]
    for name, (low, high) in first.intervals.iterrows():
        assert low <= first.as_dict()[name] <= high, name
    assert (
        first.intervals.loc["median_error_m", "high"]
        > first.intervals.loc["median_error_m", "low"]
    )


def test_the_bootstrap_resamples_whole_clips():
    # Every clip is the same, so every draw is too and the interval is a point;
    # resampling frames instead would spread it.
    pairs = [_pair([0, 2, 4])] * 3

    metrics = evaluate_homography_clips(pairs, draws=100)

    assert metrics.intervals["low"].to_dict() == pytest.approx(
        metrics.intervals["high"].to_dict()
    )
    assert metrics.intervals.loc["median_error_m", "low"] == pytest.approx(2)


def test_the_aggregate_refuses_no_clips_and_mixed_truths():
    positions = _positions_clip([_people([(10, 10)])])
    with pytest.raises(ValueError, match="at least one"):
        evaluate_homography_clips([])
    with pytest.raises(ValueError, match="calibration.*pitch_positions"):
        evaluate_homography_clips([_pair([0]), (_result([TRUE]), positions)])
