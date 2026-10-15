"""Homography error in metres of a PipelineResult against a clip's calibration, on hand-computed fixtures."""

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation import HomographyMetrics, evaluate_homography
from tactifoot_vision.pipeline import FrameResult, PipelineResult
from tactifoot_vision.pitch import SoccerPitch

# A top-down camera at 8 px per pitch unit: the 840 x 544 frame shows the
# whole 105 x 68 pitch, so every landmark and grid point is in view.
FRAME = (840, 544)
TRUE = np.diag([1 / 8, 1 / 8, 1.0])  # frame -> pitch


def _shift(dx, dy=0.0):
    """Pitch -> pitch translation: a frame -> pitch homography followed by it is off by (dx, dy)."""
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def _clip(homographies, frame_size=FRAME, truth=frozenset({Truth.CALIBRATION})):
    """A labelled clip of frames 0, 1, ... with these true homographies (``None``: not calibrated)."""
    empty = sv.Detections.empty()
    return LabelledClip(
        name="fixture",
        video_path=Path("clip.mp4"),
        fps=25.0,
        width=frame_size[0],
        height=frame_size[1],
        frames=tuple(
            LabelledFrame(index, empty, empty, homography=h)
            for index, h in enumerate(homographies)
        ),
        truth=truth,
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


def test_a_point_beyond_the_predicted_horizon_is_an_infinite_error():
    # The predicted view tilts away: frame rows below 300 px fall beyond its
    # horizon, so the points seen there have no predicted position.
    tilted = TRUE @ np.array([[1, 0, 0], [0, 1, 0], [0, -1 / 300, 1.0]])
    metrics = evaluate_homography(_result([tilted, TRUE]), _clip([TRUE] * 2))

    assert metrics.mean_error_m == np.inf
    assert metrics.p90_error_m == np.inf
    assert metrics.median_error_m == np.inf  # halfway between 0 and infinity
    assert metrics.under_5m == pytest.approx(1 / 2)


def test_an_infinite_error_does_not_move_a_percentile_that_lands_on_a_finite_one():
    tilted = TRUE @ np.array([[1, 0, 0], [0, 1, 0], [0, -1 / 300, 1.0]])
    metrics = evaluate_homography(_result([tilted, TRUE, TRUE]), _clip([TRUE] * 3))

    assert metrics.median_error_m == pytest.approx(0, abs=1e-9)  # of 0, 0, inf
    assert metrics.p90_error_m == np.inf


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
    ],
)
def test_refuses_a_clip_it_cannot_score_the_result_on(clip, match):
    with pytest.raises(ValueError, match=match):
        evaluate_homography(_result([TRUE]), clip)
