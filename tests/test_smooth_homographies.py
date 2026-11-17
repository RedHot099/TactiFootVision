"""The ``smooth_homographies`` post-process: the temporal filter's RTS pass over a saved result, scored in metres."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation import evaluate_homography
from tactifoot_vision.pipeline import (
    POSTPROCESSES,
    AppliedPostProcess,
    FrameResult,
    PipelineResult,
    reproject,
    smooth_homographies,
)
from tactifoot_vision.pitch import HomographyFit, SoccerPitch
from tactifoot_vision.pitch.markings import standard_to_pitch

FRAME = (840, 544)
# pitch -> frame: 8 px per metre with a mild perspective (the far side smaller)
CAMERA = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.004, 1.0]])


def _shift(dx, dy=0.0):
    """Pitch -> pitch translation."""
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def _pan(frames, speed=10.0, fps=25.0):
    """Frame -> pitch homographies of a camera panning along the pitch at ``speed`` m/s."""
    base = np.linalg.inv(CAMERA)
    return [_shift(speed * i / fps) @ base for i in range(frames)]


def _jitter(homographies, metres=0.5, seed=0):
    """Each homography off by a random pitch shift of ``metres`` standard deviation."""
    rng = np.random.default_rng(seed)
    return [_shift(*rng.normal(0, metres, 2)) @ h for h in homographies]


def _fit(matrix):
    fresh = matrix is not None
    return HomographyFit(fresh, 0, 40, 0.1 if fresh else np.inf, 1.0, "lines" if fresh else "",
                         "" if fresh else "no evidence", matrix=matrix)  # fmt: skip


def _held(matrix, age=1):
    """A frame without a fit of its own, whose estimator holds ``matrix``."""
    return HomographyFit(False, age, 0, np.inf, 0.0, "", "no evidence", matrix=matrix)


def _result(homographies, pitch=None, fits=True):
    """A PipelineResult with one player and one ball per frame, these homographies and (``fits``) the same as each frame's own fit."""
    frames = []
    for i, h in enumerate(homographies):
        people = sv.Detections(
            xyxy=np.array([[400, 200, 420, 260]], np.float32),
            tracker_id=np.array([1]),
            data={"class_name": np.array(["player"]), "team_id": np.zeros(1, int)},
        )
        balls = sv.Detections(
            xyxy=np.array([[300, 300, 306, 306], [500, 100, 506, 106]], np.float32),
            confidence=np.array([0.8, 0.3], np.float32),
            data={"class_name": np.array(["ball"] * 2)},
        )
        frames.append(
            FrameResult(
                i,
                i / 25.0,
                people,
                balls[:1],
                ball_candidates=balls,
                homography_fit=_fit(h) if fits else None,
            )  # fmt: skip
        )
    result = PipelineResult(
        frames, 25.0, FRAME, ["ball", "player"], pitch or SoccerPitch()
    )
    return reproject(result, homographies)


def _clip(homographies):
    """A labelled clip whose calibration is these homographies."""
    empty = sv.Detections.empty()
    return LabelledClip(
        name="pan",
        video_path=Path("pan.mp4"),
        fps=25.0,
        width=FRAME[0],
        height=FRAME[1],
        frames=tuple(
            LabelledFrame(i, empty, empty, homography=h)
            for i, h in enumerate(homographies)
        ),
        truth=frozenset({Truth.CALIBRATION}),
    )


def _errors(result, truth):
    """Per frame, the largest error in metres of the result's homography against the truth, over the frame."""
    corners = np.array([[0, 0], [FRAME[0], 0], [0, FRAME[1]], [*FRAME]], float)
    points = np.column_stack([corners, np.ones(4)])
    errors = []
    for frame, true in zip(result, truth, strict=True):
        got, want = points @ frame.homography.T, points @ true.T
        got, want = got[:, :2] / got[:, 2:], want[:, :2] / want[:, 2:]
        errors.append(np.linalg.norm(got - want, axis=1).max())
    return np.array(errors)


def test_a_jittered_pan_gets_smoother_and_closer_to_the_truth():
    truth = _pan(50)
    jittered = _result(_jitter(truth))

    smoothed = smooth_homographies(jittered)

    before = evaluate_homography(jittered, _clip(truth))
    after = evaluate_homography(smoothed, _clip(truth))
    assert after.jitter_m < before.jitter_m / 2
    assert after.median_error_m < before.median_error_m


def test_a_steady_pan_is_followed_without_lag():
    truth = _pan(50, speed=10.0)  # 0.4 m a frame: a lag of one frame is 0.4 m

    smoothed = smooth_homographies(_result(truth))

    assert _errors(smoothed, truth).max() < 0.01


def test_pitch_xy_is_reprojected_with_the_smoothed_homographies():
    jittered = _result(_jitter(_pan(30)))

    smoothed = smooth_homographies(jittered)

    expected = reproject(jittered, [f.homography for f in smoothed])
    for got, want, before in zip(smoothed, expected, jittered, strict=True):
        assert not np.allclose(got.homography, before.homography)
        np.testing.assert_allclose(got.pitch_xy, want.pitch_xy)
        np.testing.assert_allclose(got.ball_xy, want.ball_xy)
        np.testing.assert_allclose(
            got.ball_candidates.data["pitch_xy"], want.ball_candidates.data["pitch_xy"]
        )


def test_a_cut_starts_a_new_segment_that_nothing_is_smoothed_across():
    first = _pan(30)
    second = [_shift(40, 10) @ h for h in _pan(30)]  # another view: a cut
    truth = first + second
    result = _result(truth)
    # FrameResult.homography is set to the old view there, unlike the pass's own
    # output (the causal filter returns a rejected frame's fit), so that the test
    # sees which matrix the RTS pass falls back to: it must be homography_fit.matrix.
    for i in (30, 31):
        result.frames[i].homography = first[-1]
    smoothed = smooth_homographies(result)
    errors = _errors(smoothed, truth)
    # the frames before the cut and from the reset on are each smoothed on their own side
    assert errors[:30].max() < 0.01 and errors[32:].max() < 0.01
    # the two frames the filter rejects before its reset get their fit from before the filter
    assert errors[30:32].max() < 1e-6


def test_frames_without_a_fit_are_predicted_inside_a_segment_and_kept_outside():
    truth = _pan(40)
    result = _result(truth)
    for i in (10, 11):  # a short gap inside a segment: the filter predicts
        result.frames[i].homography_fit = _held(truth[9], i - 9)
    for i in range(20, 40):  # a long gap: the segment ends after max_gap frames
        result.frames[i].homography_fit = _held(truth[19], i - 19)

    smoothed = smooth_homographies(result, max_gap=12)

    assert _errors(smoothed.frames[:20], truth[:20]).max() < 0.01
    # outside every segment: the estimator's held matrix, from before the filter
    np.testing.assert_array_equal(smoothed.frames[39].homography, truth[19])


def test_smoothing_works_in_the_pitch_s_units():
    pitch = SoccerPitch(120, 80)
    jittered = _jitter(_pan(30), seed=3)
    scale = standard_to_pitch(pitch)

    standard = smooth_homographies(_result(jittered))
    in_units = smooth_homographies(_result([scale @ h for h in jittered], pitch=pitch))

    for a, b in zip(
        in_units, standard, strict=True
    ):  # the same filter, in metres, whatever the units
        np.testing.assert_allclose(a.homography, scale @ b.homography, atol=1e-9)


def test_a_result_without_fits_comes_back_unchanged_with_a_warning(caplog):
    result = _result(_jitter(_pan(10)), fits=False)

    with caplog.at_level(logging.WARNING):
        smoothed = smooth_homographies(result)

    assert "no per-frame fits" in caplog.text
    for got, want in zip(smoothed, result, strict=True):
        np.testing.assert_array_equal(got.homography, want.homography)


@pytest.mark.parametrize("name", ["track_ball", "join_tracks", "assign_roles"])
def test_warns_when_a_post_process_that_reads_positions_ran_before_it(name, caplog):
    result = _result(_pan(10))
    result = dataclasses.replace(result, postprocesses=[AppliedPostProcess(name, {})])

    with caplog.at_level(logging.WARNING):
        smooth_homographies(result)

    assert name in caplog.text


def test_refuses_frames_out_of_time_order():
    result = _result(_pan(5))
    result.frames.reverse()

    with pytest.raises(ValueError, match="increasing timestamps"):
        smooth_homographies(result)


def test_it_is_a_recorded_post_process_that_leaves_its_input_alone():
    result = _result(_jitter(_pan(20)))
    before = [f.homography.copy() for f in result]

    smoothed = POSTPROCESSES.get("smooth_homographies")(result, accel_m=0.15)

    assert smoothed.postprocesses == [
        AppliedPostProcess(
            "smooth_homographies",
            {  # the settings given; the rest are the ones the result records
                "accel_m": 0.15,
                "meas_m": None,
                "gate_chi2": None,
                "reset_after": None,
                "max_gap": None,
                "rate_hz": None,
            },
        )
    ]
    for frame, matrix in zip(result, before, strict=True):
        np.testing.assert_array_equal(frame.homography, matrix)


@pytest.mark.parametrize(
    ("options", "match"),
    [({"accel_m": 0.0}, "> 0"), ({"meas_m": -1.0}, "> 0"), ({"reset_after": 0}, "reset_after"), ({"max_gap": -1}, "max_gap")],
)  # fmt: skip
def test_refuses_options_it_cannot_smooth_with(options, match):
    with pytest.raises(ValueError, match=match):
        smooth_homographies(_result(_pan(10)), **options)


def test_a_gap_inside_the_run_of_rejects_still_moves_every_reject_to_the_new_view():
    first = _pan(30)
    second = [_shift(40, 10) @ h for h in _pan(30)]
    truth = first + second
    result = _result(truth)
    result.frames[31].homography_fit = _fit(None)  # reject, no fit, reject, reject
    result.frames[31].homography_fit = dataclasses.replace(
        result.frames[31].homography_fit, matrix=second[0]
    )  # the estimator holds the new view's first fit there
    for i in (30, 31):  # the old view, so the test sees which matrix the pass uses
        result.frames[i].homography = first[-1]
    smoothed = smooth_homographies(result)
    errors = _errors(smoothed, truth)
    # frame 30 (the first reject) and 31 (the gap) take the matrices from before the filter
    assert errors[30] < 1e-6
    np.testing.assert_array_equal(smoothed.frames[31].homography, second[0])
    assert errors[:30].max() < 0.01 and errors[33:].max() < 0.01


def test_a_strided_result_is_filtered_in_physical_time():
    truth = _pan(60)
    jittered = _jitter(truth, seed=5)
    strided = _result(jittered[::2])
    for k, frame in enumerate(strided.frames):  # source frames 0, 2, 4, ... at 25 fps
        frame.index, frame.timestamp = 2 * k, 2 * k / 25.0
    every = smooth_homographies(strided)
    from tactifoot_vision.pitch.temporal import HomographyFilter

    kalman = HomographyFilter(FRAME)
    for frame in strided.frames:
        kalman.step(frame.homography_fit.matrix, 2.0)
    for got, want in zip(every, kalman.smoothed(), strict=True):
        np.testing.assert_allclose(got.homography, want, rtol=1e-9, atol=1e-12)


def test_a_frame_whose_estimator_holds_nothing_ends_the_segment_as_in_the_causal_pass():
    from line_points import line_points
    from tactifoot_vision.pitch import HomographyEstimator, fit

    size = (1920, 1080)
    camera = fit._camera(target=(30.0, 34.0, 0.0), focal_px=1600)
    estimator = HomographyEstimator(max_age_s=2 / 25)  # two frames held, then nothing
    empty = sv.Detections.empty()
    frames = []
    for i, seen in enumerate([True, True, True, False, False, False]):
        points = line_points(camera, size) if seen else None
        matrix = estimator.update(points, size)
        frames.append(FrameResult(i, i / 25.0, empty, empty, ball_candidates=empty,
                                  homography=matrix, homography_fit=estimator.fit))  # fmt: skip
    result = PipelineResult(frames, 25.0, size, [], estimator.pitch)
    causal = [f.homography is not None for f in result.frames]
    final = [f.homography is not None for f in smooth_homographies(result).frames]
    assert causal == final == [True, True, True, True, True, False]
