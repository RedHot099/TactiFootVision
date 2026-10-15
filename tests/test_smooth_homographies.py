"""The ``smooth_homographies`` post-process: lag-free smoothing of the homographies, scored in metres."""

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
from tactifoot_vision.pitch import SoccerPitch

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


def _result(homographies, fps=25.0, stride=1):
    """A PipelineResult with one player and one ball per frame and these homographies."""
    frames = []
    for i in range(len(homographies)):
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
        index = i * stride
        frames.append(
            FrameResult(index, index / fps, people, balls[:1], ball_candidates=balls)
        )
    result = PipelineResult(frames, fps, FRAME, ["ball", "player"], SoccerPitch())
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


def test_a_linear_pan_is_not_lagged():
    truth = _pan(50, speed=10.0)  # 0.4 m a frame: a lag of one frame is 0.4 m

    smoothed = smooth_homographies(_result(truth))

    assert _errors(smoothed, truth).max() < 0.01


def test_a_jittered_linear_pan_is_not_lagged_on_average():
    truth = _pan(200, speed=10.0)
    smoothed = smooth_homographies(_result(_jitter(truth)))

    centre = np.array([*FRAME, 2.0]) / 2
    along = []
    for frame, true in zip(smoothed, truth, strict=True):
        got, want = frame.homography @ centre, true @ centre
        along.append(got[0] / got[2] - want[0] / want[2])
    # a lag puts every frame behind the pan, by 0.4 m a frame of lag (1.6 m
    # for a causal average of 9 frames); jitter alone averages out
    assert abs(np.mean(along)) < 0.1


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


def test_frames_without_a_homography_keep_none_and_split_the_smoothing():
    first = _pan(20)
    # another camera after the gap: a cut the smoothing must not blend across
    second = [_shift(40, 10) @ h for h in _pan(20)]
    result = _result([*first, None, *second])

    smoothed = smooth_homographies(result)

    assert smoothed.frames[20].homography is None
    assert np.isnan(smoothed.frames[20].pitch_xy).all()
    kept = [f for f in smoothed if f.homography is not None]
    assert _errors(kept, first + second).max() < 0.01


def test_a_matrix_scaled_or_negated_smooths_like_the_matrix_itself():
    truth = _pan(30)
    scaled = [h * (-3.0 if i % 2 else 0.5) for i, h in enumerate(truth)]

    smoothed = smooth_homographies(_result(scaled))

    assert _errors(smoothed, truth).max() < 0.01


def test_the_window_is_in_seconds_whatever_the_stride():
    truth = _pan(60, speed=2.0)
    jittered = _jitter(truth)
    # 0.36 s is 9 frames at stride 1 but only 3 at stride 3: 3 frames fit
    # a quadratic exactly, so they are left as they were
    unstrided = smooth_homographies(_result(jittered), window=0.36)
    strided = smooth_homographies(_result(jittered, stride=3), window=0.36)

    assert _errors(unstrided, truth).mean() < _errors(_result(jittered), truth).mean()
    for got, want in zip(strided, jittered, strict=True):
        np.testing.assert_allclose(
            got.homography / got.homography[2, 2], want / want[2, 2]
        )


def test_a_stretch_shorter_than_the_window_is_smoothed_over_itself():
    truth = _pan(6)  # an even stretch: 5 of its frames make each window
    jittered = _result(_jitter(truth))

    smoothed = smooth_homographies(jittered, window=2.0)

    assert _errors(smoothed, truth).mean() < _errors(jittered, truth).mean()


def test_the_filter_is_centred_so_a_frame_sees_the_frames_after_it():
    still = _pan(21, speed=0.0)
    still[10] = _shift(2.0) @ still[10]  # one bad frame in a still shot

    smoothed = smooth_homographies(_result(still))

    errors = _errors(smoothed, _pan(21, speed=0.0))
    # a causal filter would leave the frames before the bad one untouched
    assert errors[6:10].min() > 0.01
    np.testing.assert_allclose(errors[6:10], errors[11:15][::-1])


def test_a_frame_corner_on_the_horizon_does_not_upset_the_scaling():
    # frame -> pitch whose top edge is the horizon: H[2, 2] is 0
    horizon = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.01, 0.0]])
    truth = [_shift(0.4 * i) @ horizon for i in range(30)]

    smoothed = smooth_homographies(_result(truth))

    centre = np.array([*FRAME, 2.0]) / 2
    for frame, true in zip(smoothed, truth, strict=True):
        got, want = frame.homography @ centre, true @ centre
        np.testing.assert_allclose(got[:2] / got[2], want[:2] / want[2], atol=1e-6)


def test_a_frame_centre_on_the_horizon_keeps_its_matrix_and_splits_the_smoothing():
    truth = _pan(41)
    # the frame centre's row is the horizon: its homogeneous coordinate is 0
    on_horizon = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 1.0, -FRAME[1] / 2]])
    homographies = [*truth[:20], on_horizon, *truth[21:]]

    smoothed = smooth_homographies(_result(homographies))

    np.testing.assert_array_equal(smoothed.frames[20].homography, on_horizon)
    kept = [f for i, f in enumerate(smoothed) if i != 20]
    assert _errors(kept, truth[:20] + truth[21:]).max() < 0.01


def test_refuses_frames_out_of_time_order():
    result = _result(_pan(5))
    result.frames.reverse()

    with pytest.raises(ValueError, match="increasing timestamps"):
        smooth_homographies(result)


def test_it_is_a_recorded_post_process_that_leaves_its_input_alone():
    result = _result(_jitter(_pan(20)))
    before = [f.homography.copy() for f in result]

    smoothed = POSTPROCESSES.get("smooth_homographies")(result)

    assert smoothed.postprocesses == [
        AppliedPostProcess("smooth_homographies", {"window": 0.36, "order": 2})
    ]
    for frame, matrix in zip(result, before, strict=True):
        np.testing.assert_array_equal(frame.homography, matrix)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"window": 0.0}, "window"),
        ({"window": -1.0}, "window"),
        ({"order": -1}, "order"),
    ],
)
def test_refuses_options_it_cannot_smooth_with(options, match):
    with pytest.raises(ValueError, match=match):
        smooth_homographies(_result(_pan(10)), **options)
