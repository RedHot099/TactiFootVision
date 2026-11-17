"""The temporal filter (``pitch/temporal.py``) and its causal use in HomographyEstimator: noise, a pan, a cut, rejects, causality."""

import numpy as np
import pytest

from line_points import line_points
from tactifoot_vision.pitch import HomographyEstimator, frame_to_pitch
from tactifoot_vision.pitch import fit as S
from tactifoot_vision.pitch.temporal import FilterConfig, HomographyFilter

SIZE = (1920, 1080)
GRID = np.array([[x, y] for x in (200, 960, 1720) for y in (600, 800, 1000)], float)
SECOND = (52.5, -40.0, 18.0)  # a camera on the other side: a cut


def _pan(frames, x0=30.0, speed=0.5):
    return [
        S._camera(target=(x0 + speed * k, 34.0, 0.0), focal_px=2800)
        for k in range(frames)
    ]


def _errors(matrices, truth):
    """Per frame, the largest pitch distance (m) over a grid of the lower frame; NaN without a matrix."""
    return np.array([
        np.nan if m is None
        else np.linalg.norm(frame_to_pitch(GRID, m) - frame_to_pitch(GRID, t), axis=1).max()
        for m, t in zip(matrices, truth, strict=True)
    ])  # fmt: skip


def _noisy(matrix, rng, px=15.0):
    """The matrix seen through image noise: the frame shifted by a small random homography."""
    shift = np.eye(3)
    shift[:2, 2] = rng.normal(0, px, 2)
    return matrix @ shift


def test_the_filter_smooths_noise_and_keeps_up_with_a_pan():
    rng = np.random.default_rng(0)
    truth = _pan(60)
    noisy = [_noisy(m, rng) for m in truth]
    kalman = HomographyFilter(SIZE, FilterConfig(accel_m=0.05, meas_m=1.0))
    causal = [kalman.step(m) for m in noisy]
    final = kalman.smoothed()
    raw = np.median(_errors(noisy, truth)[10:])
    assert np.median(_errors(final, truth)[10:]) < raw
    assert np.median(_errors(causal, truth)[10:]) < raw


def test_a_cut_resets_the_filter_and_rts_does_not_cross_it():
    a = _pan(30, speed=0.3)
    b = [S._camera(position=SECOND, target=(70.0, 30.0, 0.0), focal_px=2600)] * 30
    kalman = HomographyFilter(SIZE)
    for m in a + b:
        kalman.step(m)
    errors = _errors(kalman.smoothed(), a + b)
    assert errors[:28].max() < 0.05 and errors[34:].max() < 0.05  # each side on its own
    assert np.isnan(errors[30:32]).all()  # the rejects before the reset: in no segment
    assert (
        sum(r is not None and r[4] for r in kalman.records) >= 2
    )  # a restart after the cut


def test_causal_output_ignores_future_frames():
    a = _pan(30, speed=0.3)
    b = a[:15] + [S._camera(target=(80.0, 30.0, 0.0), focal_px=2800)] * 15
    ka, kb = HomographyFilter(SIZE), HomographyFilter(SIZE)
    out_a, out_b = [ka.step(m) for m in a], [kb.step(m) for m in b]
    for x, y in zip(out_a[:15], out_b[:15], strict=True):
        assert np.array_equal(x, y)


def test_a_rejected_frame_keeps_its_own_fit():
    kalman = HomographyFilter(SIZE)
    for m in _pan(20, speed=0.3):
        kalman.step(m)
    other = S._camera(position=SECOND, target=(80.0, 30.0, 0.0), focal_px=2600)
    out = kalman.step(
        other
    )  # the first frame of another view is rejected, not predicted
    assert _errors([out], [other])[0] < 1e-6


def test_a_matrix_beyond_the_horizon_is_no_measurement():
    kalman = HomographyFilter(SIZE)
    assert kalman.measure(None) is None
    assert kalman.measure(np.full((3, 3), np.nan)) is None
    horizon = np.array(
        [[1, 0, 0], [0, 1, 0], [0, -1 / 700, 1.0]]
    )  # the horizon at row 700: half the points behind
    assert kalman.measure(horizon) is None
    assert kalman.step(None) is None and kalman.records == [None]


def test_the_estimator_filters_causally_and_keeps_each_frame_s_own_fit():
    rng = np.random.default_rng(1)
    truth = _pan(40)
    filtered, raw = HomographyEstimator(), HomographyEstimator(temporal_filter=None)
    out_f, out_r, own = [], [], []
    for m in truth:
        points = line_points(m, SIZE, noise_px=2.0, rng=rng)
        out_f.append(filtered.update(points, SIZE))
        own.append(filtered.fit.matrix)
        out_r.append(raw.update(points, SIZE))
    # without the filter the matrix is the frame's own fit; with it, steadier
    for a, b in zip(own, out_r, strict=True):
        np.testing.assert_allclose(
            a / np.linalg.norm(a), b / np.linalg.norm(b), atol=1e-9
        )
    jitter = lambda ms: np.median(np.abs(np.diff(_errors(ms, truth[10:]))))  # noqa: E731
    assert jitter(out_f[10:]) < jitter(out_r[10:])
    assert filtered.temporal_filter == FilterConfig() and raw.temporal_filter is None
    assert (
        HomographyEstimator(temporal_filter={"accel_m": 0.15}).temporal_filter.accel_m
        == 0.15
    )


def test_a_reset_hands_the_whole_run_of_rejects_to_the_new_view():
    """reject, no fit, reject, reject: the first rejected frame must leave the old segment too."""
    a = _pan(20, speed=0.3)
    other = S._camera(position=SECOND, target=(80.0, 30.0, 0.0), focal_px=2600)
    kalman = HomographyFilter(SIZE)
    for m in a:
        kalman.step(m)
    for m in (other, None, other, other):  # the third reject resets the filter
        kalman.step(m)
    assert all(
        r is None for r in kalman.records[20:23]
    )  # the rejects and the gap between them
    assert kalman.records[23][4]  # the reset frame starts the new segment
    assert kalman.records[19] is not None


def test_time_steps_scale_the_model_and_one_step_is_today_s_filter():
    rng = np.random.default_rng(2)
    noisy = [_noisy(m, rng) for m in _pan(30)]
    default, explicit, half = (
        HomographyFilter(SIZE),
        HomographyFilter(SIZE),
        HomographyFilter(SIZE),
    )
    for m in noisy:
        assert np.array_equal(default.step(m), explicit.step(m, 1.0))
        half.step(m, 0.5)
    assert not np.allclose(default.smoothed()[-1], half.smoothed()[-1])
    F, Q = half._model(2.0)
    assert F[0, 8] == 2.0 and Q[0, 0] == pytest.approx(0.15**2 / 4 * 16)
    with pytest.raises(ValueError, match="dt"):
        default.step(noisy[0], 0.0)


def test_without_history_nothing_is_kept_and_rts_is_refused():
    kalman = HomographyFilter(SIZE, keep_history=False)
    for m in _pan(10):
        kalman.step(m)
    assert kalman.records == []
    with pytest.raises(RuntimeError, match="keep_history"):
        kalman.smoothed()


def test_the_history_packs_covariances_exactly_and_keeps_others_whole():
    """A covariance of the filter's pattern (``kron(B, I_8)``) is stored as its 2 x 2 block and comes back bit for bit; any other is kept whole."""
    from tactifoot_vision.pitch.temporal import _pack, _unpack

    block = np.array([[0.09, 0.012], [0.011, 1.44]])
    pattern = np.kron(block, np.eye(8))
    packed = _pack(pattern)
    assert isinstance(packed, tuple) and len(packed) == 4
    np.testing.assert_array_equal(_unpack(packed), pattern)
    off = pattern.copy()
    off[3, 3] += 1e-12  # one coordinate differs in the last bits
    assert _pack(off) is off and _unpack(_pack(off)) is off
    # the filter's own covariances follow the pattern, so its history is packed
    kalman = HomographyFilter(SIZE)
    for matrix in _pan(5) + [None] + _pan(3):
        kalman.step(matrix)
    assert all(
        isinstance(r[1], tuple) and isinstance(r[3], tuple) for r in kalman.records if r
    )
