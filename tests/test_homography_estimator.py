"""HomographyEstimator on synthetic line points: recovery, the pitch's units, holding, the template backoff, settings."""

import numpy as np
import pytest

from line_points import line_points
from tactifoot_vision.enums import PitchMarking
from tactifoot_vision.pitch import (
    FitConfig,
    HomographyEstimator,
    HomographyFit,
    SoccerPitch,
    fit,
    frame_to_pitch,
)
from tactifoot_vision.pitch.markings import standard_to_pitch

SIZE = (1920, 1080)
TRUTH = fit._camera(target=(30.0, 34.0, 0.0), focal_px=1600)  # frame -> standard pitch
CIRCLE_ONLY = [
    PitchMarking.CIRCLE_CENTRAL,
    PitchMarking.MIDDLE_LINE,
]  # leaves the matrix undetermined
PROBES = np.array([[400.0, 900], [960, 700], [1500, 1000], [960, 500]])


def _error_m(matrix, truth=TRUTH) -> float:
    return float(
        np.linalg.norm(
            frame_to_pitch(PROBES, matrix) - frame_to_pitch(PROBES, truth), axis=1
        ).max()
    )


def test_the_fit_recovers_pitch_positions_and_says_how_it_went():
    estimator = HomographyEstimator()
    matrix = estimator.update(line_points(TRUTH, SIZE), SIZE)
    assert matrix is estimator.matrix and _error_m(matrix) < 1e-3
    fit_info = estimator.fit
    assert isinstance(fit_info, HomographyFit)
    assert fit_info.fresh and fit_info.age == 0 and fit_info.reason == ""
    assert (
        fit_info.points > 20
        and fit_info.uncertainty_m < 0.1
        and fit_info.inlier_share == 1.0
    )
    assert fit_info.start in (
        "lines",
        "template",
    )  # exact points: both starts end on the same matrix
    # the next frame starts from this one
    estimator.update(line_points(TRUTH, SIZE), SIZE)
    assert _error_m(estimator.matrix) < 1e-3


def test_matrices_map_to_the_pitch_s_units():
    pitch = SoccerPitch(120, 80)
    in_units = standard_to_pitch(pitch) @ TRUTH
    matrix = HomographyEstimator(pitch).update(
        line_points(in_units, SIZE, pitch=pitch), SIZE
    )
    assert _error_m(matrix, in_units) < 1e-3
    np.testing.assert_allclose(
        frame_to_pitch([[960, 700]], matrix),
        frame_to_pitch([[960, 700]], in_units),
        atol=1e-3,
    )


def test_noisy_points_and_low_probabilities():
    rng = np.random.default_rng(4)
    points = line_points(TRUTH, SIZE, noise_px=1.5, rng=rng)
    assert _error_m(HomographyEstimator().update(points, SIZE)) < 0.5
    faint = line_points(TRUTH, SIZE, prob=0.2)  # below min_prob 0.3: no evidence
    estimator = HomographyEstimator()
    assert (
        estimator.update(faint, SIZE) is None and estimator.fit.reason == "no evidence"
    )
    assert HomographyEstimator(min_prob=0.1).update(faint, SIZE) is not None


def test_with_the_filter_a_frame_without_a_fit_gets_its_prediction():
    estimator = HomographyEstimator(max_age_s=2 / 25)
    good = estimator.update(line_points(TRUTH, SIZE), SIZE)
    predicted = estimator.update(
        None, SIZE
    )  # a camera at rest: the prediction is the last state
    assert not estimator.fit.fresh and _error_m(predicted, good) < 1e-6


def test_the_last_matrix_is_held_for_max_age_s():
    estimator = HomographyEstimator(max_age_s=2 / 25, temporal_filter=None)
    good = estimator.update(line_points(TRUTH, SIZE), SIZE)
    for age in (1, 2):
        held = estimator.update(None, SIZE)
        np.testing.assert_array_equal(held, good)
        assert not estimator.fit.fresh and estimator.fit.age == age
        assert estimator.fit.reason == "no evidence"
    assert estimator.update(None, SIZE) is None
    assert (
        HomographyEstimator(max_age_s=None).update(None, SIZE) is None
    )  # nothing to hold yet


def test_the_hold_and_the_back_off_are_in_seconds_at_the_processed_rate():
    estimator = HomographyEstimator(max_age_s=2.0, backoff_max_s=1.28)
    assert (estimator._max_age, estimator._backoff_max) == (50, 32)  # 25 fps
    estimator.reset(frame_rate=12.5)  # stride 2
    assert (estimator._max_age, estimator._backoff_max) == (25, 16)
    estimator.reset(frame_rate=50.0)
    assert (estimator._max_age, estimator._backoff_max) == (100, 64)
    short = HomographyEstimator(
        max_age_s=0.01, backoff_max_s=0.01
    )  # below one frame at 25 fps
    assert (short._max_age, short._backoff_max) == (1, 1)  # still one frame, not off
    off = HomographyEstimator(max_age_s=0, backoff_max_s=0)
    assert (off._max_age, off._backoff_max) == (0, 0)
    held = HomographyEstimator(max_age_s=4 / 25, temporal_filter=None)
    held.reset(frame_rate=12.5)  # 0.16 s is two frames at 12.5 fps
    held.update(line_points(TRUTH, SIZE), SIZE)
    assert held.update(None, SIZE) is not None and held.update(None, SIZE) is not None
    assert held.update(None, SIZE) is None


def test_the_template_search_backs_off_after_it_fails(monkeypatch):
    calls = []
    original = fit.template_starts
    monkeypatch.setattr(
        fit,
        "template_starts",
        lambda prob: calls.append(estimator._frame - 1) or original(prob),
    )
    estimator = HomographyEstimator(backoff_max_s=4 / 25)
    points = line_points(TRUTH, SIZE, names=CIRCLE_ONLY)
    for _ in range(12):
        assert estimator.update(points, SIZE) is None
    assert estimator.fit.reason == "undetermined"
    assert calls == [0, 1, 3, 7, 11]  # skipped 1, 2, 4, then at most 4 frames
    calls.clear()
    estimator = HomographyEstimator(backoff_max_s=0)
    for _ in range(3):
        estimator.update(points, SIZE)
    assert calls == [0, 1, 2]


def test_templates_run_only_without_a_previous_matrix_or_when_it_fails(monkeypatch):
    estimator = HomographyEstimator()
    held = estimator.update(line_points(TRUTH, SIZE), SIZE)
    calls = []
    monkeypatch.setattr(fit, "template_starts", lambda prob: calls.append(1) or [])
    estimator.update(line_points(TRUTH, SIZE), SIZE)  # fitted from the previous matrix
    assert not calls
    # refused from the previous matrix: solved again on its own (templates once), still refused
    estimator.update(line_points(TRUTH, SIZE, names=CIRCLE_ONLY), SIZE)
    assert calls == [1] and not estimator.fit.fresh and estimator.matrix is not None
    assert held is not None
    estimator.reset()
    assert estimator.matrix is None and estimator.fit is None
    estimator.update(line_points(TRUTH, SIZE, names=CIRCLE_ONLY), SIZE)
    assert calls == [1, 1]


def test_fit_settings_come_as_a_config_or_its_fields():
    assert HomographyEstimator(fit={"gate_px": 5.0}).fit_config == FitConfig(
        gate_px=5.0
    )
    config = FitConfig(template_top=1)
    assert HomographyEstimator(fit=config).fit_config is config
    assert "HomographyEstimator(" in repr(HomographyEstimator())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_prob": -0.1}, "min_prob must be in"),
        ({"min_prob": 1.5}, "min_prob must be in"),
        ({"per_class": 1}, "per_class must be >= 2"),
        ({"max_age_s": -1}, "max_age_s must be None or >= 0"),
        ({"backoff_max_s": -1}, "backoff_max_s must be >= 0"),
    ],
)
def test_out_of_range_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HomographyEstimator(**kwargs)


def test_the_edges_of_the_ranges_are_accepted():
    HomographyEstimator(min_prob=0, max_age_s=None, backoff_max_s=0, per_class=2)
    HomographyEstimator(min_prob=1, max_age_s=0)


def test_a_frame_refused_from_the_previous_matrix_is_solved_on_its_own(monkeypatch):
    """After a cut the previous matrix can lead the fit astray: the frame is solved again without it."""
    from tactifoot_vision.pitch import homography as module

    other = fit._camera(
        position=(52.5, -40.0, 18.0), target=(70.0, 30.0, 0.0), focal_px=2600
    )
    real = module.solve
    calls = []

    def refuse_from_previous(ev, config, previous=None, *, templates=True):
        calls.append(previous is not None)
        if previous is not None:
            return fit.Fit(None, reason="no valid fit")
        return real(ev, config, None, templates=templates)

    estimator = HomographyEstimator(temporal_filter=None)
    estimator.update(line_points(TRUTH, SIZE), SIZE)
    monkeypatch.setattr(module, "solve", refuse_from_previous)
    matrix = estimator.update(line_points(other, SIZE), SIZE)
    assert calls == [True, False] and estimator.fit.fresh
    assert _error_m(matrix, other) < 1e-2
    # a fit on its own above half the rejection limit is not taken: the last matrix is held
    assert (
        estimator.restart_uncertainty_m
        == estimator.fit_config.max_uncertainty_m / 2
        == 1.0
    )
    loose = fit.Fit(other.copy(), uncertainty_m=1.5, start="template")

    def loose_alone(ev, config, previous=None, *, templates=True):
        return fit.Fit(None, reason="no valid fit") if previous is not None else loose

    strict = HomographyEstimator(temporal_filter=None)
    monkeypatch.setattr(module, "solve", real)
    good = strict.update(line_points(TRUTH, SIZE), SIZE)
    monkeypatch.setattr(module, "solve", loose_alone)
    np.testing.assert_array_equal(strict.update(line_points(other, SIZE), SIZE), good)
    assert not strict.fit.fresh
    # the threshold follows the limit
    wide = HomographyEstimator(temporal_filter=None, fit={"max_uncertainty_m": 4.0})
    monkeypatch.setattr(module, "solve", real)
    wide.update(line_points(TRUTH, SIZE), SIZE)
    monkeypatch.setattr(module, "solve", loose_alone)
    assert wide.update(line_points(other, SIZE), SIZE) is not None and wide.fit.fresh


def test_the_filter_s_prediction_never_outlives_max_age_s():
    estimator = HomographyEstimator(
        max_age_s=2 / 25
    )  # max_age_s below the filter's 12-step gap
    estimator.update(line_points(TRUTH, SIZE), SIZE)
    assert estimator.update(None, SIZE) is not None
    assert estimator.update(None, SIZE) is not None
    assert (
        estimator.update(None, SIZE) is None
    )  # the hold is dropped: so is the prediction
    assert estimator.update(None, SIZE) is None
    assert estimator.update(line_points(TRUTH, SIZE), SIZE) is not None  # a new segment


def test_the_estimator_keeps_nothing_per_frame():
    import tracemalloc

    estimator = HomographyEstimator()
    points = line_points(TRUTH, SIZE)
    for _ in range(20):
        estimator.update(points, SIZE)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(300):
        estimator.update(points, SIZE)
    grown = sum(
        s.size_diff for s in tracemalloc.take_snapshot().compare_to(before, "filename")
    )
    tracemalloc.stop()
    assert estimator._filter.records == []
    assert grown < 200_000  # 300 frames of RTS history would hold about 2 MB


def test_the_frame_rate_sets_the_filter_s_time_step():
    estimator = HomographyEstimator()
    assert estimator._dt == 1.0
    estimator.reset(frame_rate=25.0)
    assert estimator._dt == 1.0  # 25 fps, every frame: exactly the declared filter
    estimator.reset(frame_rate=12.5)
    assert estimator._dt == 2.0  # stride 2 at 25 fps
    estimator.reset(frame_rate=50.0)
    assert estimator._dt == 0.5
    with pytest.raises(ValueError, match="frame_rate"):
        estimator.reset(frame_rate=0)


def test_a_restart_fit_above_the_threshold_still_backs_off_the_template_search(
    monkeypatch,
):
    """The frame was solved on its own with templates and the fit was not taken: the back-off counts it."""
    from tactifoot_vision.pitch import homography as module

    other = fit._camera(target=(70.0, 30.0, 0.0), focal_px=2600)
    loose = fit.Fit(other, uncertainty_m=1.5, start="template", tried_templates=True)

    def loose_alone(ev, config, previous=None, *, templates=True):
        if previous is not None:
            return fit.Fit(None, reason="no valid fit")
        return loose if templates else fit.Fit(None, reason="undetermined")

    estimator = HomographyEstimator(temporal_filter=None)
    estimator.update(line_points(TRUTH, SIZE), SIZE)
    monkeypatch.setattr(module, "solve", loose_alone)
    estimator.update(
        line_points(other, SIZE), SIZE
    )  # frame 1: refused, solved alone, too loose
    assert not estimator.fit.fresh
    assert (
        estimator._skip_until == 2 and estimator._wait == 2
    )  # the back-off advanced, as after any failed search


def _shifted(matrix, dx):
    """A frame -> standard pitch matrix whose pitch positions are moved ``dx`` metres along the pitch."""
    return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]]) @ matrix


def test_a_rejected_fit_is_returned_as_it_is(monkeypatch):
    """The filter rejects a fit 8 m off; until a reset confirms another view, the frame keeps its own fit (ADR 0005).

    This is what makes the causal track depend on the training run: a network
    that gives a wrong single fit puts it on the causal track; ``final`` does
    not use it.
    """
    from tactifoot_vision.pitch import homography as module

    estimator = HomographyEstimator()
    points = line_points(TRUTH, SIZE)
    for _ in range(10):
        estimator.update(points, SIZE)
    real = module.solve
    wrong = fit.Fit(
        _shifted(TRUTH, 8.0), uncertainty_m=0.1, inlier_share=1.0, start="lines"
    )
    monkeypatch.setattr(module, "solve", lambda *a, **k: wrong)
    emitted = estimator.update(points, SIZE)
    assert estimator.fit.fresh
    assert _error_m(emitted, _shifted(TRUTH, 8.0)) < 1e-6  # the wrong fit itself
    monkeypatch.setattr(module, "solve", real)
    assert _error_m(estimator.update(points, SIZE)) < 0.05  # the filter's track resumes


def test_a_cut_the_old_view_still_partly_explains_returns_the_new_fit_at_once():
    """The third council review's case: straight markings only, a cut to the same camera moved 8 m along the pitch.

    The old prediction still puts more than half of the new points on their
    markings (the lines parallel to the shift). A rejected fit goes out as it
    is, so the new view is returned on the cut's first frame.
    """
    from tactifoot_vision.pitch.markings import GEOMETRY, INDEX

    straight = [m for m in PitchMarking if GEOMETRY[INDEX[m]].kind == "line"]
    moved = _shifted(TRUTH, 8.0)
    estimator = HomographyEstimator()
    for _ in range(10):
        estimator.update(line_points(TRUTH, SIZE, names=straight), SIZE)
    first = estimator.update(line_points(moved, SIZE, names=straight), SIZE)
    assert estimator._filter.rejects == 1  # rejected, not yet a reset
    assert _error_m(first, moved) < 0.05
    for _ in range(3):
        out = estimator.update(line_points(moved, SIZE, names=straight), SIZE)
        assert _error_m(out, moved) < 0.05


def test_a_cut_the_old_view_does_not_explain_switches_at_once():
    """After a cut the old view explains none of the new points: the new view's fit is returned on its first frame."""
    other = fit._camera(
        position=(52.5, -40.0, 18.0), target=(70.0, 30.0, 0.0), focal_px=2600
    )
    estimator = HomographyEstimator()
    for _ in range(10):
        estimator.update(line_points(TRUTH, SIZE), SIZE)
    first = estimator.update(line_points(other, SIZE), SIZE)
    assert estimator._filter.rejects == 1  # rejected, not yet a reset
    assert _error_m(first, other) < 0.05
    for _ in range(4):
        assert _error_m(estimator.update(line_points(other, SIZE), SIZE), other) < 0.05


def test_a_returned_matrix_does_not_change_with_later_frames():
    """Causality: what the estimator returned for a frame stays as it was whatever frames follow (no shared, mutated state)."""
    other = fit._camera(target=(70.0, 30.0, 0.0), focal_px=2600)
    common = [line_points(TRUTH, SIZE)] * 3 + [None] + [line_points(TRUTH, SIZE)] * 2
    futures = {
        "a cut": [line_points(other, SIZE)] * 5,
        "no fit": [None] * 5,
        "the same view": [line_points(TRUTH, SIZE)] * 5,
    }
    runs = {}
    for name, future in futures.items():
        estimator = HomographyEstimator()
        returned = [estimator.update(p, SIZE) for p in common]
        copies = [m.copy() for m in returned]
        for p in future:
            estimator.update(p, SIZE)
        for m, c in zip(
            returned, copies, strict=True
        ):  # nothing returned was changed in place
            np.testing.assert_array_equal(m, c)
        runs[name] = copies
    first = runs["a cut"]
    for (
        copies
    ) in runs.values():  # the same past gives the same outputs, whatever the future
        for x, y in zip(first, copies, strict=True):
            np.testing.assert_array_equal(x, y)
