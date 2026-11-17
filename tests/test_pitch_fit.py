"""The homography fit on synthetic cameras: exact recovery, noise, outliers, the weak cases, the mirror, the Jacobian."""

import numpy as np
import pytest

from line_points import marking_points, project
from tactifoot_vision.enums import PitchMarking
from tactifoot_vision.pitch import fit as S
from tactifoot_vision.pitch.markings import (
    FLIP_ACROSS,
    FLIP_ALONG,
    GEOMETRY,
    INDEX,
    MARKINGS,
    NUM_MARKINGS,
)

SIZE = (1920, 1080)
camera = S._camera


def evidence(
    matrix, size=SIZE, names=None, noise_px=0.0, per_class=20, rng=None
) -> S.Evidence:
    """What a frame of ``matrix`` shows: points of its visible markings, with noise."""
    rng = rng or np.random.default_rng(0)
    ev = S.Evidence(size)
    cls, pts = [], []
    for name in names or MARKINGS:
        xy, ok = project(matrix, marking_points(PitchMarking(name)), size)
        xy = xy[ok]
        if len(xy) < 2:
            continue
        take = np.linspace(0, len(xy) - 1, min(per_class, len(xy))).astype(int)
        pts.append(xy[take] + rng.normal(0, noise_px, (len(take), 2)))
        cls.append(np.full(len(take), INDEX[PitchMarking(name)]))
    if pts:
        ev.pt_xy = np.vstack(pts)
        ev.pt_class = np.concatenate(cls)
        ev.pt_weight = np.ones(len(ev.pt_class))
    return ev


def pitch_error(a: np.ndarray, b: np.ndarray, size=SIZE) -> float:
    """Largest distance (m) between two frame -> pitch matrices over the lower frame."""
    ph = np.column_stack([S.lower_frame(size), np.ones(20)])
    pa, pb = ph @ a.T, ph @ b.T
    return float(
        np.linalg.norm(pa[:, :2] / pa[:, 2:] - pb[:, :2] / pb[:, 2:], axis=1).max()
    )


VIEWS = {
    "centre": camera(),
    "left box": camera(target=(14.0, 34.0, 0.0), focal_px=2200),
    "right box rolled": camera(target=(92.0, 30.0, 0.0), focal_px=1800, roll_deg=3),
    "wide": camera(target=(40.0, 34.0, 0.0), focal_px=1100),
}
BOX = [
    PitchMarking.BIG_RECT_LEFT_TOP,
    PitchMarking.BIG_RECT_LEFT_BOTTOM,
    PitchMarking.BIG_RECT_LEFT_MAIN,
    PitchMarking.SMALL_RECT_LEFT_MAIN,
]
SYMMETRIC = [
    PitchMarking.CIRCLE_CENTRAL,
    PitchMarking.MIDDLE_LINE,
    PitchMarking.SIDE_LINE_TOP,
]


def test_markings_lie_on_the_standard_pitch_and_flips_are_involutions():
    assert NUM_MARKINGS == len(PitchMarking) == 20
    for m in GEOMETRY:
        if m.kind == "line":
            h = np.array([[*m.start, 1.0], [*m.end, 1.0]])
            assert np.abs(h @ m.line).max() < 1e-9, m.name
            assert 0 <= m.start.min() and m.end[0] <= 105 and m.end[1] <= 68
    for flip in (FLIP_ALONG, FLIP_ACROSS):
        assert all(flip[flip[i]] == i for i in range(NUM_MARKINGS))
    assert (
        MARKINGS[FLIP_ALONG[INDEX[PitchMarking.CIRCLE_LEFT]]]
        == PitchMarking.CIRCLE_RIGHT
    )
    assert (
        MARKINGS[FLIP_ACROSS[INDEX[PitchMarking.SIDE_LINE_TOP]]]
        == PitchMarking.SIDE_LINE_BOTTOM
    )


@pytest.mark.parametrize("view", VIEWS)
def test_exact_recovery(view):
    truth = VIEWS[view]
    fit = S.solve(evidence(truth), S.FitConfig())
    assert fit.matrix is not None, fit.reason
    assert pitch_error(fit.matrix, truth) < 1e-3


@pytest.mark.parametrize("view", VIEWS)
def test_noise_gives_a_small_error_and_honest_uncertainty(view):
    truth = VIEWS[view]
    fit = S.solve(
        evidence(truth, noise_px=1.5, rng=np.random.default_rng(1)), S.FitConfig()
    )
    assert fit.matrix is not None, fit.reason
    assert pitch_error(fit.matrix, truth) < 0.5
    assert fit.uncertainty_m < 0.5


def test_outliers_are_gated_out():
    truth = VIEWS["left box"]
    ev = evidence(truth, noise_px=0.5, rng=np.random.default_rng(2))
    rng = np.random.default_rng(3)
    bad = rng.random(len(ev.pt_xy)) < 0.15  # moved anywhere in the frame
    ev.pt_xy[bad] = rng.random((bad.sum(), 2)) * SIZE
    fit = S.solve(ev, S.FitConfig())
    assert fit.matrix is not None, fit.reason
    assert pitch_error(fit.matrix, truth) < 0.5
    assert "gated" in fit.start


def test_four_lines_two_directions_suffice():
    truth = VIEWS["left box"]
    fit = S.solve(evidence(truth, names=BOX), S.FitConfig())
    assert fit.matrix is not None, fit.reason
    assert pitch_error(fit.matrix, truth) < 1e-3


def test_minimal_linear_system_four_lines_two_points_each():
    """Eight equations (four lines, two points each) determine the matrix exactly: the 8 x 9 system has rank 8."""
    truth = VIEWS["left box"]
    ev = evidence(truth, names=BOX, per_class=2)
    assert len(ev.pt_class) == 8
    prob = S._Problem(ev, S.FitConfig())
    hn = S.line_start(ev, prob)
    assert hn is not None
    assert pitch_error(S.to_pixels(hn, prob.t_img), truth) < 1e-6


def test_parallel_lines_only_are_rejected():
    names = [
        PitchMarking.BIG_RECT_LEFT_MAIN,
        PitchMarking.SMALL_RECT_LEFT_MAIN,
        PitchMarking.SIDE_LINE_LEFT,
    ]
    assert (
        S.solve(evidence(VIEWS["left box"], names=names), S.FitConfig()).matrix is None
    )


def test_circle_and_halfway_line_alone_are_undetermined():
    """A circle and a line through its centre leave one degree of freedom: no matrix, whatever the start."""
    truth = VIEWS["centre"]
    names = [PitchMarking.CIRCLE_CENTRAL, PitchMarking.MIDDLE_LINE]
    fit = S.solve(evidence(truth, names=names), S.FitConfig(), previous=truth)
    assert fit.matrix is None
    assert fit.reason == "undetermined"


def test_symmetric_evidence_solves_from_templates_and_the_mirror_is_refused():
    """The circle, halfway line and a touchline fit the true view and its mirror image; only the true one keeps the orientation."""
    truth = VIEWS["centre"]
    ev = evidence(truth, names=SYMMETRIC)
    mirrored = np.array([[-1.0, 0, 105], [0, 1, 0], [0, 0, 1]]) @ truth
    assert S.orientation(truth, (960, 900)) > 0 > S.orientation(mirrored, (960, 900))
    fit = S.solve(ev, S.FitConfig())  # no previous matrix: the templates
    assert fit.matrix is not None, fit.reason
    assert fit.tried_templates
    assert pitch_error(fit.matrix, truth) < 1e-2
    with_previous = S.solve(ev, S.FitConfig(), previous=mirrored)
    assert (
        with_previous.matrix is None or pitch_error(with_previous.matrix, truth) < 1e-2
    )


def test_templates_only_without_a_previous_matrix(monkeypatch):
    truth = VIEWS["centre"]
    ev = evidence(truth, names=SYMMETRIC)
    calls = []
    original = S.template_starts
    monkeypatch.setattr(
        S, "template_starts", lambda prob: calls.append(1) or original(prob)
    )
    stale = np.array([[1.0, 0, 60.0], [0, 1, 0], [0, 0, 1]]) @ truth
    fit = S.solve(ev, S.FitConfig(), previous=stale)
    assert not calls and not fit.tried_templates
    S.solve(ev, S.FitConfig(), templates=False)
    assert not calls
    S.solve(ev, S.FitConfig())
    assert calls


def test_template_starts_are_at_least_10_m_apart():
    ev = evidence(VIEWS["centre"], names=SYMMETRIC)
    prob = S._Problem(ev, S.FitConfig(template_top=3))
    starts = S.template_starts(prob)
    assert len(starts) == 3
    px = np.column_stack([S.lower_frame(SIZE), np.ones(20)])
    views = []
    for _, hn in starts:
        p = px @ S.to_pixels(hn, prob.t_img).T
        views.append(p[:, :2] / p[:, 2:])
    for i in range(3):
        for j in range(i):
            assert np.nanmedian(np.linalg.norm(views[i] - views[j], axis=1)) > 10.0


def test_previous_start_is_refined_even_when_other_starts_rank_lower(monkeypatch):
    truth = camera()
    prob = S._Problem(evidence(truth), S.FitConfig())
    hn = S.to_normalised(truth, prob.t_img)
    refined = []
    original = prob.refine
    monkeypatch.setattr(
        prob, "cost", lambda batch: np.array([0.0, 1.0, 2.0])[: len(batch)]
    )
    monkeypatch.setattr(prob, "refine", lambda h0: refined.append(h0) or original(h0))
    starts = [
        ("lines", hn),
        ("lines-mirrored", hn),
        ("other", hn),
        ("previous", hn * 2),
    ]
    S._best_refined(prob, starts, S.FitConfig(), top=2)
    assert any(np.allclose(h, hn * 2) for h in refined)


@pytest.mark.parametrize("view", VIEWS)
def test_closed_form_jacobian_matches_finite_differences(view):
    truth = VIEWS[view]
    problem = S._Problem(
        evidence(truth, noise_px=1.0, rng=np.random.default_rng(5)), S.FitConfig()
    )
    hn0 = S.to_normalised(truth, problem.t_img)
    delta = np.random.default_rng(6).normal(
        0, 1e-3, 8
    )  # away from the start, as inside the LM
    h = problem.matrix(hn0, delta)
    eps = 1e-7
    steps = problem.matrix(hn0, delta[None] + eps * np.eye(8))
    numeric = (problem.residuals_of(steps) - problem.residuals_of(h)[None]).T / eps
    closed = problem.jacobian(hn0, h)
    assert np.abs(closed - numeric).max() < 1e-4 * np.abs(numeric).max()


def test_view_without_evidence_gives_none():
    fit = S.solve(S.Evidence(SIZE), S.FitConfig())
    assert fit.matrix is None and fit.reason == "no evidence"


def test_select_points_keeps_the_probable_spread_points():
    xy = np.column_stack([np.arange(40.0), np.full(40, 5.0)])
    points = S.LinePoints(
        np.zeros(40, np.int64), xy, np.r_[np.full(30, 0.9), np.full(10, 0.1)]
    )
    ev = S.select_points(points, SIZE, min_prob=0.3, per_class=10)
    assert len(ev.pt_class) == 10
    assert (
        ev.pt_xy[:, 0].min() == 0 and ev.pt_xy[:, 0].max() == 29
    )  # spread over the kept ones, ends included
    assert np.all(ev.pt_weight == np.float32(0.9))
    with pytest.raises(ValueError):
        S.LinePoints(np.array([20]), np.zeros((1, 2)), np.ones(1))  # no such marking
