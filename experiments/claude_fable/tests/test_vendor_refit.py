"""Per-draw refit of the cross-fitted recalibration and stacks (xg-vendor ticket 14 fix round, finding 1)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from test_vendor_ladder import primary_ladder, synthetic_table
from xgpoc.eval.calibration import logit, sigmoid
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import refit
from xgpoc.vendor.draws import table_draws


def stack_inputs(table):
    y = table.goal_sn.to_numpy(int)
    features = np.column_stack([ld.frame_feature(table.frame_logit_0.to_numpy()), logit(table.pred_freeze_frame.to_numpy())])
    return y, features, ld.match_folds(table.match.to_numpy())


def sklearn_crossfit(features: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """An independent cross-fit: sklearn with C = 1e6 at a tight tolerance."""
    out = np.empty(len(y))
    for fold in np.unique(folds):
        held_out = folds == fold
        model = LogisticRegression(C=1e6, tol=1e-12, max_iter=100_000).fit(features[~held_out], y[~held_out])
        out[held_out] = model.predict_proba(features[held_out])[:, 1]
    return out


def test_the_crossfit_is_sklearns_fit_to_convergence():
    table = synthetic_table(n_matches=30, per_match=20, frame_signal=0.3)
    y, features, folds = stack_inputs(table)
    np.testing.assert_allclose(ld.crossfit(features, y, folds)[0], sklearn_crossfit(features, y, folds), atol=1e-6)


def test_a_unit_weight_draw_reproduces_the_observed_crossfit():
    table = synthetic_table(n_matches=30, per_match=20, frame_signal=0.3)
    y, features, folds = stack_inputs(table)
    observed, coefficients = ld.crossfit(features, y, folds)
    again = refit.crossfit_draws(features, y, folds, np.ones((1, len(y))), coefficients)
    np.testing.assert_allclose(again[0], observed, atol=1e-9)


def test_a_draw_equals_crossfitting_the_resampled_matches_with_duplicates():
    table = synthetic_table(n_matches=24, per_match=15, frame_signal=0.3)
    y, features, folds = stack_inputs(table)
    _, coefficients = ld.crossfit(features, y, folds)
    draws = table_draws(table, n_boot=3, seed=7)
    weights = draws.weights(np.arange(len(table)))
    refitted = refit.crossfit_draws(features, y, folds, weights, coefficients)
    for b in range(3):   # duplicate each shot by its match's count; copies keep their match's fold
        copies = np.repeat(np.arange(len(y)), weights[b].astype(int))
        brute = sklearn_crossfit(features[copies], y[copies], folds[copies])
        np.testing.assert_allclose(refitted[b, copies], brute, atol=1e-6)


def test_a_constant_feature_gets_a_zero_slope_like_the_penalised_fit():
    y = np.r_[np.ones(10), np.zeros(30)]
    x = np.column_stack([np.ones(40), np.full(40, logit(0.12))])
    beta = refit.logistic_newton(x, y, np.ones((2, 40)), np.zeros(2))
    np.testing.assert_allclose(sigmoid(beta @ x[0]), 0.25, atol=1e-9)
    assert np.abs(beta[:, 1]).max() < 1e-6


def test_refitting_in_every_draw_widens_the_h2_interval_over_fixed_predictions():
    """Planted: a small population and a frame without signal. The stack's second coefficient is fitted on a few hundred
    shots, so its fitting noise is a visible part of the H2 spread; holding the cross-fitted predictions fixed hides it."""
    table = synthetic_table(n_matches=30, per_match=12, frame_signal=0.0, seed=3)
    result = primary_ladder(table, 0.005, 1000)
    boot = ld.bootstrap(table.match.to_numpy(), n_boot=1000)
    point, fixed = ld.h2_draws(table.goal_sn.to_numpy(), table.pred_freeze_frame.to_numpy(), table.frame_logit_0.to_numpy(),
                               table.match.to_numpy(), boot)
    h2 = result.summary["h2"]
    assert point == pytest.approx(h2["diff"], abs=1e-12)
    assert h2["boot_sd"] > 1.2 * np.std(fixed, ddof=1)
    assert result.summary["bootstrap"] == ld.BOOTSTRAP_NOTES["refit"]


def test_newton_refuses_to_report_a_fit_that_did_not_converge():
    y = np.r_[np.ones(5), np.zeros(5)]
    x = np.column_stack([np.ones(10), np.r_[np.full(5, 2.0), np.full(5, -2.0)]])   # perfectly separated
    with pytest.raises(ValueError, match="did not converge"):
        refit.logistic_newton(x, y, np.ones((1, 10)), np.zeros(2), penalty=0.0)


def test_the_per_row_auc_equals_the_fixed_auc_with_ties():
    rng = np.random.default_rng(5)
    y = rng.random(60) < 0.3
    p = np.round(rng.random(60), 1)   # many ties
    weights = rng.integers(0, 3, (4, 60)).astype(float)
    np.testing.assert_allclose(ld._auc_rows(y, np.broadcast_to(p, weights.shape), weights), ld._auc(y, p, weights), atol=1e-12)
