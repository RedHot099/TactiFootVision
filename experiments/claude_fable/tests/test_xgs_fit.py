"""xg-stack arms and models (ticket 01, councils C5 and C6): the natural spline, the candidates' arms, the zero-variance
rule, the Newton solver against sklearn, the inner match-grouped CV with z unpenalised, fold isolation of the imputer,
the knots and the scaler, recovery of a planted effect, the GBM's monotone constraint and R's fit on the raw scale."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import stats as ss
from xgs_fixtures import outcomes, synthetic_table


def test_natural_spline_basis_is_linear_beyond_the_boundary_knots_and_holds_the_line():
    knots = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    x = np.linspace(-6, 20, 261)
    basis = sf.natural_spline_basis(x, knots)
    assert basis.shape == (261, 4)
    for side in (x <= 0, x >= 8):
        assert np.allclose(np.diff(basis[side], n=2, axis=0), 0.0, atol=1e-9)    # linear outside [0, 8]
    assert np.allclose(basis[:, 0], x)
    inner = (x > 0.1) & (x < 7.9)
    assert not np.allclose(np.diff(basis[inner, 1:], n=2, axis=0), 0.0)          # curved inside
    assert sf.natural_spline_basis(x, knots[[0, 1, 3, 4]]).shape == (261, 3)      # 3 df: 4 knots
    with pytest.raises(ValueError):
        sf.natural_spline_basis(x, np.array([0.0, 1.0, 1.0, 2.0]))


def test_the_candidates_hold_the_columns_c6_lists():
    table = synthetic_table()
    b = sf.spec("b", table).arms
    assert b["G"].columns == [sf.Z, "angle@0", "shot_missing@0", "ball_none@0", "dist@0_ns1", "dist@0_ns2", "dist@0_ns3"]
    assert b["G"].knot_quantiles == pytest.approx((0, 1 / 3, 2 / 3, 1))
    c = sf.spec("c", table).arms
    assert c["G"].spline == ("dist@0", "angle@0") and c["G"].df == 4 and c["G"].columns[-4:] == [f"angle@0_ns{k}" for k in range(1, 5)]
    constant = set(sf.constant_columns(table, sf.C5_LINEAR))                            # the fixture's all-zero indicators
    assert constant == {"frame_missing@0", "kp_conf_missing@0"} and set(c["G"].linear) == {sf.Z, *sf.C5_LINEAR} - constant
    bc = sf.spec("bc", table)
    assert bc.arms["G"].columns == b["G"].columns and bc.model is sf.INNER_CV and sf.spec("b", table).model is sf.FIXED
    assert set(b["Q"].linear) == set(b["G"].linear) | set(sf.QUALITY) - constant
    assert set(c["Q"].columns) == set(c["G"].columns)                                  # under (c), Q is G
    assert set(b["GS"].columns) == set(b["G"].columns) | set(si.S_FLAGS) and b["S"].columns == [sf.Z, *si.S_FLAGS]
    assert set(b["ALL"].columns) > set(b["Q"].columns) | set(b["GS"].columns)
    assert sf.spec("b", table).gbm_columns["G"] == [sf.Z, "angle@0", "ball_none@0", "dist@0"]


def test_the_zero_variance_rule_drops_constant_inputs_and_is_asserted():
    table = synthetic_table()
    table["sn_team_unknown"] = 0.0
    s = sf.spec("b", table)
    assert s.dropped["GS"] == ["sn_team_unknown"] and "sn_team_unknown" not in s.arms["GS"].columns
    assert "frame_missing@0" in s.dropped["Q"] and s.dropped["G"] == []
    assert sf.constant_columns(table.assign(x=np.nan, w=[1.0] + [np.nan] * (len(table) - 1)), ["x", "w", sf.Z]) == ["x", "w"]
    with pytest.raises(ValueError, match="constant input"):
        sf.check_no_constant(table, sf.Arm("GS", "confirmatory", (sf.Z, "sn_team_unknown")))


def test_the_newton_fit_equals_sklearn_s_l2_logistic_regression():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 5))
    y = (rng.random(800) < sigmoid(-1 + x @ [0.5, -0.3, 0.0, 0.2, 0.1])).astype(int)
    for c in (1.0, 0.01):
        beta = sf.logistic_fit(x, y, np.full(5, 1 / c))
        model = LogisticRegression(C=c, max_iter=5000, tol=1e-10).fit(x, y)
        np.testing.assert_allclose(beta, np.r_[model.intercept_, model.coef_[0]], atol=1e-5)
    free = sf.logistic_fit(x, y, sf.INNER_CV.penalties(5, 0.001))                       # z (column 0) unpenalised
    assert abs(free[1]) > 0.3 and np.abs(free[2:]).max() < 0.05
    np.testing.assert_array_equal(sf.FIXED.penalties(3, 1.0), [1.0, 1.0, 1.0])


def test_the_inner_cv_uses_the_outer_training_rows_only_and_picks_by_inner_log_loss():
    table = synthetic_table(n_matches=50, per_match=30, seed=2)
    folds = table.fold.to_numpy()
    arm = sf.spec("c", table).arms["G"]
    designs = sf.fold_designs(table, folds, arm, inner=True)
    d = designs[0]
    assert len(d.inner) == 4 and all(set(i.train) | set(i.test) == set(d.train) for i in d.inner)
    assert all(set(i.test) == set(np.flatnonzero(folds == i.fold)) for i in d.inner)
    y = outcomes(table, seed=3)
    changed = y.copy()
    changed[d.test] = 1 - changed[d.test]                                  # the scored fold's outcomes
    assert sf.chosen_c(d, y, sf.INNER_CV) == sf.chosen_c(d, changed, sf.INNER_CV)
    losses = {}
    for c in sf.C_GRID:
        total = 0.0
        for i in d.inner:
            beta = sf.logistic_fit(i.x_train, y[i.train], sf.INNER_CV.penalties(i.x_train.shape[1], c))
            total += sf.mean_log_loss(y[i.test], sf.predict(beta, i.x_test)) * len(i.test)
        losses[c] = total
    assert sf.chosen_c(d, y, sf.INNER_CV) == min(sf.C_GRID, key=lambda c: (losses[c], c))
    assert sf.chosen_c(d, y, sf.INNER_CV) == pytest.approx(0.001)          # pure noise: the strongest penalty
    r = sf.fold_designs(table, folds, sf.spec("c", table).arms["R"])[0]
    assert sf.chosen_c(r, y, sf.INNER_CV) == 1.0 and not r.inner


def test_the_preprocessor_is_fitted_on_the_training_folds_only():
    table = synthetic_table(seed=3)
    folds = table.fold.to_numpy()
    arm = sf.spec("c", table).arms["G"]
    before = sf.fold_designs(table, folds, arm)
    changed = table.copy()
    scored = folds == 0
    changed.loc[scored, ["dist@0", "angle@0", "shot_x@0", "n_players@0"]] = np.nan      # the scored fold's inputs
    changed.loc[scored, sf.Z] = 50.0
    after = sf.fold_designs(changed, folds, arm)
    np.testing.assert_array_equal(before[0].x_train, after[0].x_train)
    train = table.iloc[before[0].train]
    pre = sf.Preprocessor.fit(arm, train)
    assert pre.means["dist@0"] == pytest.approx(train["dist@0"].mean())
    np.testing.assert_allclose(pre.knots["dist@0"], np.quantile(train["dist@0"].dropna(), [0, 0.25, 0.5, 0.75, 1]))
    np.testing.assert_allclose(before[0].x_train.mean(axis=0)[before[0].x_train.std(axis=0) > 0], 0.0, atol=1e-9)
    x_test = after[0].x_test
    mean_row = pre(train.iloc[:1].assign(**{"dist@0": np.nan, "angle@0": np.nan, "shot_x@0": np.nan, "n_players@0": np.nan}))
    dist = [arm.columns.index(f"dist@0_ns{k}") for k in range(1, 5)]
    np.testing.assert_allclose(x_test[:, dist], np.repeat(mean_row[:, dist], len(x_test), axis=0))
    assert x_test[:, 0] == pytest.approx((50.0 - pre.centre[0]) / pre.scale[0])


@pytest.mark.parametrize("candidate", ["b", "c", "bc"])
def test_a_planted_distance_effect_is_recovered_and_detected(candidate):
    table = synthetic_table(n_matches=80, per_match=40, seed=4)
    folds, y = table.fold.to_numpy(), outcomes(table, beta=-0.6, seed=5)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=400, seed=0)
    spec = sf.spec(candidate, table)
    designs = sf.all_designs(table, folds, spec, ("R", "G", "GS"))
    p = {name: sf.out_of_fold(designs[name], y, spec.model) for name in ("R", "G", "GS")}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in ("G", "GS")}
    assert deltas["G"]["delta_ll"] < -0.01 and deltas["G"]["hi95"] < 0
    assert ss.readings(deltas, ("G", "GS"))["G"]["detectable_improvement"]
    d = designs["G"][0]
    beta = sf.logistic_fit(d.x_train, y[d.train], spec.model.penalties(d.x_train.shape[1], sf.chosen_c(d, y, spec.model)))
    assert beta[1 + spec.arms["G"].columns.index("dist@0_ns1")] < 0           # farther shots score less, as planted


def test_the_gbm_is_monotone_increasing_in_the_frame_logit():
    table = synthetic_table(seed=6)
    folds, y = table.fold.to_numpy(), outcomes(table, seed=7)
    columns = sf.spec("c", table).gbm_columns["GS"]
    model = sf.gbm(columns).fit(table[columns].to_numpy(float), y)
    rows = np.repeat(table[columns].to_numpy(float)[:20], 50, axis=0)
    rows[:, 0] = np.tile(np.linspace(-5, 1, 50), 20)
    p = model.predict_proba(rows)[:, 1].reshape(20, 50)
    assert (np.diff(p, axis=1) >= -1e-12).all()
    oof = sf.out_of_fold_gbm(table, folds, y, columns)
    assert oof.shape == y.shape and np.isfinite(oof).all()
    with pytest.raises(ValueError):
        sf.gbm(["dist@0", sf.Z])


def test_reference_fits_return_the_raw_scale_coefficients():
    table = synthetic_table(n_matches=200, per_match=50, seed=8)
    rng = np.random.default_rng(9)
    y = (rng.random(len(table)) < sigmoid(0.4 + 1.3 * table[sf.Z].to_numpy())).astype(int)
    for a, b in (sf.reference_fit(table, y, sf.FIXED), sf.reference_fit(table, y, sf.INNER_CV), sf.unpenalised_reference(table, y)):
        assert a == pytest.approx(0.4, abs=0.15) and b == pytest.approx(1.3, abs=0.1)
    assert sf.reference_fit(table, y, sf.INNER_CV) == pytest.approx(sf.unpenalised_reference(table, y), abs=1e-6)


def test_out_of_fold_scores_each_fold_with_the_other_folds_model():
    table = synthetic_table(seed=10)
    folds, y = table.fold.to_numpy(), outcomes(table, seed=11)
    spec = sf.spec("c", table)
    designs = sf.fold_designs(table, folds, spec.arms["G"], inner=True)
    chosen = []
    p = sf.out_of_fold(designs, y, spec.model, chosen)
    d = designs[2]
    beta = sf.logistic_fit(d.x_train, y[d.train], spec.model.penalties(d.x_train.shape[1], chosen[2]))
    np.testing.assert_allclose(p[d.test], sf.predict(beta, d.x_test))
    assert len(chosen) == 5 and set(chosen) <= set(sf.C_GRID)
    assert set(d.train).isdisjoint(d.test) and np.all(folds[d.test] == 2) and np.all(folds[d.train] != 2)
