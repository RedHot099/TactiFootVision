"""xg-stack arms and models (ticket 01, council C5): the natural spline, fold isolation of the imputer, the knots and
the scaler, recovery of a planted effect, the GBM's monotone constraint and R's fit on the raw scale."""

import numpy as np
import pytest
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import stats as ss
from xgs_fixtures import outcomes, synthetic_table


def test_natural_spline_basis_is_linear_beyond_the_boundary_knots_and_holds_the_line():
    knots = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    basis = sf.natural_spline_basis(np.linspace(-6, 20, 261), knots)
    assert basis.shape == (261, 4)
    x = np.linspace(-6, 20, 261)
    for side in (x <= 0, x >= 8):
        assert np.allclose(np.diff(basis[side], n=2, axis=0), 0.0, atol=1e-9)    # linear outside [0, 8]
    assert np.allclose(basis[:, 0], x)
    inner = (x > 0.1) & (x < 7.9)
    assert not np.allclose(np.diff(basis[inner, 1:], n=2, axis=0), 0.0)          # curved inside
    with pytest.raises(ValueError):
        sf.natural_spline_basis(x, np.array([0.0, 1.0, 1.0, 2.0]))


def test_arm_columns_are_the_linear_columns_then_four_basis_columns_per_spline():
    g = sf.ARMS["G"]
    assert g.columns[0] == sf.Z and len(g.columns) == len(g.linear) + 4 * len(g.spline)
    assert g.columns[-4:] == ["angle@0_ns1", "angle@0_ns2", "angle@0_ns3", "angle@0_ns4"]
    assert set(sf.ARMS["GS"].columns) == set(g.columns) | set(si.S_FLAGS)
    assert set(sf.ARMS["ALL"].columns) > set(sf.ARMS["GS"].columns)
    assert sf.ARMS["R"].columns == [sf.Z] and sf.ARMS["S"].columns == [sf.Z, *si.S_FLAGS]
    assert sf.GBM_COLUMNS["G"][0] == sf.Z and not set(sf.GBM_COLUMNS["G"]) & sf.INDICATORS


def test_the_preprocessor_is_fitted_on_the_training_folds_only():
    table = synthetic_table(seed=3)
    folds = table.fold.to_numpy()
    arm = sf.ARMS["G"]
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
    # the scored rows use the training fold's parameters: a missing value is the training mean, so its basis is the mean's
    x_test = after[0].x_test
    mean_row = pre(train.iloc[:1].assign(**{"dist@0": np.nan, "angle@0": np.nan, "shot_x@0": np.nan, "n_players@0": np.nan}))
    dist = [arm.columns.index(f"dist@0_ns{k}") for k in range(1, 5)]
    np.testing.assert_allclose(x_test[:, dist], np.repeat(mean_row[:, dist], len(x_test), axis=0))
    assert x_test[:, 0] == pytest.approx((50.0 - pre.centre[0]) / pre.scale[0])


def test_a_planted_distance_effect_is_recovered_and_detected():
    table = synthetic_table(n_matches=80, per_match=40, seed=4)
    folds, y = table.fold.to_numpy(), outcomes(table, beta=-0.6, seed=5)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=400, seed=0)
    p = {name: sf.out_of_fold(sf.fold_designs(table, folds, sf.ARMS[name]), y) for name in ("R", "G", "GS")}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in ("G", "GS")}
    assert deltas["G"]["delta_ll"] < -0.01 and deltas["G"]["hi95"] < 0
    assert ss.readings(deltas, ("G", "GS"))["G"]["detectable_improvement"]
    model = sf.fitted_logistic(sf.fold_designs(table, folds, sf.ARMS["G"])[0].x_train, y[folds != 0])
    dist_linear = sf.ARMS["G"].columns.index("dist@0_ns1")
    assert model.coef_[0, dist_linear] < 0                                    # farther shots score less, as planted


def test_the_gbm_is_monotone_increasing_in_the_frame_logit():
    table = synthetic_table(seed=6)
    folds, y = table.fold.to_numpy(), outcomes(table, seed=7)
    columns = sf.GBM_COLUMNS["GS"]
    model = sf.gbm(columns).fit(table[columns].to_numpy(float), y)
    rows = np.repeat(table[columns].to_numpy(float)[:20], 50, axis=0)
    rows[:, 0] = np.tile(np.linspace(-5, 1, 50), 20)
    p = model.predict_proba(rows)[:, 1].reshape(20, 50)
    assert (np.diff(p, axis=1) >= -1e-12).all()
    oof = sf.out_of_fold_gbm(table, folds, y, "G")
    assert oof.shape == y.shape and np.isfinite(oof).all()
    with pytest.raises(ValueError):
        sf.gbm(["dist@0", sf.Z])


def test_reference_fit_returns_the_raw_scale_coefficients():
    table = synthetic_table(n_matches=200, per_match=50, seed=8)
    rng = np.random.default_rng(9)
    y = (rng.random(len(table)) < sigmoid(0.4 + 1.3 * table[sf.Z].to_numpy())).astype(int)
    a, b = sf.reference_fit(table, y)
    assert a == pytest.approx(0.4, abs=0.15) and b == pytest.approx(1.3, abs=0.1)


def test_out_of_fold_scores_each_fold_with_the_other_folds_model():
    table = synthetic_table(seed=10)
    folds, y = table.fold.to_numpy(), outcomes(table, seed=11)
    designs = sf.fold_designs(table, folds, sf.ARMS["R"])
    p = sf.out_of_fold(designs, y)
    d = designs[2]
    direct = sf.fitted_logistic(d.x_train, y[d.train]).predict_proba(d.x_test)[:, 1]
    np.testing.assert_allclose(p[d.test], direct)
    assert set(d.train).isdisjoint(d.test) and np.all(folds[d.test] == 2) and np.all(folds[d.train] != 2)
