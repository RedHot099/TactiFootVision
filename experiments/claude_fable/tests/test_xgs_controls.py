"""Controls 2-4 of xg-stack (ticket 01, council C5): the β solver, the oracle, the permutation and the summaries."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.compare.stats import permute_within_matches
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgs_fixtures import outcomes, standardised_distance, synthetic_table


def test_beta_solves_the_true_delta_and_is_negative():
    table = synthetic_table(n_matches=100, per_match=40, seed=20)
    z, d = table[sf.Z].to_numpy(), sc.signal(table)
    np.testing.assert_allclose(d, standardised_distance(table))
    assert sc.true_delta(0.0, -0.2, 1.1, z, d) == pytest.approx(0.0, abs=1e-12)
    beta = sc.solve_beta(-0.2, 1.1, z, d)
    assert beta < 0 and sc.true_delta(beta, -0.2, 1.1, z, d) == pytest.approx(sc.POSITIVE_TARGET, abs=1e-9)
    assert sc.true_delta(beta * 1.5, -0.2, 1.1, z, d) < sc.POSITIVE_TARGET               # a stronger signal, a larger gain


def test_expected_log_loss_is_the_mean_bernoulli_log_loss():
    p, q = np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.7])
    direct = -np.mean(p * np.log(q) + (1 - p) * np.log(1 - q))
    assert sc.expected_log_loss(p, q) == pytest.approx(direct)


def designs_of(table):
    folds = table.fold.to_numpy()
    return {name: sf.fold_designs(table, folds, sf.ARMS[name]) for name in ("R", *sf.CONFIRMATORY)}


def test_the_positive_control_reports_beta_power_and_the_oracle():
    table = synthetic_table(n_matches=60, per_match=40, seed=21)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    frame, summary = sc.positive_control(designs_of(table), weights, table, -0.1, 1.0, n=8, seed=3)
    assert len(frame) == 8 and summary["beta"] < 0 and summary["true_delta"] == pytest.approx(-0.002, abs=1e-9)
    assert 0 <= summary["power_G"] <= 1 and summary["passed"] == (-0.003 <= summary["mean_recovered_delta_G"] <= -0.001)
    assert summary["diagnostic_mean_oracle_delta"] < 0                       # knowing the truth beats R


def test_the_synthetic_nulls_summary_applies_the_criterion():
    table = synthetic_table(n_matches=40, per_match=30, seed=22)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    p_r = sigmoid(table[sf.Z].to_numpy())
    frame, summary = sc.synthetic_nulls(designs_of(table), weights, p_r, n=10, seed=4)
    assert len(frame) == 10 and summary["passed"] == (summary["detectable_rate"] <= 0.09)
    again, _ = sc.synthetic_nulls(designs_of(table), weights, p_r, n=10, seed=4)
    pd.testing.assert_frame_equal(frame, again)                               # one generator per control, replicate order


def test_the_permutation_keeps_each_match_s_goals_and_runs_the_full_read_out():
    table = synthetic_table(n_matches=30, per_match=20, seed=23)
    y = outcomes(table, seed=24)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    result, summary = sc.permutation(table, table.fold.to_numpy(), y, weights)
    assert set(result["deltas"]) == {"G", "GS", "S", "ALL"} and summary["passed"] == (not summary["holm_rejected"])
    permuted = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), sc.PERMUTATION_SEED).goal.to_numpy()
    assert (pd.Series(permuted).groupby(table.game.to_numpy()).sum() == pd.Series(y).groupby(table.game.to_numpy()).sum()).all()
    assert not np.array_equal(permuted, y)
