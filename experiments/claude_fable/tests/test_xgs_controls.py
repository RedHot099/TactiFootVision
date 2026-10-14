"""Controls 2-4 and the simulation reports of xg-stack (ticket 01, councils C5 and C6): the plants and the β solver,
the oracle and its power, the Monte Carlo SE, the power curve's smallest detectable effect, the clustered nulls'
coverage, the permutation and the summaries."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.compare.stats import permute_within_matches
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgs_fixtures import outcomes, standardised_distance, synthetic_table


def designs_of(table, candidate="b"):
    spec = sf.spec(candidate, table)
    return sf.all_designs(table, table.fold.to_numpy(), spec, ("R", *sf.CONFIRMATORY)), spec


def test_the_plants_and_beta_solve_the_true_delta():
    table = synthetic_table(n_matches=100, per_match=40, seed=20)
    z = table[sf.Z].to_numpy()
    d = sc.plant(table, "dist")
    np.testing.assert_allclose(d, standardised_distance(table))
    s = sc.plant(table, "dist_angle")
    e = sc.standardised(table["angle@0"].to_numpy())
    np.testing.assert_allclose(s, sc.standardised(d - e))
    assert s.std() == pytest.approx(1.0) and np.corrcoef(s, d)[0, 1] > 0.5 and np.corrcoef(s, e)[0, 1] < -0.5
    with pytest.raises(ValueError):
        sc.plant(table, "angle")
    assert sc.true_delta(0.0, -0.2, 1.1, z, d) == pytest.approx(0.0, abs=1e-12)
    for signal in (d, s):
        for target in sc.CURVE:
            beta = sc.solve_beta(-0.2, 1.1, z, signal, target)
            assert beta < 0 and sc.true_delta(beta, -0.2, 1.1, z, signal) == pytest.approx(target, abs=1e-9)


def test_the_projection_is_the_expected_log_likelihood_fit():
    rng = np.random.default_rng(0)
    z = rng.normal(-2, 1, 3000)
    p = sigmoid(0.3 + 0.8 * z + 0.4 * rng.normal(size=3000))
    q = sc.projection(z, p)
    x = np.column_stack([np.ones(3000), z])
    np.testing.assert_allclose(x.T @ (p - q), 0.0, atol=1e-8)                 # the score equations
    np.testing.assert_allclose(sc.projection(z, sigmoid(0.3 + 0.8 * z)), sigmoid(0.3 + 0.8 * z), atol=1e-9)


def test_expected_log_loss_and_mc_se():
    p, q = np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.7])
    assert sc.expected_log_loss(p, q) == pytest.approx(-np.mean(p * np.log(q) + (1 - p) * np.log(1 - q)))
    assert sc.mc_se(0.5, 100) == pytest.approx(0.05) and sc.mc_se(0.0, 10) == 0.0


def test_the_positive_control_reports_beta_power_and_the_oracle():
    table = synthetic_table(n_matches=60, per_match=40, seed=21)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    designs, spec = designs_of(table)
    frame, summary = sc.positive_control(designs, weights, spec.model, table, -0.1, 1.0, "dist", 8, 3)
    assert len(frame) == 8 and summary["beta"] < 0 and summary["true_delta"] == pytest.approx(-0.002, abs=1e-9)
    assert summary["power_G_mc_se"] == pytest.approx(sc.mc_se(summary["power_G"], 8))
    assert summary["passed"] == (-0.003 <= summary["mean_recovered_delta_G"] <= -0.001)
    assert summary["mean_oracle_delta"] < 0 and 0 <= summary["oracle_power"] <= 1
    _, other = sc.positive_control(designs, weights, spec.model, table, -0.1, 1.0, "dist_angle", 4, 3)
    _, stronger = sc.positive_control(designs, weights, spec.model, table, -0.1, 1.0, "dist", 4, 3, target=-0.004)
    assert "passed" not in other and "passed" not in stronger                  # only the dist plant at -0.002 is gated


def test_the_smallest_detectable_effect_interpolates_the_power_curve():
    curve = [{"target": -0.002, "power_G": 0.3}, {"target": -0.003, "power_G": 0.7}, {"target": -0.004, "power_G": 0.9}]
    assert sc.smallest_detectable(curve) == pytest.approx(-0.0035)
    assert sc.smallest_detectable([{**c, "power_G": 0.5} for c in curve]) is None
    assert sc.smallest_detectable([{**c, "power_G": 0.85} for c in curve]) == -0.002


def test_the_synthetic_nulls_summary_applies_the_criterion_and_is_reproducible():
    table = synthetic_table(n_matches=40, per_match=30, seed=22)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    designs, spec = designs_of(table)
    p_r = sigmoid(table[sf.Z].to_numpy())
    frame, summary = sc.synthetic_nulls(designs, weights, spec.model, p_r, 10, 4)
    assert len(frame) == 10 and summary["passed"] == (summary["detectable_rate"] <= 0.09)
    assert set(summary["coverage"]) == {"G", "GS"}
    again, _ = sc.synthetic_nulls(designs, weights, spec.model, p_r, 10, 4)
    pd.testing.assert_frame_equal(frame, again)                               # one generator per control, replicate order


def test_the_clustered_nulls_draw_one_intercept_per_match_and_report_coverage(monkeypatch):
    table = synthetic_table(n_matches=40, per_match=30, seed=25)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    designs, spec = designs_of(table)
    seen = []
    original = sc.simulate

    def spy(designs_, weights_, model, truth, n, seed, log=sc.quiet):
        rng = np.random.default_rng(seed)
        seen.append(truth(rng))
        return original(designs_, weights_, model, truth, n, seed, log)
    monkeypatch.setattr(sc, "simulate", spy)
    frame, summary = sc.clustered_nulls(designs, weights, spec.model, table, -0.2, 1.0, 12, 6, sd=0.5)
    u = np.log(seen[0] / (1 - seen[0])) - (-0.2 + table[sf.Z].to_numpy())
    per_match = pd.Series(u).groupby(table.game.to_numpy()).agg(["std", "mean"])
    assert per_match["std"].max() < 1e-9 and per_match["mean"].std() == pytest.approx(0.5, rel=0.4)
    cov = summary["coverage"]["G"]
    assert cov["coverage_of_expected"] == pytest.approx((frame.bound_G <= frame.delta_G.mean()).mean())
    assert cov["floor"] == pytest.approx(0.95 - 2 * sc.mc_se(0.95, 12))
    assert cov["wording_allowed"] == (cov["coverage_of_expected"] >= cov["floor"])


def test_the_permutation_keeps_each_match_s_goals_and_runs_the_full_read_out():
    table = synthetic_table(n_matches=30, per_match=20, seed=23)
    y = outcomes(table, seed=24)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    result, summary = sc.permutation(table, table.fold.to_numpy(), y, weights, sf.spec("b", table), 0)
    assert set(result["deltas"]) == {"G", "GS", "S", "Q", "ALL"} and summary["passed"] == (not summary["holm_rejected"])
    permuted = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), 0).goal.to_numpy()
    assert (pd.Series(permuted).groupby(table.game.to_numpy()).sum() == pd.Series(y).groupby(table.game.to_numpy()).sum()).all()
    assert not np.array_equal(permuted, y)
