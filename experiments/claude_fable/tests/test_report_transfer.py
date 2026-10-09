"""The transfer-gap analysis (xg-credibility 13) on synthetic test sets whose gap is known."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_transfer import (  # noqa: E402
    ARM_IX,
    ECE_EDGES,
    P_COLS,
    Relabelling,
    bootstrap,
    calibrate,
    curve,
    gap_table,
    k_cap,
    log_losses,
    reading,
    recalibrated,
    recalibration_rows,
    reliability_table,
    sensitivity_table,
    stratified_draws,
)
from xgpoc.eval.calibration import PlattScaler, logit, sigmoid  # noqa: E402

DROP = 0.08  # "without" predicts this much less than "with" on every shot


def uniform_pattern_frame() -> pd.DataFrame:
    """Two leagues, 6 and 4 matches of 10 identical shots each: goals on every fifth shot
    (20%), the last shot a penalty, "with" predicts 0.15-0.19 by shot position and
    "without" DROP less, the same in both seeds. Every match is alike, so every bootstrap
    draw has the same gap."""
    rows = []
    for league, n in (("la", 6), ("lb", 4)):
        for m in range(n):
            for s in range(10):
                p = 0.15 + 0.02 * (s % 3)
                rows.append({"shot_id": f"{league}{m}_{s}", "league": league, "game": f"{league}/x/{m}",
                             "goal": s % 5 == 0, "is_penalty": s == 9, "p_with": p, "p_without": p - DROP})
    test = pd.DataFrame(rows)
    for col in P_COLS:
        test[col] = test.p_without if col.startswith("p_without") else test.p_with
    return test


def expected_gap(test: pd.DataFrame) -> float:
    y = test.goal.to_numpy().astype(float)
    return float(np.mean(log_losses(y, test.p_without.to_numpy()) - log_losses(y, test.p_with.to_numpy())))


def test_pooled_gap_equals_the_known_gap():
    test = uniform_pattern_frame()
    truth = expected_gap(test)
    gaps = gap_table(bootstrap(test, n_boot=40), test).set_index("group")
    assert truth > 0
    assert gaps.loc["pooled", "gap"] == pytest.approx(truth, abs=1e-12)
    assert gaps.loc["pooled", "gap_seed0"] == pytest.approx(truth) and gaps.loc["pooled", "seed_difference"] == 0
    assert gaps.loc["pooled", "relative_gap"] == pytest.approx(truth / gaps.loc["pooled", "log_loss_with"])
    assert gaps.loc["pooled", "gap_lo"] == pytest.approx(truth) and gaps.loc["pooled", "gap_hi"] == pytest.approx(truth)
    assert gaps.loc["pooled", "reading"] == "gap_shown"
    assert gaps.loc["equal_weight", "gap"] == pytest.approx(truth)
    assert gaps.loc["open_play", "gap"] == pytest.approx(expected_gap(test[~test.is_penalty]))
    assert list(gaps.loc[["la", "lb"], "matches"]) == [6, 4]


def test_pooled_gap_weights_leagues_by_shots_and_equal_weight_by_league():
    test = uniform_pattern_frame()
    lb = test.league == "lb"
    for col in (c for c in P_COLS if c.startswith("p_without")):
        test.loc[lb, col] = test.loc[lb, "p_with"]
    gaps = gap_table(bootstrap(test, n_boot=10), test).set_index("group")
    la = expected_gap(test[~lb])
    assert gaps.loc["lb", "gap"] == pytest.approx(0)
    assert gaps.loc["pooled", "gap"] == pytest.approx(la * 60 / 100)
    assert gaps.loc["equal_weight", "gap"] == pytest.approx(la / 2)


def test_seed_gaps_of_opposite_sign_are_reported_and_read_as_ambiguous():
    assert reading(0.001, 0.003, [0.004, -0.001]) == "ambiguous_between_seeds"
    assert reading(0.001, 0.003, [0.004, 0.001]) == "gap_shown"
    assert reading(-0.003, -0.001, [-0.004, -0.001]) == "gap_shown"
    assert reading(-0.001, 0.003, [0.004, 0.001]) == "no_gap_shown"


def test_stratified_draws_keep_the_number_of_matches_per_league():
    games = np.array([f"g{i}" for i in range(9)])
    strata = np.array(["a"] * 6 + ["b"] * 3)
    for rows in stratified_draws(games, strata, n_boot=50, seed=3):
        assert (strata[rows] == "a").sum() == 6 and (strata[rows] == "b").sum() == 3
    first = next(stratified_draws(games, strata, 1, seed=0))
    rng = np.random.default_rng(0)
    assert np.array_equal(first, np.concatenate([rng.choice(6, 6, replace=True), 6 + rng.choice(3, 3, replace=True)]))


def test_calibrate_applies_platt_prior_shift_and_penalty_rate():
    rng = np.random.default_rng(0)
    z = rng.normal(-2, 1, 4000)
    y = (rng.random(z.size) < sigmoid(z)).astype(float)
    platt = PlattScaler().fit(z, y)
    arm = {"arm_goal_rate": float(y.mean()), "arm_penalties": 10, "arm_penalty_goals": 10}
    pen = np.array([False, True])
    p = calibrate(platt, float(y.mean()), np.array([-1.0, -1.0]), pen, arm)
    assert p[0] == pytest.approx(platt(np.array([-1.0]))[0])  # equal rates: no shift
    assert p[1] == pytest.approx((0.76 * 40 + 10) / (40 + 10))
    shifted = calibrate(platt, float(y.mean()), np.array([-1.0]), np.array([False]), arm | {"arm_goal_rate": 0.3})
    assert logit(shifted)[0] - logit(p[:1])[0] == pytest.approx(logit(0.3) - logit(y.mean()))


@pytest.mark.parametrize(("n", "cap"), [(48, 20), (47, 20), (58, 20), (19, 5), (27, 10), (26, 10), (30, 20), (29, 10)])
def test_k_cap_matches_the_protocol(n, cap):
    assert k_cap(n) == cap


def shifted_cell(n_matches: int = 40, shots: int = 40, shift: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """One half-set: "with" predicts the true probability, "without" the same shifted by
    `shift` in logit; one penalty per match at a fixed rate in both arms."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        z = rng.normal(-2.2, 1.0, shots)
        y = rng.random(shots) < sigmoid(z)
        for s in range(shots):
            rows.append({"shot_id": f"{m}_{s}", "game": f"l/x/{m:02d}", "goal": bool(y[s]), "is_penalty": s == 0, "z": z[s]})
    cell = pd.DataFrame(rows)
    pen = cell.is_penalty.to_numpy()
    for seed_ in (0, 1):
        cell[f"p_with_s{seed_}"] = np.where(pen, 0.75, sigmoid(cell.z))
        cell[f"p_without_s{seed_}"] = np.where(pen, 0.75, sigmoid(cell.z + shift))
    return cell


def test_recalibration_draws_share_the_evaluation_set_and_recover_a_pure_shift():
    cell = shifted_cell()
    rows = recalibration_rows(cell, "l", "A", n_draws=20)
    assert set(rows.k) == {0, 5, 10, 20}
    assert (rows.groupby("draw").n_eval.nunique() == 1).all()  # one evaluation set per draw, for every k
    assert (rows.n_eval == 40 * 20).all()  # 40 matches - K = 20 candidates, 40 shots each
    k0 = rows[rows.k == 0]
    assert (k0.without_platt == k0.without_raw).all() and (k0.with_platt == k0.with_raw).all()
    c = curve(rows, ("l",), (0, 5, 10, 20)).set_index(["variant", "k"])
    assert c.loc[("without_platt", 0), "mean"] > 0.005
    # a pure logit shift is what the intercept-only map undoes: most of the gap comes back
    assert c.loc[("without_intercept", 20), "recovered"] > 0.8
    assert c.loc[("without_platt", 20), "recovered"] > 0.7
    assert np.isnan(c.loc[("with_platt", 20), "recovered"])
    assert c.loc[("with_platt", 20), "mean"] > 0  # recalibrating the true probability can only add noise
    assert (c.fallback_draws == 0).all()


def test_draws_without_an_open_play_goal_stay_at_k0_and_are_counted():
    cell = shifted_cell(n_matches=20)
    cell.loc[~cell.is_penalty, "goal"] = cell.loc[~cell.is_penalty, "game"] >= "l/x/15"  # matches 0-14: no open-play goal
    rows = recalibration_rows(cell, "l", "A", n_draws=30)
    k5 = rows[rows.k == 5]
    assert k5.fallback.any()
    fb = k5[k5.fallback]
    assert (fb.without_platt == fb.without_raw).all() and (fb.without_intercept == fb.without_raw).all()
    assert curve(rows, ("l",), (0, 5)).set_index(["variant", "k"]).loc[("without_platt", 5), "fallback_draws"] == \
        k5[k5.seed == 0].fallback.sum()


def test_curve_refuses_a_league_without_the_requested_k():
    rows = recalibration_rows(shifted_cell(n_matches=19), "l", "A", n_draws=2)
    with pytest.raises(ValueError, match="not every league"):
        curve(rows, ("l",), (0, 5, 10))


def test_sensitivity_nearest_row_reproduces_the_gap_and_exclude_drops_shots():
    rng = np.random.default_rng(1)
    test = uniform_pattern_frame()
    for col in P_COLS:  # the pattern plus noise, so every column and seed differs
        test[col] = test.p_with + rng.uniform(-0.03, 0.03, len(test))
    gaps = gap_table(bootstrap(test, n_boot=30), test).set_index("group")
    goal = test.goal.to_numpy()
    keep = np.ones(len(test), bool)
    keep[:7] = False
    labels = [Relabelling("nearest", "rule", goal, np.ones(len(test), bool)), Relabelling("exclude", "rule", goal & keep, keep)]
    sens = sensitivity_table(test, labels, n_boot=30)
    near = sens[sens.rule == "nearest"].set_index("group")
    for g in ("pooled", "la", "lb"):
        for c in ("gap", "gap_lo", "gap_hi", "gap_seed0", "gap_seed1"):
            assert near.loc[g, c] == pytest.approx(gaps.loc[g, c], abs=1e-12)
    ex = sens[(sens.rule == "exclude") & (sens.group == "pooled")].iloc[0]
    p = test[P_COLS].to_numpy()[keep]
    losses = log_losses(goal[keep].astype(float), p).mean(0)
    gap = losses[ARM_IX["without"]].mean() - losses[ARM_IX["with"]].mean()
    assert ex.shots == keep.sum() and ex.gap == pytest.approx(gap)
    assert "reading" not in sens.columns  # the pre-registered reading belongs to the main row only


def test_only_the_main_row_carries_the_reading_and_the_relative_gap():
    test = uniform_pattern_frame()
    gaps = gap_table(bootstrap(test, n_boot=5), test).set_index("group")
    assert gaps.reading.notna().sum() == 1 and gaps.relative_gap.notna().sum() == 1
    assert pd.notna(gaps.loc["pooled", "reading"])


def test_reliability_bins_are_the_ece_bins_with_both_seeds_stacked():
    test = uniform_pattern_frame()
    rel = reliability_table(bootstrap(test, n_boot=5))
    la = test[test.league == "la"]
    p = la[[P_COLS[i] for i in ARM_IX["without"]]].to_numpy().ravel()
    y = np.tile(la.goal.to_numpy(), 2)
    b = np.digitize(p, ECE_EDGES) - 1
    got = rel[(rel.league == "la") & (rel.arm == "without")].set_index("bin")
    assert set(got.index) == set(b)
    for k in got.index:
        assert got.loc[k, "shots"] == (b == k).sum() and got.loc[k, "goal_rate"] == pytest.approx(y[b == k].mean())


def test_recalibrated_keeps_penalties_and_refuses_unknown_modes():
    p = np.array([0.1, 0.2, 0.3, 0.75])
    pen = np.array([False, False, False, True])
    fit = np.ones(4, bool)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert recalibrated(p, pen, fit, y, "intercept")[3] == 0.75
    with pytest.raises(ValueError, match="unknown recalibration"):
        recalibrated(p, pen, fit, y, "isotonic")
