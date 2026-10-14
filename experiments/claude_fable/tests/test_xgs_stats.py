"""xg-stack read-out statistics (ticket 01, council C5): against direct computations, Holm, the readings, the
categories, and the long-run Holm rate of a pure-noise feature set."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.detector.testrule import shot_log_loss
from xgpoc.stack import analysis as sa
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgs_fixtures import outcomes, synthetic_table


def test_paired_delta_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    n, games = 300, np.repeat([f"m{i:02d}" for i in range(30)], 10)
    y = (rng.random(n) < 0.2).astype(int)
    p_ref, p_arm = rng.uniform(0.05, 0.5, n), rng.uniform(0.05, 0.5, n)
    weights = ss.bootstrap_weights(games, draws=500, seed=0)
    out = ss.paired_delta(y, p_arm, p_ref, weights)
    d = shot_log_loss(y, p_arm, ss.CLIP) - shot_log_loss(y, p_ref, ss.CLIP)
    boot = np.array([np.average(d, weights=w) for w in weights])
    assert out["delta_ll"] == pytest.approx(d.mean())
    assert out["lo95"] == pytest.approx(np.percentile(boot, 2.5)) and out["hi95"] == pytest.approx(np.percentile(boot, 97.5))
    assert out["bound_one_sided_95"] == pytest.approx(np.percentile(boot, 5))
    assert out["upper_one_sided_95"] == pytest.approx(np.percentile(boot, 95))
    assert out["p_one_sided"] == pytest.approx((1 + np.sum(boot >= 0)) / 501)
    inside = np.arange(n) < 100
    sub = ss.paired_delta(y, p_arm, p_ref, weights, inside)
    assert sub["delta_ll"] == pytest.approx(d[inside].mean())


def test_holm_steps_down_and_stops_at_the_first_failure():
    assert ss.holm({"G": 0.01, "GS": 0.04}) == {"G": True, "GS": True}
    assert ss.holm({"G": 0.03, "GS": 0.04}) == {"G": False, "GS": False}        # 0.03 > 0.05 / 2
    assert ss.holm({"G": 0.02, "GS": 0.06}) == {"G": True, "GS": False}


def test_readings_need_holm_and_the_practical_bar():
    deltas = {"G": {"p_one_sided": 0.001, "delta_ll": -0.0025}, "GS": {"p_one_sided": 0.01, "delta_ll": -0.001}}
    read = ss.readings(deltas, ("G", "GS"))
    assert read["G"] == {"holm_rejected": True, "detectable_improvement": True, "practical_candidate": True}
    assert read["GS"]["detectable_improvement"] and not read["GS"]["practical_candidate"]
    deltas["G"]["p_one_sided"] = 0.2
    assert not ss.readings(deltas, ("G", "GS"))["G"]["practical_candidate"]


def test_category_masks_leave_missing_inputs_out():
    table = pd.DataFrame({"shot_x@0": [80.0, 95.0, np.nan], "angle@0": [0.1, 0.5, np.nan], "dist@0": [5.0, 20.0, np.nan],
                          "sn_corner_20s": [1.0, 0.0, 0.0], "sn_direct_fk": [0.0, 0.0, 1.0]})
    masks = ss.category_masks(table)
    assert masks["e3_outside_box"][0].tolist() == [True, True, False] and masks["e3_outside_box"][1].tolist() == [True, False, False]
    assert masks["e3_angle_lt_15deg"][1].tolist() == [True, False, False]                 # 0.1 rad < 15 deg < 0.5 rad
    assert masks["e3_dist_lt_6m"][1].tolist() == [True, False, False]
    assert masks["sn_corner_20s"][1].tolist() == [True, False, False] and masks["sn_direct_fk"][1].tolist() == [False, False, True]


def test_the_read_out_reports_every_arm_and_the_categories():
    table = synthetic_table(n_matches=40, per_match=20, seed=12)
    y = outcomes(table, seed=13)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=200, seed=0)
    spec = sf.spec("c", table)
    result = sa.analyse(table, table.fold.to_numpy(), y, weights, spec)
    assert set(result["deltas"]) == {"G", "GS", "S", "Q", "ALL"} and set(result["readings"]) == {"G", "GS"}
    assert set(result["gbm"]) == {"G", "GS"} and set(result["scores"]["R"]) >= {"log_loss", "auc", "brier", "cal_slope", "cal_intercept"}
    assert result["chosen_c"]["R"] == [1.0] * 5 and set(result["chosen_c"]["G"]) <= set(sf.C_GRID)
    categories = result["categories"]
    assert set(categories.category) == set(ss.CATEGORIES) and set(categories.arm) == set(sf.ARM_NAMES)
    assert categories.loc[categories.arm == "R", "delta_ll"].isna().all()


@pytest.mark.parametrize("candidate", ["b", "c"])
def test_a_pure_noise_feature_set_gives_no_holm_rejection_in_the_long_run(candidate):
    table = synthetic_table(n_matches=40, per_match=30, seed=14)
    spec = sf.spec(candidate, table)
    designs = sf.all_designs(table, table.fold.to_numpy(), spec, ("R", *sf.CONFIRMATORY))
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=300, seed=0)
    p_true = 1 / (1 + np.exp(-table[sf.Z].to_numpy()))
    frame = sc.simulate(designs, weights, spec.model, sc.fixed(p_true), 30, seed=15)
    assert frame.detectable_any.mean() <= 0.05
    assert frame.delta_G.mean() > 0                       # noise inputs cost log-loss out of fold
