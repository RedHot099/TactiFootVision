"""xg-temporal controls and read-out rules (ticket 03) on synthetic inputs: the label-free truths hit their analytic
targets, the motion plant is invisible to S, the fits do not depend on the number of worker processes, the gates'
criteria, the permutation keeps each match's labels, the readability veto, the claims and the leak-curve rules."""

import json

import numpy as np
import pandas as pd
import pytest
from xgpoc.stack import stats as ss
from xgpoc.temporal import controls as tco
from xgpoc.temporal import features as tf
from xgpoc.temporal import readout as tr
from xgt_fixtures import synthetic_inputs

GAMES = np.repeat(np.arange(40), 50).astype(str)


@pytest.fixture(scope="module")
def built():
    data, _ = synthetic_inputs(k=4)
    designs, _ = tf.fold_designs(data)
    return data, designs, tco.Truths.build(data, GAMES), ss.bootstrap_weights(GAMES)


def test_the_seeds_are_fresh_and_the_settings_serialise():
    settings = tco.settings()
    assert settings["seeds_fresh"]["distinct"] and len(set(tco.SEEDS.values())) == len(tco.SEEDS)
    assert tco.TRACER_SEED not in tco.SEEDS.values() and settings["replicates"] == 200
    assert not set(tco.REMEDY_SEEDS.values()) & set(tco.SEEDS.values()) and "positive_-0.002" in tco.REMEDY_SEEDS
    assert settings["remedy"]["seeds"] == tco.REMEDY_SEEDS and settings["positive"]["second_plant"]["gating"] is False
    assert settings["rich_nulls"]["scale"] == -0.004 and settings["positive"]["gated_contrast"] == "T-S"
    assert json.loads(json.dumps(settings)) == settings


def test_the_rich_null_hits_its_oracle_s_minus_r_scale_without_outcomes(built):
    _, _, truths, _ = built
    assert abs(truths.u.mean()) < 1e-12 and abs(truths.u.std() - 1) < 1e-12
    for scale in (tco.RICH_SCALE, tco.RICH_SCALE_EXTRA):
        p, record = truths.rich(scale)
        assert record["a"] < 0 and record["oracle_S-R"] == pytest.approx(scale, abs=1e-9)
        assert p.shape == truths.z.shape and ((p > 0) & (p < 1)).all()
    again = tco.Truths.build(built[0], GAMES)
    assert np.array_equal(again.u, truths.u) and np.array_equal(again.m, truths.m)


@pytest.mark.parametrize("plant", ["motion", "pc01"])
def test_both_motion_plants_are_invisible_to_s_and_hit_their_oracle_t_minus_s_target(built, plant):
    _, _, truths, _ = built
    m = truths.m if plant == "motion" else truths.m_pc01
    x = np.column_stack([np.ones(len(m)), truths.x_static])
    assert np.abs(x.T @ m).max() / len(m) < 1e-10                         # residualised on S's pooled inputs
    p, record = truths.positive(tco.POSITIVE_TARGET, plant)
    assert record["plant"] == plant and record["oracle_T-S"] == pytest.approx(tco.POSITIVE_TARGET, abs=1e-9)
    assert record["oracle_T-R"] == pytest.approx(tco.POSITIVE_TARGET, abs=2e-4)
    assert record["beta"] < 0


def test_the_gating_plant_combines_every_motion_column(built):
    data, _, truths, _ = built
    columns = tco.motion_columns(data, tco.pooled_scores(data))
    assert columns.shape[1] == data.k * len(tf.motion_kinds(data.window))
    residual = tco.residualised(columns, truths.x_static)
    weights = np.linalg.lstsq(residual, truths.m, rcond=None)[0]
    assert np.allclose(residual @ weights, truths.m, atol=1e-8)           # m lies in the span of the motion columns
    assert np.count_nonzero(np.abs(weights) > 1e-6) == columns.shape[1]
    assert abs(np.corrcoef(truths.m, truths.m_pc01)[0, 1]) < 0.9


def test_dependent_missing_indicators_are_dropped_before_the_unpenalised_projection():
    x = np.column_stack([np.arange(10.0), np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5), np.zeros(5)]])
    kept = tco.independent_columns((x - x.mean(0)) / x.std(0))
    assert kept.shape[1] == 2


def test_the_fits_do_not_depend_on_the_number_of_worker_processes(built):
    _, designs, truths, weights = built
    p, _ = truths.positive(tco.POSITIVE_TARGET)
    frames = []
    for jobs in (1, 2):
        with tco.Fitter(designs, jobs) as fitter:
            frames.append(tco.simulate(fitter, weights, tco.fixed(p), 2, 7))
    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert set(frames[0].columns) >= {"delta_T-S", "holm_T-S", "delta_oracle", "floor_share_T", "holm_any"}


def test_outcomes_are_drawn_in_replicate_order_from_the_seed():
    p = np.full(50, 0.3)
    a, b = tco.draw(tco.fixed(p), 3, 11, 50), tco.draw(tco.fixed(p), 3, 11, 50)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))
    rng = np.random.default_rng(11)
    assert np.array_equal(a[0][1], (rng.random(50) < p).astype(int))


def null_frame(rejections: int, n: int = 100, column: str = "holm_any") -> pd.DataFrame:
    frame = pd.DataFrame({name: np.zeros(n) for name in [f"{k}_{c}" for c in tco.CONTRASTS
                                                         for k in ("delta", "bound", "hi95", "p")]})
    for c in tr.CONFIRMATORY:
        frame[f"holm_{c}"] = False
    frame["holm_any"] = False
    frame.loc[: rejections - 1, column] = True
    return frame.assign(floor_share_S=1.0, floor_share_T=1.0)


def test_the_null_gates_compare_the_holm_rejection_rate_with_0_09():
    assert tco.null_summary(null_frame(9), 1, "any")["passed"]
    assert not tco.null_summary(null_frame(10), 1, "any")["passed"]
    rich = tco.null_summary(null_frame(10, column="holm_T-R"), 1, "T-S")
    assert rich["passed"] and rich["holm_rejection"]["T-R"]["rate"] == 0.1      # only a T - S rejection fails the rich gate
    assert not tco.null_summary(null_frame(10, column="holm_T-S"), 1, "T-S")["passed"]


def test_the_positive_gate_is_the_mean_recovered_t_minus_s_in_its_band():
    frame = null_frame(0).assign(delta_oracle=-0.002, p_oracle=0.01)
    for value, passed in ((-0.0015, True), (-0.0031, False), (-0.0009, False)):
        summary = tco.positive_summary(frame.assign(**{"delta_T-S": value}), 1, {"target": -0.002, "plant": "motion"})
        assert summary["passed"] is passed
    assert "passed" not in tco.positive_summary(frame, 1, {"target": -0.004})
    assert "passed" not in tco.positive_summary(frame, 1, {"target": -0.002, "plant": "pc01"})


def test_the_smallest_detectable_effect_interpolates_the_t_minus_s_power():
    curve = [{"target": -0.002, "power_T-S": 0.4}, {"target": -0.003, "power_T-S": 0.7}, {"target": -0.004, "power_T-S": 0.9}]
    assert tco.smallest_detectable(curve) == pytest.approx(-0.0035)
    assert tco.smallest_detectable([{"target": -0.002, "power_T-S": 0.1}]) is None


def test_the_label_free_run_holds_every_gate_and_refuses_to_drop_one(built):
    data, designs, truths, weights = built
    provenance = {"passed": True, "checks": []}
    with tco.Fitter(designs) as fitter:
        frames, summary = tco.run_label_free(fitter, weights, truths, provenance, 2)
        with pytest.raises(ValueError, match="gating"):
            tco.run_label_free(fitter, weights, truths, provenance, 2, simulations=("plain_nulls", "rich_nulls"))
    assert set(frames) == set(tco.SIMULATIONS) and set(summary["gates"]) == set(tco.LABEL_FREE_GATES)
    assert summary["reports"]["power_curve"]["targets"] == [-0.002, -0.003, -0.004]
    assert summary["gates"]["positive"]["plant"] == "motion"
    assert summary["reports"]["positive_pc01_-0.002"]["plant"] == "pc01"
    assert "passed" not in summary["reports"]["positive_pc01_-0.002"]
    assert set(summary["reports"]["clustered_nulls"]["coverage"]) == set(tco.CONTRASTS)
    assert summary["label_free_passed"] == all(summary["gates"][g]["passed"] for g in tco.LABEL_FREE_GATES)
    json.dumps(summary, default=float)


def test_a_strong_motion_plant_is_recovered_by_t_minus_s(built):
    _, designs, truths, weights = built
    p, _ = truths.positive(-0.02)
    with tco.Fitter(designs) as fitter:
        frame = tco.simulate(fitter, weights, tco.fixed(p), 3, 5)
    assert (frame["delta_T-S"] < 0).all() and frame["delta_T-S"].mean() < -0.008 and frame["holm_T-S"].sum() >= 2


def test_the_permutation_keeps_each_matchs_labels_and_returns_no_label(built, monkeypatch):
    _, designs, _, weights = built
    y = (np.random.default_rng(3).random(len(GAMES)) < 0.2).astype(int)
    seen, original = {}, tr.out_of_fold

    def fits(d, permuted):
        seen["y"] = permuted
        return original(d, permuted)
    monkeypatch.setattr(tco.tr, "out_of_fold", fits)
    result = tco.permutation(designs, weights, y, GAMES)
    assert not np.array_equal(seen["y"], y)
    for g in np.unique(GAMES):
        assert seen["y"][GAMES == g].sum() == y[GAMES == g].sum()
    assert set(result) == {"seed", "deltas", "holm_rejected", "chosen_c", "criterion", "passed"}
    assert result["passed"] == (not result["holm_rejected"])


def test_the_readability_veto_fires_only_on_a_lower_bound_above_0_55():
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(60), 30).astype(str)
    goal = (rng.random(len(games)) < 0.1).astype(int)
    on_target = (rng.random(len(games)) < 0.35).astype(int)
    p_s = np.full(len(games), 0.1)
    readable = tr.readability_veto(goal, on_target, 1 / (1 + np.exp(-(-2.2 + 1.5 * on_target))), p_s, games)
    unreadable = tr.readability_veto(goal, on_target, 1 / (1 + np.exp(-(-2.2 + 0.1 * rng.normal(size=len(games))))), p_s,
                                     games)
    assert readable["vetoed"] and readable["auc"] > 0.9 and readable["n"] == int((goal == 0).sum())
    assert not unreadable["vetoed"] and readable["lower_one_sided_95"] <= readable["auc"]


def test_the_claims_need_holm_the_practical_threshold_no_veto_and_no_leak_jump():
    result = {"readings": {"T-R": {"holm_rejected": True, "detectable_improvement": True, "practical_candidate": True},
                           "T-S": {"holm_rejected": False, "detectable_improvement": False, "practical_candidate": False}}}
    chosen = {"R": [None] * 5, "S": [1e-5] * 3 + [1.0] * 2, "T": [1e-5] * 2 + [1.0] * 3}
    calm = {"jump": {"T-R": {"signal": False}, "T-S": {"signal": False}}}
    jump = {"jump": {"T-R": {"signal": True}, "T-S": {"signal": False}}}
    open_ = tr.claims(result, {"vetoed": False}, chosen, calm)
    vetoed = tr.claims(result, {"vetoed": True}, chosen, calm)
    jumped = tr.claims(result, {"vetoed": False}, chosen, jump)
    assert open_["T-R"]["practical_candidate"] and not open_["motion_helps"] and open_["T-R"]["wording"] is None
    assert not vetoed["T-R"]["practical_candidate"] and vetoed["T-R"]["practical_blocked_by_veto"]
    assert not jumped["T-R"]["practical_candidate"] and jumped["T-R"]["wording"] == tr.AMBIGUOUS
    assert jumped["T-R"]["detectable_improvement"]                          # the Holm reading is unchanged
    assert open_["arm_near_reference"] == {"S": True, "T": False}


def test_the_trend_is_the_least_squares_line_through_three_points():
    for at in (-400, -200):
        c = tr.trend_weights(at)
        assert c.sum() == pytest.approx(1.0)
        assert c @ np.array([-1000.0, -800.0, -600.0]) == pytest.approx(at)          # the line y = x


def test_the_leak_rules_read_the_method_control_and_the_jump_with_its_interval():
    rng = np.random.default_rng(0)

    def curve(late: float, positive_hi: float) -> dict:
        values = {-1000: 0.0, -800: -0.0005, -600: -0.001, -400: -0.0015, -200: late, 0: -0.01, 400: -0.02}
        noise = rng.normal(0, 1e-4, 200)
        return {o: {n: {"delta_ll": v, "draws": v + noise, "hi95": positive_hi if (o, n) == (400, "S-R") else 0.0}
                    for n in ("T-R", "T-S", "S-R")} for o, v in values.items()}
    flat = tr.leak_rules(curve(-0.002, -0.01))
    assert flat["method_positive"]["passed"] and not flat["leak_signal"]
    assert flat["jump"]["T-S"]["excess"]["-200"]["excess"] == pytest.approx(0.0, abs=1e-12)
    jump = tr.leak_rules(curve(-0.0035, 0.001))
    excess = jump["jump"]["T-S"]["excess"]["-200"]
    assert jump["leak_signal"] and not jump["method_positive"]["passed"]
    assert excess["excess"] == pytest.approx(-0.0015) and excess["lo95"] <= excess["excess"] <= excess["hi95"]


def test_a_curve_point_keeps_its_draws_paired_with_the_main_weights():
    games = np.repeat(np.arange(30), 20).astype(str)
    weights = ss.bootstrap_weights(games)
    rng = np.random.default_rng(1)
    y = (rng.random(len(games)) < 0.2).astype(int)
    probabilities = {arm: np.clip(0.2 + 0.05 * rng.normal(size=len(games)), 0.01, 0.99) for arm in ("R", "S", "T")}
    point = tr.curve_point(y, probabilities, weights)
    assert set(point) == {"T-R", "T-S", "S-R"} and point["T-S"]["draws"].shape == (ss.N_BOOT,)
    assert point["T-R"]["delta_ll"] == pytest.approx(ss.paired_delta(y, probabilities["T"], probabilities["R"],
                                                                     weights)["delta_ll"])
