"""The frozen rule of the xg-stack test (ticket 04, council C7): fixed parameters, the paired ΔLL at every level, the
label precedence with both margins, the pre-written sentences, the synthetic-null gate and the power scenarios, on
synthetic data with known answers."""

import json
from statistics import NormalDist

import numpy as np
import pytest
from xgpoc.detector.testrule import match_weights, shot_log_loss
from xgpoc.stack_heldout import rule as tr


@pytest.fixture
def rule() -> dict:
    return tr.load_rule()


def result(point: float, lo: float, hi: float, lo90: float, hi90: float, level: float = 0.95) -> dict:
    return {"delta_ll": point, "intervals": {tr.key(level): [lo, hi], tr.key(0.9): [lo90, hi90]}}


def test_the_committed_rule_holds_c7s_values_and_a_changed_value_is_refused(rule, tmp_path):
    assert rule["confirmatory"] == "G" and rule["bootstrap"] == rule["bootstrap"] | {"draws": 5000, "seed": 0}
    assert set(rule["sentences"]) == set(tr.LABELS)
    for keys, wrong in ((("equivalence", "margin"), 0.005), (("success", "ladder"), [0.99]), (("gate", "max_false_success"), 0.1)):
        changed = json.loads(json.dumps(rule))
        target = changed
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = wrong
        path = tmp_path / "rule.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="council C7 fixed"):
            tr.load_rule(path)


def test_the_parameters_digest_ignores_only_the_before_scoring_block(rule):
    digest = tr.parameters_digest(rule)
    assert tr.parameters_digest(rule | {tr.BEFORE_SCORING: {"level_in_force": 0.99}}) == digest
    assert tr.parameters_digest(rule | {"scope": "other"}) != digest


def test_the_level_in_force_must_be_stated_and_on_the_ladder():
    assert tr.level_in_force({tr.BEFORE_SCORING: {"level_in_force": 0.975}}) == 0.975
    for block in ({}, {tr.BEFORE_SCORING: {"level_in_force": None}}, {tr.BEFORE_SCORING: {"level_in_force": 0.9}}):
        with pytest.raises(ValueError, match="no success level"):
            tr.level_in_force(block)


def test_delta_is_the_shot_weighted_paired_difference_with_intervals_at_every_level():
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(30), 10)
    y = rng.random(300) < 0.2
    p_base, p_var = np.full(300, 0.2), np.clip(0.2 + rng.normal(0, 0.05, 300), 0.01, 0.99)
    weights = rng.poisson(1.0, (400, 30))[:, games].astype(float) + 1e-9
    out = tr.delta(y, p_var, p_base, weights, 1e-6)
    d = shot_log_loss(y, p_var, 1e-6) - shot_log_loss(y, p_base, 1e-6)
    boot = weights @ d / weights.sum(axis=1)
    assert out["delta_ll"] == pytest.approx(d.mean()) and out["boot_sd"] == pytest.approx(boot.std(ddof=1))
    assert set(out["intervals"]) == {"0.95", "0.975", "0.99", "0.995", "0.9"}
    assert out["intervals"]["0.99"] == pytest.approx([np.percentile(boot, 0.5), np.percentile(boot, 99.5)])
    lo90, hi90 = out["intervals"]["0.9"]
    lo95, hi95 = out["intervals"]["0.95"]
    assert lo95 <= lo90 <= hi90 <= hi95
    mask = games < 10
    masked = tr.delta(y, p_var, p_base, weights, 1e-6, mask)
    assert masked["shots"] == 100 and masked["delta_ll"] == pytest.approx(d[mask].mean())


def test_the_bootstrap_weights_are_a1s_draws(rule):
    games = np.array(["b", "a", "a", "c"])
    weights = tr.bootstrap_weights(games, rule)
    assert weights.shape == (5000, 4) and np.array_equal(weights[:, 1], weights[:, 2])
    assert (weights[:, [0, 1, 3]].sum(axis=1) == 3).all()                     # three matches drawn per replicate
    assert np.array_equal(weights, match_weights(games, 5000, 0))


@pytest.mark.parametrize(("interval", "label", "practical"), [
    ((-0.004, -0.003, -0.0005, -0.0035, -0.0010), "success", True),       # 90% outside ±0.002
    ((-0.0012, -0.0019, -0.0001, -0.0017, -0.0003), "success and equivalent", False),
    ((0.003, 0.0001, 0.006, 0.0005, 0.0055), "worse", False),
    ((-0.0003, -0.0016, 0.0011, -0.0014, 0.0009), "equivalent within δ", False),
    ((-0.0006, -0.0031, 0.0018, -0.0027, 0.0014), "inconclusive", False),
])
def test_the_label_follows_the_precedence_at_delta_0_002_and_reports_0_005(interval, label, practical):
    read = tr.reading(result(*interval), 0.95)
    assert read["label"] == label and read["practical"] is practical
    assert read["equivalent"]["0.005"] == (-0.005 < interval[3] and interval[4] < 0.005)
    assert read["smallest_equivalence_margin"] == pytest.approx(max(abs(interval[3]), abs(interval[4])))


def test_the_label_reads_the_interval_of_the_level_in_force():
    out = result(-0.0010, -0.0018, -0.0001, -0.0016, -0.0003, level=0.95)
    out["intervals"]["0.99"] = [-0.0022, 0.0002]
    assert tr.reading(out, 0.95)["success"] and not tr.reading(out, 0.99)["success"]
    assert tr.reading(out, 0.99)["label"] == "equivalent within δ"


def test_the_sentence_is_the_labels_with_the_numbers_and_the_practical_clause(rule):
    out = result(-0.00062, -0.00118, -0.00004, -0.00110, -0.00015)
    read = tr.reading(out, 0.95)
    text = tr.sentence(rule, out, read)
    assert read["label"] == "success and equivalent"
    assert "−0.00062 [−0.00118; −0.00004] (95% przedział" in text and "±0.002" in text
    assert text.endswith("poniżej progu praktycznego.")
    out = result(-0.0003, -0.0016, 0.0011, -0.0014, 0.0009)
    text = tr.sentence(rule, out, tr.reading(out, 0.95))
    assert "równoważne w granicy δ = 0.002" in text and "próg" not in text


def test_choose_level_takes_the_first_level_at_or_under_the_ceiling():
    rates = {"0.95": 0.07, "0.975": 0.052, "0.99": 0.03, "0.995": 0.01}
    assert tr.choose_level(rates, 0.05) == 0.99
    assert tr.choose_level(rates | {"0.95": 0.05}, 0.05) == 0.95
    assert tr.choose_level(dict.fromkeys(rates, 0.2), 0.05) is None


def test_the_null_gate_is_deterministic_reads_no_label_and_keeps_95_for_a_worse_g(rule):
    rng = np.random.default_rng(3)
    games = np.repeat([f"m{i:02d}" for i in range(40)], 20)
    p_r = 1 / (1 + np.exp(-rng.normal(-2, 1, 800)))
    p_g = np.clip(p_r * np.exp(rng.normal(0, 0.3, 800)), 1e-3, 0.99)      # noisier than the truth p_R
    small = rule | {"gate": rule["gate"] | {"replicates": 300}, "bootstrap": rule["bootstrap"] | {"draws": 500}}
    weights = tr.bootstrap_weights(games, small)
    first = tr.null_gate(p_r, p_g, games, weights, small, chunk=64)
    assert first == tr.null_gate(p_r, p_g, games, weights, small, chunk=100)
    assert first["level_in_force"] == 0.95 and first["passed"]
    assert first["mean_null_delta"] > 0 and first["false_success_rate"]["0.95"] <= 0.05
    rates = [first["false_success_rate"][tr.key(level)] for level in tr.LEVELS]
    assert rates == sorted(rates, reverse=True)


def test_the_null_gate_flags_an_identical_g_as_never_successful(rule):
    games = np.repeat(np.arange(20), 10)
    p = np.full(200, 0.15)
    small = rule | {"gate": rule["gate"] | {"replicates": 50}, "bootstrap": rule["bootstrap"] | {"draws": 200}}
    out = tr.null_gate(p, p, games, tr.bootstrap_weights(games, small), small)
    assert out["false_success_rate"]["0.95"] == 0.0 and out["mean_null_delta"] == 0.0


def test_the_power_scenarios_are_the_normal_approximation_of_each_scenario():
    dev = {"shots": 7062, "matches": 328, "delta_ll": -0.00065, "boot_sd": 0.00043}
    out = tr.power_scenarios(dev, {"shots": 3819, "matches": 172}, 0.001082, 0.79)
    se = out["standard_errors"]
    assert se["dev_sd_shot_scaled"] == pytest.approx(0.00043 * np.sqrt(7062 / 3819))
    assert se["dev_sd_match_scaled"] == pytest.approx(0.00043 * np.sqrt(328 / 172)) and se["a1_test_se"] == 0.001082
    assert out["effects"]["practical_recovered"] == pytest.approx(-0.002 * 0.79)
    assert len(out["scenarios"]) == 9
    head = next(r for r in out["scenarios"] if r["se"] == "dev_sd_shot_scaled" and r["effect"] == "development_point")
    n = NormalDist()
    for level in tr.LEVELS:
        z = n.inv_cdf(1 - (1 - level) / 2)
        assert head[f"p_success_{tr.key(level)}"] == pytest.approx(n.cdf(0.00065 / se["dev_sd_shot_scaled"] - z))
    assert head["p_success_0.95"] > head["p_success_0.995"]
    assert head["p_equivalent_0.005"] >= head["p_equivalent_0.002"]
