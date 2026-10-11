"""Power simulation of the H2 margin (xg-vendor ticket 08) on planted effects whose power is known by construction."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from xgpoc.eval.calibration import sigmoid
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import power as pw

ROOT = Path(__file__).resolve().parents[1]


def planted_power(true_diff: float, margin: float, sd_match: float, n_matches: int = 135, n_studies: int = 600) -> float:
    """Studies whose per-match H2 differences are N(true_diff, sd_match^2), equal match sizes, through the power code."""
    rng = np.random.default_rng(1)
    boot = ld.bootstrap(np.arange(n_matches), n_boot=2000)
    records = []
    for _ in range(n_studies):
        d = rng.normal(true_diff, sd_match, n_matches)
        records.append({"scenario": "planted", "n": n_matches, **pw.study_record(d.mean(), boot.counts @ d / n_matches, margins=(margin,))})
    return float(pw.power_table(pd.DataFrame(records), margins=(margin,)).power_tost.iloc[0])


def test_power_is_about_80_percent_at_a_margin_matched_to_a_zero_effect():
    """True difference 0, SE s: TOST power = 2 * Phi(margin / s - z_0.95) - 1, which is 0.80 at margin = 2.926 s."""
    sd_match, n = 0.01, 135
    margin = (norm.ppf(0.95) + norm.ppf(0.90)) * sd_match / np.sqrt(n)
    assert planted_power(0.0, margin, sd_match, n) == pytest.approx(0.80, abs=0.06)


def test_power_is_about_80_percent_at_a_margin_matched_to_a_planted_gain():
    """True difference -0.001 (the frame helps): power ~ Phi((margin - 0.001) / s - z_0.95), 0.80 at margin = 0.001 + 2.486 s."""
    sd_match, n = 0.01, 135
    margin = 0.001 + (norm.ppf(0.95) + norm.ppf(0.80)) * sd_match / np.sqrt(n)
    assert planted_power(-0.001, margin, sd_match, n) == pytest.approx(0.80, abs=0.06)
    assert planted_power(-0.001, 0.0005, sd_match, n) == 0.0


def power_frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame([{"scenario": name, "margin": m, "power_tost": p} for name, powers in rows.items() for m, p in zip(pw.MARGINS, powers, strict=True)])


def test_choose_margin_takes_the_smallest_margin_with_enough_decision_power():
    archived, weaker = (s.name for s in pw.DECISION_SCENARIOS)
    table = power_frame({archived: [0.5, 0.85, 0.99], weaker: [0.6, 0.79, 0.98], "primary/null/archived": [0.95, 1.0, 1.0]})
    decision = pw.choose_margin(table)
    assert decision["chosen_margin"] == 0.005 and decision["decision_power"]["0.003"] == 0.79
    table.loc[table.scenario.eq(weaker) & table.margin.eq(0.003), "power_tost"] = 0.80
    assert pw.choose_margin(table)["chosen_margin"] == 0.003
    assert pw.choose_margin(power_frame({archived: [0.1, 0.2, 0.7], weaker: [0.1, 0.2, 0.9]}))["chosen_margin"] is None
    with pytest.raises(ValueError, match="decision scenario"):
        pw.choose_margin(power_frame({archived: [0.9, 0.9, 0.9]}))


def synthetic_basis(n_matches: int = 60, per_match: int = 20, frame_signal: float = 0.5, seed: int = 0) -> pw.Basis:
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    z_pos, s = rng.normal(-2.2, 1.1, n), rng.normal(0, 1, n)
    y = (rng.random(n) < sigmoid(z_pos + frame_signal * s)).astype(int)
    matches = np.repeat([f"m{k:03d}" for k in range(n_matches)], per_match + rng.integers(-5, 6, n_matches))[:n]
    return pw.Basis(y, sigmoid(z_pos), 0.6 * z_pos + frame_signal * s + rng.normal(0, 0.5, n), matches)


def test_weaker_frame_sigma_raises_the_frame_log_loss_by_the_target():
    basis = synthetic_basis(n_matches=150)
    q, _ = pw.truth_probabilities(basis, "development")
    sigma = pw.weaker_frame_sigma(q, basis.frame_logit, 0.002, np.random.default_rng(0))
    fresh = np.random.default_rng(9).standard_normal((64, len(q)))
    base = pw.expected_recalibrated_log_loss(q, basis.frame_logit)
    worse = np.mean([pw.expected_recalibrated_log_loss(q, basis.frame_logit + sigma * e) for e in fresh])
    assert sigma > 0 and worse - base == pytest.approx(0.002, abs=5e-4)


def test_combined_studies_add_shots_inside_the_same_matches():
    basis = synthetic_basis()
    q, _ = pw.truth_probabilities(basis, "null")
    groups = basis.groups
    rate = pw.EXTRA_SHOTS / len(basis.y)
    rows, copies, y, frame = pw.simulate_study(np.random.default_rng(0), basis, groups, q, pw.Scenario("combined", "null", "archived"), 0.0)
    assert len(np.unique(copies)) == len(groups)
    assert (pd.Series(basis.matches[rows]).groupby(copies).nunique() == 1).all()
    per_copy = np.bincount(copies)
    original = np.array([len(groups[np.flatnonzero(np.unique(basis.matches) == basis.matches[rows][copies == k][0])[0]]) for k in range(len(groups))])
    np.testing.assert_array_equal(per_copy, original + np.floor(rate * original + 0.5).astype(int))
    np.testing.assert_array_equal(frame, basis.frame_logit[rows])
    primary_rows, *_ = pw.simulate_study(np.random.default_rng(0), basis, groups, q, pw.Scenario("primary", "null", "weaker"), 0.0)
    assert len(primary_rows) == sum(len(groups[k]) for k in np.random.default_rng(0).integers(0, len(groups), len(groups)))


def test_null_truth_drops_the_frame_and_an_informative_frame_moves_h2_down():
    basis = synthetic_basis(frame_signal=1.0)
    q_null, coef_null = pw.truth_probabilities(basis, "null")
    q_dev, coef_dev = pw.truth_probabilities(basis, "development")
    assert len(coef_null) == 2 and len(coef_dev) == 3 and coef_dev[1] > 0.5
    null = pw.run_scenario(basis, pw.Scenario("primary", "null", "archived"), q_null, 0.0, n_sims=25, n_boot=300)
    dev = pw.run_scenario(basis, pw.Scenario("primary", "development", "archived"), q_dev, 0.0, n_sims=25, n_boot=300)
    assert dev["diff"].mean() < -0.01 < null["diff"].mean() and dev.superior.mean() > 0.8
    assert abs(null["diff"].mean()) < 0.003
    power = pw.power_table(pd.concat([null, dev]))
    assert list(power.scenario.unique()) == ["primary/null/archived", "primary/development/archived"] and (power.n_sims == 25).all()


def test_the_rule_note_is_pinned(tmp_path):
    assert pw.rule_note_sha256(ROOT) == pw.RULE_SHA256
    (tmp_path / pw.RULE_NOTE).parent.mkdir(parents=True)
    shutil.copy(ROOT / pw.RULE_NOTE, tmp_path / pw.RULE_NOTE)
    with (tmp_path / pw.RULE_NOTE).open("a") as note:
        note.write("\nzmiana\n")
    with pytest.raises(ValueError, match="pinned"):
        pw.rule_note_sha256(tmp_path)
