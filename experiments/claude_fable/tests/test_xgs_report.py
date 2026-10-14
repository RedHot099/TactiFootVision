"""The report readings of xg-stack (ticket 03): the recomputed power on the 172 matches and the arms' forest plot."""

import sys
from pathlib import Path
from statistics import NormalDist

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from xgs_report import PRACTICAL, ROWS, arm_rows, forest, power_on_test  # noqa: E402

RULE = {"success": {"interval": 0.95}, "equivalence": {"interval": 0.9, "margin": 0.005}}


def delta(point: float, sd: float) -> dict:
    return {"delta_ll": point, "lo95": point - 2 * sd, "hi95": point + 2 * sd, "boot_sd": sd, "draws": 2000}


def test_power_scales_the_se_by_the_shot_ratio_and_reads_the_normal_approximation():
    deltas = {"G": delta(-0.001, 0.0005), "GS": delta(0.0, 0.0005)}
    out = power_on_test(deltas, n_dev=400, n_test=100, rule=RULE)
    assert out["scale"] == pytest.approx(2.0)
    g = out["arms"]["G"]
    assert g["se_test"] == pytest.approx(0.001)
    n = NormalDist()
    # P(upper bound < 0) = Φ(−Δ/se − z), z = 1.96 for one variant and 2.24 for two under Bonferroni
    assert g["at"]["observed"]["one_variant"] == pytest.approx(n.cdf(1.0 - n.inv_cdf(0.975)))
    assert g["at"]["practical"]["two_variants_bonferroni"] == pytest.approx(n.cdf(-PRACTICAL / 0.001 - n.inv_cdf(0.9875)))
    assert out["arms"]["GS"]["at"]["observed"]["one_variant"] == pytest.approx(0.025)     # no effect: the one-sided level
    assert g["at"]["practical"]["two_variants_bonferroni"] < g["at"]["practical"]["one_variant"]


def summary() -> dict:
    deltas = {arm: delta(-0.0001 * i, 0.0004) for i, arm in enumerate(("G", "GS", "S", "Q", "ALL"), 1)}
    gbm = {arm: {"delta_vs_R": delta(0.001 * i, 0.001)} for i, arm in enumerate(("G", "GS"), 1)}
    return {"deltas": deltas, "gbm": gbm}


def test_rows_take_the_logistic_arms_then_the_gbm_sensitivity():
    rows = arm_rows(summary())
    assert [(r["arm"], r["model"]) for r in rows] == [(arm, model) for arm, model, _, _ in ROWS]
    assert rows[2]["delta_ll"] == pytest.approx(-0.0003)                 # S, logistic
    assert rows[-1]["delta_ll"] == pytest.approx(0.002)                  # GS, GBM against R


def test_the_forest_is_drawn(tmp_path):
    forest(arm_rows(summary()), -0.0025, 0.8, 2000, tmp_path / "forest.png")
    assert (tmp_path / "forest.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
