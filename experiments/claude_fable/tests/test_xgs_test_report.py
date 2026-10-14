"""The readings of the xg-stack test report (ticket 04): keys without dots, every interval and scenario carried over."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgs_test_report as rp  # noqa: E402


def delta(point: float) -> dict:
    return {"shots": 10, "delta_ll": point, "ll_variant": 0.3, "ll_base": 0.31, "boot_sd": 0.001, "draws": 5000,
            "intervals": {"0.95": [-0.002, 0.001], "0.975": [-0.003, 0.002], "0.99": [-0.004, 0.003],
                          "0.995": [-0.005, 0.004], "0.9": [-0.0015, 0.0005]}}


def test_levels_and_margins_lose_their_dots_without_colliding():
    assert [rp.undotted(k) for k in ("0.95", "0.975", "0.99", "0.995", "0.9")] == ["95", "975", "99", "995", "90"]
    assert rp.undotted("p_equivalent_0.002") == "p_equivalent_002" and rp.undotted("p_success_0.9") == "p_success_90"


def test_a_delta_keeps_every_interval_as_lo_hi_per_level():
    flat = rp.flat_delta(delta(-0.001))
    assert (flat["lo95"], flat["hi95"], flat["lo90"], flat["hi90"], flat["lo995"]) == (-0.002, 0.001, -0.0015, 0.0005, -0.005)
    assert not any("." in k for k in flat)


def test_the_power_and_the_readings_carry_every_scenario_and_label():
    power = {"standard_errors": {"a": 0.1, "b": 0.2}, "scenarios": [
        {"se": s, "se_value": 0.1, "effect": e, "delta": -0.001, "p_success_0.95": 0.3, "p_equivalent_0.002": 0.5}
        for s in ("a", "b") for e in ("x", "y")], "recovery": 0.8, "development": {"shots": 7}}
    flat = rp.flat_power(power)
    assert flat["b"]["se"] == 0.2 and flat["a"]["y"] == {"delta": -0.001, "p_success_95": 0.3, "p_equivalent_002": 0.5}
    result = {"level_in_force": 0.95, "population": {"shots": 10},
              "primary": {"delta": delta(-0.001), "sentence": "s",
                          "reading": {"label": "inconclusive", "equivalent": {"0.002": False, "0.005": True},
                                      "smallest_equivalence_margin": 0.0015}},
              "descriptive": {"GS_vs_R": delta(0.0), "G_vs_g_combo": delta(0.0), "R_vs_g_combo": delta(0.0),
                              "G_vs_R_by_role": {"evaluation": delta(-0.002), "gate": delta(0.001)}, "scores": {"R": {}}}}
    out = rp.readings(result, {"shots": 3, "spearman": {"R": 0.6}})
    assert out["primary"]["label"] == "inconclusive" and out["primary"]["equivalent_005"] is True
    assert out["descriptive"]["evaluation"]["delta_ll"] == -0.002 and out["statsbomb"]["spearman"]["R"] == 0.6
    gate = {"false_success_rate": {"0.95": 0.01, "0.975": 0.0}, "mc_se": {"0.95": 0.001}, "false_worse_rate_95": 0.2,
            "mean_null_delta": 0.0006, "replicates": 5000, "seed": 301, "level_in_force": 0.95}
    assert rp.before_scoring(gate, power)["gate"]["false_success"] == {"95": 0.01, "975": 0.0}
