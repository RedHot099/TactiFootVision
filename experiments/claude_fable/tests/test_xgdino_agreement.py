"""xgpoc.dino.agreement on synthetic data (ticket 07)."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgpoc.dino import agreement as ag
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg

PROTOCOL = Path(__file__).resolve().parents[1] / "research/xgdino_ablation_protocol.json"


def test_tolerances_are_the_frozen_protocol_values():
    frozen = json.loads(PROTOCOL.read_text())["frozen"]["agreement"]
    tau = ag.tolerances(frozen)
    assert tau == {"distance": 2.0, "angle": 5.0, "near_opp_m": 1.5, "opp_3m": 0.0, "def_cone": 0.0,
                   "gk_depth_m": 1.0, "gk_lateral_m": 1.0}
    edited = json.loads(json.dumps(frozen))
    edited["tolerances"]["distance_m"] = 2.5
    with pytest.raises(ValueError):
        ag.tolerances(edited)


def test_known_error_correlation_recovered():
    rng = np.random.default_rng(1)
    n, rho = 4000, 0.6
    shared = rng.normal(size=n)
    e_det = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=n)
    e_probe = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=n)
    sb = rng.uniform(5, 30, n)
    games = np.repeat(np.arange(200), n // 200)
    out = ag.error_table(ag.BY_NAME["distance"], sb, sb + e_det, sb + 2 * e_probe, pr.draws(games, 200))
    # Spearman of a bivariate normal with Pearson rho: 6/pi * arcsin(rho / 2)
    expected = 6 / np.pi * np.arcsin(rho / 2)
    s = out["spearman_signed_errors"]
    assert s["lo"] < expected < s["hi"] and abs(s["point"] - expected) < 0.03
    assert out["probe_minus_detector_mae"]["lo"] > 0          # probe errors are twice as large
    assert abs(out["detector_mae"]["point"] - np.sqrt(2 / np.pi)) < 0.05


def test_cells_apply_the_frozen_thresholds():
    tau = 2.0
    det = np.array([0.5, 1.9, 2.0, 2.5, 3.0, 9.0, 10.0, 12.0, np.nan])   # q75 of present |e| = 9.25
    probe = np.array([0.1, 3.0, 0.2, 0.3, 5.0, 1.0, 2.0, 4.0, 1.0])
    dist = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
    rows, n = ag.cells(det, probe, dist, tau)
    assert n["detector_q75"] == pytest.approx(np.quantile(det[:8], 0.75))
    assert list(rows.detector) == ["right", "right", "right", "between", "between", "between", "wrong", "wrong", "missing"]
    assert n["detector_wrong__probe_right"] == 1 and n["detector_wrong__probe_wrong"] == 1
    assert n["detector_right__probe_wrong"] == 1 and n["detector_missing__probe_right"] == 1
    # the "wrong / right" shot (probe 2.0) does not beat the distance-only reference (1.0): not counted
    assert n["detector_wrong__probe_right__beats_distance"] == 0
    rows, n = ag.cells(det, probe, np.full(9, 5.0), tau)
    assert n["detector_wrong__probe_right__beats_distance"] == 1
    assert sum(n[f"detector_{s}"] for s in ag.DETECTOR_STATUS) == n["rows"] == 9


def test_count_reading_is_exact():
    f = ag.BY_NAME["opp_3m"]
    assert list(ag.reading(f, [0.4, 0.6, -0.7, 2.49, np.nan])[:4]) == [0, 1, 0, 2]
    _, n = ag.cells(np.abs(ag.reading(f, [0, 1, 3]) - [0, 0, 0]), np.abs(ag.reading(f, [0.4, 0.2, 0.3])), np.ones(3), 0.0)
    assert n["detector_right__probe_right"] == 1 and n["detector_wrong__probe_right"] == 1


def test_angle_errors_in_degrees():
    sb = np.full(50, 0.5)
    out = ag.error_table(ag.BY_NAME["angle"], sb, sb + np.deg2rad(3.0), sb + np.deg2rad(6.0), pr.draws(np.arange(50) // 5, 50))
    assert out["detector_mae"]["point"] == pytest.approx(3.0) and out["probe_mae"]["point"] == pytest.approx(6.0)


def test_strata_merge_small_ones():
    d = np.r_[np.arange(950.0), np.nan]
    s = ag.distance_strata(d, 10, 100)
    assert s[-1] == -1 and min(np.bincount(s[:-1])) >= 100 and set(s[:-1]) == set(range(s.max() + 1))
    assert len(set(ag.distance_strata(np.arange(1000.0)))) == 10


def test_permutation_null_calibrated_under_independence():
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(300):
        dist = rng.uniform(5, 30, 600)
        strata = ag.distance_strata(dist, 10, 50)
        # both flags depend on distance (as in the data) but are independent given it
        dw = rng.random(600) < 0.1 + dist / 100
        pr_ok = rng.random(600) < 0.6 - dist / 100
        ps.append(ag.permutation_null(dw, pr_ok, strata, n_perm=499, seed=int(rng.integers(1 << 30)))["p_one_sided"])
    ps = np.array(ps)
    assert np.mean(ps <= 0.05) <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / 300)
    assert np.mean(ps <= 0.5) == pytest.approx(0.5, abs=0.1)
    # and it sees a planted association
    dw = rng.random(600) < 0.2
    out = ag.permutation_null(dw, dw | (rng.random(600) < 0.3), ag.distance_strata(rng.uniform(5, 30, 600), 10, 50), 999)
    assert out["p_one_sided"] < 0.01 and out["fraction_permutable"] == 1.0


def test_timing_bands_merge_rule():
    r = np.r_[np.full(500, 10.0), np.full(150, 100.0), np.full(40, 400.0), [np.nan]]
    label, counts = ag.timing_bands(r)
    assert counts["raw"] == {"<=50": 500, "50-200": 150, ">200": 40, "missing": 1}
    assert counts["merged"] == {"<=50": 500, ">50": 190, "missing": 1}
    label, counts = ag.timing_bands(np.r_[np.full(500, 10.0), np.full(30, 100.0), np.full(40, 400.0)])
    assert counts["merged"] == {"<=50": 500, ">50": 70} or counts["merged"] == {"all": 570}
    label, counts = ag.timing_bands(np.r_[np.full(50, 10.0), np.full(150, 100.0), np.full(140, 400.0)])
    assert counts["merged"] == {"<=200": 200, ">200": 140}


def test_combination_rules_are_fixed():
    det = np.array([10.0, np.nan, 20.0])
    probe = np.array([12.0, 7.0, 18.0])
    assert list(ag.combine("mean", det, probe)) == [11.0, 7.0, 19.0]
    assert list(ag.combine("fill", det, probe)) == [10.0, 7.0, 20.0]


def test_nothing_fitted_to_statsbomb():
    """Every output that is a prediction is a function of the sources only: permuting StatsBomb changes errors, never
    predictions; a StatsBomb table is refused as a prediction source and by the probe data."""
    rng = np.random.default_rng(0)
    det, probe, dist = rng.normal(10, 3, 40), rng.normal(10, 3, 40), rng.normal(10, 3, 40)
    det[::7] = np.nan
    for rule in ("mean", "fill", "detector"):
        assert np.array_equal(ag.combine(rule, det, probe), ag.combine(rule, det, probe), equal_nan=True)
    sb = rng.normal(10, 3, 40)
    d = pr.draws(np.arange(40) // 4, 50)
    a = ag.flatten(ag.combination_table(ag.BY_NAME["distance"], sb, det, probe, dist, d))
    b = ag.flatten(ag.combination_table(ag.BY_NAME["distance"], rng.permutation(sb), det, probe, dist, d))
    assert not np.allclose(a.mae, b.mae)                 # scored against StatsBomb ...
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
             for n in ast.walk(ast.parse(Path(ag.__file__).read_text())) if isinstance(n, ast.Call)}
    assert not calls & {"statsbomb_targets", "fit", "lstsq", "solve", "ridge_paths", "run"}   # ... never fitted to it
    sbt = tg.tagged(pd.DataFrame({"distance": sb}), "statsbomb")
    with pytest.raises(ValueError):
        ag.check_sources(sbt)
    with pytest.raises(ValueError):
        pr.ProbeData(sbt, np.zeros(40), np.zeros(40))
