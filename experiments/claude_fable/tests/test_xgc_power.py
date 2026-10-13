"""The xg-compare power simulation (ticket 01): its statistics against direct computations, and its error rates on
synthetic data where the truth is known."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from xgpoc.bundle import sha256_file
from xgpoc.compare import categories as cat
from xgpoc.compare import power as pw
from xgpoc.compare import protocol as pr
from xgpoc.vendor import ladder as ld


def synthetic_basis(n_matches: int = 60, per_match: int = 30, n_boot: int = 300, seed: int = 0) -> pw.Basis:
    """Shots with match-clustered method logits and categories drawn at random (no category effect)."""
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    match = np.repeat(np.arange(n_matches), per_match)
    v = rng.normal(-2, 1, n)
    z = {"V": v, "D": 0.8 * v + rng.normal(0, 0.5, n), "P": v + rng.normal(0, 0.3, n),
         "F": v + rng.normal(0, 0.3, n_matches)[match] + rng.normal(0, 0.6, n)}
    categories = pd.DataFrame({c.column: rng.choice(np.array(c.levels, dtype=object), n) for c in cat.PRE_SHOT})
    counts = ld.bootstrap(np.arange(n_matches), n_boot).counts
    half = np.where(match % 2 == 0, "discovery", "validation")
    return pw.Basis(z, {k: v + 0.1 for k, v in z.items()}, match, half, categories, np.vstack([np.ones(n_matches), counts]).astype(float))


# ------------------------------------------------------------------ statistics
def test_adjusted_effect_equals_least_squares_on_the_resampled_rows():
    rng = np.random.default_rng(3)
    n, m = 300, 30
    match = rng.integers(0, m, n)
    control, y = rng.normal(size=n), rng.normal(size=n)
    inside = rng.random((n, 2)) < 0.3
    defined = np.column_stack([np.ones(n, bool), rng.random(n) < 0.8])
    counts = rng.integers(0, 3, (4, m)).astype(float)
    effects = pw.AdjustedEffects(inside, defined, control, match, np.vstack([np.ones(m), counts]))
    point, draws = effects(y)
    for level in range(2):
        for row, weights in enumerate([np.ones(m), *counts]):
            keep = defined[:, level]
            repeat = weights[match[keep]].astype(int)
            x = np.column_stack([np.ones(keep.sum()), inside[keep, level], control[keep]])
            beta = np.linalg.lstsq(np.repeat(x, repeat, axis=0), np.repeat(y[keep], repeat), rcond=None)[0]
            got = point[level] if row == 0 else draws[row - 1, level]
            assert got == pytest.approx(beta[1], abs=1e-9)


def test_planting_moves_the_adjusted_effect_by_exactly_the_planted_amount():
    rng = np.random.default_rng(4)
    n, m = 400, 40
    match = rng.integers(0, m, n)
    inside = rng.random((n, 1)) < 0.25
    effects = pw.AdjustedEffects(inside, np.ones((n, 1), bool), rng.normal(size=n), match,
                                 np.vstack([np.ones(m), ld.bootstrap(np.arange(m), 50).counts]))
    y = rng.normal(size=n)
    (p0, d0), (p1, d1) = effects(y), effects(y + 0.15 * inside[:, 0])
    assert p1 - p0 == pytest.approx(0.15) and np.allclose(d1 - d0, 0.15)


def test_weighted_spearman_equals_spearman_of_the_resampled_shots():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=40), rng.normal(size=40)
    a[3] = a[7]                                         # a tie
    weights = rng.integers(0, 4, (3, 40)).astype(float)
    got = pw.spearman(a, b, weights)
    for row, w in enumerate(weights):
        repeat = w.astype(int)
        assert got[row] == pytest.approx(spearmanr(np.repeat(a, repeat), np.repeat(b, repeat)).statistic, abs=1e-12)


def test_p_values_follow_the_protocol_sides():
    draws = np.column_stack([np.r_[np.full(90, 0.2), np.full(10, -0.1)]] * 3)
    assert list(pw.p_values(draws, ["greater", "less", "two-sided"])) == pytest.approx([0.10, 0.90, 0.20])


def test_bh_by_and_holm_on_known_p_values():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    assert list(np.flatnonzero(pw.bh(p, 0.05))) == [0, 1]                       # 0.039 > 3/8 * 0.05
    assert list(np.flatnonzero(pw.bh(np.array([0.01, 0.02, 0.03]), 0.05))) == [0, 1, 2]
    assert list(np.flatnonzero(pw.bh(np.array([0.01, 0.02, 0.03]), 0.05, dependent=True))) == []   # c(3) = 1.833
    assert list(np.flatnonzero(pw.holm(np.array([0.01, 0.04, 0.03, np.nan]), 0.05))) == [0]          # 0.03 > 0.05 / 2


def test_frozen_main_scale_applies_each_folds_coefficients():
    rows = pd.DataFrame({"match": ["a", "b", "a"], "z_vendor": [0.0, 1.0, 2.0], "z_positions": 0.0, "z_dist_angle": 0.0, "z_frame": 1.0})
    folds = pd.DataFrame({"match": ["a", "b"], "fold": [1, 0]})
    coefficients = {rung: [[0.0, 1.0], [0.0, 1.0]] for rung, _ in pw.MAIN_SCALE.values()} | {"vendor": [[1.0, 2.0], [0.5, 3.0]]}
    z = pw.frozen_main_scale(rows, coefficients, folds)
    assert list(z["V"]) == [0.5, 3.0, 6.5]
    with pytest.raises(ValueError, match="fold"):
        pw.frozen_main_scale(rows.assign(match=["a", "c", "a"]), coefficients, folds)


# ------------------------------------------------------------------ error rates
def test_zero_injected_effect_rejects_about_alpha():
    """Per test about alpha. The percentile bootstrap's far tail is a little thin with clustered data, so BH's chance of
    any rejection under the global null runs above q (about 0.08-0.10 here); the real run reports its own rate."""
    basis = synthetic_basis(n_matches=135, per_match=20, n_boot=500)
    deviations = pw.correlation_difference_deviations(basis)
    records = pd.concat([pw.Study(basis, deviations, np.random.default_rng([7, s])).run_family(()).assign(study=s) for s in range(150)])
    assert records.tested.all()
    assert 0.03 <= (records.p <= 0.05).mean() <= 0.08
    assert records.groupby("study").reject_bh.any().mean() <= 0.15


def test_a_planted_effect_well_above_the_noise_is_found_by_the_family_and_the_scan():
    basis = synthetic_basis(n_matches=80, per_match=40)
    deviations = pw.correlation_difference_deviations(basis)
    study = pw.Study(basis, deviations, np.random.default_rng([0, 0]))
    assert study.run_family(("3", "8")).set_index("item").loc[["3", "8"]].reject_bh.all()
    level = pr.scan_levels().index(("body", "Head"))
    gap_items = [i.id for i in pr.FAMILY]
    assert not study.run_scan().any()
    assert study.run_scan(planted_level=level)[level]
    assert study.run_scan(gap_items).sum() >= 1


def test_summary_reports_every_item_and_the_scan():
    basis = synthetic_basis(n_matches=40, per_match=20, n_boot=100)
    family, scan, levels = pw.simulate(basis, [0, 1])
    summary = pw.summarise(family, scan, levels)
    assert set(summary["items"]) == {i.id for i in pr.FAMILY} and summary["n_studies"] == 2
    assert set(summary["scan"]) == {"null", "all"}
    assert {"power_bh_logit_shift"} <= set(summary["items"]["10"]) and "power_bh_logit_shift" not in summary["items"]["3"]
    assert len(levels) and {lv["category"] for lv in summary["scan_levels"]["levels"]} <= {c for c, _ in pr.scan_levels()}
    assert all(0 <= lv["power"] <= lv["meets_floors_share"] <= 1 for lv in summary["scan_levels"]["levels"])
    json.dumps(summary, allow_nan=False)


# ------------------------------------------------------------------ protocol
def test_every_family_item_names_a_dictionary_level_and_the_scan_skips_post_shot_categories():
    for item in pr.FAMILY:
        if item.category is not None:
            assert item.level in cat.BY_NAME[item.category].levels
    scanned = {c for c, _ in pr.scan_levels()}
    assert not scanned & {"on_target_sn", "sb_outcome", "half", "rebound_3s", "rebound_subtype"}
    assert len(pr.FAMILY) == 14 and [i.id for i in pr.FAMILY if i.side != "two-sided"] == ["2", "8", "9", "10", "16"]
    assert pr.BY_ID["2"].injection == -0.15 and pr.BY_ID["10"].injection == 0.005 and pr.BY_ID["9"].injection == -0.10


def test_the_design_transcribes_c4():
    design = pr.design()
    assert design["bootstrap"]["draws"] == 2000 and design["bootstrap"]["seed"] == 0 and design["scan"]["split"]["seed"] == 20261013
    assert design["floors"]["category_in"] == 50 and design["floors"]["scan_half_in"] == 50 and design["floors"]["label_goals"] == 15
    assert design["thresholds"] == {**pr.THRESHOLDS, "meaning": design["thresholds"]["meaning"]}
    items = {i["id"]: i for i in design["family"]["items"]}
    assert set(items) == {"2", "3", "4", "5", "6a", "7", "8", "9", "10", "11", "12", "14", "15", "16"}
    assert (items["2"]["side"], items["8"]["side"], items["9"]["side"], items["10"]["side"]) == ("less", "greater", "less", "greater")
    assert items["5"]["threshold"] == 0.15 and items["16"]["threshold"] == 0.005 and items["8"]["threshold"] == 0.10
    assert set(design["outside_family"]) == {"1", "13", "6b", "negative_control"}
    json.dumps(design, allow_nan=False)


def test_an_item_must_name_a_dictionary_category():
    with pytest.raises(ValueError, match="not a level"):
        pr.Item("x", "x", "gap", "weather", "rain", "two-sided")
    with pytest.raises(ValueError, match="kind or side"):
        pr.Item("x", "x", "gap", "body", "Head", "sideways")


def test_code_hashes_cover_the_compare_package_and_the_script():
    from xgpoc.compare import records as xr
    record = xr.code_hashes(xr.EXPERIMENT / "scripts/xgc_power.py")
    assert {"xgpoc/compare/power.py", "xgpoc/vendor/ladder.py", "scripts/xgc_power.py"} <= set(record["files"])
    assert len(record["digest"]) == 64


def test_protocol_load_refuses_another_hash(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(pr.design()))
    assert pr.load(path, sha256_file(path))["family"]["population"] == "primary"
    with pytest.raises(ValueError, match="SHA-256"):
        pr.load(path, "0" * 64)
