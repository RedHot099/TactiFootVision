"""Information ladder and H2 (xg-vendor ticket 08) on synthetic evaluation tables whose answer is known by construction."""

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from xgpoc.eval.calibration import logit, sigmoid
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import evaltable_dev, ladder_dev
from xgpoc.vendor import ladder as ld
from xgpoc.vendor.draws import table_draws

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_complexity import game_draws  # noqa: E402
from significance import paired_bootstrap  # noqa: E402
from vendor_ladder import analysis_parameters  # noqa: E402


def synthetic_table(n_matches: int = 100, per_match: int = 25, frame_signal: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Labels from sigmoid(z_pos + frame_signal * s). Positions see z_pos only; the frame sees part of z_pos, noise and,
    when `frame_signal` > 0, the frame-only signal s. dist_angle is a miscalibrated monotone copy of the positions."""
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    match = np.repeat([f"m{k:03d}" for k in range(n_matches)], per_match)
    z_pos = rng.normal(-2.2, 1.1, n)
    s = rng.normal(0, 1, n)
    q = sigmoid(z_pos + frame_signal * s)
    y = rng.random(n) < q
    return pd.DataFrame({
        "key": [f"s{i:05d}" for i in range(n)], "match": match, "population": "primary",
        "split": np.where(np.arange(n) % 4 == 0, "test", "train"), "frame_status": "ok",
        "frame_logit_0": 0.6 * z_pos + frame_signal * s + rng.normal(0, 0.5, n),
        "pred_constant": 0.12, "pred_dist_angle": sigmoid(0.5 * z_pos - 1.5), "pred_event": sigmoid(z_pos),
        "pred_freeze_frame": sigmoid(z_pos), "pred_vendor_xg": sigmoid(z_pos + rng.normal(0, 0.3, n)),
        "goal_sn": y, "on_target_sn": y, "sb_outcome": np.where(y, "Goal", "Saved"), "is_penalty": False, "q": q})


def primary_ladder(table: pd.DataFrame, margin: float, n_boot: int, **kwargs) -> ld.LadderResult:
    return ld.analyse(table, "primary", ("primary",), margin, table_draws(table, n_boot), **kwargs)


def kl(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


def contrast_row(result: ld.LadderResult, variant: str, a: str, b: str, metric: str = "log_loss") -> pd.Series:
    c = result.contrasts
    hit = c[c.variant_a.eq(variant) & c.a.eq(a) & c.b.eq(b) & c.metric.eq(metric)]
    assert len(hit) == 1
    return hit.iloc[0]


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_counts_follow_the_significance_draw_sequence():
    games = np.array(["c", "a", "b", "a", "c", "c", "d"])
    boot = ld.bootstrap(games, n_boot=40, seed=0)
    assert list(boot.keys) == ["a", "b", "c", "d"] and list(boot.match) == [2, 0, 1, 0, 2, 2, 3]
    per_match = np.bincount(boot.match)
    for b, rows in enumerate(game_draws(games, 40, seed=0)):
        np.testing.assert_array_equal(np.bincount(boot.match[rows], minlength=4) // per_match, boot.counts[b])


def test_metric_draws_reproduce_the_significance_module():
    """Same matches drawn, same metrics: the paired deltas of scripts/significance.py are the negated draws here."""
    table = synthetic_table(n_matches=30, per_match=12)
    y, games = table.goal_sn.to_numpy(int), table.match.to_numpy()
    pa, pb = table.pred_event.to_numpy(), table.pred_dist_angle.to_numpy()
    boot = ld.bootstrap(games, n_boot=300)
    da, db = ld.metric_draws(y, pa, boot), ld.metric_draws(y, pb, boot)
    frame = lambda p: pd.DataFrame({"shot_id": table.key, "goal": y, "p": p})   # noqa: E731
    reference = paired_bootstrap(frame(pa), frame(pb), table[["key", "match"]].rename(columns={"key": "shot_id", "match": "game"}), n_boot=300, seed=0)
    for metric, sign in (("log_loss", -1), ("brier", -1), ("auc", 1)):   # significance: positive = a better
        mine = sign * (da[metric] - db[metric])
        np.testing.assert_allclose([np.median(mine), *np.percentile(mine, [2.5, 97.5])],
                                   [reference[f"d_{metric}"], reference[f"d_{metric}_lo"], reference[f"d_{metric}_hi"]], atol=1e-12)


def test_point_metrics_equal_the_unweighted_metrics():
    table = synthetic_table(n_matches=20, per_match=15)
    y, p = table.goal_sn.to_numpy(int), table.pred_dist_angle.to_numpy()
    point = ld.metric_draws(y, p, ld.observed(ld.bootstrap(table.match.to_numpy(), n_boot=1)))
    assert point["log_loss"][0] == pytest.approx(log_loss(y, p), abs=1e-12)
    assert point["auc"][0] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert point["cal_slope"][0] == pytest.approx(evaltable_dev.calibration_slope(y.astype(float), p), abs=1e-7)
    assert point["mean_p_minus_rate"][0] == pytest.approx(p.mean() - y.mean(), abs=1e-12)


def test_a_population_set_reads_the_shared_table_draws():
    """A subset (here the test matches) is weighted by the whole table's draws, as every other analysis of the table."""
    table = synthetic_table(n_matches=40, per_match=10)
    shared = table_draws(table, n_boot=100)
    test_rows = table[table.split.eq("test")]
    subset = ld.draws_for(shared, test_rows.match.to_numpy())
    np.testing.assert_array_equal(subset.counts, shared.counts)
    np.testing.assert_array_equal(shared.keys[subset.match], test_rows.match.to_numpy())
    y, p = test_rows.goal_sn.to_numpy(float), test_rows.pred_event.to_numpy()
    weights = shared.weights(np.flatnonzero(table.split.eq("test")))
    expected = (weights * ld.per_shot_log_loss(y, p)).sum(1) / weights.sum(1)
    np.testing.assert_allclose(ld.metric_draws(y, p, subset, metrics=("log_loss",))["log_loss"], expected, rtol=1e-12)
    with pytest.raises(ValueError, match="not among"):
        ld.draws_for(shared, np.array(["zzz"]))


def test_a_draw_without_goals_is_refused():
    games = np.repeat(["a", "b", "c"], 4)
    y = np.r_[np.ones(4), np.zeros(8)]
    with pytest.raises(ValueError, match="no goal"):
        ld.metric_draws(y, np.full(12, 0.3), ld.bootstrap(games, n_boot=50), metrics=("log_loss",))


# ---------------------------------------------------------------- cross-fitting

def test_match_folds_put_each_match_in_one_fold_fixed_by_seed():
    games = np.repeat([f"m{k:02d}" for k in range(23)], 3)
    folds = ld.match_folds(games, seed=4)
    assert (pd.Series(folds).groupby(games).nunique() == 1).all()
    assert sorted(np.bincount(folds[::3])) == [4, 4, 5, 5, 5]
    np.testing.assert_array_equal(folds, ld.match_folds(games, seed=4))
    assert not np.array_equal(folds, ld.match_folds(games, seed=5))


def test_crossfitting_never_uses_a_shots_own_match():
    """Flipping every label of one match leaves that match's recalibrated and stacked predictions unchanged."""
    table = synthetic_table(n_matches=30, per_match=15, frame_signal=0.5)
    folds = ld.match_folds(table.match.to_numpy())
    before, _ = ld.rung_predictions(table, folds)
    own = table.match.eq("m007").to_numpy()
    flipped = table.assign(goal_sn=np.where(own, ~table.goal_sn, table.goal_sn))
    after, _ = ld.rung_predictions(flipped, folds)
    same_fold = (folds == folds[own][0]) & ~own
    for rung in before:
        if rung.variant == "recalibrated":
            np.testing.assert_array_equal(before[rung][own], after[rung][own])
            np.testing.assert_array_equal(before[rung][same_fold], after[rung][same_fold])
            assert not np.allclose(before[rung][~same_fold & ~own], after[rung][~same_fold & ~own])


# ---------------------------------------------------------------- the ladder

def test_ladder_recovers_planted_gaps():
    """Raw: the miscalibrated copy loses KL(q || p) to the truth. Recalibrated: the copy ties the truth, and the
    constant loses the information KL(q || mean q)."""
    table = synthetic_table(n_matches=120, per_match=25)
    result = primary_ladder(table, 0.003, 400)
    q = table.q.to_numpy()
    planted_raw = kl(q, table.pred_dist_angle.to_numpy()).mean()
    planted_constant = kl(q, np.full(len(q), q.mean())).mean()

    raw = contrast_row(result, "raw", "event", "dist_angle")       # event - dist_angle: truth minus the copy
    assert raw["lo95"] < -planted_raw < raw["hi95"] and raw["hi95"] < 0
    recal = contrast_row(result, "recalibrated", "event", "dist_angle")
    assert recal["lo95"] < 0 < recal["hi95"] and abs(recal["diff"]) < 0.002
    constant = contrast_row(result, "recalibrated", "event", "constant")
    assert constant["lo95"] < -planted_constant < constant["hi95"] and constant["hi95"] < 0
    m = result.metrics.set_index(["variant", "rung"])
    assert np.isnan(m.loc[("raw", "constant"), "cal_slope"]) and m.loc[("raw", "event"), "auc"] > 0.7
    assert set(m.index) == {(v, r) for v in ld.VARIANTS for r in ld.LADDER} - {("raw", s) for s in ld.STACKS}


def test_tost_on_planted_draws():
    rng = np.random.default_rng(0)
    sd, margin = 0.001, 0.004
    zero = ld.contrast(0.0, rng.normal(0, sd, 5000), "log_loss", margin)
    twice = ld.contrast(2 * margin, rng.normal(2 * margin, sd, 5000), "log_loss", margin)
    assert zero["equivalent"] and zero["p_tost"] < 0.01 and zero["smallest_equivalence_margin"] == pytest.approx(0.0017, abs=2e-4)
    assert not twice["equivalent"] and twice["p_tost"] > 0.99 and not twice["superior"]
    better = ld.contrast(-2 * margin, rng.normal(-2 * margin, sd, 5000), "log_loss", margin)
    assert better["superior"] and not better["equivalent"]


def best_logistic_kl(q: np.ndarray, x: np.ndarray) -> float:
    """Expected excess log-loss over the truth q of the population-best logistic model on features x (soft labels)."""
    x = x.reshape(len(q), -1)
    model = LogisticRegression(C=1e6, max_iter=5000).fit(np.r_[x, x], np.r_[np.ones(len(q)), np.zeros(len(q))], sample_weight=np.r_[q, 1 - q])
    return float(kl(q, model.predict_proba(x)[:, 1]).mean())


def oracle_gain(table: pd.DataFrame) -> float:
    """The planted H2 gap: expected log-loss of the best positions-only Platt minus that of the best stack."""
    q, z_pos = table.q.to_numpy(), logit(table.pred_freeze_frame.to_numpy())
    return best_logistic_kl(q, z_pos) - best_logistic_kl(q, np.column_stack([table.frame_logit_0.to_numpy(), z_pos]))


def test_h2_concludes_equivalence_for_a_useless_frame_and_not_at_twice_the_margin():
    useless = primary_ladder(synthetic_table(n_matches=120, per_match=25), 0.005, 400).summary["h2"]
    assert useless["equivalent"] and not useless["superior"] and useless["lo90"] > -0.005 and useless["hi90"] < 0.005

    informative = synthetic_table(n_matches=120, per_match=25, frame_signal=0.6)
    gain = oracle_gain(informative)
    h2 = primary_ladder(informative, gain / 2, 400).summary["h2"]
    assert gain > 0.005
    assert not h2["equivalent"] and h2["superior"] and h2["lo95"] < -gain < h2["hi95"]


def test_h2_draws_are_the_analysis_numbers():
    table = synthetic_table(n_matches=40, per_match=20, frame_signal=0.3)
    h2 = primary_ladder(table, 0.003, 300, fold_seed=3).summary["h2"]
    boot = ld.bootstrap(table.match.to_numpy(), n_boot=300)
    point, draws = ld.h2_draws(table.goal_sn.to_numpy(), table.pred_freeze_frame.to_numpy(), table.frame_logit_0.to_numpy(),
                               table.match.to_numpy(), boot, fold_seed=3)
    assert point == pytest.approx(h2["diff"], abs=1e-12)
    np.testing.assert_allclose(np.percentile(draws, [5, 95]), [h2["lo90"], h2["hi90"]], atol=1e-12)


def test_analysis_population_drops_penalties_and_unscored_frames():
    table = synthetic_table(n_matches=10, per_match=6)
    table.loc[0, "is_penalty"] = True
    table.loc[1, ["frame_status", "frame_logit_0"]] = ["0:anchor_off_tolerance", np.nan]
    table.loc[2, "population"] = "statsbomb_only"
    rows, exclusions = ld.analysis_population(table, ("primary",))
    assert exclusions == {"chosen": 59, "penalty": 1, "no_frame": 1, "analysed": 57}
    assert ld.analysis_population(table, ld.POPULATION_SETS["combined"])[1]["analysed"] == 58
    assert ld.analysis_population(table, ("primary",), splits=("test",))[0].split.eq("test").all()
    with pytest.raises(ValueError, match="labels joined"):
        ld.analysis_population(table.drop(columns="goal_sn"), ("primary",))


def test_contrast_pairs_keep_stacks_recalibrated_and_include_h2():
    pairs = ld.contrast_pairs()
    assert len(pairs) == 28 + 15 and ld.H2 in pairs and ld.H2_VENDOR in pairs
    assert all(a.variant == b.variant for a, b in pairs)
    assert not any(r.variant == "raw" and r.name in ld.STACKS for pair in pairs for r in pair)


def test_prototype_comparison_flags_cells_beyond_one_standard_error():
    result = primary_ladder(synthetic_table(n_matches=30, per_match=12), 0.003, 200)
    names = {v: k for k, v in ladder_dev.PROTOTYPE_RUNGS.items()}
    proto = result.metrics.assign(population="pooled_power_projection", rung=[names.get((v, r)) for v, r in zip(result.metrics.variant, result.metrics.rung, strict=True)])
    proto = proto[proto.rung.notna()].copy()
    same = ladder_dev.compare_metrics(result.metrics, proto)
    assert len(same) == 11 * 3 and same.within_noise.all() and (same.point == same.proto_point).all()
    se = (proto.log_loss_hi - proto.log_loss_lo) / ladder_dev.Z95_WIDTH
    proto.loc[proto.rung.eq("r_event"), "log_loss"] += 2 * se[proto.rung.eq("r_event")]
    moved = ladder_dev.compare_metrics(result.metrics, proto)
    assert set(moved[~moved.within_noise].prototype_rung + "/" + moved[~moved.within_noise].metric) == {"r_event/log_loss"}


def ladder_args(**given) -> argparse.Namespace:
    return argparse.Namespace(**({"margin": None, "n_boot": None, "fold_seed": None, "boot_seed": None, "prototype": None,
                                  "split_groups": False} | given))


def test_parameters_come_from_the_preregistration_record_for_confirmatory_and_dryrun_tables():
    prereg = et.PreRegistration("a" * 64, datetime(2026, 10, 12, tzinfo=UTC), "b" * 64, margin=0.002, n_boot=6000, fold_seed=3,
                                boot_seed=4, draft=True)
    for kind in ("confirmatory", "dryrun"):
        params = analysis_parameters(kind, prereg, ladder_args())
        assert (params["margin"], params["n_boot"], params["fold_seed"], params["boot_seed"]) == (0.002, 6000, 3, 4)
        with pytest.raises(ValueError, match="needs --prereg"):
            analysis_parameters(kind, None, ladder_args(margin=0.005))
        with pytest.raises(ValueError, match=r"fixes the parameters; drop \['margin'\]"):
            analysis_parameters(kind, prereg, ladder_args(margin=0.005))
        with pytest.raises(ValueError, match="no development options"):
            analysis_parameters(kind, prereg, ladder_args(split_groups=True))
        with pytest.raises(ValueError, match="at least 5000 draws"):
            analysis_parameters(kind, replace(prereg, n_boot=1000), ladder_args())


def test_a_development_run_takes_the_margin_from_the_command_line():
    params = analysis_parameters("development", None, ladder_args(margin=0.005, n_boot=200))
    assert (params["margin"], params["n_boot"], params["fold_seed"], params["boot_seed"]) == (0.005, 200, ld.FOLD_SEED, ld.BOOT_SEED)
    with pytest.raises(ValueError, match="needs --margin"):
        analysis_parameters("development", None, ladder_args())
