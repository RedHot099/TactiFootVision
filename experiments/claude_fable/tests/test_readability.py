"""Readability curves, probes, event study and controls of the direction probe (xg-credibility 09) on synthetic
tables whose answer is known by construction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from xgpoc.probe import readability as rd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_complexity import game_draws  # noqa: E402

GRID = np.array(rd.GRID, float)
J0 = rd.OFFSET_INDEX[0]


def synthetic_shots(n_matches: int = 80, per_match: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    game = np.repeat([f"m{k:03d}" for k in range(n_matches)], per_match)
    p_match = rng.uniform(0.3, 0.6, n_matches)
    y = (rng.random(len(game)) < np.repeat(p_match, per_match)).astype(np.int8)
    split = np.where(np.repeat(np.arange(n_matches) % 4 == 0, per_match), "test", "train")
    return pd.DataFrame({"shot_id": [f"s{i:05d}" for i in range(len(game))], "game": game, "split": split, "y": y,
                         "in_population": True, "goal": False, "rebound_chain": rng.random(len(game)) < 0.2,
                         "no_cut": rng.random(len(game)) < 0.9})


def centring_factor(shots: pd.DataFrame, member: np.ndarray | None = None) -> float:
    """D of z = y exactly: centring shrinks the per-shot contrast to this factor."""
    p = shots.groupby("game").y.transform("mean").to_numpy()
    m = np.ones(len(shots), bool) if member is None else member
    y = shots.y.to_numpy()
    return float((1 - p)[m & (y == 1)].mean() + p[m & (y == 0)].mean())


def planted_logits(shots: pd.DataFrame, gap: np.ndarray, noise: float = 0.05, seed: int = 1) -> np.ndarray:
    """z = match effect + gap(o) * y + noise."""
    rng = np.random.default_rng(seed)
    match_effect = shots.game.map({g: rng.normal(0, 2) for g in shots.game.unique()}).to_numpy()
    return match_effect[:, None] + gap[None, :] * shots.y.to_numpy()[:, None] + rng.normal(0, noise, (len(shots), len(gap)))


def event_gap(pre_level: float = 0.3, slope: float = 2e-4, step: float = 1.0) -> np.ndarray:
    """A linear pre-contact trend plus a step of `step` from +40 ms on (contacts all after the anchor)."""
    return pre_level + slope * GRID + step * (GRID > 0)


# ---------------------------------------------------------------- bootstrap

def test_match_draw_counts_match_the_significance_draw_sequence():
    games = np.array(["c", "a", "b", "a", "c", "c", "d"])
    keys, counts = rd.match_draw_counts(games, n_boot=50, seed=0)
    assert list(keys) == ["a", "b", "c", "d"]
    inv = np.searchsorted(keys, games)
    per_match = np.bincount(inv, minlength=len(keys))
    for b, rows in enumerate(game_draws(games, 50, seed=0)):
        np.testing.assert_array_equal(np.bincount(inv[rows], minlength=len(keys)) // per_match, counts[b])


def test_weighted_gap_equals_the_gap_on_explicitly_resampled_matches():
    shots = synthetic_shots(n_matches=12, per_match=6)
    z = planted_logits(shots, event_gap(), noise=0.5)
    y = shots.y.to_numpy()
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=5)
    match = np.searchsorted(keys, shots.game.to_numpy())
    pop = np.ones(len(shots), bool)
    draws = rd.gap(rd.match_sums(z, y, match, len(keys), pop, pop), counts)
    zc = z - pd.DataFrame(z).groupby(match).transform("mean").to_numpy()   # centring uses the original match
    for b in range(5):
        rows = np.concatenate([np.flatnonzero(match == m) for m in range(len(keys)) for _ in range(counts[b, m])])
        expected = zc[rows][y[rows] == 1].mean(0) - zc[rows][y[rows] == 0].mean(0)
        np.testing.assert_allclose(draws[b], expected, atol=1e-12)


def test_weighted_auc_equals_sklearn_with_sample_weights_and_ties():
    rng = np.random.default_rng(3)
    score = np.round(rng.normal(size=300), 1)          # many ties
    y = (rng.random(300) < 0.4).astype(int)
    match = rng.integers(0, 30, 300)
    counts = rng.integers(0, 3, (4, 30))
    assert rd.weighted_auc(score, y, match) == pytest.approx(roc_auc_score(y, score))
    got = rd.weighted_auc(score, y, match, counts)
    for b in range(4):
        assert got[b] == pytest.approx(roc_auc_score(y, score, sample_weight=counts[b, match]))


def test_draw_without_a_class_is_skipped():
    shots = synthetic_shots(n_matches=3, per_match=4)
    shots["y"] = np.repeat([1, 0, 0], 4).astype(np.int8)       # only match 0 has on-target shots
    z = planted_logits(shots, event_gap())
    keys = np.unique(shots.game)
    counts = np.array([[1, 1, 1], [0, 2, 1]])
    pop = np.ones(len(shots), bool)
    d = rd.gap(rd.match_sums(z, shots.y.to_numpy(), np.searchsorted(keys, shots.game), 3, pop, pop), counts)
    assert np.isfinite(d[0]).all() and np.isnan(d[1]).all()
    assert rd.interval(d)[2] == 1


# ---------------------------------------------------------------- planted effects

def test_planted_gap_and_post_contact_jump_are_recovered():
    shots = synthetic_shots()
    gap = event_gap(pre_level=0.3, slope=2e-4, step=1.0)
    z = planted_logits(shots, gap)
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=200)
    curves = rd.readability_curves({"X": (z, shots.y.to_numpy())}, shots, keys, counts, {"X": ("all",)})
    d = curves.point["X", "all"]
    factor = centring_factor(shots)
    np.testing.assert_allclose(d, gap * factor, atol=0.01)
    tab = rd.curve_table(curves)
    pre = tab[tab.offset_ms.isin(rd.PRE)]
    post = tab[tab.offset_ms.isin(rd.POST)]
    assert np.abs(pre.D_minus_P).max() < 0.01
    np.testing.assert_allclose(post.D_minus_P, factor, atol=0.02)
    assert (post.D_minus_P_lo > 0).all() and (pre.D_minus_P_lo < 0).all() and (pre.D_minus_P_hi > 0).all()
    assert rd.jump(d) == pytest.approx(factor * (1 + 2e-4 * np.mean(rd.POST)), abs=0.02)   # P(0), not P(POST)


def test_stratum_is_centred_on_the_whole_match_population():
    shots = synthetic_shots(seed=4)
    z = np.repeat(shots.y.to_numpy()[:, None].astype(float), len(GRID), 1)       # D = centring factor exactly
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=3)
    curves = rd.readability_curves({"G": (z, shots.y.to_numpy())}, shots, keys, counts)
    for s, m in rd.strata_masks(shots).items():
        np.testing.assert_allclose(curves.point["G", s], centring_factor(shots, m), atol=1e-12)


def test_controls_pass_on_a_clean_planted_effect_and_fail_on_planted_artefacts():
    shots = synthetic_shots(n_matches=150)
    y = shots.y.to_numpy()
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=300)
    clean = planted_logits(shots, event_gap())
    y_perm = rd.permute_within_match(shots)
    readings = {"F": (clean, y), "F_neg": (clean, y_perm)}
    rec = rd.control_record(rd.readability_curves(readings, shots, keys, counts, {"F_neg": ("all",)}))
    assert rec["placebo"]["passed"] and rec["negative"]["passed"]
    assert rec["positive_jump"]["F"]["delta_lower_positive"]

    bent = event_gap() + 0.5 * (GRID >= -800) * (GRID < 0)        # a jump at -800, inside the placebo window
    broken = planted_logits(shots, bent)
    leak = planted_logits(shots, event_gap()) + 0.5 * y_perm[:, None] * (GRID > 0)   # pipeline "knows" the shuffle
    readings = {"F": (broken, y), "F_neg": (leak, y_perm)}
    rec = rd.control_record(rd.readability_curves(readings, shots, keys, counts, {"F_neg": ("all",)}))
    assert not rec["placebo"]["passed"] and not rec["negative"]["passed"]
    flat = rd.positive_jump(float(rd.jump(np.full(len(GRID), 0.3))), rd.jump(np.full((10, len(GRID)), 0.3)))
    assert not flat["delta_lower_positive"]


def test_gates_follow_the_protocol_thresholds():
    draws = np.column_stack([np.linspace(0.05, 0.25, 100), np.linspace(-0.05, 0.05, 100)])
    gate = rd.placebo_gate(np.array([0.15, 0.0]), draws)
    assert not gate["passed"] and gate["pi_-800"]["fails"] and not gate["pi_-600"]["fails"]
    assert rd.placebo_gate(np.array([0.09, 0.0]), draws)["passed"]          # small, even if the CI excludes 0
    neg = rd.negative_gate({"delta_neg": (0.2, np.linspace(-0.1, 0.5, 100))}, delta_f=1.0)
    assert neg["passed"]                                                      # large, but the CI covers 0
    assert not rd.negative_gate({"delta_neg": (0.2, np.linspace(0.05, 0.4, 100))}, delta_f=1.0)["passed"]


def test_trend_jump_placebo_and_plateau_on_an_exact_curve():
    d = 0.5 + 1e-4 * GRID + 2.0 * (GRID > 0) - 0.4 * (GRID > 400)
    a, b = rd.line(d)
    assert a == pytest.approx(0.5) and b == pytest.approx(1e-4)
    delta = (2.0 + 2.0 - 0.4 + 2.0 - 0.4) / 3 + 1e-4 * np.mean(rd.POST)
    assert rd.jump(d) == pytest.approx(delta)
    np.testing.assert_allclose(rd.placebo(d), 0, atol=1e-12)
    assert rd.plateau_slope(d) == pytest.approx((d[rd.OFFSET_INDEX[800]] - d[rd.OFFSET_INDEX[400]]) / delta)


# ---------------------------------------------------------------- linear probes

def synthetic_features(shots: pd.DataFrame, labels: np.ndarray, d: int = 6, seed: int = 5) -> np.ndarray:
    """(n, 30, d): dim 0 carries the label from +40 ms on (the ball's direction), dim 1 weakly before; persistent
    noise per shot plus fresh noise per frame."""
    rng = np.random.default_rng(seed)
    n = len(shots)
    s = 2 * labels[:, None] - 1
    x = 0.8 * rng.normal(size=(n, 1, d)) + 0.6 * rng.normal(size=(n, len(GRID), d))
    x[:, :, 0] += 1.5 * s * (GRID > 0)
    x[:, :, 1] += 0.3 * s
    return x


def array_loader(x: np.ndarray):
    return lambda rows, j: x[rows, j]


def test_probe_reads_the_planted_direction_and_shuffled_labels_give_auc_one_half():
    shots = synthetic_shots(n_matches=60)
    y = shots.y.to_numpy().astype(int)
    fold = np.where(shots.split.eq("test"), -1, shots.game.str[1:].astype(int) % 5)
    x = synthetic_features(shots, y)
    res = rd.probe_readout(array_loader(x), y, np.ones(len(y), bool), fold, c_grid=(1e-2, 1e-1))
    assert res.logits.shape == (2, len(y), len(GRID)) and np.isfinite(res.logits).all()
    assert res.cv_loss.shape == (5, 2) and (res.n_iter < rd.MAX_ITER).all()
    z = res.logits[res.selected()]
    post = [rd.OFFSET_INDEX[o] for o in rd.POST]
    assert min(roc_auc_score(y, z[:, j]) for j in post) > 0.85
    assert roc_auc_score(y, z[:, rd.OFFSET_INDEX[-1600]]) < 0.7

    y_perm = rd.permute_within_match(shots)
    neg = rd.probe_readout(array_loader(x), y_perm, np.ones(len(y), bool), fold, c_grid=(1e-2,))
    match = np.searchsorted(np.unique(shots.game), shots.game)
    for j in post:
        assert rd.weighted_auc(neg.logits[0][:, j], y_perm, match) == pytest.approx(0.5, abs=0.06)


def test_probe_curve_recovers_the_jump_after_contact():
    shots = synthetic_shots(n_matches=60, seed=7)
    y = shots.y.to_numpy().astype(int)
    fold = shots.game.str[1:].astype(int).to_numpy() % 5
    res = rd.probe_readout(array_loader(synthetic_features(shots, y, seed=8)), y, np.ones(len(y), bool), fold, (1e-2,))
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=200)
    curves = rd.readability_curves({"F": (res.logits[0], y)}, shots, keys, counts, {"F": ("all",)})
    tab = rd.curve_table(curves)
    dense = tab.set_index("offset_ms").D_minus_P
    assert (tab[tab.offset_ms.isin(rd.POST)].D_minus_P_lo > 0).all()
    assert abs(dense[0]) < 0.25 * dense[400] and dense[40] > 0.75 * dense[400]   # the step sits between 0 and +40


def test_labels_of_non_training_shots_never_enter_a_probe():
    shots = synthetic_shots(n_matches=20, per_match=10)
    y = shots.y.to_numpy().astype(int)
    fold = np.where(shots.split.eq("test"), -1, shots.game.str[1:].astype(int) % 5)
    x = synthetic_features(shots, y)
    train = np.ones(len(y), bool)
    train[:15] = False                      # e.g. goal shots: scored, never trained on
    res = rd.probe_readout(array_loader(x), y, train, fold, (1e-1,))
    flipped = y.copy()
    flipped[:15] = 1 - flipped[:15]
    same = rd.probe_readout(array_loader(x), flipped, train, fold, (1e-1,))
    np.testing.assert_allclose(res.logits, same.logits)      # labels of non-training shots never enter


def test_nested_probe_scores_every_train_valid_shot_and_no_test_shot():
    shots = synthetic_shots(n_matches=50, per_match=12, seed=9)
    y = shots.y.to_numpy().astype(int)
    fold = np.where(shots.split.eq("test"), -1, shots.game.str[1:].astype(int) % 5)
    fold_matches = {k: sorted(set(shots.game[fold == k])) for k in range(5)}
    res = rd.nested_probe_readout(array_loader(synthetic_features(shots, y)), shots, fold, fold_matches, (1e-2,))
    assert np.isnan(res.logits[:, fold < 0]).all() and np.isfinite(res.logits[:, fold >= 0]).all()
    assert res.cv_loss.shape == (25, 1)


def test_inner_folds_are_sorted_match_positions_mod_five_over_the_whole_fold():
    games = pd.Series(["b", "a", "g", "c", "a", "f", "e", "d"])
    np.testing.assert_array_equal(rd.inner_folds(games, list("gfedcba")), [1, 0, 1, 2, 0, 0, 4, 3])
    # a fold match without analysed shots ("c") keeps its position: later matches do not shift
    np.testing.assert_array_equal(rd.inner_folds(pd.Series(["d", "e"]), list("abcde")), [3, 4])
    with pytest.raises(ValueError, match="outer fold"):
        rd.inner_folds(pd.Series(["z"]), list("abc"))


def test_permutation_keeps_match_label_counts_and_is_deterministic():
    shots = synthetic_shots(n_matches=10).sample(frac=1, random_state=0)     # row order must not matter
    a = rd.permute_within_match(shots)
    assert np.array_equal(a, rd.permute_within_match(shots))
    by = pd.DataFrame({"g": shots.game, "y": shots.y, "p": a}).groupby("g").sum()
    assert (by.y == by.p).all() and not np.array_equal(a, shots.y.to_numpy())
    ordered = shots.sort_values("shot_id")
    np.testing.assert_array_equal(pd.Series(a, index=shots.shot_id)[ordered.shot_id].to_numpy(),
                                  rd.permute_within_match(ordered))


# ---------------------------------------------------------------- population

def tiny_table_and_index():
    shots = ["a", "b", "c", "g", "p"]
    index = pd.DataFrame({"shot_id": shots, "game": ["m1", "m1", "m2", "m2", "m2"], "half": 1,
                          "position_ms": [1000, 9000, 1000, 5000, 8000], "split": ["train", "train", "test", "test", "test"],
                          "goal": [False, False, False, True, False], "is_penalty": [False, False, False, False, True],
                          "label": ["Shots on target", "Shots off target", "Shots on target", "Shots on target",
                                    "Shots on target"]})
    rows = []
    for s in shots[:4]:
        for o in rd.GRID:
            rows.append({"shot_id": s, "offset_ms": o, "frame_status": "ok", "decoded_relative_ms": float(o),
                         "cut_between_anchor": False, "encoder": "dinov2_frozen"})
    table = pd.DataFrame(rows)
    table.loc[(table.shot_id == "b") & (table.offset_ms == 0), "decoded_relative_ms"] = 40.0     # 40 ms off
    table.loc[(table.shot_id == "c") & (table.offset_ms == 400), "decoded_relative_ms"] = 420.0  # within 20 ms
    table.loc[(table.shot_id == "c") & (table.offset_ms == 800), "cut_between_anchor"] = True
    return table, index


def test_probe_shots_apply_the_time_tolerance_and_count_exclusions():
    table, index = tiny_table_and_index()
    shots, exc = rd.probe_shots(table, index)
    assert list(shots.shot_id) == ["a", "c", "g"]
    assert list(shots.in_population) == [True, True, False] and list(shots.y) == [1, 1, 1]
    assert list(shots.no_cut) == [True, False, True]
    assert exc.set_index("offset_ms").excluded.to_dict()[0] == 1
    assert exc.excluded.iloc[-1] == 1 and exc.fraction.iloc[-1] == pytest.approx(1 / 3)
    assert rd.stop_rule_triggered(exc)


def test_probe_shots_refuse_a_table_without_population_shots():
    table, index = tiny_table_and_index()
    with pytest.raises(ValueError, match="missing"):
        rd.probe_shots(table[table.shot_id != "a"], index)


# ---------------------------------------------------------------- loaders, tables, cache

def test_embedding_loader_returns_rows_in_the_asked_order():
    emb = np.arange(40, dtype=np.float16).reshape(20, 2)
    refs = np.array([[5, 1], [19, 0], [2, -1]])
    load = rd.embedding_loader(emb, refs)
    np.testing.assert_array_equal(load(np.array([1, 0, 2]), 0), emb[[19, 5, 2]])
    np.testing.assert_array_equal(load(np.array([1, 0]), 1), emb[[0, 1]])
    with pytest.raises(ValueError, match="no embedding"):
        load(np.array([2]), 1)


def test_auc_and_jump_tables_cover_every_reading_stratum_and_offset():
    shots = synthetic_shots(n_matches=40)
    y = shots.y.to_numpy()
    z = planted_logits(shots, event_gap())
    keys, counts = rd.match_draw_counts(shots.game.to_numpy(), n_boot=50)
    gp = z.copy()
    gp[shots.split.eq("test").to_numpy()] = np.nan                # like Gp: no test shots
    readings = {"G": (z, y), "Gp": (gp, y)}
    auc = rd.auc_table(readings, shots, keys, counts)
    assert len(auc[auc.reading == "G"]) == len(rd.STRATA) * len(GRID)
    assert set(auc[auc.reading == "Gp"].stratum) == {"all", "rebound_chain", "no_cut"}
    post = auc[(auc.reading == "G") & (auc.stratum == "all") & auc.offset_ms.isin(rd.POST)]
    pre = auc[(auc.reading == "G") & (auc.stratum == "all") & auc.offset_ms.isin(rd.PRE)]
    assert (post.auc_lo > 0.6).all() and post.auc.min() > pre.auc.max()    # uncentred: the match effect stays
    jumps = rd.jump_table(rd.readability_curves(readings, shots, keys, counts))
    assert set(jumps.role) == {"gate_input", "descriptive"}
    assert jumps.set_index(["reading", "stratum"]).loc[("G", "all"), "role"] == "gate_input"
    assert jumps.set_index(["reading", "stratum"]).loc[("Gp", "test_split")].isna()["delta"]


def test_probe_cache_round_trips_and_refits_on_a_new_identity(tmp_path):
    import report_readability as script

    calls = []

    def fit():
        calls.append(1)
        return rd.ProbeResult(np.ones((2, 3, len(GRID))), np.zeros((5, 2)), np.full((5, 2), 7), (1e-4, 1e-3))

    first = script.cached_probe(tmp_path / "p.npz", {"a": 1}, fit)
    again = script.cached_probe(tmp_path / "p.npz", {"a": 1}, fit)
    assert len(calls) == 1 and again.c_grid == (1e-4, 1e-3) and np.array_equal(again.logits, first.logits)
    script.cached_probe(tmp_path / "p.npz", {"a": 2}, fit)
    assert len(calls) == 2


def test_fold_probe_never_sees_its_own_fold_and_test_shots_get_the_fold_mean():
    shots = synthetic_shots(n_matches=30, per_match=10, seed=11)
    y = shots.y.to_numpy().astype(int)
    fold = np.where(shots.split.eq("test"), -1, shots.game.str[1:].astype(int) % 5)
    x = synthetic_features(shots, y)
    train = np.ones(len(y), bool)
    res = rd.probe_readout(array_loader(x), y, train, fold, (1e-1,))
    for k in range(5):
        flipped = y.copy()
        flipped[fold == k] = 1 - flipped[fold == k]
        other = rd.probe_readout(array_loader(x), flipped, train, fold, (1e-1,))
        np.testing.assert_allclose(other.logits[0][fold == k], res.logits[0][fold == k])   # own labels never enter
    probes = []
    for k in range(5):                      # each fold probe applied directly
        rows = np.flatnonzero((fold >= 0) & (fold != k))
        xt = np.concatenate([x[rows, rd.OFFSET_INDEX[o]] for o in rd.PROBE_TRAIN])
        w, b, _ = rd.fit_probe(xt, np.tile(y[rows], len(rd.PROBE_TRAIN)), (1e-1,))
        probes.append(x[fold < 0] @ w[:, 0] + b[0])
    np.testing.assert_allclose(res.logits[0][fold < 0], np.mean(probes, 0), atol=1e-8)


def test_nested_probe_ignores_labels_of_non_population_shots():
    shots = synthetic_shots(n_matches=40, per_match=10, seed=12)
    y = shots.y.to_numpy().astype(int)
    fold = np.where(shots.split.eq("test"), -1, shots.game.str[1:].astype(int) % 5)
    fold_matches = {k: sorted(set(shots.game[fold == k])) for k in range(5)}
    goals = np.zeros(len(shots), bool)
    goals[::7] = True
    shots["in_population"] = ~goals
    x = synthetic_features(shots, y)
    res = rd.nested_probe_readout(array_loader(x), shots, fold, fold_matches, (1e-1,))
    flipped = shots.assign(y=np.where(goals, 1 - y, y).astype(np.int8))
    other = rd.nested_probe_readout(array_loader(x), flipped, fold, fold_matches, (1e-1,))
    np.testing.assert_allclose(other.logits, res.logits)
    assert np.isfinite(res.logits[:, goals & (fold >= 0)]).all()          # goal shots are scored


def test_outer_folds_put_test_shots_at_minus_one_and_refuse_inconsistent_folds():
    shots = pd.DataFrame({"game": ["a", "b", "t"], "split": ["train", "valid", "test"]})
    folds = pd.DataFrame({"game": ["a", "b"], "fold": [3, 0]})
    np.testing.assert_array_equal(rd.outer_folds(shots, folds), [3, 0, -1])
    with pytest.raises(ValueError, match="test match"):
        rd.outer_folds(shots, pd.concat([folds, pd.DataFrame({"game": ["t"], "fold": [1]})]))
    with pytest.raises(ValueError, match="no fold"):
        rd.outer_folds(shots, folds.iloc[:1])


def test_a_goal_only_match_outside_the_bootstrap_keys_is_allowed():
    shots = synthetic_shots(n_matches=10, per_match=6)
    extra = shots[shots.game == "m000"].assign(game="zzz", in_population=False, goal=True)
    both = pd.concat([shots, extra], ignore_index=True)
    keys, counts = rd.match_draw_counts(both.game[both.in_population].to_numpy(), n_boot=20)
    z = planted_logits(both, event_gap())
    curves = rd.readability_curves({"X": (z, both.y.to_numpy())}, both, keys, counts, {"X": ("all",)})
    alone = rd.readability_curves({"X": (z[:len(shots)], shots.y.to_numpy())}, shots, keys, counts, {"X": ("all",)})
    np.testing.assert_allclose(curves.point["X", "all"], alone.point["X", "all"])
    with pytest.raises(ValueError, match="population shot"):
        rd.match_positions(both.assign(in_population=True), keys)
