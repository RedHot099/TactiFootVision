"""H3 probes and their targets (xg-vendor ticket 09, seam 2): planted targets, negative control, fold hygiene, geometry."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgpoc.probe.readability import match_draw_counts, weighted_auc
from xgpoc.vendor import probe_targets as pt
from xgpoc.vendor import probes
from xgpoc.vendor.draws import table_draws


def synthetic(n_matches=30, per_match=12, dim=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"m{i:02d}" for i in range(n_matches)], per_match)
    x = rng.normal(size=(len(groups), dim))
    return x, groups, rng


def test_planted_linear_target_is_recovered():
    x, groups, rng = synthetic()
    w = rng.normal(size=x.shape[1])
    y = np.column_stack([x @ w + 0.3 * rng.normal(size=len(x)), rng.normal(size=len(x))])
    pred, alphas, fold = probes.probe_predictions(x, y, groups)
    r2 = probes.weighted_r2(y[:, 0], pred[:, 0])
    assert r2 > 0.9
    assert probes.weighted_r2(y[:, 1], pred[:, 1]) <= 0          # pure noise decodes at R^2 <= 0
    assert alphas.shape == (probes.N_OUTER, 2) and alphas[:, 1].min() > alphas[:, 0].max()   # noise picks heavier shrinkage
    for k in range(probes.N_OUTER):   # outer folds are match-grouped
        assert not set(groups[fold == k]) & set(groups[fold != k])


def test_outer_test_rows_enter_no_fit_or_standardisation():
    """The prediction of a test row depends on the training rows and its own features only: perturbing the other rows of its
    outer test fold (features and targets, far outside the training scale) leaves it unchanged."""
    x, groups, rng = synthetic()
    y = (x @ rng.normal(size=x.shape[1]))[:, None]
    pred, alphas, fold = probes.probe_predictions(x, y, groups)
    test = np.flatnonzero(fold == 2)
    i, others = test[0], test[1:]
    x2, y2 = x.copy(), y.copy()
    x2[others] = 1000 * rng.normal(size=(len(others), x.shape[1])) + 500
    y2[others] = 1e6
    pred2, alphas2, fold2 = probes.probe_predictions(x2, y2, groups)
    assert np.array_equal(fold, fold2)
    assert np.array_equal(alphas[2], alphas2[2])
    assert pred2[i, 0] == pytest.approx(pred[i, 0], abs=1e-9)
    assert not np.allclose(pred2[fold != 2], pred[fold != 2])   # not vacuous: the perturbed rows train the other folds


def test_nested_cv_matches_a_scaler_ridge_pipeline():
    """Reference: sklearn StandardScaler + Ridge refitted on the inner training rows of every inner fold (alpha choice) and on the
    outer training rows (prediction). Equal answers mean the standardisation is fitted where the docstring says, never on
    inner validation or outer test rows."""
    x, groups, rng = synthetic(n_matches=15, per_match=8, dim=12)
    y = x @ rng.normal(size=x.shape[1]) + rng.normal(size=len(x))
    pred, alphas, fold = probes.probe_predictions(x, y, groups)
    expected = np.empty(len(y))
    for k in range(probes.N_OUTER):
        train, test = fold != k, fold == k
        xa, ya, ga = x[train], y[train], groups[train]
        sse = np.zeros(len(probes.ALPHAS))
        for inner_train, inner_valid in GroupKFold(probes.N_INNER).split(xa, groups=ga):
            for j, alpha in enumerate(probes.ALPHAS):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(xa[inner_train], ya[inner_train])
                sse[j] += ((model.predict(xa[inner_valid]) - ya[inner_valid]) ** 2).sum()
        alpha = probes.ALPHAS[int(np.argmin(sse))]
        assert alphas[k, 0] == alpha
        expected[test] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(xa, ya).predict(x[test])
    assert np.allclose(pred[:, 0], expected, atol=1e-5)


def test_binary_target_scored_by_auc():
    x, groups, rng = synthetic()
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(float)
    pred, _, _ = probes.probe_predictions(x, y, groups)
    keys = np.unique(groups)
    assert weighted_auc(pred[:, 0], y, np.searchsorted(keys, groups)) > 0.9


def make_table(n_matches=24, per_match=10):
    games = np.repeat([f"g{i:02d}" for i in range(n_matches)], per_match)
    n = len(games)
    return pd.DataFrame({"key": [f"s{i}" for i in range(n)], "statsbomb_event_id": [f"e{i}" for i in range(n)], "match": games,
                         "split": np.where(np.arange(n) < n // 2, "test", "train")})


def make_targets(table, rng, latent):
    n = len(table)
    dist = 5 + 3 * latent[:, 0] + 0.2 * rng.normal(size=n)
    cone = np.clip(np.round(1 + latent[:, 1]), 0, 4)
    targets = pd.DataFrame({"dist": dist, "angle": 0.5 + 0.1 * latent[:, 2], "ff_def_cone": cone, "def_cone_ge1": (cone >= 1).astype(float),
                            "def_cone_ge2": (cone >= 2).astype(float), "gk_depth": 2 + latent[:, 3], "gk_off_line_abs": np.abs(latent[:, 4]),
                            "gk_dist_goal": 3 + latent[:, 3], "ff_opp_3m": rng.poisson(1, n).astype(float),
                            "head": (latent[:, 5] > 0.5).astype(float)})
    targets.loc[:9, ["gk_depth", "gk_off_line_abs", "gk_dist_goal"]] = np.nan   # no goalkeeper in a few freeze frames
    return targets[pt.TARGET_COLUMNS]


def test_probe_table_difference_and_control():
    """G-combo carries the latent quantities cleanly, the reference only through noise: the difference is positive with an
    interval above zero on the shots both cover; the shuffled control decodes at R^2 <= 0 for every encoder."""
    table = make_table()
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(len(table), 6))
    targets = make_targets(table, rng, latent)
    g = np.hstack([latent, 0.1 * rng.normal(size=(len(table), 20))])
    ref = np.hstack([latent + 2.0 * rng.normal(size=latent.shape), rng.normal(size=(len(table), 20))])
    g_rows = np.flatnonzero(np.arange(len(table)) % 4 != 0)   # G covers three quarters of the shots
    embeddings = {"emb_g_fold0": (g_rows, g[g_rows]), "emb_dinov2": (np.arange(len(table)), ref)}
    draws = table_draws(table, n_boot=200)
    res = probes.probe_table(table, targets, embeddings, draws)

    m = res.metrics.set_index(["encoder", "coverage", "target"])
    assert set(res.metrics.coverage) == {"own", "shared_with_emb_g_fold0"}
    assert m.loc[("emb_g_fold0", "own", "dist"), "n"] == len(g_rows)
    assert m.loc[("emb_dinov2", "own", "dist"), "n"] == len(table)
    assert m.loc[("emb_g_fold0", "own", "gk_depth"), "n"] == len(np.setdiff1d(g_rows, np.arange(10)))
    assert m.loc[("emb_g_fold0", "own", "dist"), "value"] > 0.9
    assert m.loc[("emb_g_fold0", "own", "head"), "metric"] == "AUC"
    assert (res.metrics.loc[res.metrics.negative_control, "value"] <= 0).all()

    d = res.differences.set_index("target")
    assert d.loc["dist", "n"] == len(g_rows)
    assert d.loc["dist", "lo"] > 0
    assert d.loc["dist", "reference_value"] == pytest.approx(m.loc[("emb_dinov2", "shared_with_emb_g_fold0", "dist"), "value"])
    assert d.loc["dist", "difference"] == pytest.approx(d.loc["dist", "encoder_value"] - d.loc["dist", "reference_value"])
    assert set(res.predictions.coverage) == {"own", "shared_with_emb_g_fold0"}


def test_the_control_gate_fails_an_encoder_that_decodes_the_shuffled_target():
    """Planted leak: one encoder carries the shuffled distance itself, so its negative control decodes far above R^2 = 0
    and the gate fails it; the clean encoder passes."""
    table = make_table()
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(len(table), 6))
    targets = make_targets(table, rng, latent)
    everything = np.arange(len(table))
    leak = probes.control_target(targets.dist.to_numpy(float))
    embeddings = {"emb_g_fold0": (everything, np.column_stack([latent, leak])), "emb_dinov2": (everything, latent)}
    res = probes.probe_table(table, targets, embeddings, table_draws(table, n_boot=50))
    gate = probes.control_gate(res.metrics)
    assert gate["encoders"]["emb_g_fold0"]["max_r2"] > 0.5 and not gate["encoders"]["emb_g_fold0"]["passed"]
    assert gate["encoders"]["emb_dinov2"]["passed"] and not gate["passed"]
    assert gate["threshold"] == probes.CONTROL_MAX_R2 == 0.0
    clean = probes.probe_table(table, targets, {"emb_g_fold0": (everything, latent + 0.1), "emb_dinov2": (everything, latent)},
                               table_draws(table, n_boot=50))
    assert probes.control_gate(clean.metrics)["passed"]


def test_probe_table_full_coverage_reuses_own_runs_and_subsets():
    table = make_table()
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(len(table), 6))
    targets = make_targets(table, rng, latent)
    everything = np.arange(len(table))
    embeddings = {"emb_g_fold0": (everything, latent + 0.1 * rng.normal(size=latent.shape)), "emb_dinov2": (everything, latent)}
    subsets = {"all": np.ones(len(table), bool), "test": table.split.eq("test").to_numpy()}
    res = probes.probe_table(table, targets, embeddings, table_draws(table, n_boot=50), subsets)
    assert set(res.metrics.coverage) == {"own"}
    assert set(res.differences.subset) == {"all", "test"}
    assert res.metrics[res.metrics.subset.eq("test") & res.metrics.target.eq("dist")].n.tolist() == [len(table) // 2] * 2


def test_bootstrap_uses_the_shared_draws():
    table = make_table()
    draws = table_draws(table, n_boot=30, seed=0)
    keys, counts = match_draw_counts(table.match.to_numpy(), 30, 0)
    assert np.array_equal(draws.keys, keys) and np.array_equal(draws.counts, counts)
    rows = np.array([0, 15, 239])
    assert np.array_equal(draws.weights(rows), counts[:, [0, 1, 23]])


def test_weighted_r2_matches_row_duplication():
    rng = np.random.default_rng(3)
    y, p = rng.normal(size=6), rng.normal(size=6)
    w = np.array([[2, 0, 1, 1, 3, 1]], float)
    dup = np.repeat(np.arange(6), w[0].astype(int))
    expected = 1 - ((y[dup] - p[dup]) ** 2).sum() / ((y[dup] - y[dup].mean()) ** 2).sum()
    assert probes.weighted_r2(y, p, w)[0] == pytest.approx(expected)


def test_thresholds_are_reported_not_decided():
    metrics = pd.DataFrame({"target": ["dist", "head"], "value": [0.6, 0.65], "lo": [0.45, 0.6]})
    out = probes.apply_thresholds(metrics, {"dist": 0.5})
    assert out.point_meets.tolist()[0] is True and out.lower_meets.tolist()[0] is False
    assert pd.isna(out.threshold[1])
    with pytest.raises(ValueError):
        probes.apply_thresholds(metrics, {"xg": 0.1})


# ------------------------------------------------------------------ targets
def player(x, y, teammate=False, position="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": position}}


def test_targets_of_a_known_freeze_frame():
    frame = [player(114, 40), player(114, 50), player(116, 40, teammate=True), player(110, 41), player(119, 41, position="Goalkeeper")]
    shots = pd.DataFrame({"event_id": ["a", "b", "c"], "x": [108.0, 108.0, 100.0], "y": [40.0, 40.0, 30.0],
                          "body_part": ["Head", "Right Foot", "Left Foot"],
                          "freeze_frame_json": [json.dumps(frame), None, json.dumps([player(119, 40, teammate=True)])]})
    t = pt.probe_targets(shots)
    a = t.loc["a"]
    assert a.dist == pytest.approx(12.0)
    assert a.angle == pytest.approx(2 * np.arctan(4 / 12))
    assert a.ff_def_cone == 1 and a.def_cone_ge1 == 1 and a.def_cone_ge2 == 0   # (110, 41) is outside the cone, the teammate does not count
    assert a.ff_opp_3m == 1                                                     # (110, 41) only; the goalkeeper is 11 away
    assert a.gk_depth == pytest.approx(1.0) and a.gk_dist_goal == pytest.approx(np.sqrt(2)) and a.gk_off_line_abs == pytest.approx(1.0)
    assert a["head"] == 1
    assert t.loc["b", ["ff_def_cone", "def_cone_ge1", "ff_opp_3m", "gk_depth"]].isna().all()   # no freeze frame
    assert t.loc["b", "head"] == 0 and t.loc["b"].dist == pytest.approx(12.0)
    assert t.loc["c", "ff_def_cone"] == 0 and np.isnan(t.loc["c", "gk_depth"])                # freeze frame without a goalkeeper


def test_targets_for_table_needs_every_event():
    targets = pd.DataFrame({c: [1.0, 2.0] for c in pt.TARGET_COLUMNS}, index=pd.Index(["e1", "e0"], name="event_id"))
    table = pd.DataFrame({"statsbomb_event_id": ["e0", "e1"]})
    assert pt.targets_for_table(table, targets).dist.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="no StatsBomb shot"):
        pt.targets_for_table(pd.DataFrame({"statsbomb_event_id": ["e0", "e9"]}), targets)


def test_compare_with_prototype_features():
    shots = pd.DataFrame({"event_id": ["a"], "x": [108.0], "y": [40.0], "body_part": ["Head"],
                          "freeze_frame_json": [json.dumps([player(114, 40), player(119, 42, position="Goalkeeper")])]})
    t = pt.probe_targets(shots)
    features = pd.DataFrame({"event_id": ["a"], "body": ["Head"], "dist": [12.0], "angle": [t.angle.iloc[0]], "ff_def_cone": [1.0],
                             "gk_depth": [1.0], "gk_off_line": [-2.0], "gk_dist_goal": [np.hypot(1, 2)], "ff_opp_3m": [0.0]})
    cmp = pt.compare_with_features(t, features).set_index("target")
    assert cmp.max_abs_diff.max() < 1e-8 and cmp.nan_mismatch.sum() == 0
