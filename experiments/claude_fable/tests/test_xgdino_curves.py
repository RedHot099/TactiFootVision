"""xg-dino ticket 06: eigendecomposition ridge, per-layer curves and the best-layer rule (spec §Testing Decisions;
councils C1, C2)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from test_xgdino_probes import planted
from xgpoc.dino import curves as cv
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg


def exact_ridge(x, y, z, alpha, xe, ze):
    """Dense float64 ridge with unpenalised z: the reference solution."""
    m = np.eye(len(x)) - z @ np.linalg.pinv(z)
    beta = np.linalg.solve(x.T @ m @ x + alpha * np.eye(x.shape[1]), x.T @ m @ y)
    return xe @ beta + ze @ np.linalg.lstsq(z, y - x @ beta, rcond=None)[0]


@pytest.mark.parametrize("seed", [0, 1])
def test_eigendecomposition_ridge_equals_the_exact_ridge_for_every_penalty_and_target(seed):
    rng = np.random.default_rng(seed)
    n, d, t = 300, 40, 4
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) / 3      # correlated columns
    x = ((x - x.mean(0)) / x.std(0)).astype(np.float32)
    z = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = x @ rng.normal(size=(d, t)) + z @ rng.normal(size=(4, t)) + rng.normal(size=(n, t))
    xe, ze = rng.normal(size=(50, d)), np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    zs = {"embedding": z[:, :1], "distance_embedding": z}
    evals = [(xe, {"embedding": ze[:, :1], "distance_embedding": ze}), (x, zs)]
    alphas = (1e-3, 1.0, 31.6, 1e4, 1e6)
    fast = cv.ridge_paths(x, y, zs, evals, alphas)
    ref = pr.ridge_paths(x, y, zs, evals, alphas)          # ticket 04: float32 Gram matrix, Cholesky per penalty
    for v, zv in zs.items():
        for (xev, zev), got, old in zip(evals, fast[v], ref[v], strict=True):
            assert got.shape == old.shape == (len(alphas), len(xev), t)
            for i, alpha in enumerate(alphas):
                want = exact_ridge(x.astype(float), y, zv, alpha, xev.astype(float), zev[v])
                np.testing.assert_allclose(got[i], want, rtol=0, atol=1e-9 * np.abs(want).max())
                if alpha >= 1:                             # the float32 Gram rounding of ticket 04 only shows below
                    np.testing.assert_allclose(old[i], want, rtol=0, atol=1e-4 * np.abs(want).max())


def test_fast_run_reproduces_ticket_04_ridge_and_puts_binary_targets_in_the_row_group():
    t, games, folds, signal, rng = planted()
    n, d = len(t), 24
    t["near_opp_m2"] = t.near_opp_m * 2 + rng.normal(size=n)
    data = pr.ProbeData(tg.tagged(t, "automatic"), games, folds)
    x = np.outer(signal, rng.normal(size=d)) + rng.normal(size=(n, d))
    enc = pr.Encoder("x", lambda k: x)
    targets = ["distance", "angle", "near_opp_m", "opp_3m_ge1", "near_opp_m2"]
    ref, fast = pr.run(data, enc, targets), cv.run(data, enc, targets)
    for name in ("distance", "angle", "near_opp_m", "near_opp_m2"):
        keys = ["embedding", "distance", "constant", "embedding_shuffled", "shuffled_target"]
        if name != "distance":                             # the spline explains distance: a flat penalty path
            keys.append("distance_embedding")
            assert fast.targets[name].penalty == ref.targets[name].penalty
        for key in keys:
            np.testing.assert_allclose(fast.targets[name].oof[key], ref.targets[name].oof[key], rtol=1e-5, atol=1e-7)
    b = fast.targets["opp_3m_ge1"]
    assert pr.point_metric(b.y, b.oof["embedding"], True) > 0.85             # ridge output scored by AUC
    np.testing.assert_allclose(b.oof["distance"], ref.targets["opp_3m_ge1"].oof["distance"], atol=1e-8)
    assert "embedding_shuffled" not in b.oof
    groups = cv.mask_groups(data, targets)
    assert [names for _, names in groups] == [["distance", "angle", "opp_3m_ge1"], ["near_opp_m", "near_opp_m2"]]


def test_no_match_lies_on_both_sides_of_a_split_in_any_row_group():
    t, games, folds, *_ = planted()
    data = pr.ProbeData(tg.tagged(t, "automatic"), games, folds)
    for rows, _ in cv.mask_groups(data, list(t.columns)):
        for k in range(5):
            train, test = rows[folds[rows] != k], rows[folds[rows] == k]
            assert not set(games[train]) & set(games[test])
            for itr, iva in pr.inner_splits(games[train]):
                assert not set(games[train][itr]) & set(games[train][iva])


def test_best_layer_ties_go_to_the_later_block_over_the_depth():
    scores = {f"block{b}": s for b, s in enumerate([0.10, 0.40, 0.505, 0.51, 0.50, 0.30])}
    assert pr.best_layer(scores, binary=False) == "block4"              # 0.50 within 0.01 of 0.51
    assert pr.best_layer(scores | {"block5": 0.499}, binary=False) == "block4"
    assert pr.best_layer({"block0": 0.80, "block1": 0.7951, "block2": 0.79}, binary=True) == "block1"
    assert cv.first_readable(scores, 0.30) == "block1" and cv.first_readable(scores, 0.9) is None


def test_weighted_spearman_equals_spearman_on_the_replicated_rows():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 60).astype(float), rng.normal(size=60)      # ties in y
    w = rng.integers(0, 3, size=(3, 60)).astype(float)
    got = cv.weighted_spearman(y, p, w)
    for b in range(3):
        rep = np.repeat(np.arange(60), w[b].astype(int))
        assert got[b] == pytest.approx(spearmanr(y[rep], p[rep]).statistic, abs=1e-12)
    assert float(cv.weighted_spearman(y, p)) == pytest.approx(spearmanr(y, p).statistic)


def test_fast_run_reads_paired_shots_with_each_fold_probe():
    t, games, folds, signal, rng = planted()
    n, d = len(t), 24
    paired_signal = rng.normal(size=60)
    paired = tg.tagged(pd.DataFrame({"distance": rng.uniform(5, 30, 60)}), "automatic")
    data = pr.ProbeData(tg.tagged(t, "automatic"), games, folds, paired)
    directions = rng.normal(size=(5, d))
    enc = pr.Encoder("g", lambda k: np.outer(signal, directions[k]) + rng.normal(size=(n, d)),
                     lambda k: np.outer(paired_signal, directions[k]) + rng.normal(size=(60, d)))
    r = cv.run(data, enc, ["near_opp_m"], paired=True)
    assert len(r.targets["near_opp_m"].paired_folds["embedding"]) == 5
    assert np.corrcoef(r.targets["near_opp_m"].paired["embedding"], paired_signal)[0, 1] > 0.9
