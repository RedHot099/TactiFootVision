"""xg-dino-info ticket 12: probe targets, weighted read-out, BH, the geometry probe and the provenance of header labels
(spec §Decisions C3.4, C3.7; the pinned protocol's `probes` families)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_xgdino_probes import planted
from xgpoc.dino import curves as cv
from xgpoc.dino import info_probes as ip
from xgpoc.dino import labels as lb
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg
from xgpoc.probe.readability import weighted_auc

EXPERIMENT = Path(__file__).resolve().parents[1]


def brute_auc(score, y, w):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    num = den = 0.0
    for i in pos:
        for j in neg:
            ww = w[i] * w[j]
            num += ww * ((score[i] > score[j]) + 0.5 * (score[i] == score[j]))
            den += ww
    return num / den


def test_weighted_auc_rows_equals_brute_force_replication_and_the_match_bootstrap_auc():
    rng = np.random.default_rng(0)
    n = 60
    score, y = np.round(rng.normal(size=n), 1), (rng.random(n) < 0.4).astype(float)
    w = rng.integers(0, 4, size=(3, n)).astype(float)
    got = ip.weighted_auc_rows(score, y, w)
    for b in range(3):
        assert got[b] == pytest.approx(brute_auc(score, y, w[b]))
    match = np.arange(n) // 5
    counts = rng.integers(0, 3, size=(4, match.max() + 1)).astype(float)
    np.testing.assert_allclose(ip.weighted_auc_rows(score, y, counts[:, match]), weighted_auc(score, y, match, counts))
    assert np.isnan(ip.weighted_auc_rows(score, np.zeros(n), w)).all()


def test_row_weights_that_are_constant_change_nothing_and_stratum_weights_move_the_metric():
    t, games, folds, signal, rng = planted()
    n = len(t)
    d = pr.draws(games, 40)
    y, pred = (signal > 0).astype(float), signal + rng.normal(size=n)
    rows = np.arange(n)
    tg.BINARY_EXTRA.add("hdr")
    plain = ip.metric_draws_w("hdr", y, pred, rows, d, None)["AUC"]
    const = ip.metric_draws_w("hdr", y, pred, rows, d, np.full(n, 3.0))["AUC"]
    assert const[0] == pytest.approx(plain[0])
    np.testing.assert_allclose(const[1], plain[1])
    w = np.where(y == 1, 5.0, 1.0)
    assert ip.metric_draws_w("hdr", y, pred, rows, d, w)["AUC"][0] == pytest.approx(plain[0])     # AUC is class-weight free
    w2 = np.where(pred > 0, 4.0, 1.0)
    assert ip.metric_draws_w("hdr", y, pred, rows, d, w2)["AUC"][0] == pytest.approx(brute_auc(pred, y, w2))
    cont = ip.metric_draws_w("near_opp_m", signal, pred, rows, d, np.full(n, 2.0))
    base = ip.metric_draws_w("near_opp_m", signal, pred, rows, d, None)
    assert cont["R2"][0] == pytest.approx(base["R2"][0])
    np.testing.assert_allclose(cont["MAE"][1], base["MAE"][1])


def test_bh_and_the_one_sided_bootstrap_p():
    assert ip.bh([0.001, 0.03, 0.02, 0.5]) == [True, True, True, False]       # the step-up rule
    assert ip.bh([0.001, 0.04, 0.03, 0.5]) == [True, False, False, False]
    assert ip.bh([0.02, 0.03, 0.04, 0.9], q=0.05) == [False] * 4
    assert ip.bh([0.2, float("nan"), 0.9]) == [False, False, False]
    assert ip.bh([0.01, float("nan")]) == [True, False]
    assert ip.one_sided_p(np.array([1.0, 2.0, np.nan, -0.5, 0.0])) == pytest.approx((1 + 2) / 5)
    assert ip.one_sided_p(np.full(2000, 0.3)) == pytest.approx(1 / 2001)


def test_set_piece_header_labels_and_z_difference():
    a = pd.DataFrame({"sn_direct_fk": [1, 0, 0, np.nan, np.nan], "sn_indirect_fk_3s": [0, 0, 0, 0, 1],
                      "sn_corner_20s": [0, 1, 0, 0, np.nan]}, index=list("abcde"))
    sp = ip.set_piece(a)
    assert sp.tolist()[:3] == [1.0, 1.0, 0.0] and np.isnan(sp["d"]) and sp["e"] == 1.0
    gpt = pd.DataFrame({"shot_id": ["a", "b", "c"], "header_gpt": [1.0, 0.0, np.nan], "source": "gpt"})
    cap = pd.DataFrame({"shot_id": ["a", "b", "c", "d"], "header_caption": ["header", "foot", "unknown", "header"]})
    h = ip.header_targets(gpt, cap, pd.Index(list("abcde")))
    assert h.header_gpt.tolist()[:2] == [1.0, 0.0] and h.header_gpt.isna().tolist() == [False, False, True, True, True]
    assert h.header_caption.tolist()[:2] == [1.0, 0.0] and h.header_caption.isna().tolist() == [False, False, True, False, True]
    z = ip.z_difference(pd.DataFrame({"shot_id": ["a", "b"], "p_B": [0.1, 0.5], "p_C": [0.2, 0.5]}))
    assert z["a"] == pytest.approx(np.log(0.2 / 0.8) - np.log(0.1 / 0.9)) and z["b"] == 0.0


def small_tables(n=200):
    rng = np.random.default_rng(1)
    ids = pd.Index([f"s{i}" for i in range(n)])
    attrs = pd.DataFrame({"a_full": rng.normal(size=n), "a_almost": rng.normal(size=n), "a_flag": (rng.random(n) < 0.3).astype(float),
                          "a_ball": rng.normal(size=n)}, index=ids)
    attrs.loc[ids[:3], "a_almost"] = np.nan
    attrs.loc[ids[::2], "a_ball"] = np.nan
    for c in ip.SET_PIECE:
        attrs[c] = 0.0
    labels = pd.DataFrame({"header_gpt": np.where(rng.random(n) < 0.5, 1.0, np.nan), "header_caption": np.nan}, index=ids)
    dist = pd.Series(rng.uniform(5, 30, n), index=ids)
    dist.iloc[5] = np.nan
    z = pd.Series(rng.normal(size=n), index=ids, name=ip.ZDIFF)
    return attrs, labels, z, dist, ids


def test_probe_table_shares_rows_across_near_complete_targets_and_masks_missing_distance():
    attrs, labels, z, dist, ids = small_tables()
    t = ip.probe_table(attrs, labels, z, dist, ["a_full", "a_almost", "a_flag", "a_ball", ip.ZDIFF], ids)
    assert t.attrs["source"] == "automatic" and list(t.columns[:1]) == ["distance"]
    main = t[["a_full", "a_almost", "a_flag"]].notna()
    assert main.all(axis=1).equals(main.any(axis=1))                  # one row set for the near-complete targets
    assert not main.loc[ids[[0, 1, 2, 5]]].any().any()                 # missing a_almost / distance leave it
    assert t.a_ball.notna().sum() == 99 and np.isnan(t.a_ball.iloc[5])
    assert t[ip.ZDIFF].notna().sum() == len(ids) - 1                  # z_diff only loses the missing-distance row
    assert "a_flag" in tg.BINARY_EXTRA and tg.kind("a_flag") == "binary" and tg.kind("a_full") == "continuous"
    g = ip.probe_table(attrs, labels, z, dist, ["header_gpt"], ids)
    assert g.attrs["source"] == "gpt"
    with pytest.raises(ValueError, match="one label source"):
        ip.source_of(["header_gpt", "header_caption"])


def test_header_labels_are_probe_targets_only():
    t, games, folds, *_ = planted()
    for source in ("gpt", "caption"):
        pr.ProbeData(tg.tagged(t.copy(), source), games, folds)               # accepted by the probes
    with pytest.raises(ValueError, match="evaluation only"):
        pr.ProbeData(_statsbomb(t), games, folds)
    with pytest.raises(ValueError, match="unknown target source"):
        tg.tagged(t.copy(), "human")
    for name in ("header_gpt", "header_caption"):
        with pytest.raises(ValueError, match="not allowed in a model design"):
            lb.assert_not_design(["dist@0", name])


def _statsbomb(t):
    out = t.copy()
    out.attrs["source"] = "statsbomb"
    return out


def test_geometry_basis_has_full_rank_and_survives_missing_constant_and_collinear_columns():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 4))
    geo = np.column_stack([base, base[:, 0] * 2 + base[:, 1], np.zeros(120), rng.normal(size=120)])
    geo[::9, 6] = np.nan
    train, other = geo[:100], geo[100:]
    ztr, zo = cv.geometry_basis(train, other)
    assert ztr.shape[1] == 6 and zo.shape == (20, 6) and np.isfinite(ztr).all() and np.isfinite(zo).all()
    assert np.linalg.matrix_rank(ztr) == ztr.shape[1]
    np.testing.assert_allclose(ztr[:, 0], 1.0)


def test_geometry_probe_credits_the_embedding_only_for_what_geometry_does_not_hold():
    t, games, folds, signal, rng = planted()
    n, d = len(t), 24
    geo = np.column_stack([t.distance.to_numpy(), rng.normal(size=n)])
    hidden = rng.normal(size=n)
    t["zdiff"] = 0.8 * geo[:, 1] + 0.6 * hidden + 0.1 * rng.normal(size=n)       # geometry holds part of it
    t["zpure"] = 0.8 * geo[:, 1] + 0.1 * rng.normal(size=n)                      # geometry holds all of it
    data = pr.ProbeData(tg.tagged(t, "automatic"), games, folds)
    x = np.outer(hidden, rng.normal(size=d)) + np.outer(geo[:, 1], rng.normal(size=d)) + 0.3 * rng.normal(size=(n, d))
    enc = pr.Encoder("x", lambda k: x)
    r = cv.run(data, enc, ["zdiff", "zpure"], geometry=geo)
    assert {"geometry_embedding", "geometry", "embedding", "distance"} <= set(r.targets["zdiff"].oof)
    d_ = pr.draws(games, 100)
    gains = {}
    for name in ("zdiff", "zpure"):
        md = ip.target_metrics_w(r.targets[name], d_, ("geometry_embedding", "geometry"))
        gains[name] = ip.geometry_decision(md)
    assert gains["zdiff"]["gain"]["point"] > 0.25 and gains["zdiff"]["encodes_beyond_geometry"]
    assert gains["zpure"]["gain"]["point"] < 0.05 and not gains["zpure"]["encodes_beyond_geometry"]
    plain = cv.run(data, enc, ["zdiff", "zpure"])                                  # no geometry: ticket 06 unchanged
    assert "geometry" not in plain.targets["zdiff"].oof
    for key in ("embedding", "distance_embedding", "distance"):
        np.testing.assert_allclose(plain.targets["zdiff"].oof[key], r.targets["zdiff"].oof[key], rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError, match="continuous columns"):
        cv.run(data, enc, ["opp_3m_ge1"], geometry=geo)


def test_protocol_pin_in_the_script_matches_the_file():
    protocol = EXPERIMENT / "research/xgdino_info_protocol.json"
    script = (EXPERIMENT / "scripts/xgdino_info_probes.py").read_text()
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() in script
