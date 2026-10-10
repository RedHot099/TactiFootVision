"""Post-contact share, known-answer test and verdict of the direction probe (xg-credibility 10).

Synthetic tables with a planted post-contact share and blur, and a reproduction of the protocol's validation
simulation. The prototype (`.scratch/xg-credibility/prototype/sim.py`, sha256 `PROTOTYPE_SHA256`) lives in the
untracked agent folder, which the worktree does not see, so the parts the estimator was validated with (generative
model, Newton probe, estimator, KA statistic) are copied verbatim below (`proto_*`, rng call order kept). One test
checks the copy against the real file when it is present.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgpoc.bundle import sha256_file
from xgpoc.probe import readability as rd
from xgpoc.probe import share as sh

PROTOTYPE = Path("/home/kuba/projects/tactifoot_vision/.scratch/xg-credibility/prototype/sim.py")
PROTOTYPE_SHA256 = "4b1257709634edc24507b48ae795a444c46199c2864d8b0fa3696b91b2f9d9a8"
GRID = np.array(rd.GRID, float)
J = rd.OFFSET_INDEX

# ---------------------------------------------------------------- prototype copy (sim.py, verbatim logic)

P_PRE = [-1600, -1400, -1200, -1000, -800, -600]
P_POST = [400, 600, 800]
P_DENSE = list(range(-400, 401, 40))
P_GRID = sorted(set(P_PRE) | set(P_DENSE) | set(P_POST) | {1000})
P_KA = [o for o in P_DENSE if o <= 200]
P_GI = {o: i for i, o in enumerate(P_GRID)}
P_G = np.array(P_GRID, float)
P_TRAIN = P_PRE + P_POST
CONTACTS = {"n+80_80": ("normal", 80.0, 80.0), "n0_80": ("normal", 0.0, 80.0), "tail15": ("mix", 0.15, 60.0, 50.0, -300.0, 150.0),
            "null": ("uniform", 40.0, 400.0)}


def proto_scenario(contact: str, blur: bool, a_pre: float = 0.3, a_post: float = 0.6) -> dict:
    """sim.Scenario defaults (trend 0.15, rho 0.7, no wind-up or decay, 500 matches); medium signal by default."""
    return {"contact": CONTACTS[contact], "ramp": 3 if blur else 1, "never": 0.3 if blur else 0.0, "a_pre": a_pre,
            "a_post": a_post, "trend": 0.15, "rho": 0.7, "n_matches": 500}


def proto_contacts(rng, sc, n):
    kind, *p = sc["contact"]
    if kind == "normal":
        return rng.normal(p[0], p[1], n)
    if kind == "uniform":
        return rng.uniform(p[0], p[1], n)
    w, m1, s1, m2, s2 = p
    z = rng.random(n) < w
    return np.where(z, rng.normal(m2, s2, n), rng.normal(m1, s1, n))


def proto_readability(o, c, ramp):
    d = o - c
    r = np.clip((d + 40.0) / (40.0 * ramp), 0, 1)
    return np.where(d >= 0, r, 0.0)


def proto_population(rng, sc):
    m = sc["n_matches"]
    sizes = np.clip(np.round(rng.normal(19, 5.7, m)), 5, 39).astype(int)
    match = np.repeat(np.arange(m), sizes)
    n = match.size
    prev_logit = np.log(0.45 / 0.55) + rng.normal(0, 0.25, m)
    y = (rng.random(n) < 1 / (1 + np.exp(-prev_logit[match]))).astype(float)
    k = np.exp(rng.normal(0, 0.25, m))[match]
    return {"match": match, "y": y, "c": proto_contacts(rng, sc, n), "never": rng.random(n) < sc["never"], "k": k,
            "eta": rng.normal(size=(n, 5))}


def proto_features(rng, pop, sc, o):
    s = 2 * pop["y"] - 1
    t = np.minimum(o - pop["c"], 0.0)
    g = 1 + sc["trend"] * t / 1000
    r = proto_readability(o, pop["c"], sc["ramp"]) * (~pop["never"])
    fresh = rng.normal(size=pop["eta"].shape)
    x = np.sqrt(sc["rho"]) * pop["eta"] + np.sqrt(1 - sc["rho"]) * fresh
    x[:, 0] += s * sc["a_pre"] * pop["k"] * g
    x[:, 1] += s * sc["a_post"] * pop["k"] * r
    return x


def proto_fit(x, y, lam=1e-3, iters=25):
    b, n, p = x.shape
    w = np.zeros((b, p))
    eye = np.eye(p) * lam * n
    eye[-1, -1] = 0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(x @ w[..., None])[..., 0]))
        g = (x.transpose(0, 2, 1) @ (mu - y)[..., None])[..., 0] + (eye @ w.T).T
        h = x.transpose(0, 2, 1) @ (x * (mu * (1 - mu))[..., None]) + eye
        step = np.linalg.solve(h, g[..., None])[..., 0]
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def proto_probe(feats, y, match, folds=5):
    fold = (np.arange(match.max() + 1) % folds)[match]
    z = np.empty((y.size, len(P_GRID)))
    for f in range(folds):
        tr, te = fold != f, fold == f
        xtr = np.concatenate([feats[o][tr] for o in P_TRAIN])
        ytr = np.tile(y[tr], len(P_TRAIN))
        mu, sd = xtr.mean(0), xtr.std(0)
        xs = np.concatenate([(xtr - mu) / sd, np.ones((len(xtr), 1))], 1)
        w = proto_fit(xs[None], ytr[None])[0]
        for j, o in enumerate(P_GRID):
            z[te, j] = (feats[o][te] - mu) / sd @ w[:-1] + w[-1]
    return z


def proto_gap_sums(z, y, match, m):
    cnt = np.bincount(match, minlength=m).astype(float)
    msum = np.zeros((m, z.shape[1]))
    np.add.at(msum, match, z)
    zc = z - (msum / cnt[:, None])[match]
    s_on, s_off = np.zeros((m, z.shape[1])), np.zeros((m, z.shape[1]))
    np.add.at(s_on, match, zc * y[:, None])
    np.add.at(s_off, match, zc * (1 - y)[:, None])
    return s_on, s_off, np.bincount(match, weights=y, minlength=m), np.bincount(match, weights=1 - y, minlength=m)


def proto_gap_curve(sums, w=None):
    s_on, s_off, non, noff = sums
    if w is None:
        return s_on.sum(0) / non.sum() - s_off.sum(0) / noff.sum()
    return (w @ s_on) / (w @ non)[:, None] - (w @ s_off) / (w @ noff)[:, None]


def proto_s_curve(curve):
    """sim.s_curve on the loss-like curve L = -D."""
    x = np.array(P_PRE, float)
    yv = curve[..., [P_GI[o] for o in P_PRE]]
    b = ((yv - yv.mean(-1, keepdims=True)) * (x - x.mean())).sum(-1) / ((x - x.mean()) ** 2).sum()
    a = yv.mean(-1) - b * x.mean()
    p = a[..., None] + b[..., None] * P_G
    denom = a - curve[..., [P_GI[o] for o in P_POST]].mean(-1)
    return (p - curve) / denom[..., None], denom


def proto_median_time(s):
    d = np.array(P_DENSE, float)
    sv = s[[P_GI[o] for o in P_DENSE]]
    idx = np.where(sv >= 0.5)[0]
    if idx.size == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return d[0]
    return d[j - 1] + (0.5 - sv[j - 1]) / (sv[j] - sv[j - 1]) * 40


def proto_ka_stats(d, dk):
    s0, _ = proto_s_curve(-d)
    sk, _ = proto_s_curve(-dk)
    ka, kp = [P_GI[o] for o in P_KA], [P_GI[o + 200] for o in P_KA]
    return ((sk[..., ka] - s0[..., ka]).mean(-1), 0.5 * (s0[..., kp] - s0[..., ka]).mean(-1),
            sk[..., P_GI[0]] - s0[..., P_GI[0]], 0.5 * (s0[..., P_GI[200]] - s0[..., P_GI[0]]))


def proto_replicate(sc, seed, n_boot):
    """sim.run_rep's main path and KA block (do_alt=False), same rng order; returns its outputs and the inputs."""
    rng = np.random.default_rng(seed)
    pop = proto_population(rng, sc)
    m = sc["n_matches"]
    feats = {o: proto_features(rng, pop, sc, o) for o in P_GRID}
    w = rng.multinomial(m, np.full(m, 1 / m), size=n_boot).astype(float)
    z = proto_probe(feats, pop["y"], pop["match"])
    sums = proto_gap_sums(z, pop["y"], pop["match"], m)
    d, db = proto_gap_curve(sums), proto_gap_curve(sums, w)
    s, denom = proto_s_curve(-d)
    sb, _ = proto_s_curve(-db)
    half = rng.random(pop["y"].size) < 0.5
    fk = {o: np.where(half[:, None], feats[o + 200], feats[o]) for o in P_GRID if o + 200 in feats}
    for o in P_GRID:
        fk.setdefault(o, feats[o])
    zk = proto_probe(fk, pop["y"], pop["match"])
    sk = proto_gap_sums(zk, pop["y"], pop["match"], m)
    dk, dkb = proto_gap_curve(sk), proto_gap_curve(sk, w)
    a, ap, d0, d0p = proto_ka_stats(d, dk)
    ab, _, _, _ = proto_ka_stats(db, dkb)
    out = {"np": s[P_GI[0]], "np_lo": np.quantile(sb[:, P_GI[0]], 0.025), "np_hi": np.quantile(sb[:, P_GI[0]], 0.975),
           "denom": denom, "S": s, "tmed": proto_median_time(s), "ka_A": a, "ka_Ap": ap, "ka_A_lo": np.quantile(ab, 0.025),
           "ka_d0": d0, "ka_d0p": d0p, "share_eff": float(proto_readability(0.0, pop["c"], sc["ramp"]).mean())}
    return out, {"pop": pop, "feats": feats, "w": w, "z": z, "zk": zk, "half": half}


# ---------------------------------------------------------------- helpers

def curves_of(z, y, match, n_matches, counts):
    every = np.ones(len(y), bool)
    sums = rd.match_sums(z, y, match, n_matches, every, every)
    return rd.gap(sums), rd.gap(sums, counts)


def table_logits(pop, sc, rng, a_pre=0.3, a_post=0.6):
    """A synthetic table: per-shot logits whose readability follows the planted contact times, ramp and never-readable
    share (`proto_features`' two signal features read with unit weights), at every grid offset."""
    return np.stack([proto_features(rng, pop, sc, o)[:, :2].sum(1) for o in rd.GRID], 1)


# ---------------------------------------------------------------- prototype reproduction

def test_grid_and_windows_are_the_prototypes():
    assert tuple(P_GRID) == rd.GRID and tuple(P_KA) == sh.KA
    assert tuple(P_PRE) == rd.PRE and tuple(P_POST) == rd.POST and tuple(P_DENSE) == rd.DENSE


@pytest.mark.parametrize(("contact", "blur"), [("n+80_80", False), ("n0_80", True)])
def test_estimator_reproduces_the_prototype_on_its_synthetic_data(contact, blur):
    """Same logits and bootstrap weights in, the prototype's U, interval, S, t50 and KA statistics out (exactly)."""
    sc = proto_scenario(contact, blur)
    out, inp = proto_replicate(sc, seed=11, n_boot=200)
    pop, w = inp["pop"], inp["w"]
    d, db = curves_of(inp["z"], pop["y"], pop["match"], sc["n_matches"], w)
    dk, dkb = curves_of(inp["zk"], pop["y"], pop["match"], sc["n_matches"], w)
    np.testing.assert_allclose(sh.normalised(d), out["S"], atol=1e-10)
    assert sh.share(d) == pytest.approx(out["np"], abs=1e-10)
    assert rd.jump(d) == pytest.approx(out["denom"], abs=1e-12)
    u = sh.estimate(float(sh.share(d)), sh.share(db))
    assert (u["lo"], u["hi"]) == pytest.approx((out["np_lo"], out["np_hi"]), abs=1e-10)
    assert float(sh.t50(sh.normalised(d))) == pytest.approx(out["tmed"], abs=1e-9)
    ka, kab = sh.known_answer(d, dk), sh.known_answer(db, dkb)
    assert (ka.a, ka.a_pred, ka.anchor, ka.anchor_pred) == pytest.approx(
        (out["ka_A"], out["ka_Ap"], out["ka_d0"], out["ka_d0p"]), abs=1e-10)
    assert sh.estimate(float(ka.a), kab.a)["lo"] == pytest.approx(out["ka_A_lo"], abs=1e-10)
    # the anchor move used by the real pipeline equals the prototype's feature swap
    stacked = np.stack([inp["feats"][o] for o in P_GRID], 1)
    moved = sh.move_anchor(stacked, inp["half"])
    for o in P_GRID:
        expected = np.where(inp["half"][:, None], inp["feats"][o + 200], inp["feats"][o]) if o + 200 in inp["feats"] \
            else inp["feats"][o]
        np.testing.assert_array_equal(moved[:, J[o]], expected)


def test_sklearn_probe_pipeline_reproduces_the_prototype_share_and_known_answer():
    """The real probe (`readability.probe_readout`, StandardScaler + lbfgs) at the prototype's penalty gives the
    prototype's U and A to within the probe solvers' difference."""
    sc = proto_scenario("tail15", blur=False)
    out, inp = proto_replicate(sc, seed=5, n_boot=100)
    pop = inp["pop"]
    y, match = pop["y"].astype(int), pop["match"]
    fold = match % 5
    rows = 0.8 * len(y) * len(rd.PROBE_TRAIN)
    c = 1 / (1e-3 * rows)                      # sim: penalty lam * n on standardised weights, n = training rows

    def fit(features):
        stacked = np.stack([features[o] for o in P_GRID], 1)
        res = rd.probe_readout(lambda r, j: stacked[r, j], y, np.ones(len(y), bool), fold, (c,))
        return res.logits[0]

    z = fit(inp["feats"])
    assert np.abs(z - inp["z"]).max() < 0.02
    stacked = np.stack([inp["feats"][o] for o in P_GRID], 1)
    moved = sh.move_anchor(stacked, inp["half"])
    zk = fit({o: moved[:, J[o]] for o in P_GRID})
    d, _ = curves_of(z, y, match, sc["n_matches"], inp["w"])
    dk, _ = curves_of(zk, y, match, sc["n_matches"], inp["w"])
    assert sh.share(d) == pytest.approx(out["np"], abs=0.005)
    assert sh.known_answer(d, dk).a == pytest.approx(out["ka_A"], abs=0.005)
    assert sh.share(d) == pytest.approx(out["share_eff"], abs=0.05)


@pytest.mark.skipif(not PROTOTYPE.exists(), reason="the untracked prototype folder is not on this machine")
def test_the_copy_matches_the_prototype_file(monkeypatch):
    import hashlib

    assert hashlib.sha256(PROTOTYPE.read_bytes()).hexdigest() == PROTOTYPE_SHA256
    spec = importlib.util.spec_from_file_location("proto_sim", PROTOTYPE)
    sim = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "proto_sim", sim)         # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(sim)
    sc = proto_scenario("n+80_80", blur=False)
    real = sim.run_rep((sim.Scenario("t", contact=sc["contact"], a_pre=0.3, a_post=0.6), 3, 100, True, False))
    copy, _ = proto_replicate(sc, seed=3, n_boot=100)
    for k in ("np", "np_lo", "np_hi", "denom", "tmed", "ka_A", "ka_Ap", "ka_A_lo", "ka_d0", "ka_d0p", "share_eff"):
        assert copy[k] == pytest.approx(real[k], abs=1e-12), k
    np.testing.assert_allclose(copy["S"], real["S"], atol=1e-12)


# ---------------------------------------------------------------- planted share and blur on synthetic tables

@pytest.mark.parametrize(("contact", "blur", "eff"), [("null", False, 0.0), ("n+80_80", False, 0.158),
                                                        ("n0_80", False, 0.5), ("n0_80", True, 0.377)])
def test_planted_share_and_blur_are_recovered_from_a_table(contact, blur, eff):
    """U recovers the effective share (blur: below the true post-contact fraction 0.5), the KA anchor move is detected
    with A / A_pred near 1, and the gates pass."""
    sc = proto_scenario(contact, blur)
    rng = np.random.default_rng(21)
    pop = proto_population(rng, sc)
    z = table_logits(pop, sc, rng)
    y, match = pop["y"].astype(int), pop["match"]
    keys, counts = rd.match_draw_counts(match, n_boot=300)
    d, db = curves_of(z, y, match, len(keys), counts)
    summary = sh.reading_summary(d, db)
    true_eff = float(proto_readability(0.0, pop["c"], sc["ramp"]).mean())
    assert true_eff == pytest.approx(eff, abs=0.02)
    assert summary["u"]["value"] == pytest.approx(true_eff, abs=0.03)
    assert summary["u"]["lo"] <= true_eff + 0.01 and true_eff - 0.01 <= summary["u"]["hi"]
    assert sh.positive_control(summary)["passed"]
    if blur:
        assert summary["u"]["value"] < 0.45                     # effective share, not the fraction after contact

    half = sh.ka_half(len(y))
    dk, dkb = curves_of(sh.move_anchor(z, half), y, match, len(keys), counts)
    gate = sh.known_answer_gate(sh.known_answer(d, dk), sh.known_answer(db, dkb))
    assert gate["passed"], gate
    assert 0.8 < gate["ratio"] < 1.25


def test_t50_tracks_the_median_contact_time():
    for mu in (-200.0, 0.0, 80.0):
        sc = proto_scenario("n0_80", blur=False) | {"contact": ("normal", mu, 60.0)}
        rng = np.random.default_rng(2)
        pop = proto_population(rng, sc)
        z = table_logits(pop, sc, rng, a_post=1.4)
        d, _ = curves_of(z, pop["y"].astype(int), pop["match"], sc["n_matches"], np.ones((1, sc["n_matches"])))
        assert float(sh.t50(sh.normalised(d))) == pytest.approx(mu, abs=25)


def test_an_exact_step_gives_the_planted_share_exactly():
    """D = level + Delta * F(o), F the share of shots past contact: S = F on the grid, U = F(0)."""
    frac = np.clip((GRID + 120) / 400, 0, 1)                    # contacts uniform on -120..+280
    d = 0.2 + 0.5 * frac
    np.testing.assert_allclose(sh.normalised(d)[[J[o] for o in rd.DENSE]], frac[[J[o] for o in rd.DENSE]], atol=1e-12)
    assert sh.share(d) == pytest.approx(0.3)
    assert float(sh.t50(sh.normalised(d))) == pytest.approx(80.0)


# ---------------------------------------------------------------- t50, anchor move, KA half

def test_t50_codes_and_interpolation():
    s = np.zeros((4, len(GRID)))
    s[0, J[-400]:] = 0.6                                         # already above at -400
    s[2, J[-40]] = 0.2                                           # crosses between -40 and 0
    s[2, J[0]:] = 0.8
    s[3] = np.nan
    t = sh.t50(s)
    assert t[0] == sh.T50_BELOW and t[1] == sh.T50_ABOVE and np.isnan(t[3])
    assert t[2] == pytest.approx(-40 + 0.3 / 0.6 * 40)
    assert [sh.t50_label(v) for v in t] == ["<= -400", "> +400", "-20", "nan"]
    s5 = np.zeros(len(GRID))
    s5[J[400] + 1:] = 1.0                                        # only after +400: outside DENSE, still "> +400"
    assert sh.t50(s5) == sh.T50_ABOVE


def test_anchor_move_replaces_partnered_offsets_of_the_moved_rows_only():
    values = np.arange(3 * len(GRID)).reshape(3, len(GRID)) * 1.0
    moved = np.array([True, False, True])
    out = sh.move_anchor(values, moved)
    np.testing.assert_array_equal(out[1], values[1])
    for o in rd.GRID:
        expected = values[0, J[o + 200]] if o + 200 in J else values[0, J[o]]
        assert out[0, J[o]] == expected
    assert sh.NO_PARTNER == (240, 280, 320, 360, 1000)
    refs = np.stack([values, values], -1).astype(int)            # trailing axes (e.g. refs) travel with the offset
    np.testing.assert_array_equal(sh.move_anchor(refs, moved)[..., 0], out.astype(int))


def test_ka_half_is_the_protocols_draw():
    half = sh.ka_half(1000)
    np.testing.assert_array_equal(half, np.random.default_rng(1).random(1000) < 0.5)
    assert 400 < half.sum() < 600


# ---------------------------------------------------------------- gates and verdict

def est(value, lo, hi):
    return {"value": value, "lo": lo, "hi": hi, "skipped_draws": 0}


def g_rule(positive=True, u=(0.3, 0.2, 0.4), diff=(0.2, 0.05, 0.3)):
    return {"positive_passed": positive, "u": est(*u), "diff": est(*diff)}


PASS = dict.fromkeys(sh.F_GATES, True)


def test_positive_control_needs_both_conditions():
    base = {"delta": est(0.1, 0.05, 0.15), "u_half_width": 0.15}
    assert sh.positive_control(base)["passed"]
    assert not sh.positive_control(base | {"u_half_width": 0.1501})["passed"]
    assert not sh.positive_control(base | {"delta": est(0.1, -0.01, 0.2)})["passed"]


def test_known_answer_gate_thresholds():
    def gate(a, a_lo, a_pred):
        draws = sh.KnownAnswer(np.linspace(a_lo, a + (a - a_lo), 41), np.full(41, a_pred), np.zeros(41), np.zeros(41))
        return sh.known_answer_gate(sh.KnownAnswer(np.float64(a), np.float64(a_pred), np.float64(0), np.float64(0)), draws)

    assert gate(0.16, 0.05, 0.16)["passed"]
    assert not gate(0.16, -0.01, 0.16)["passed"]                # not detected
    assert not gate(0.16, 0.05, 0.32 + 1e-6)["passed"]          # ratio just below 0.5
    assert gate(0.16, 0.05, 0.32)["passed"] and gate(0.24, 0.05, 0.16)["passed"]     # ratio 0.5 and 1.5 pass
    assert not gate(0.25, 0.05, 0.16)["passed"]
    assert not gate(0.16, 0.05, -0.1)["passed"]                 # a negative prediction never passes


def test_verdict_rules_apply_in_order():
    u_low = est(0.02, -0.05, 0.08)
    r = sh.verdict(PASS | {"known_answer": False, "positive_control": False}, est(0.5, 0.3, 0.7), {"G": g_rule()})
    assert (r["rule"], r["verdict"], r["failed_gates"]) == (1, "inconclusive", ["positive_control", "known_answer"])
    assert sh.verdict(PASS, est(0.3, 0.11, 0.5), {"G": g_rule()})["rule"] == 2
    assert sh.verdict(PASS, est(0.3, 0.10, 0.5), {"G": g_rule(positive=False)})["rule"] == 4   # 0.10 is not above
    r = sh.verdict(PASS, u_low, {"G": g_rule(positive=False), "X": g_rule()})
    assert (r["rule"], r["verdict"]) == (3, "post_contact_leakage_learned_only_by_g") and r["rule3"]["X"]["met"]
    for bad in (g_rule(positive=False), g_rule(u=(0.2, 0.1, 0.3)), g_rule(diff=(0.1, 0.0, 0.2))):
        r = sh.verdict(PASS, u_low, {"G": bad, "X": bad})
        assert r["rule"] == 4 and r["sentence"] == "udział powyżej 0,10 wykluczony"
    r = sh.verdict(PASS, est(0.05, -0.02, 0.17), {"G": g_rule(positive=False)})
    assert r["rule"] == 4 and r["sentence"] == "udział do 0,17 nie wykluczony"
    with pytest.raises(ValueError, match="gates"):
        sh.verdict({"placebo": True}, u_low, {})


def test_summary_of_a_reading_and_paired_difference():
    rng = np.random.default_rng(0)
    d = 0.1 + 0.4 * (GRID > 0)                                   # S jumps from 0 to 1 between 0 and +40
    draws = d + rng.normal(0, 0.01, (200, len(GRID)))
    s = sh.reading_summary(d, draws)
    assert s["u"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert s["u_half_width"] == pytest.approx((s["u"]["hi"] - s["u"]["lo"]) / 2)
    assert s["t50"]["label"] == "+20"
    diff = sh.paired_difference((sh.share(d), sh.share(draws)), (sh.share(d), sh.share(draws)))
    assert diff["value"] == 0 and diff["lo"] == 0 and diff["hi"] == 0
    assert sh.flat("U", s["u"]) == {"U": s["u"]["value"], "U_lo": s["u"]["lo"], "U_hi": s["u"]["hi"]}


# ---------------------------------------------------------------- script guards

def _script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import report_post_contact_share as script

    return script


def test_work_logits_round_trip_and_refuse_a_reordered_file(tmp_path):
    script = _script()
    n = 3
    long = pd.DataFrame({c: np.repeat([f"s{i}" for i in range(n)], len(rd.GRID)) if c == "shot_id" else 0
                         for c in script.SHOT_COLUMNS})
    long.insert(5, "offset_ms", np.tile(rd.GRID, n))
    for r in (*script.READINGS, "F_neg"):
        long[r] = np.arange(n * len(rd.GRID), dtype=float)
    long.to_parquet(tmp_path / "logits.parquet")
    shots, logits = script.load_logits(tmp_path / "logits.parquet")
    assert list(shots.shot_id) == ["s0", "s1", "s2"]
    np.testing.assert_array_equal(logits["F"][1], np.arange(len(rd.GRID), 2 * len(rd.GRID)))
    long.iloc[::-1].to_parquet(tmp_path / "logits.parquet")
    with pytest.raises(SystemExit, match="shot-major"):
        script.load_logits(tmp_path / "logits.parquet")


def test_ticket09_guard_refuses_a_changed_work_file(tmp_path):
    script = _script()
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    out.mkdir()
    (work / "curves.npz").write_bytes(b"a")
    record = {"protocol_sha256": rd.PROTOCOL_SHA256, "work_files": {"curves.npz": sha256_file(work / "curves.npz")},
              "sources": {s: sha256_file(script.EXP / s) for s in script.TICKET09_SOURCES}, "inputs": {"x": "1"}}
    (out / "complete.json").write_text(json.dumps(record))
    assert script.verify_ticket09(out, work, {"x": "1"}) == record
    with pytest.raises(SystemExit, match="other ticket-08 inputs"):
        script.verify_ticket09(out, work, {"x": "2"})
    (work / "curves.npz").write_bytes(b"b")
    with pytest.raises(SystemExit, match="does not match"):
        script.verify_ticket09(out, work, {"x": "1"})
