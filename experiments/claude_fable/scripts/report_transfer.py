"""Leave-one-league-out transfer gap, calibration and recalibration curve (xg-credibility 13).

    CUDA_VISIBLE_DEVICES="" ./run.sh python scripts/report_transfer.py

CPU only, from the 40 finished fits of ticket 12 (results/xgc_transfer), exactly as the
approved protocol research/xgc_transfer_protocol.md fixes it (section numbers below are
the protocol's):

  5  calibration per fit, on its own arm only: Platt on the calibration slice's logits,
     prior shift from the slice's goal rate to the rate of all 300 arm matches, penalties
     at the Beta posterior rate of the arm's penalties;
  6  pooled log-loss gap "without" minus "with" on every test shot of the five leagues,
     per training seed and as their mean, 95% percentile interval from 2 000 match draws
     stratified by league (seed 0, the same draws for both seeds), relative gap inside the
     bootstrap, the pre-registered reading; secondary metrics per league and pooled for
     both arms and their difference (log-loss, AUC, Brier, ECE on 10 equal-width bins,
     calibration slope, intercept and mean xG minus goal rate on open play), the
     equal-weight and open-play pooled gaps;
  7  recalibration curve, k = 0, 5, 10, 20 matches of the test half-set, 200 draws per
     (league, half-set), Platt (primary) and intercept-only (secondary) on the "without"
     arm, plus the "with" arm recalibrated by Platt on the same matches;
  8  the gap under the ticket-03 attribution rules and bounds (test labels only);
  9  reliability bins of both arms on every league.

Before anything is read, the plan is re-derived and checked against the approved hash
(`arms.verify_plan`) and every fit is checked against its completion record
(`arms.completeness`: plan and protocol hash, file hashes, shot coverage). Labels come
from the index; the CSV `goal` column must agree with it.

Every match was seen in earlier rounds and the recipe was chosen on validation data from
all leagues, so every number here is a development result.

Writes the result folder results_reports/transfer_2026-10 (small CSVs, figures,
summary.json, complete.json with source, input and output hashes and the protocol hash)
and the per-draw recalibration rows to results/xgc_transfer_report (git-ignored).
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib
import numpy as np
import pandas as pd
from report_complexity import (
    GRID,
    INK,
    INK2,
    MUTED,
    SURFACE,
    _axes_style,
    interval,
    recalibration,
    scores,
)
from xgpoc.bundle import canonical_digest, code_version, sha256_file
from xgpoc.eval.attribution import attribute, bound_labels, relabel, source_events
from xgpoc.eval.calibration import (
    PlattScaler,
    logit,
    penalty_rate,
    prior_shift,
    route_penalties,
    sigmoid,
)
from xgpoc.eval.metrics import expected_calibration_error
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.transfer import arms
from xgpoc.transfer import plan as split_plan

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results" / "xgc_transfer"
WORK = EXP / "results" / "xgc_transfer_report"
FIGURES = EXP / "results_reports" / "transfer_2026-10"

LEAGUES = split_plan.DOMESTIC_LEAGUES
NAMES = {"england_epl": "Premier League", "france_ligue-1": "Ligue 1", "germany_bundesliga": "Bundesliga",
         "italy_serie-a": "Serie A", "spain_laliga": "LaLiga"}
INDICATIVE = frozenset({"france_ligue-1", "germany_bundesliga"})  # protocol 6: few goals, small dose
COLUMNS = [(arm, seed) for arm in split_plan.ARMS for seed in arms.SEEDS]  # prediction columns, arm-major


def p_col(arm: str, seed: int) -> str:
    """Name of the calibrated prediction column of one arm and training seed."""
    return f"p_{arm}_s{seed}"


P_COLS = [p_col(arm, seed) for arm, seed in COLUMNS]
ARM_IX = {arm: [i for i, (a, _) in enumerate(COLUMNS) if a == arm] for arm in split_plan.ARMS}
ARM_COLS = {arm: [P_COLS[i] for i in ix] for arm, ix in ARM_IX.items()}
METRICS = ("log_loss", "auc", "brier", "ece", "cal_large", "intercept", "slope")
N_BOOT = 2000
BOOT_SEED = 0
K_GRID = (0, 5, 10, 20)
MIN_EVAL = 10        # matches of the half-set that must stay for scoring the curve
N_RECAL_DRAWS = 200
MAIN_CURVE = ("england_epl", "italy_serie-a", "spain_laliga")  # leagues with K = 20
ECE_EDGES = np.linspace(0, 1, 11)  # the 10 equal-width bins of xgpoc.eval.metrics
# point-1 reference for training noise (protocol 6): G5 seed-to-seed gap, single fold models vs G-combo
NOISE_REFERENCE = {"g5_seed_to_seed": 0.0005, "single_fold_model_vs_g_combo": [0.0010, 0.0083]}
ARM_COLOR = {"without": "#eb6834", "with": "#2a78d6"}
INTERCEPT_COLOR = "#1baf7a"  # dataviz reference slot 3 (validated with slots 1-2)
ARM_LABEL = {"without": "bez ligi", "with": "z ligą"}
SOURCES = ("scripts/report_transfer.py", "scripts/report_complexity.py", "scripts/eval_subgroups.py", "xgpoc/bundle.py",
           "xgpoc/eval/attribution.py", "xgpoc/eval/calibration.py", "xgpoc/eval/metrics.py", "xgpoc/paths.py",
           "xgpoc/data/soccernet_labels.py", "xgpoc/transfer/arms.py", "xgpoc/transfer/plan.py")


# ---------------------------------------------------------------- calibration (protocol 5)

def calibrate(platt: PlattScaler, calibration_goal_rate: float, logits: np.ndarray, is_penalty: np.ndarray,
              arm: dict) -> np.ndarray:
    """Probabilities for `logits` of one fit: `platt` (fitted on every shot of the arm's
    calibration slice), prior shift from the slice's goal rate to the arm's (all 300
    matches), penalties at the Beta posterior rate of the arm's penalties. `arm` is
    `arms.arm_constants`."""
    p = prior_shift(platt(logits), calibration_goal_rate, arm["arm_goal_rate"])
    return route_penalties(p, is_penalty, penalty_rate(arm["arm_penalty_goals"], arm["arm_penalties"]))


# ---------------------------------------------------------------- bootstrap and metrics (protocol 6)

def stratified_draws(games: np.ndarray, strata: np.ndarray, n_boot: int, seed: int = BOOT_SEED):
    """Row indices of match resamples stratified by `strata` (row-aligned with `games`): in
    every stratum, in sorted order, as many matches as it has, drawn with replacement."""
    cells = []
    for s in sorted(set(strata)):
        rows = np.flatnonzero(strata == s)
        keys, inv = np.unique(games[rows], return_inverse=True)
        cells.append([rows[inv == k] for k in range(len(keys))])
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([c[k] for c in cells for k in rng.choice(len(c), len(c), replace=True)])


def league_groups(league: np.ndarray) -> dict[str, np.ndarray]:
    """Row masks: "pooled" (every row), then each league in sorted order."""
    return {"pooled": np.ones(len(league), bool)} | {lg: league == lg for lg in sorted(set(league))}


def log_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-shot log-loss, clipped as everywhere in the project; works column-wise on 2-D p."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    y = y.reshape(-1, *([1] * (q.ndim - 1)))
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def sample_metrics(y: np.ndarray, p: np.ndarray, open_play: np.ndarray) -> np.ndarray:
    """METRICS of one prediction column: log-loss, AUC, Brier and ECE on every shot; mean xG
    minus goal rate, recalibration intercept and slope on open play (penalties have a
    constant rate)."""
    auc, ll, brier = scores(y, p)
    a, b = recalibration(y[open_play], p[open_play])
    return np.array([ll, auc, brier, expected_calibration_error(y, p),
                     p[open_play].mean() - y[open_play].mean(), a, b])


def ece_bin(p: np.ndarray) -> np.ndarray:
    """Equal-width bin of every prediction, as `expected_calibration_error` bins them."""
    return np.clip(np.digitize(p, ECE_EDGES) - 1, 0, len(ECE_EDGES) - 2)


def reliability_sums(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(2, bins): shots and goals per equal-width ECE bin of the given predictions."""
    b, n_bins = ece_bin(p), len(ECE_EDGES) - 1
    return np.stack([np.bincount(b, minlength=n_bins), np.bincount(b, weights=y, minlength=n_bins)])


def bootstrap(test: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Point values and stratified draws of every per-group quantity. Groups: "pooled" and
    each league. Per group: METRICS for every prediction column (shape (4, M)), open-play
    log-loss per column, reliability sums per arm (seeds stacked)."""
    y = test.goal.to_numpy().astype(float)
    op = ~test.is_penalty.to_numpy().astype(bool)
    league = test.league.to_numpy()
    p = test[P_COLS].to_numpy()
    groups = league_groups(league)

    def draw_values(rows: np.ndarray) -> dict:
        out = {}
        for g, mask in groups.items():
            r = rows[mask[rows]]
            yy, oo, pp = y[r], op[r], p[r]
            out[g] = {"m": np.array([sample_metrics(yy, pp[:, j], oo) for j in range(len(P_COLS))]),
                      "open_ll": log_losses(yy[oo], pp[oo]).mean(0),
                      "rel": np.array([reliability_sums(np.tile(yy, len(ix)), pp[:, ix].T.ravel())
                                       for ix in ARM_IX.values()])}
        return out

    point = draw_values(np.arange(len(test)))
    boot = [draw_values(r) for r in stratified_draws(test.game.to_numpy(), league, n_boot, seed)]
    return {"groups": list(groups), "point": point,
            "boot": {g: {k: np.array([b[g][k] for b in boot]) for k in point[g]} for g in groups}}


def reading(lo: float, hi: float, seed_gaps: list[float]) -> str:
    """The pre-registered reading of protocol 6."""
    if lo <= 0 <= hi:
        return "no_gap_shown"
    return "gap_shown" if len({np.sign(g) for g in seed_gaps}) == 1 and seed_gaps[0] != 0 else "ambiguous_between_seeds"


def arm_mean(x: np.ndarray, arm: str) -> np.ndarray:
    """Mean over the arm's seed columns (axis -2 holds the columns)."""
    return x[..., ARM_IX[arm], :].mean(-2)


def metric_table(res: dict) -> pd.DataFrame:
    """Per group and metric: both arms (seed mean) and without - with, with 95% intervals,
    plus each seed's point value."""
    rows = []
    for g in res["groups"]:
        pt, bt = res["point"][g]["m"], res["boot"][g]["m"]
        for j, m in enumerate(METRICS):
            row = {"group": g, "metric": m}
            for arm in split_plan.ARMS:
                row[arm] = arm_mean(pt, arm)[j]
                row[f"{arm}_lo"], row[f"{arm}_hi"] = interval(arm_mean(bt, arm)[:, j])
            row["diff"] = row["without"] - row["with"]
            row["diff_lo"], row["diff_hi"] = interval(arm_mean(bt, "without")[:, j] - arm_mean(bt, "with")[:, j])
            for i, (arm, seed) in enumerate(COLUMNS):
                row[f"{arm}_seed{seed}"] = pt[i, j]
            rows.append(row)
    return pd.DataFrame(rows)


def gap_row(name: str, pt: np.ndarray, bt: np.ndarray) -> dict:
    """pt: log-loss per prediction column (4,), bt: the same per draw (B, 4). Gap = without -
    with per seed; the result is their mean, its interval comes from the seed-mean draws."""
    seed_gaps = [pt[ARM_IX["without"][s]] - pt[ARM_IX["with"][s]] for s in range(len(arms.SEEDS))]
    lo, hi = interval(bt[:, ARM_IX["without"]].mean(1) - bt[:, ARM_IX["with"]].mean(1))
    return {"group": name, "gap": float(np.mean(seed_gaps)), "gap_lo": lo, "gap_hi": hi,
            **{f"gap_seed{s}": float(g) for s, g in zip(arms.SEEDS, seed_gaps, strict=True)},
            "seed_difference": float(seed_gaps[0] - seed_gaps[1]),
            "log_loss_without": float(pt[ARM_IX["without"]].mean()), "log_loss_with": float(pt[ARM_IX["with"]].mean())}


def headline_extras(row: dict, bt: np.ndarray) -> dict:
    """What protocol 6 and 9 add to the main row only: the pre-registered reading and the
    relative gap (gap / log-loss of "with", the ratio taken inside every draw)."""
    with_bt = bt[:, ARM_IX["with"]].mean(1)
    rel_lo, rel_hi = interval((bt[:, ARM_IX["without"]].mean(1) - with_bt) / with_bt)
    seed_gaps = [row[f"gap_seed{s}"] for s in arms.SEEDS]
    return {"reading": reading(row["gap_lo"], row["gap_hi"], seed_gaps), "relative_gap": row["gap"] / row["log_loss_with"],
            "relative_gap_lo": rel_lo, "relative_gap_hi": rel_hi}


def gap_table(res: dict, test: pd.DataFrame) -> pd.DataFrame:
    """Log-loss gaps: pooled by shots (the primary result), equal-weight leagues, open play,
    then each league."""
    ll = METRICS.index("log_loss")
    leagues = [g for g in res["groups"] if g != "pooled"]

    def log_loss_values(g: str, key: str = "m") -> tuple[np.ndarray, np.ndarray]:
        """(point (4,), draws (B, 4)) of the log-loss per prediction column."""
        pt, bt = res["point"][g][key], res["boot"][g][key]
        return (pt[:, ll], bt[:, :, ll]) if key == "m" else (pt, bt)

    pooled = gap_row("pooled", *log_loss_values("pooled"))
    rows = [pooled | headline_extras(pooled, log_loss_values("pooled")[1])]
    rows.append(gap_row("equal_weight", np.mean([log_loss_values(g)[0] for g in leagues], axis=0),
                        np.mean([log_loss_values(g)[1] for g in leagues], axis=0)))
    rows.append(gap_row("open_play", *log_loss_values("pooled", "open_ll")))
    rows += [gap_row(g, *log_loss_values(g)) for g in leagues]
    out = pd.DataFrame(rows)
    by = test.groupby("league")
    counts = {"pooled": (len(test), test.game.nunique(), int(test.goal.sum()))} | {
        g: (len(d), d.game.nunique(), int(d.goal.sum())) for g, d in by}
    counts["equal_weight"] = counts["pooled"]
    counts["open_play"] = (int((~test.is_penalty).sum()), test.game.nunique(), int(test.goal[~test.is_penalty].sum()))
    out[["shots", "matches", "goals"]] = [counts[g] for g in out.group]
    out["indicative"] = out.group.isin(INDICATIVE)
    return out


def reliability_table(res: dict) -> pd.DataFrame:
    """Per league and arm (both seeds stacked): shots, mean xG and goal rate with a 95%
    interval per equal-width bin, the bins of the ECE."""
    rows = []
    for g in res["groups"]:
        if g == "pooled":
            continue
        pt, bt = res["point"][g]["rel"], res["boot"][g]["rel"]
        for a, arm in enumerate(split_plan.ARMS):
            n, goals = pt[a]
            with np.errstate(invalid="ignore", divide="ignore"):
                rate_bt = bt[:, a, 1] / bt[:, a, 0]
            for b in range(len(n)):
                if n[b] == 0:
                    continue
                draws = rate_bt[:, b][np.isfinite(rate_bt[:, b])]
                rows.append({"league": g, "arm": arm, "bin": b, "bin_lo": ECE_EDGES[b], "bin_hi": ECE_EDGES[b + 1],
                             "shots": int(n[b]), "goal_rate": goals[b] / n[b],
                             "goal_rate_lo": interval(draws)[0], "goal_rate_hi": interval(draws)[1]})
    return pd.DataFrame(rows)


def bin_mean_xg(test: pd.DataFrame, rel: pd.DataFrame) -> pd.Series:
    """Mean prediction in each reliability row's bin (seeds stacked)."""
    out = []
    for r in rel.itertuples():
        d = test[test.league == r.league]
        p = d[ARM_COLS[r.arm]].to_numpy().ravel()
        out.append(p[ece_bin(p) == r.bin].mean())
    return pd.Series(out, index=rel.index)


# ---------------------------------------------------------------- recalibration curve (protocol 7)

def k_cap(n_matches: int, grid: tuple[int, ...] = K_GRID, min_eval: int = MIN_EVAL) -> int:
    """K: the largest k of the grid that leaves at least `min_eval` matches for scoring."""
    return max(k for k in grid if n_matches - k >= min_eval)


def draw_rng(league: str, half_set: str) -> np.random.Generator:
    """The recalibration draws' generator, derived from (plan seed 0, L, h) as the plan's own
    streams are (`plan._rng`)."""
    return split_plan._rng(0, league, half_set)


def fit_intercept(z: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """b of y ~ sigmoid(z + b) (slope fixed at 1) by Newton; needs both classes in y."""
    b = 0.0
    for _ in range(iters):
        q = sigmoid(z + b)
        step = float((y - q).sum() / (q * (1 - q)).sum())
        b += step
        if abs(step) < 1e-12:
            break
    return b


def recalibrated(p: np.ndarray, is_penalty: np.ndarray, fit: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    """p re-mapped on its logit by a map fitted on the open-play rows `fit`: "platt" (a z + b)
    or "intercept" (z + b). Penalties keep their rate."""
    z = logit(p)
    sel = fit & ~is_penalty
    if mode == "platt":
        scaler = PlattScaler().fit(z[sel], y[sel])
        q = scaler(z)
    elif mode == "intercept":
        q = sigmoid(z + fit_intercept(z[sel], y[sel]))
    else:
        raise ValueError(f"unknown recalibration {mode!r}")
    return np.where(is_penalty, p, q)


def recalibration_rows(cell: pd.DataFrame, league: str, half_set: str, n_draws: int = N_RECAL_DRAWS) -> pd.DataFrame:
    """Per draw, k and seed: evaluation shots and summed log-loss of "without" (raw, Platt,
    intercept-only) and "with" (raw, Platt). One draw = a permutation of the sorted
    half-set matches; its first K are the candidates, k takes their first k, the rest is
    the evaluation set of every k. When the k matches hold no open-play goal the draw stays
    at k = 0 for that k (`fallback`)."""
    games = np.array(sorted(cell.game.unique()))
    cap = k_cap(len(games))
    rng = draw_rng(league, half_set)
    y = cell.goal.to_numpy().astype(float)
    pen = cell.is_penalty.to_numpy().astype(bool)
    g = cell.game.to_numpy()
    rows = []
    for d in range(n_draws):
        order = rng.permutation(games)
        evaluate = ~np.isin(g, order[:cap])
        for k in (k for k in K_GRID if k <= cap):
            fit = np.isin(g, order[:k])
            yf = y[fit & ~pen]
            fallback = k > 0 and not yf.any()
            for seed in arms.SEEDS:
                row = {"league": league, "half_set": half_set, "draw": d, "k": k, "seed": seed, "fallback": fallback,
                       "n_eval": int(evaluate.sum())}
                for arm in split_plan.ARMS:
                    p = cell[p_col(arm, seed)].to_numpy()
                    row[f"{arm}_raw"] = log_losses(y[evaluate], p[evaluate]).sum()
                    modes = ("platt", "intercept") if arm == "without" else ("platt",)
                    for mode in modes:
                        q = p if k == 0 or fallback else recalibrated(p, pen, fit, y, mode)
                        row[f"{arm}_{mode}"] = log_losses(y[evaluate], q[evaluate]).sum()
                rows.append(row)
    return pd.DataFrame(rows)


CURVE_VARIANTS = {"without_platt": "gap", "without_intercept": "gap", "with_platt": "with_change"}


def curve(rows: pd.DataFrame, leagues: tuple[str, ...], ks: tuple[int, ...]) -> pd.DataFrame:
    """Curve of the given leagues (shots pooled across their half-sets per draw index):
    per k and variant the log-loss minus that of "with" without recalibration, mean over
    seeds, then mean and 5-95% spread over draws. For the "without" variants this is the
    gap; recovered share 1 - gap(k) / gap(0) on the draw means, only when gap(0) > 0."""
    sub = rows[rows.league.isin(leagues) & rows.k.isin(ks)]
    if set(sub.k) != set(ks) or sub.groupby("k").league.nunique().min() != len(leagues):
        raise ValueError(f"curve over {leagues}: not every league has every k of {ks}")
    cols = ["n_eval", "with_raw", *CURVE_VARIANTS]
    agg = sub.groupby(["draw", "k", "seed"])[cols].sum()
    gaps = pd.DataFrame({v: (agg[v] - agg["with_raw"]) / agg["n_eval"] for v in CURVE_VARIANTS})
    per_draw = gaps.groupby(["draw", "k"]).mean()
    out = []
    for v in CURVE_VARIANTS:
        stats = per_draw[v].groupby("k").agg(mean="mean", p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
        gap0 = stats.loc[0, "mean"]
        for k, s in stats.iterrows():
            recovered = 1 - s["mean"] / gap0 if CURVE_VARIANTS[v] == "gap" and gap0 > 0 else np.nan
            out.append({"variant": v, "k": k, "mean": s["mean"], "p5": s["p5"], "p95": s["p95"], "recovered": recovered})
    fb = sub[sub.seed == arms.SEEDS[0]].groupby("k").fallback.sum()
    res = pd.DataFrame(out)
    res["fallback_draws"] = res.k.map(fb).astype(int)
    res["eval_shots"] = int(sub[(sub.draw == 0) & (sub.k == 0) & (sub.seed == arms.SEEDS[0])].n_eval.sum())
    return res


# ---------------------------------------------------------------- attribution sensitivity (protocol 8)

def mean_logit_p(test: pd.DataFrame, arm: str) -> np.ndarray:
    """sigmoid of the seed-mean logit: with it, `bound_labels`' per-shot objective is the
    seed mean of the per-seed objectives."""
    return sigmoid(logit(test[ARM_COLS[arm]].to_numpy()).mean(1))


class Relabelling(NamedTuple):
    """Test labels under one ticket-03 rule or bound; `keep` is False for dropped shots."""

    name: str
    kind: str  # "rule" or "bound"
    goal: np.ndarray
    keep: np.ndarray


def sensitivity_labels(test: pd.DataFrame, attr) -> list[Relabelling]:
    """Every ticket-03 rule, then the per-arm bounds and the gap bounds, on the test rows;
    the first entry is the nearest rule (the index labels)."""
    out = [Relabelling(rule, "rule", *relabel(test, attr, rule)) for rule in ("nearest", "earliest", "last", "exclude")]
    keep_all = np.ones(len(test), bool)
    p = {arm: mean_logit_p(test, arm) for arm in split_plan.ARMS}
    for arm in split_plan.ARMS:
        for bound in ("worst", "best"):
            out.append(Relabelling(f"{bound} for {arm}", "bound", bound_labels(test, attr, p[arm], bound), keep_all))
    out.append(Relabelling("largest gap", "bound", bound_labels(test, attr, p["without"], "worst", p["with"]), keep_all))
    out.append(Relabelling("smallest gap", "bound", bound_labels(test, attr, p["without"], "best", p["with"]), keep_all))
    return out


def sensitivity_table(test: pd.DataFrame, labels: list[Relabelling], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Log-loss of both arms and the gap per rule or bound, pooled and per league, on the
    match draws of the main bootstrap (dropped shots leave their match's weight)."""
    p = test[P_COLS].to_numpy()
    league = test.league.to_numpy()
    near = test.goal.to_numpy().astype(bool)
    groups = league_groups(league)
    losses = [(*lab, log_losses(lab.goal.astype(float), p)) for lab in labels]
    w_pt = np.ones(len(test))
    # match multiplicity per shot and draw (small integers, int16 keeps 2 000 draws at ~35 MB)
    draws = [np.bincount(r, minlength=len(test)).astype(np.int16)
             for r in stratified_draws(test.game.to_numpy(), league, n_boot, seed)]
    rows = []
    for name, kind, goal, keep, loss in losses:
        for g, mask in groups.items():
            m = (mask & keep).astype(float)

            def ll(w, m=m, loss=loss):
                return (w * m) @ loss / (w * m).sum()
            pt = ll(w_pt)
            bt = np.array([ll(w) for w in draws])
            row = gap_row(g, pt, bt)
            rows.append({"rule": name, "kind": kind, "shots": int(m.sum()), "goals": int((goal & (m > 0)).sum()),
                         "goals_moved": int((goal & ~near & (m > 0)).sum()), **row})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- inputs

def load_test(index: pd.DataFrame, plan: pd.DataFrame, results: Path = RESULTS) -> tuple[pd.DataFrame, list[dict]]:
    """Test shots of every (league, half-set) with the calibrated prediction of each arm and
    seed (P_COLS), and one row of calibration constants per fit."""
    shots = index[["shot_id", "game", "goal", "is_penalty", "half", "position_ms"]]
    labels = shots.set_index("shot_id").goal
    cells, fits = [], []
    for league in LEAGUES:
        for h in split_plan.HALF_SETS:
            test = None
            for arm in split_plan.ARMS:
                sets = arms.arm_sets(shots, plan, arms.FitKey(league, h, arm, arms.SEEDS[0]))
                const = arms.arm_constants(sets)
                if test is None:
                    test = sets["test"].sort_values("shot_id").reset_index(drop=True)
                for seed in arms.SEEDS:
                    run = results / arms.FitKey(league, h, arm, seed).path
                    pred = {r: pd.read_csv(run / f"{r}_predictions.csv").set_index("shot_id") for r in ("calibration", "test")}
                    for r, d in pred.items():
                        if not (d.goal.astype(bool) == labels.loc[d.index]).all():
                            raise ValueError(f"{run}: {r} goal column disagrees with the index")
                    cal = pred["calibration"]
                    cal_goal = labels.loc[cal.index].to_numpy().astype(float)
                    platt = PlattScaler().fit(cal.logit.to_numpy(), cal_goal)
                    z = pred["test"].logit.loc[test.shot_id].to_numpy()
                    test[p_col(arm, seed)] = calibrate(platt, float(cal_goal.mean()), z, test.is_penalty.to_numpy(), const)
                    fits.append({"league": league, "half_set": h, "arm": arm, "seed": seed, "platt_a": platt.a,
                                 "platt_b": platt.b, "calibration_goal_rate": float(cal_goal.mean()), **const,
                                 "penalty_rate": penalty_rate(const["arm_penalty_goals"], const["arm_penalties"])})
            cells.append(test.assign(league=league, half_set=h))
    test = pd.concat(cells, ignore_index=True)
    if test.shot_id.duplicated().any():
        raise ValueError("a test shot belongs to two half-sets")
    return test, fits


def half_set_shots(test: pd.DataFrame, league: str, half_set: str) -> pd.DataFrame:
    return test[(test.league == league) & (test.half_set == half_set)].reset_index(drop=True)


def input_hashes(results: Path) -> dict[str, str]:
    """sha256 of every file the analysis reads from the fits (predictions and records)."""
    out = {}
    for key in arms.fit_grid():
        run = results / key.path
        for f in ("complete.json", "calibration_predictions.csv", "test_predictions.csv"):
            out[str((key.path / f).as_posix())] = sha256_file(run / f)
    return out


# ---------------------------------------------------------------- figures

def _label(g: str) -> str:
    return {"pooled": "łącznie (wg strzałów)", "equal_weight": "łącznie (ligi po równo)",
            "open_play": "łącznie, strzały z gry"}.get(g, NAMES.get(g, g) + (" (orientacyjnie)" if g in INDICATIVE else ""))


def plot_gaps(gaps: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    """Left: log-loss gap per group with the interval and both seeds. Right: calibration
    slope per league for both arms."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
    ys = np.arange(len(gaps))[::-1]
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    x, lo, hi = (1000 * gaps[c].to_numpy() for c in ("gap", "gap_lo", "gap_hi"))
    ax.errorbar(x, ys, xerr=[x - lo, hi - x], fmt="o", ms=7, color=INK2, elinewidth=1.5, capsize=0, mec=SURFACE, mew=1.5,
                zorder=3, label="średnia z ziaren, 95% CI")
    for s, marker in zip(arms.SEEDS, ("^", "v"), strict=True):
        ax.plot(1000 * gaps[f"gap_seed{s}"], ys, marker, ms=6, mfc=SURFACE, mec=INK2, mew=1.2, ls="none", zorder=4,
                label=f"ziarno {s}")
    ax.set_yticks(ys, [f"{_label(g)}\n{n} strz., {gl} goli" for g, n, gl in zip(gaps.group, gaps.shots, gaps.goals, strict=True)],
                  fontsize=8.5)
    ax.axhline(ys[2] - 0.5, color=GRID, lw=1)
    ax.set_xlabel("luka log-loss: bez ligi − z ligą [×10⁻³]   (> 0: transfer kosztuje)")
    ax.set_title("Luka transferu", fontsize=10.5, loc="left", color=INK, pad=24)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, labelcolor=INK2)
    _axes_style(ax)
    ax.grid(axis="y", visible=False)

    sl = metrics[(metrics.metric == "slope") & (metrics.group != "pooled")].reset_index(drop=True)
    sl = pd.concat([metrics[(metrics.metric == "slope") & (metrics.group == "pooled")], sl], ignore_index=True)
    yb = np.arange(len(sl))[::-1]
    bx.axvline(1, color=MUTED, lw=1, zorder=1)
    for off, arm in ((0.15, "without"), (-0.15, "with")):
        v, lo, hi = (sl[c].to_numpy() for c in (arm, f"{arm}_lo", f"{arm}_hi"))
        bx.errorbar(v, yb + off, xerr=[v - lo, hi - v], fmt="o", ms=6, color=ARM_COLOR[arm], elinewidth=1.5, capsize=0,
                    mec=SURFACE, mew=1.5, zorder=3, label=ARM_LABEL[arm])
    bx.set_yticks(yb, [_label(g).replace(" (wg strzałów)", "") for g in sl.group], fontsize=8.5)
    bx.set_xlabel("nachylenie kalibracji, strzały z gry (1 = skalibrowany)")
    bx.set_title("Nachylenie kalibracji na lidze testowej", fontsize=10.5, loc="left", color=INK, pad=24)
    bx.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, labelcolor=INK2)
    _axes_style(bx)
    bx.grid(axis="y", visible=False)
    fig.suptitle("Wynik rozwojowy. 95% CI: bootstrap po meczach warstwowany po lidze (2 000 losowań)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(rel: pd.DataFrame, test: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    """Reliability per league for both arms on the ECE's equal-width bins, with a
    log-scale histogram of the predictions (seeds stacked)."""
    fig, axes = plt.subplots(2, len(LEAGUES), figsize=(3.3 * len(LEAGUES), 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    for col, lg in enumerate(LEAGUES):
        ax, hx = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, zorder=1)
        for off, arm in ((-0.006, "without"), (0.006, "with")):
            r = rel[(rel.league == lg) & (rel.arm == arm)]
            ax.errorbar(r.mean_xg + off, r.goal_rate, yerr=[r.goal_rate - r.goal_rate_lo, r.goal_rate_hi - r.goal_rate],
                        fmt="o", ms=5, color=ARM_COLOR[arm], elinewidth=1.2, capsize=0, mec=SURFACE, mew=1.2, zorder=3,
                        label=ARM_LABEL[arm])
            p = test.loc[test.league == lg, ARM_COLS[arm]].to_numpy().ravel()
            hx.hist(p, bins=np.linspace(0, 1, 51), histtype="step", color=ARM_COLOR[arm], lw=1.5)
        ece = metrics[(metrics.group == lg) & (metrics.metric == "ece")].iloc[0]
        ax.text(0.03, 0.97, f"ECE bez {ece.without:.3f}\nECE z {ece['with']:.3f}", transform=ax.transAxes, va="top",
                fontsize=8, color=INK2, linespacing=1.4)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(_label(lg), fontsize=9.5, loc="left", color=INK)
        hx.set_yscale("log")
        hx.set_xlabel("przewidywane xG")
        if col == 0:
            ax.set_ylabel("odsetek goli w przedziale")
            hx.set_ylabel("strzały (log)")
            ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
        _axes_style(ax)
        _axes_style(hx)
    fig.suptitle("Kalibracja na lidze testowej: 10 równych przedziałów ECE, wszystkie strzały, oba ziarna razem, "
                 "95% CI po meczach", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_curve(curves: pd.DataFrame, out: Path) -> None:
    """Small multiples: main curve (EPL + Serie A + LaLiga), all five leagues (k <= 5),
    each league to its K. y = log-loss minus that of "with" without recalibration."""
    panels = ["main", "all_five", *LEAGUES]
    titles = {"main": "Premier League + Serie A + LaLiga", "all_five": "pięć lig (k ≤ 5)"} | {lg: _label(lg) for lg in LEAGUES}
    styles = {"without_platt": ("bez ligi + Platt", ARM_COLOR["without"], "-"),
              "without_intercept": ("bez ligi + sam wyraz wolny", INTERCEPT_COLOR, "-"),
              "with_platt": ("z ligą + Platt", ARM_COLOR["with"], "-")}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), facecolor=SURFACE)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels, strict=False):
        c = curves[curves.group == panel]
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        for v, (label, color, ls) in styles.items():
            d = c[c.variant == v].sort_values("k")
            ax.fill_between(d.k, 1000 * d.p5, 1000 * d.p95, color=color, alpha=0.1, lw=0)
            ax.plot(d.k, 1000 * d["mean"], ls, color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, label=label)
        ax.set_title(titles[panel], fontsize=9.5, loc="left", color=INK)
        ax.set_xticks(K_GRID)
        ax.set_xlim(-1, 21)
        _axes_style(ax)
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    for ax in axes[3:len(panels)]:
        ax.set_xlabel("k meczów nowej ligi do rekalibracji")
    for ax in axes[::4]:
        ax.set_ylabel("log-loss − log-loss „z ligą” [×10⁻³]")
    fig.suptitle("Krzywa rekalibracji: średnia z 200 losowań i ziaren, pasmo 5–95% losowań", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--figures", type=Path, default=FIGURES)
    a = ap.parse_args()
    for d in (a.work, a.figures):
        d.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(a.index).drop_duplicates("shot_id").reset_index(drop=True)
    index["goal"], index["is_penalty"] = index.goal.astype(bool), index.is_penalty.astype(bool)
    plan, hashes = arms.verify_plan(index)
    problems = arms.completeness(a.results, index, plan)
    if problems:
        raise SystemExit("fits incomplete or changed:\n" + "\n".join(problems))
    inputs = input_hashes(a.results)
    test, fits = load_test(index, plan, a.results)
    if set(test.game) != set(index.game[index.game.str.split("/").str[0].isin(LEAGUES)]):
        raise ValueError("the test half-sets do not cover every match of the five leagues once")
    print(f"test: {len(test)} shots, {test.game.nunique()} matches, {int(test.goal.sum())} goals")

    res = bootstrap(test)
    metrics = metric_table(res)
    gaps = gap_table(res, test)
    metrics["indicative"] = metrics.group.isin(INDICATIVE)
    rel = reliability_table(res)
    rel.insert(rel.columns.get_loc("shots") + 1, "mean_xg", bin_mean_xg(test, rel))
    head = gaps.iloc[0]
    print(gaps.round(5).to_string(index=False))

    recal_draws = pd.concat([recalibration_rows(half_set_shots(test, lg, h), lg, h)
                             for lg in LEAGUES for h in split_plan.HALF_SETS], ignore_index=True)
    recal_draws.to_csv(a.work / "recalibration_draws.csv.gz", index=False)
    caps = {lg: {k_cap(half_set_shots(test, lg, h).game.nunique()) for h in split_plan.HALF_SETS} for lg in LEAGUES}
    if any(len(c) != 1 for c in caps.values()):
        raise ValueError(f"half-sets of one league have different K: {caps}")
    caps = {lg: c.pop() for lg, c in caps.items()}
    curve_parts = [curve(recal_draws, MAIN_CURVE, K_GRID).assign(group="main"),
                   curve(recal_draws, LEAGUES, (0, 5)).assign(group="all_five")]
    curve_parts += [curve(recal_draws, (lg,), tuple(k for k in K_GRID if k <= caps[lg])).assign(group=lg) for lg in LEAGUES]
    curves = pd.concat(curve_parts, ignore_index=True)
    curves["indicative"] = curves.group.isin(INDICATIVE)
    print(curves.round(5).to_string(index=False))

    attr = attribute(source_events(test.game, LABELS_ROOT))
    labels = sensitivity_labels(test, attr)
    if labels[0].name != "nearest" or not np.array_equal(labels[0].goal, test.goal.to_numpy()):
        raise ValueError("the nearest rule does not reproduce the index labels on the test shots")
    sens = sensitivity_table(test, labels)
    near = sens[(sens.rule == "nearest")].set_index("group")
    for g in near.index:
        ref = gaps.set_index("group").loc[g]
        if not np.allclose(near.loc[g, ["gap", "gap_lo", "gap_hi"]].astype(float), ref[["gap", "gap_lo", "gap_hi"]].astype(float),
                           atol=1e-12):
            raise ValueError(f"nearest sensitivity row does not reproduce the gap of {g}")
    print(sens[sens.group == "pooled"].round(5).to_string(index=False))

    tables = {"gaps.csv": gaps, "metrics.csv": metrics, "recalibration_curve.csv": curves,
              "attribution_sensitivity.csv": sens, "reliability_bins.csv": rel, "calibration_fits.csv": pd.DataFrame(fits)}
    for f, t in tables.items():
        t.to_csv(a.figures / f, index=False)
    summary = {"development_result": True, "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED, "recalibration_draws": N_RECAL_DRAWS,
               "k_cap": caps, "test_shots": len(test), "test_matches": int(test.game.nunique()),
               "test_goals": int(test.goal.sum()), "indicative": sorted(INDICATIVE),
               "headline": {k: (v.item() if hasattr(v, "item") else v) for k, v in head.items()},
               "noise_reference": NOISE_REFERENCE,
               "recalibration_fallbacks": {f"{lg}/{h}/k{k}": int(n) for (lg, h, k), n in
                                           recal_draws[recal_draws.seed == arms.SEEDS[0]].groupby(["league", "half_set", "k"]).fallback.sum()
                                           .items() if n > 0}}
    (a.figures / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_gaps(gaps, metrics, a.figures / "transfer_gap.png")
    plot_reliability(rel, test, metrics, a.figures / "reliability_by_league.png")
    plot_curve(curves, a.figures / "recalibration_curve.png")

    outputs = {f.name: sha256_file(f) for f in sorted(a.figures.iterdir())
               if f.name not in ("complete.json", "REPORT.md")}
    record = {"ticket": "xg-credibility 13", "protocol_sha256": hashes["protocol_sha256"], "plan_sha256": hashes["plan_sha256"],
              "index_sha256": sha256_file(a.index), "split_plan_code": hashes["split_plan_code"],
              "sources": {s: sha256_file(EXP / s) for s in SOURCES}, "commit": code_version(EXP),
              "inputs": inputs, "inputs_digest": canonical_digest(inputs),
              "labels_digest": canonical_digest({g: sha256_file(LABELS_ROOT / g / "Labels-v2.json")
                                                 for g in sorted(test.game.unique())}),
              "recalibration_draws_sha256": sha256_file(a.work / "recalibration_draws.csv.gz"), "files": outputs}
    (a.figures / "complete.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(summary["headline"], indent=2, default=float))


if __name__ == "__main__":
    main()
