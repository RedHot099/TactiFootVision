"""Does the final stack earn its complexity over the fine-tuned DINOv2 (G-combo) alone?

    ./run.sh python scripts/report_complexity.py --out results/v16_complexity \
        --figures results_reports/ensemble_vs_g_2026-10

Answers the external reviewer (2026-10-02) on the v1.2 test set without training. Every
G-combo configuration is rebuilt from the fold predictions in results/v12_cv and gets the
deployed post-processing: Platt on its own out-of-fold predictions, prior shift to the
pooled train+valid rate, penalty routing. Configurations (ViT-L passes per shot):

  g1   one fold model (fold k, seed s), Platt on its own held-out fold    10 instances
  g2   one fold, both seeds, Platt on that fold                            5 instances
  g5   five folds, one seed, Platt on that seed's OOF (400 games)          2 instances
  g10  five folds x two seeds = deployed G-combo                           1
  ens  final stack C-combo + G-combo + G-combo-mix + GBM                    1

What is computed, all on one set of game-clustered bootstrap draws (2 000, the draws of
significance.py), so every difference is paired:
  * ens vs g10 on all 2 229 test shots (penalties routed in both): AUC, log-loss, Brier,
    and the log-loss / Brier reduction as a share of g10's skill over a constant forecast,
    the ratio taken inside every draw;
  * an equivalence curve for the log-loss and Brier reductions: two one-sided tests at 5%
    each over margins 0 to 0.005, read off the bootstrap distribution, and the smallest
    margin at which equivalence holds;
  * every g instance against g10 on its own, and each rung as the mean of its saved
    instances (conditional on those instances, not a statement about retraining);
  * calibration on open-play shots (penalties get a constant rate): mean xG - goal rate,
    recalibration intercept and slope, ECE on the reliability plot's equal-count bins (and
    on equal-width bins for comparison), the paired ens - g10 differences of all of them;
  * a split-half recalibration diagnostic: refit on half of the test games, score on the
    other half, over random halves;
  * exploratory subgroups (distance band, venue) with AUC next to the calibration slope.

History: the ens vs g10 comparison and its intervals were known since 2026-09-03
(results/v12_report/significance_final). The decision rule written here on 2026-10-02 came
after that, so it is not pre-registered; the equivalence curve replaces any practical
threshold. The test set has been looked at in earlier rounds (research/audit_2026-09-07.md):
every number here is a development result.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from eval_subgroups import DIST_BINS, DIST_LABELS, TAB
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgpoc.eval.calibration import (
    PlattScaler,
    logit,
    penalty_rate,
    prior_shift,
    route_penalties,
    sigmoid,
)
from xgpoc.eval.metrics import expected_calibration_error
from xgpoc.paths import INDEX

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CV = Path("results/v12_cv/g_combo")
MODEL = "g_finetune_dinov2_l_b4_overlay_combo"
ENS = Path("results/v12_cv_ensemble/best_ensemble_test_predictions.csv")
G10_REF = Path("results/v12_cv_ensemble/g_combo_test_predictions.csv")
SIGNIFICANCE = Path("results/v12_report/significance_final/paired_tests.csv")
HOME_RE = r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ "
N_REL_BINS = 10
MARGINS = np.linspace(0, 0.005, 51)  # equivalence grid, step 1e-4
ALPHA = 0.05                         # each one-sided test
ISO_CLIP = 1e-3                      # isotonic output floor / ceiling, so a held-out goal in a 0-rate bin stays finite
SCORES = ("auc", "log_loss", "brier")
LOSSES = SCORES[1:]                  # the order of constant_scores
CALIB = ("cal_large", "intercept", "slope", "ece", "ece_width")
SUB = ("cal_large", "auc", "log_loss", "slope")

# dataviz reference palette, slots 1-3 (validated all-pairs, light surface)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"g1": "#eb6834", "g10": "#2a78d6", "ens": "#1baf7a"}


# ---------------------------------------------------------------- configurations

def load_runs(root: Path, model: str = MODEL) -> dict[tuple[int, int], dict[str, pd.DataFrame]]:
    """(fold, seed) -> {'valid': OOF frame, 'test': test frame}, both shot_id / z (logit)."""
    runs = {}
    for f in sorted(root.glob(f"fold*/seed*/{model}_test_predictions.csv")):
        fold, seed = (int(re.search(rf"{k}(\d+)", f.as_posix()).group(1)) for k in ("fold", "seed"))
        runs[fold, seed] = {split: pd.read_csv(f.parent / f"{model}_{split}_predictions.csv").drop_duplicates("shot_id")
                            .assign(z=lambda d: logit(d.p.to_numpy()))[["shot_id", "z"]]
                            for split in ("valid", "test")}
    return runs


def mean_logit(frames: list[pd.DataFrame]) -> pd.Series:
    """Logit average per shot. OOF frames of different folds are disjoint, so this both
    averages seeds of one fold and concatenates folds (as build_ensemble.collect does)."""
    return pd.concat(frames).groupby("shot_id").z.mean()


def deploy(runs: dict, keys: list[tuple[int, int]], labels: pd.Series, is_pen: pd.Series,
           pooled_rate: float, p_pen: float) -> pd.Series:
    """Calibrated test probabilities of the configuration made of the fold models `keys`."""
    oof = mean_logit([runs[k]["valid"] for k in keys])
    test = mean_logit([runs[k]["test"] for k in keys])
    y = labels.loc[oof.index].to_numpy()
    p = PlattScaler().fit(oof.to_numpy(), y)(test.to_numpy())
    p = prior_shift(p, float(y.mean()), pooled_rate)
    return pd.Series(route_penalties(p, is_pen.loc[test.index].to_numpy(), p_pen), index=test.index)


def configurations(runs: dict) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
    folds, seeds = sorted({f for f, _ in runs}), sorted({s for _, s in runs})
    return {"g1": [(f"f{f}s{s}", [(f, s)]) for f in folds for s in seeds],
            "g2": [(f"f{f}", [(f, s) for s in seeds]) for f in folds],
            "g5": [(f"s{s}", [(f, s) for f in folds]) for s in seeds],
            "g10": [("all", [(f, s) for f in folds for s in seeds])]}


# ---------------------------------------------------------------- statistics

def recalibration(y: np.ndarray, p: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """(intercept, slope) of the logistic recalibration y ~ a + b * logit(p). A calibrated
    model has b = 1, a = 0; b < 1 means over-confident (spread too wide), b > 1 too timid."""
    z = logit(p)
    x = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        q = sigmoid(x @ beta)
        step = np.linalg.solve(x.T @ (x * (q * (1 - q))[:, None]), x.T @ (y - q))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """auc, log_loss, brier - the order of SCORES."""
    return np.array([roc_auc_score(y, p), log_loss(y, p), np.mean((p - y) ** 2)])


def constant_scores(y: np.ndarray) -> np.ndarray:
    """log-loss and Brier of the constant forecast at the sample's own goal rate."""
    return np.array([log_loss(y, np.full(len(y), y.mean())), y.mean() * (1 - y.mean())])


def rel_bins(p: np.ndarray, n_bins: int = N_REL_BINS) -> np.ndarray:
    """Equal-count bin of every prediction (ties can merge bins)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)


def ece_equal_count(y: np.ndarray, p: np.ndarray, n_bins: int = N_REL_BINS) -> float:
    """ECE on the equal-count bins the reliability plot uses."""
    b = rel_bins(p, n_bins)
    n = np.bincount(b)
    gap = np.bincount(b, weights=y) - np.bincount(b, weights=p)  # per bin: n * (goal rate - mean xG)
    return float(np.abs(gap[n > 0]).sum() / len(y))


def calibration(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The quantities of CALIB: mean xG - goal rate, recalibration intercept and slope,
    ECE on equal-count bins, ECE on equal-width bins."""
    return np.array([p.mean() - y.mean(), *recalibration(y, p), ece_equal_count(y, p),
                     expected_calibration_error(y, p)])


def game_draws(games: np.ndarray, n_boot: int, seed: int = 0):
    """Row indices of game-clustered resamples; same draw sequence as significance.py."""
    keys, inv = np.unique(games, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([rows[k] for k in rng.choice(len(keys), len(keys), replace=True)])


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def equivalence_curve(draws: np.ndarray, margins: np.ndarray = MARGINS, alpha: float = ALPHA) -> pd.DataFrame:
    """Two one-sided tests per margin d on bootstrap draws of a paired difference D.
    H0-: D <= -d is rejected when at most alpha of the draws lie at or below -d, H0+:
    D >= d when at most alpha lie at or above d; equivalence at d needs both, i.e. the
    90% percentile interval inside (-d, d). The p-values only fall as d grows."""
    draws = np.asarray(draws, float)[:, None]
    if np.isnan(draws).any():
        raise ValueError("equivalence_curve: NaN bootstrap draws would count as inside every margin")
    p_lower = (draws <= -margins).mean(0)
    p_upper = (draws >= margins).mean(0)
    p_tost = np.maximum(p_lower, p_upper)
    return pd.DataFrame({"margin": margins, "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
                         "equivalent": p_tost <= alpha})


def smallest_margin(curve: pd.DataFrame) -> float:
    """Smallest grid margin with equivalence; NaN when the grid never reaches it."""
    hit = curve.margin[curve.equivalent]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def recalibration_gains(y_fit: np.ndarray, p_fit: np.ndarray, y_ev: np.ndarray, p_ev: np.ndarray) -> dict[str, float]:
    """Log-loss on the evaluation shots and its drop after a recalibration fitted on the fit
    shots. Logistic = the two-parameter `recalibration`; isotonic = monotone, outputs
    clipped to ISO_CLIP."""
    a, b = recalibration(y_fit, p_fit)
    iso = IsotonicRegression(y_min=ISO_CLIP, y_max=1 - ISO_CLIP, out_of_bounds="clip").fit(p_fit, y_fit)
    before = log_loss(y_ev, p_ev)
    return {"log_loss": before, "gain_logistic": before - log_loss(y_ev, sigmoid(a + b * logit(p_ev))),
            "gain_isotonic": before - log_loss(y_ev, iso.predict(p_ev))}


def game_halves(games: np.ndarray, n_splits: int, seed: int = 0):
    """(split, fit rows, evaluation rows): `n_splits` random halves of the games, each used
    in both directions; a game is never on both sides."""
    keys = np.unique(games)
    rng = np.random.default_rng(seed)
    for split in range(n_splits):
        in_a = np.isin(games, rng.permutation(keys)[: len(keys) // 2])
        yield split, in_a, ~in_a
        yield split, ~in_a, in_a


def split_half_recalibration(y: np.ndarray, p: np.ndarray, games: np.ndarray, n_splits: int,
                             seed: int = 0) -> pd.DataFrame:
    """recalibration_gains fitted on one half of the games and scored on the other."""
    return pd.DataFrame([{"split": split} | recalibration_gains(y[fit], p[fit], y[ev], p[ev])
                         for split, fit, ev in game_halves(games, n_splits, seed)])


def shot_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-shot log-loss and Brier, columns in LOSSES order; their means are `scores`[1:]."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack([-(y * np.log(q) + (1 - y) * np.log(1 - q)), (p - y) ** 2])


def reduction_draws(loss_a: np.ndarray, loss_b: np.ndarray, games: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap draws of the mean loss reduction loss_b - loss_a (rows: draws, columns: LOSSES)."""
    diff = loss_b - loss_a
    return np.array([diff[r].mean(0) for r in game_draws(games, n_boot, seed)])


def margin_stability(loss_a: np.ndarray, loss_b: np.ndarray, games: np.ndarray, n_boot: int,
                     seeds: range, big_n_boot: int) -> pd.DataFrame:
    """Smallest equivalence margin per LOSSES metric under other bootstrap seeds and under
    `big_n_boot` draws: how much of the grid answer is Monte Carlo noise."""
    rows = [(f"seed {s}", n_boot, reduction_draws(loss_a, loss_b, games, n_boot, s)) for s in seeds]
    rows.append(("seed 0", big_n_boot, reduction_draws(loss_a, loss_b, games, big_n_boot, 0)))
    return pd.DataFrame([{"draws": label, "n_boot": n} | {m: smallest_margin(equivalence_curve(d[:, k]))
                                                          for k, m in enumerate(LOSSES)} for label, n, d in rows])


# ---------------------------------------------------------------- context

def context(index: pd.DataFrame, tab_path: str) -> pd.DataFrame:
    """Distance band and venue per test shot. Venue counts the whole CV pool (train+valid)
    as seen, which is what the CV models trained on (audit 2026-09-07)."""
    home = index.game.str.extract(HOME_RE)[0]
    seen = set(home[index.split != "test"].dropna())
    df = index[index.split == "test"].copy()
    df["venue"] = np.where(home[df.index].isin(seen), "gospodarz znany", "gospodarz nieznany")
    tab = pd.read_parquet(tab_path).drop_duplicates("shot_id")[["shot_id", "dist_to_goal", "homography_ok", "has_ball"]]
    tab["distance"] = pd.cut(tab.dist_to_goal, DIST_BINS, labels=DIST_LABELS).astype(object)
    tab.loc[(tab.homography_ok < 0.5) | (tab.has_ball < 0.5), "distance"] = "geometria niedostępna"
    df = df.merge(tab[["shot_id", "distance"]], on="shot_id", how="left")
    df["distance"] = df.distance.fillna("geometria niedostępna")
    return df


# ---------------------------------------------------------------- analysis

def analyse(preds: dict[str, pd.Series], ctx: pd.DataFrame, rel_variants: list[str], n_boot: int) -> dict:
    """preds: variant -> p per shot_id. One set of game draws serves every quantity, so all
    differences are paired. Per draw: SCORES of every variant on all shots, the constant
    forecast, CALIB on open play, reliability-bin goal rates, SUB per subgroup for g10/ens."""
    ids = ctx.shot_id.to_numpy()
    y = ctx.goal.to_numpy().astype(float)
    open_play = ~ctx.is_penalty.to_numpy()
    p = {k: v.loc[ids].to_numpy() for k, v in preds.items()}
    names = list(p)
    bins = {k: rel_bins(p[k][open_play]) for k in rel_variants}
    rank = np.cumsum(open_play) - 1  # row -> position among open-play shots, where the bins live
    sub = {(dim, g): ctx[dim].to_numpy() == g for dim in ("distance", "venue") for g in sorted(ctx[dim].unique())}

    def one(rows: np.ndarray) -> dict:
        yy, op = y[rows], open_play[rows]
        out = {"s": np.array([scores(yy, p[k][rows]) for k in names]),
               "base": constant_scores(yy),
               "c": np.array([calibration(yy[op], p[k][rows][op]) for k in names])}
        ro = rows[op]
        for k, b in bins.items():
            bb = b[rank[ro]]
            n = np.bincount(bb, minlength=b.max() + 1)
            out[f"rel_{k}"] = np.bincount(bb, weights=y[ro], minlength=b.max() + 1) / np.maximum(n, 1)
        for key, m in sub.items():
            mm = m[rows] & op
            ys = yy[mm]
            both_classes = mm.any() and ys.min() < ys.max()
            out[key] = (np.array([[p[k][rows][mm].mean() - ys.mean(), *scores(ys, p[k][rows][mm])[:2],
                                   recalibration(ys, p[k][rows][mm])[1]] for k in ("g10", "ens")])
                        if both_classes else np.full((2, len(SUB)), np.nan))
        return out

    point = one(np.arange(len(y)))
    boot = [one(r) for r in game_draws(ctx.game.to_numpy(), n_boot)]
    return {"names": names, "ix": {n: i for i, n in enumerate(names)}, "point": point, "boot": boot, "bins": bins, "sub": sub, "y": y, "open_play": open_play, "p": p}


def stacked(res: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(point, draws) of one per-draw quantity."""
    return res["point"][key], np.array([b[key] for b in res["boot"]])


def diff_draws(res: dict, a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Paired difference in SCORES of a against b, each the mean over its instances. AUC
    as a - b, log-loss and Brier as the reduction loss(b) - loss(a): positive = a better."""
    ix = res["ix"]
    sign = np.array([1, -1, -1])
    s_pt, s_bt = stacked(res, "s")
    ia, ib = [ix[n] for n in a], [ix[n] for n in b]
    return (sign * (s_pt[ia].mean(0) - s_pt[ib].mean(0)),
            sign * (s_bt[:, ia].mean(1) - s_bt[:, ib].mean(1)))


def diff_row(res: dict, a: list[str], b: list[str]) -> dict:
    """diff_draws summarised, plus the loss reductions as a share of g10's skill over the
    constant forecast, the ratio taken inside every draw."""
    ix = res["ix"]
    d_pt, d_bt = diff_draws(res, a, b)
    s_pt, s_bt = stacked(res, "s")
    base_pt, base_bt = stacked(res, "base")
    skill_pt, skill_bt = base_pt - s_pt[ix["g10"], 1:], base_bt - s_bt[:, ix["g10"], 1:]
    row = {"a": "+".join(a) if len(a) < 3 else f"mean of {len(a)}", "b": "+".join(b) if len(b) < 3 else f"mean of {len(b)}",
           "n_a": len(a), "n_b": len(b)}
    for j, m in enumerate(SCORES):
        lo, hi = interval(d_bt[:, j])
        row |= {f"d_{m}": d_pt[j], f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"p_{m}_a_better": float((d_bt[:, j] > 0).mean())}
    for j, m in enumerate(LOSSES):
        share = d_bt[:, j + 1] / skill_bt[:, j]
        lo, hi = interval(share)
        row |= {f"share_{m}": d_pt[j + 1] / skill_pt[j], f"share_{m}_lo": lo, f"share_{m}_hi": hi}
    return row


def calibration_row(res: dict, names: list[str]) -> dict:
    """CALIB of a variant (mean over instances) with intervals; ECE intervals are not
    reported because binned ECE is biased upwards in every resample (see calibration_diff)."""
    ix = res["ix"]
    sel = [ix[n] for n in names]
    c_pt, c_bt = stacked(res, "c")
    pt, bt = c_pt[sel].mean(0), c_bt[:, sel].mean(1)
    row = {}
    for j, m in enumerate(CALIB):
        row[m] = pt[j]
        if not m.startswith("ece"):
            row[f"{m}_lo"], row[f"{m}_hi"] = interval(bt[:, j])
    return row


def calibration_diff(res: dict, a: str, b: str) -> pd.DataFrame:
    """Paired CALIB difference a - b with intervals (both ECEs share the upward bias)."""
    ix = res["ix"]
    c_pt, c_bt = stacked(res, "c")
    d_pt, d_bt = c_pt[ix[a]] - c_pt[ix[b]], c_bt[:, ix[a]] - c_bt[:, ix[b]]
    return pd.DataFrame([{"quantity": m, "a": a, "b": b, "diff": d_pt[j], "lo": interval(d_bt[:, j])[0],
                          "hi": interval(d_bt[:, j])[1]} for j, m in enumerate(CALIB)])


def subgroup_table(res: dict) -> pd.DataFrame:
    """Per subgroup (open play): SUB of g10 and ens with intervals, and paired contrasts:
    log-loss reduction by ens, signed and absolute mean-error difference ens - g10."""
    rows = []
    j = {m: i for i, m in enumerate(SUB)}
    for (dim, g), m in res["sub"].items():
        mm = m & res["open_play"]
        pt = res["point"][(dim, g)]
        bt = np.array([b[(dim, g)] for b in res["boot"]])
        bt = bt[~np.isnan(bt).any(axis=(1, 2))]
        row = {"dimension": dim, "group": g, "n": int(mm.sum()), "goals": int(res["y"][mm].sum()),
               "goal_rate": float(res["y"][mm].mean()), "valid_draws": len(bt)}
        for i, k in enumerate(("g10", "ens")):
            row[f"{k}_mean_xg"] = float(res["p"][k][mm].mean())
            for q in SUB:
                row[f"{k}_{q}"] = pt[i, j[q]]
                if q != "log_loss":
                    row[f"{k}_{q}_lo"], row[f"{k}_{q}_hi"] = interval(bt[:, i, j[q]])
        cl, ll = j["cal_large"], j["log_loss"]
        for name, d_pt, d_bt in (
                ("d_log_loss_ens", pt[0, ll] - pt[1, ll], bt[:, 0, ll] - bt[:, 1, ll]),
                ("d_cal_large_ens_minus_g10", pt[1, cl] - pt[0, cl], bt[:, 1, cl] - bt[:, 0, cl]),
                ("d_abs_cal_large_ens_minus_g10", abs(pt[1, cl]) - abs(pt[0, cl]),
                 np.abs(bt[:, 1, cl]) - np.abs(bt[:, 0, cl]))):
            row |= {name: d_pt, f"{name}_lo": interval(d_bt)[0], f"{name}_hi": interval(d_bt)[1]}
        rows.append(row)
    order = {g: i for i, g in enumerate([*DIST_LABELS, "geometria niedostępna", "gospodarz znany", "gospodarz nieznany"])}
    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.map(order)).reset_index(drop=True)


def reliability_table(res: dict) -> pd.DataFrame:
    rows = []
    y_op = res["y"][res["open_play"]]
    for k, b in res["bins"].items():
        p_op = res["p"][k][res["open_play"]]
        bt = np.array([x[f"rel_{k}"] for x in res["boot"]])
        for j in range(b.max() + 1):
            m = b == j
            lo, hi = interval(bt[:, j])
            rows.append({"variant": k, "bin": j, "n": int(m.sum()), "p_min": float(p_op[m].min()),
                         "p_max": float(p_op[m].max()), "mean_xg": float(p_op[m].mean()),
                         "goal_rate": float(y_op[m].mean()), "goal_rate_lo": lo, "goal_rate_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def _axes_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_reliability(rel: pd.DataFrame, cal: pd.DataFrame, res: dict, titles: dict[str, str], out: Path) -> None:
    """Axes cover every open-play prediction of the shown variants; histogram on a log
    scale so the few high-xG shots stay visible."""
    keys = list(titles)
    lim = np.ceil(10 * max(res["p"][k][res["open_play"]].max() for k in keys)) / 10
    fig, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 5.9), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1]}, facecolor=SURFACE)
    for col, k in enumerate(keys):
        ax, hx = axes[0, col], axes[1, col]
        color = SERIES[k.split("_")[0]]
        r = rel[rel.variant == k]
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, zorder=1)
        ax.errorbar(r.mean_xg, r.goal_rate, yerr=[r.goal_rate - r.goal_rate_lo, r.goal_rate_hi - r.goal_rate],
                    fmt="o", ms=6, color=color, ecolor=color, elinewidth=1.5, capsize=0,
                    mec=SURFACE, mew=1.5, zorder=3)
        c = cal.loc[k]
        ax.text(0.03, 0.97, f"nachylenie {c.slope:.2f} [{c.slope_lo:.2f}; {c.slope_hi:.2f}]\n"
                            f"śr. xG − częstość {100 * c.cal_large:+.1f} p.p. [{100 * c.cal_large_lo:+.1f}; {100 * c.cal_large_hi:+.1f}]\n"
                            f"ECE {c.ece:.3f} (te same kubełki)",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK2, linespacing=1.5)
        ax.set(xlim=(0, lim), ylim=(0, lim), title=titles[k])
        if col == 0:
            ax.set_ylabel("odsetek goli w kubełku")
        _axes_style(ax)
        hx.hist(res["p"][k][res["open_play"]], bins=np.linspace(0, lim, int(round(lim * 50)) + 1), color=color,
                alpha=0.9, edgecolor=SURFACE, lw=1)
        hx.set_yscale("log")
        hx.set_xlabel("przewidywane xG")
        if col == 0:
            hx.set_ylabel("strzały (skala log)")
        _axes_style(hx)
    fig.suptitle("Kalibracja na teście (strzały z gry, bez karnych; 10 kubełków o równej liczności, "
                 "95% CI z bootstrapu po meczach)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_subgroups(sub: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    ys = np.arange(len(sub))[::-1]
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for off, k, label in ((0.17, "g10", "G-combo (10 modeli)"), (-0.17, "ens", "ensemble")):
        x = 100 * sub[f"{k}_cal_large"]
        ax.errorbar(x, ys + off, xerr=[x - 100 * sub[f"{k}_cal_large_lo"], 100 * sub[f"{k}_cal_large_hi"] - x],
                    fmt="o", ms=6, color=SERIES[k], elinewidth=1.5, capsize=0, mec=SURFACE, mew=1.5, label=label, zorder=3)
    ax.set_yticks(ys, [f"{g} (n={n}, gole {gl})" for g, n, gl in zip(sub.group, sub.n, sub.goals, strict=True)])
    ax.set_xlabel("średnie xG − odsetek goli [p.p.]   (< 0: model zaniża)")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, labelcolor=INK2)
    ax.set_title("Kalibracja w podgrupach (strzały z gry, 95% CI po meczach)", fontsize=10.5, loc="left", pad=26)
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(cost: pd.DataFrame, conf: dict, batch: int, out: Path) -> None:
    """cost: rung, d_<m> = loss(variant) - loss(g10) with _lo/_hi. Filled: every model runs
    the full ViT-L; hollow: the 20 frozen blocks run once per frame."""
    labels = {"g1": "1 model", "g2": "2 modele", "g5": "5 modeli", "g10": "G-combo (10)", "ens": "ensemble"}
    nudge = {"g1": (8, 4), "g2": (8, 4), "g5": (8, 4), "g10": (8, -13), "ens": (8, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    for ax, (m, title) in zip(axes, (("log_loss", "log-loss"), ("brier", "Brier")), strict=True):
        naive = np.array([conf[r]["naive_ms"] for r in cost.rung])
        shared = np.array([conf[r]["shared_ms"] for r in cost.rung])
        y, lo, hi = (1000 * cost[f"d_{m}{s}"].to_numpy() for s in ("", "_lo", "_hi"))
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        for xs, xn, yi in zip(shared, naive, y, strict=True):
            ax.plot([xs, xn], [yi, yi], color=GRID, lw=2, zorder=2)
        ax.errorbar(naive, y, yerr=[y - lo, hi - y], fmt="o", ms=7, color=SERIES["g10"], elinewidth=1.5, capsize=0,
                    mec=SURFACE, mew=1.5, zorder=3, label="każdy model liczy cały ViT-L")
        ax.plot(shared[1:], y[1:], "o", ms=7, mfc=SURFACE, mec=SERIES["g10"], mew=1.5, zorder=3, ls="none",
                label="wspólne 20 zamrożonych bloków")
        for xi, yi, r in zip(naive, y, cost.rung, strict=True):
            dx, dy = nudge[r]
            ax.annotate(labels[r], (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2,
                        ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
        ax.set(xlabel=f"ms na strzał, przepustowość przy batchu {batch} (szacunek, fp32+TF32)",
               ylabel=f"{title}: wariant − G-combo [×10⁻³]")
        ax.set_title(f"{title} względem G-combo (niżej = lepiej)", fontsize=10.5, loc="left")
        _axes_style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_equivalence(curves: dict[str, pd.DataFrame], refs: dict[str, dict[str, float]], out: Path) -> None:
    """TOST p-value against the margin, one panel per metric; the top axis restates the
    margin as a share of G-combo's skill over the constant forecast."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for ax, (m, title) in zip(axes, (("log_loss", "log-loss"), ("brier", "Brier")), strict=True):
        cur, ref = curves[m], refs[m]
        x = 1000 * cur.margin
        ax.axhline(ALPHA, color=MUTED, lw=1, zorder=1)
        ax.text(x.max(), ALPHA, " α = 0.05", color=INK2, fontsize=8.5, va="bottom", ha="right")
        ax.plot(x, cur.p_tost, color=SERIES["ens"], lw=2, zorder=3, drawstyle="steps-post")
        d = 1000 * ref["smallest_margin"]
        lo, hi = (1000 * v for v in ref["smallest_margin_range_over_seeds"])
        seeds = f"\n{lo:.1f}–{hi:.1f} przy innych seedach" if hi > lo else ""
        ax.axvline(d, color=INK2, lw=1, zorder=2)
        ax.annotate(f"najmniejszy margines\n{d:.1f}·10⁻³ (seed 0){seeds}", (d, 0.55), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
        g5 = 1000 * ref["g5_seed_gap"]
        ax.axvline(g5, color=MUTED, lw=1, ls="--", zorder=2)
        ax.annotate(f"|G5 seed 0 − seed 1|\n{g5:.2f}·10⁻³", (g5, 0.85), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
        ax.set(xlim=(0, x.max()), ylim=(0, 1.02), xlabel=f"margines równoważności δ dla {title} [×10⁻³]",
               ylabel="p TOST (max z dwóch testów)")
        ax.set_title(f"{title}: ensemble vs G-combo", fontsize=10.5, loc="left", pad=24)
        _axes_style(ax)
        top = ax.secondary_xaxis("top", functions=(lambda v, s=ref["skill"]: 0.1 * v / s,
                                                   lambda v, s=ref["skill"]: 10 * v * s))
        top.set_xlabel("% przewagi G-combo nad stałą prognozą", color=INK2, fontsize=8.5)
        top.tick_params(colors=INK2, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/v16_complexity"))
    ap.add_argument("--figures", type=Path, default=Path("results_reports/ensemble_vs_g_2026-10"))
    ap.add_argument("--timing", type=Path, default=Path("results/v16_complexity/timing.json"))
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--big-n-boot", type=int, default=20000, help="draws for the Monte Carlo check of the equivalence margin")
    ap.add_argument("--n-splits", type=int, default=1000, help="random game halves for the split-half recalibration")
    a = ap.parse_args()
    for d in (a.out, a.figures, a.out / "predictions"):
        d.mkdir(parents=True, exist_ok=True)

    idx = pd.read_parquet(a.index).drop_duplicates("shot_id")
    idx["goal"], idx["is_penalty"] = idx.goal.astype(bool), idx.is_penalty.astype(bool)
    labels, is_pen = idx.set_index("shot_id").goal, idx.set_index("shot_id").is_penalty
    pooled_rate = float(idx[idx.split != "test"].goal.mean())
    pen_train = idx[(idx.split == "train") & idx.is_penalty]
    p_pen = penalty_rate(int(pen_train.goal.sum()), len(pen_train))

    runs = load_runs(CV)
    preds, groups = {}, {}
    for rung, members in configurations(runs).items():
        groups[rung] = []
        for name, keys in members:
            v = rung if rung == "g10" else f"{rung}_{name}"
            preds[v] = deploy(runs, keys, labels, is_pen, pooled_rate, p_pen)
            groups[rung].append(v)
    preds["ens"] = pd.read_csv(ENS).drop_duplicates("shot_id").set_index("shot_id").p
    ref = pd.read_csv(G10_REF).set_index("shot_id").p
    repro = float((preds["g10"].loc[ref.index] - ref).abs().max())
    assert repro < 1e-9, f"g10 does not reproduce {G10_REF}: max |dp| {repro}"
    for v, s in preds.items():
        s.rename("p").reset_index().to_csv(a.out / "predictions" / f"{v}.csv", index=False)

    ctx = context(idx, TAB).sort_values("shot_id").reset_index(drop=True)
    assert set(ctx.shot_id) == set(preds["ens"].index) == set(preds["g10"].index)
    g1_shown = groups["g1"][0]
    res = analyse(preds, ctx, ["g10", "ens", g1_shown], a.n_boot)
    y, op = res["y"], res["open_play"]

    # ensemble vs G-combo, equivalence curve and its reference points
    headline = pd.DataFrame([diff_row(res, ["ens"], ["g10"])])
    d_pt, d_bt = diff_draws(res, ["ens"], ["g10"])
    s_pt = res["point"]["s"]
    ix = res["ix"]
    g5_pt, g5_bt = diff_draws(res, ["g5_s0"], ["g5_s1"])
    curves, refs = {}, {}
    for k, m in enumerate(LOSSES):  # SCORES column k + 1, `base` column k
        j = k + 1
        curves[m] = equivalence_curve(d_bt[:, j]).assign(metric=m)
        refs[m] = {"smallest_margin": smallest_margin(curves[m]), "reduction": float(d_pt[j]),
                   "ci90": [float(np.percentile(d_bt[:, j], 5)), float(np.percentile(d_bt[:, j], 95))],
                   "g5_seed_gap": float(abs(g5_pt[j])), "g5_seed_gap_ci95": list(interval(g5_bt[:, j])),
                   "skill": float(res["point"]["base"][k] - s_pt[ix["g10"], j]),
                   "constant": float(res["point"]["base"][k])}
    games = ctx.game.to_numpy()
    loss_ens, loss_g10 = shot_losses(y, res["p"]["ens"]), shot_losses(y, res["p"]["g10"])
    assert np.allclose(reduction_draws(loss_ens, loss_g10, games, a.n_boot, 0), d_bt[:, 1:], atol=1e-12)
    stability = margin_stability(loss_ens, loss_g10, games, a.n_boot, range(10), a.big_n_boot)
    for m, r in refs.items():  # the report quotes the margin from the most draws, plus the seed range
        r["smallest_margin_range_over_seeds"] = [float(stability[m].min()), float(stability[m].max())]
        r["smallest_margin_most_draws"] = float(stability[m].iloc[-1])
        r["smallest_margin_most_draws_share_of_skill"] = r["smallest_margin_most_draws"] / r["skill"]
    if a.n_boot == 2000 and SIGNIFICANCE.exists():  # the report says these are the intervals of significance.py
        sig = pd.read_csv(SIGNIFICANCE).set_index(["model_a", "model_b"]).loc[("FINAL", "G_combo")]
        for m in SCORES:
            for end in ("lo", "hi"):
                assert abs(sig[f"d_{m}_{end}"] - headline[f"d_{m}_{end}"].iloc[0]) < 1e-9, (m, end)

    # every instance and every rung (mean of saved instances) against g10; positive = g10 better
    inst = pd.DataFrame([{"rung": rung, "variant": v} | diff_row(res, ["g10"], [v])
                         for rung in ("g1", "g2", "g5") for v in groups[rung]])
    ladder = pd.DataFrame([{"rung": rung} | diff_row(res, ["g10"], groups[rung]) for rung in ("g1", "g2", "g5")])

    # calibration: shown variants, every instance, rungs, paired ens - g10
    cal = pd.DataFrame({v: calibration_row(res, [v]) for v in res["names"]}).T.rename_axis("variant")
    cal_rungs = pd.DataFrame({r: calibration_row(res, g) for r, g in groups.items()}
                             | {"ens": calibration_row(res, ["ens"])}).T.rename_axis("rung")
    cal_diff = calibration_diff(res, "ens", "g10")
    sub = subgroup_table(res)
    rel = reliability_table(res)

    # why the slope: OOF (where Platt is fitted) against the test, and what recalibration recovers
    oof = mean_logit([runs[k]["valid"] for k in runs])
    oof_y = labels.loc[oof.index].to_numpy().astype(float)
    oof_op = ~is_pen.loc[oof.index].to_numpy()
    platt = PlattScaler().fit(oof.to_numpy(), oof_y)
    p_oof = platt(oof.to_numpy())
    oof_diag = {"platt_coefficient": platt.a, "platt_intercept": platt.b,
                "post_platt_slope_all": recalibration(oof_y, p_oof)[1],
                "post_platt_slope_open_play": recalibration(oof_y[oof_op], p_oof[oof_op])[1],
                "auc_all": float(roc_auc_score(oof_y, oof)), "auc_open_play": float(roc_auc_score(oof_y[oof_op], oof[oof_op])),
                "test_auc_all": float(s_pt[ix["g10"], 0]), "test_auc_open_play": float(roc_auc_score(y[op], res["p"]["g10"][op]))}
    # where the under-confidence sits: the three lowest reliability bins of g10 (30% of open play)
    low = res["bins"]["g10"] <= 2
    far = ctx.distance.to_numpy()[op] == DIST_LABELS[-1]
    low_xg = {"shots": int(low.sum()), "goals": int(y[op][low].sum()), "sum_xg": float(res["p"]["g10"][op][low].sum()),
              "mean_xg": float(res["p"]["g10"][op][low].mean()), "share_25m_plus": float(far[low].mean()),
              "share_25m_plus_open_play": float(far.mean())}
    games_op = games[op]
    split_half = pd.concat([split_half_recalibration(y[op], res["p"][k][op], games_op, a.n_splits).assign(variant=k)
                            for k in ("g10", "ens")])
    recal = {k: {"in_sample": recalibration_gains(y[op], res["p"][k][op], y[op], res["p"][k][op]),
                 "split_half": {g: {"mean": float(d[g].mean()), "p2.5": float(np.percentile(d[g], 2.5)),
                                    "p97.5": float(np.percentile(d[g], 97.5)), "share_positive": float((d[g] > 0).mean())}
                                for g in ("gain_logistic", "gain_isotonic")}}
             for k, d in split_half.groupby("variant")}

    curve_df = pd.concat(curves.values())
    tables = {"headline_ens_vs_g10.csv": headline, "equivalence_curve.csv": curve_df, "equivalence_stability.csv": stability,
              "instances_vs_g10.csv": inst,
              "ladder_vs_g10.csv": ladder, "calibration.csv": cal.reset_index(), "calibration_rungs.csv": cal_rungs.reset_index(),
              "calibration_diff.csv": cal_diff, "subgroups.csv": sub, "reliability_bins.csv": rel,
              "split_half_recalibration.csv": split_half}
    for f, t in tables.items():
        t.to_csv(a.out / f, index=False)
    summary = {"n_test": len(ctx), "games": int(ctx.game.nunique()), "open_play": int(op.sum()), "n_boot": a.n_boot,
               "n_splits": a.n_splits, "g10_reproduction_max_abs_dp": repro, "pooled_rate": pooled_rate, "p_pen": p_pen,
               "equivalence": refs, "oof_diagnostics": oof_diag, "g10_lowest_30pct": low_xg, "recalibration": recal,
               "instances_ci_including_zero_log_loss": {r: int(((d.d_log_loss_lo <= 0) & (d.d_log_loss_hi >= 0)).sum())
                                                        for r, d in inst.groupby("rung")},
               "open_play_above_0_5": {k: int((res["p"][k][op] > 0.5).sum()) for k in ("g10", "ens")}}
    timing = json.loads(a.timing.read_text()) if a.timing.exists() else None
    if timing:
        summary["timing"] = timing
        (a.figures / "timing.json").write_text(json.dumps(timing, indent=2))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for f in [*tables, "summary.json"]:
        (a.figures / f).write_bytes((a.out / f).read_bytes())

    titles = {g1_shown: "pojedynczy model G (fold 0, seed 0)", "g10": "G-combo (10 modeli)", "ens": "ensemble (30 modeli)"}
    plot_reliability(rel, cal, res, titles, a.figures / "calibration_reliability.png")
    plot_subgroups(sub, a.figures / "calibration_subgroups.png")
    plot_equivalence(curves, refs, a.figures / "equivalence.png")
    if timing:
        flip = headline.assign(rung="ens")  # ens row holds loss(g10) - loss(ens); the plot wants variant - g10
        for m in ("log_loss", "brier"):
            flip[f"d_{m}"], flip[f"d_{m}_lo"], flip[f"d_{m}_hi"] = -flip[f"d_{m}"], -flip[f"d_{m}_hi"], -flip[f"d_{m}_lo"]
        zero = {"rung": "g10"} | {f"d_{m}{s}": 0.0 for m in ("log_loss", "brier") for s in ("", "_lo", "_hi")}
        cost = pd.concat([ladder, pd.DataFrame([zero]), flip])
        plot_cost(cost, timing["configurations_fp32"], timing["batch"], a.figures / "cost_quality.png")

    pd.set_option("display.width", 250)
    for name in ("headline_ens_vs_g10.csv", "instances_vs_g10.csv", "ladder_vs_g10.csv", "calibration_diff.csv", "subgroups.csv"):
        print(name)
        print(tables[name].round(5).to_string(index=False))
    print(cal_rungs.round(4).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "timing"}, indent=2))


if __name__ == "__main__":
    main()
