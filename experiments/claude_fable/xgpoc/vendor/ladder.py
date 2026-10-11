"""Information ladder (H1) and the test of whether the frame adds to positions (H2) (xg-vendor ticket 08, test seam 2).

Pure functions of the evaluation table (`xgpoc.vendor.evaltable`) and the analysis parameters (margin, bootstrap draws,
seeds). The power simulation (`xgpoc.vendor.power`) runs H2 through `h2_draws`, the same code as the analysis.

Population
    the rows of the chosen populations (and splits) that are not penalties and have a scored frame (`frame_status`
    ok and a finite `frame_logit_0`); the label is `goal_sn`. Excluded rows are counted, never replaced.
Rungs, in ladder order (`LADDER`)
    constant, dist_angle, event, freeze_frame (the positions rung of H2), vendor, frame (sigmoid of `frame_logit_0`),
    and the stacks frame+positions and frame+vendor.
Variants
    raw            the table's probabilities (secondary, D6); the stacks have no raw variant
    recalibrated   every single rung gets Platt (logistic regression of the label on the rung's logit, unregularised),
                   cross-fitted on `N_FOLDS` match-grouped folds (D6). The folds depend only on the population's matches
                   and the fold seed (`match_folds`), so every rung and stack shares them, and no shot is scored by a
                   model fitted on its own match. A stack is a logistic regression of the label on the raw logits of
                   its two rungs, cross-fitted on the same folds; it counts as recalibrated.
Inference
    match-clustered bootstrap with count-weighted draws: the shared draws of the whole table
    (`xgpoc.vendor.draws.table_draws`, i.e. `xgpoc.probe.readability.match_draw_counts`: keys np.unique(matches),
    default_rng(seed), rng.choice(M, M) per draw, the sequence of scripts/significance.py). A population set reads
    its matches' columns of the counts, so every set, contrast and ticket uses the same draws. The cross-fitted
    predictions are fixed; each metric is recomputed per draw with the match counts as shot weights. Intervals are
    percentile intervals.
    TOST at `ALPHA` per side: equivalence at margin d when at most ALPHA of the draws of the difference lie at or
    below -d and at most ALPHA at or above d (the 90% interval inside (-d, d)); equivalence curves over
    `MARGIN_GRID` (0 to 0.01 log-loss). One-sided superiority of a over b: for a loss, p = share of draws >= 0;
    for AUC, share <= 0; superior when p <= ALPHA.
H2 (`H2`)
    log-loss of frame+positions minus positions (freeze_frame), both recalibrated; negative = the frame helps.
    Primary population: confirmatory at the pre-registered margin. Combined population: robustness. frame+vendor
    minus vendor (`H2_VENDOR`) is reported the same way.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.vendor import evaltable as et
from xgpoc.vendor.draws import TableDraws, table_draws


class Rung(NamedTuple):
    variant: str
    name: str


N_FOLDS = 5
FOLD_SEED = 0
BOOT_SEED = 0
MIN_CONFIRMATORY_BOOT = 5000
ALPHA = 0.05
MARGIN_GRID = np.round(np.linspace(0.0, 0.01, 101), 6)   # step 1e-4
CHUNK = 500                                               # bootstrap draws per vectorised block
NEWTON_ITERATIONS = 50

SINGLE_RUNGS = {"constant": "pred_constant", "dist_angle": "pred_dist_angle", "event": "pred_event",
                "freeze_frame": "pred_freeze_frame", "vendor": "pred_vendor_xg", "frame": "frame_logit_0"}
STACKS = {"frame+positions": ("frame", "freeze_frame"), "frame+vendor": ("frame", "vendor")}
LADDER = (*SINGLE_RUNGS, *STACKS)
VARIANTS = ("recalibrated", "raw")
METRICS = ("log_loss", "brier", "auc", "cal_slope", "cal_intercept", "mean_p_minus_rate")
CONTRAST_METRICS = ("log_loss", "brier", "auc")
LOWER_IS_BETTER = {"log_loss": True, "brier": True, "auc": False}
NO_CALIBRATION = ("constant",)   # a constant prediction has no calibration slope
H2 = (Rung("recalibrated", "frame+positions"), Rung("recalibrated", "freeze_frame"))
H2_VENDOR = (Rung("recalibrated", "frame+vendor"), Rung("recalibrated", "vendor"))
POPULATION_SETS = {"primary": ("primary",), "combined": et.POPULATIONS}


# ------------------------------------------------------------------ population and folds
def analysis_population(table: pd.DataFrame, populations: Sequence[str], splits: Sequence[str] = et.SPLITS) -> tuple[pd.DataFrame, dict]:
    """(rows of the analysis, exclusion counts) for a table with labels joined. Rows keep the table order."""
    if not set(et.LABEL_COLUMNS) <= set(table.columns):
        raise ValueError("the ladder needs a table with labels joined")
    if set(populations) - set(et.POPULATIONS) or set(splits) - set(et.SPLITS):
        raise ValueError(f"unknown population or split: {populations}, {splits}")
    chosen = table[table.population.isin(populations) & table.split.isin(splits)]
    penalty = chosen.is_penalty.astype(bool)
    no_frame = ~penalty & ~(chosen.frame_status.eq(et.OK) & np.isfinite(chosen.frame_logit_0.to_numpy(float)))
    rows = chosen[~penalty & ~no_frame].reset_index(drop=True)
    exclusions = {"chosen": int(len(chosen)), "penalty": int(penalty.sum()), "no_frame": int(no_frame.sum()), "analysed": int(len(rows))}
    return rows, exclusions


def match_folds(matches: np.ndarray, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold of each shot: the sorted unique matches in a seeded random order, fold = position mod `n_folds`."""
    keys, inverse = np.unique(np.asarray(matches), return_inverse=True)
    if len(keys) < n_folds:
        raise ValueError(f"{len(keys)} matches cannot fill {n_folds} folds")
    fold_of_key = np.empty(len(keys), int)
    fold_of_key[np.random.default_rng(seed).permutation(len(keys))] = np.arange(len(keys)) % n_folds
    return fold_of_key[inverse]


def fit_logistic(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> LogisticRegression:
    """The one logistic fit of the analyses: unregularised (C = 1e6), as `xgpoc.eval.calibration.PlattScaler`."""
    return LogisticRegression(C=1e6, max_iter=5000).fit(np.asarray(x, float).reshape(len(y), -1), y, sample_weight=sample_weight)


def frame_feature(frame_logit: np.ndarray) -> np.ndarray:
    """The frame rung's logit as every model sees it: the table logit clipped like any probability rung."""
    return logit(sigmoid(frame_logit))


def crossfit(features: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities of an unregularised logistic regression of `y` on `features` (n, k), and its
    coefficients per fold (n_folds, 1 + k: intercept first). A shot is scored only by the model of the other folds."""
    features = np.asarray(features, float).reshape(len(y), -1)
    out = np.full(len(y), np.nan)
    coefficients = []
    for fold in np.unique(folds):
        held_out = folds == fold
        model = fit_logistic(features[~held_out], y[~held_out])
        out[held_out] = model.predict_proba(features[held_out])[:, 1]
        coefficients.append(np.r_[model.intercept_, model.coef_[0]])
    return out, np.array(coefficients)


def raw_rungs(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """The single rungs' probabilities as stored, clipped to [EPS, 1 - EPS]; the frame is sigmoid(frame_logit_0)."""
    raw = {name: rows[column].to_numpy(float) for name, column in SINGLE_RUNGS.items()}
    raw = {name: np.clip(p, EPS, 1 - EPS) for name, p in raw.items() if name != "frame"}
    return {**raw, "frame": sigmoid(frame_feature(rows.frame_logit_0.to_numpy(float)))}


def rung_predictions(rows: pd.DataFrame, folds: np.ndarray) -> tuple[dict[Rung, np.ndarray], dict[str, list]]:
    """Every rung's probabilities over `rows` (both variants) and the per-fold coefficients of each cross-fitted model."""
    y = rows.goal_sn.to_numpy(int)
    raw = raw_rungs(rows)
    predictions = {Rung("raw", name): p for name, p in raw.items()}
    coefficients = {}
    for name, p in raw.items():
        predictions[Rung("recalibrated", name)], coefs = crossfit(logit(p), y, folds)
        coefficients[name] = coefs.round(6).tolist()
    for name, (a, b) in STACKS.items():
        predictions[Rung("recalibrated", name)], coefs = crossfit(np.column_stack([logit(raw[a]), logit(raw[b])]), y, folds)
        coefficients[name] = coefs.round(6).tolist()
    return predictions, coefficients


# ------------------------------------------------------------------ count-weighted metrics
def bootstrap(matches: np.ndarray, n_boot: int, seed: int = BOOT_SEED) -> TableDraws:
    """The shared draws (`table_draws`) of a set of shots given only their matches."""
    return table_draws(pd.DataFrame({"match": np.asarray(matches)}), n_boot, seed)


def draws_for(draws: TableDraws, matches: np.ndarray) -> TableDraws:
    """The table's draws seen from a subset of its rows: `match` becomes each given shot's position in the keys."""
    position = np.searchsorted(draws.keys, matches)
    if not np.array_equal(draws.keys[np.minimum(position, len(draws.keys) - 1)], matches):
        raise ValueError("a shot's match is not among the draws' matches")
    return draws._replace(match=position)


def observed(boot: TableDraws) -> TableDraws:
    """Every match once: the metrics of the observed sample."""
    return boot._replace(counts=np.ones((1, len(boot.keys)), np.int32))


def per_shot_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _match_mean(values: np.ndarray, boot: TableDraws) -> np.ndarray:
    """Count-weighted mean of a per-shot quantity, per draw."""
    m = len(boot.keys)
    return (boot.counts @ np.bincount(boot.match, values, m)) / (boot.counts @ np.bincount(boot.match, minlength=m))


def _auc(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted AUC per row of `weights` (b, n); ties count one half (equals roc_auc_score on the resampled shots)."""
    order = np.argsort(p, kind="stable")
    ordered, goal = p[order], y[order].astype(bool)
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    w = weights[:, order]
    pos = np.add.reduceat(np.where(goal, w, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(goal, 0.0, w), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def _calibration(y: np.ndarray, z: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intercept, slope) of the weighted logistic recalibration y ~ a + b * z per row of `weights`, by Newton."""
    beta = np.tile([0.0, 1.0], (len(weights), 1))
    for _ in range(NEWTON_ITERATIONS):
        q = sigmoid(beta[:, :1] + beta[:, 1:] * z)
        r, v = weights * (y - q), weights * q * (1 - q)
        g0, g1 = r.sum(1), r @ z
        h00, h01, h11 = v.sum(1), v @ z, v @ (z * z)
        det = h00 * h11 - h01**2
        step = np.column_stack([(h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det])
        beta += step
        if np.abs(step).max() < 1e-9:
            return beta[:, 0], beta[:, 1]
    raise ValueError("calibration Newton did not converge")


def metric_draws(y: np.ndarray, p: np.ndarray, boot: TableDraws, metrics: Sequence[str] = METRICS, calibration: bool = True) -> dict[str, np.ndarray]:
    """Each metric per draw of `boot` (shot weight = count of its match). `calibration` False gives NaN slope and intercept."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    goals = boot.counts @ np.bincount(boot.match, y, len(boot.keys))
    if (goals == 0).any() or (goals == boot.counts @ np.bincount(boot.match, minlength=len(boot.keys))).any():
        raise ValueError("a bootstrap draw has no goal or only goals; the population is too small for a match bootstrap")
    out = {}
    if "log_loss" in metrics:
        out["log_loss"] = _match_mean(per_shot_log_loss(y, p), boot)
    if "brier" in metrics:
        out["brier"] = _match_mean((p - y) ** 2, boot)
    if "mean_p_minus_rate" in metrics:
        out["mean_p_minus_rate"] = _match_mean(p - y, boot)
    blocks = [slice(i, i + CHUNK) for i in range(0, len(boot.counts), CHUNK)]
    if "auc" in metrics:
        out["auc"] = np.concatenate([_auc(y, p, boot.counts[b][:, boot.match].astype(float)) for b in blocks])
    if {"cal_slope", "cal_intercept"} & set(metrics):
        if calibration:
            fits = [_calibration(y, logit(p), boot.counts[b][:, boot.match].astype(float)) for b in blocks]
            intercept, slope = np.concatenate([f[0] for f in fits]), np.concatenate([f[1] for f in fits])
        else:
            intercept = slope = np.full(len(boot.counts), np.nan)
        out["cal_intercept"], out["cal_slope"] = intercept, slope
    return {k: out[k] for k in metrics}


# ------------------------------------------------------------------ contrasts
def equivalence_curve(draws: np.ndarray, margins: np.ndarray = MARGIN_GRID, alpha: float = ALPHA) -> pd.DataFrame:
    """TOST per margin on bootstrap draws of a paired difference (prior art: scripts/report_complexity.py)."""
    draws = np.asarray(draws, float)[:, None]
    if np.isnan(draws).any():
        raise ValueError("NaN bootstrap draws would count as inside every margin")
    p_lower = (draws <= -margins).mean(0)
    p_upper = (draws >= margins).mean(0)
    p_tost = np.maximum(p_lower, p_upper)
    return pd.DataFrame({"margin": margins, "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost, "equivalent": p_tost <= alpha})


def smallest_margin(curve: pd.DataFrame) -> float:
    """Smallest grid margin with equivalence; NaN when the grid never reaches it."""
    hit = curve.margin[curve.equivalent]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def superiority_p(draws: np.ndarray, lower_is_better: bool = True) -> float:
    """One-sided bootstrap p-value of "a is better than b" from draws of a - b."""
    draws = np.asarray(draws, float)
    return float((draws >= 0).mean() if lower_is_better else (draws <= 0).mean())


def contrast(point: float, draws: np.ndarray, metric: str, margin: float | None = None, alpha: float = ALPHA) -> dict:
    """Summary of a - b: point, 95% and 90% percentile intervals, bootstrap SD, one-sided superiority p of a; for
    log-loss also the TOST at `margin` and the smallest equivalence margin on `MARGIN_GRID`."""
    lo95, lo90, hi90, hi95 = np.percentile(draws, [2.5, 5, 95, 97.5])
    p_sup = superiority_p(draws, LOWER_IS_BETTER[metric])
    out = {"metric": metric, "diff": float(point), "lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90,
           "boot_sd": float(np.std(draws, ddof=1)), "p_superiority": p_sup, "superior": p_sup <= alpha}
    if metric == "log_loss":
        curve = equivalence_curve(draws, alpha=alpha)
        out["smallest_equivalence_margin"] = smallest_margin(curve)
        if margin is not None:
            at = equivalence_curve(draws, np.array([margin]), alpha).iloc[0]
            out.update(margin=margin, p_tost=float(at.p_tost), equivalent=bool(at.equivalent))
    return out


def contrast_pairs() -> list[tuple[Rung, Rung]]:
    """Every pair of rungs of one variant, later rung of the ladder first (a - b)."""
    pairs = []
    for variant in VARIANTS:
        names = list(LADDER) if variant == "recalibrated" else list(SINGLE_RUNGS)
        pairs += [(Rung(variant, a), Rung(variant, b)) for i, a in enumerate(names) for b in names[:i]]
    return pairs


# ------------------------------------------------------------------ H2 for the power simulation
def h2_draws(y: np.ndarray, positions: np.ndarray, frame_logit: np.ndarray, matches: np.ndarray, boot: TableDraws,
             fold_seed: int = FOLD_SEED) -> tuple[float, np.ndarray]:
    """(point, draws) of the H2 log-loss difference, frame+positions minus positions, both recalibrated on the
    population's own folds: the computation of `analyse`, without the other rungs. `boot` holds the draws with
    `match` set for these shots (`draws_for`)."""
    y = np.asarray(y, int)
    folds = match_folds(matches, seed=fold_seed)
    z_pos, z_frame = logit(positions), frame_feature(frame_logit)
    pos, _ = crossfit(z_pos, y, folds)
    stack, _ = crossfit(np.column_stack([z_frame, z_pos]), y, folds)
    diff = per_shot_log_loss(y, stack) - per_shot_log_loss(y, pos)
    return float(_match_mean(diff, observed(boot))[0]), _match_mean(diff, boot)


# ------------------------------------------------------------------ the analysis
def _plain(value):
    """A JSON-ready scalar: numpy scalars as Python ones, NaN as None."""
    value = value.item() if isinstance(value, np.generic) else value
    return None if isinstance(value, float) and np.isnan(value) else value


@dataclass
class LadderResult:
    """One population set: metrics (rung x metric, point and 95% interval), contrasts (every pair of `contrast_pairs`
    x `CONTRAST_METRICS`), log-loss equivalence curves of every pair, a summary with the parameters and the H2 and
    frame+vendor records, recalibration
    coefficients, folds per match, exclusion counts."""

    metrics: pd.DataFrame
    contrasts: pd.DataFrame
    curves: pd.DataFrame
    summary: dict
    coefficients: dict
    folds: pd.DataFrame
    exclusions: dict


def analyse(table: pd.DataFrame, name: str, populations: Sequence[str], margin: float, draws: TableDraws,
            splits: Sequence[str] = et.SPLITS, fold_seed: int = FOLD_SEED) -> LadderResult:
    """The ladder, all contrasts and H2 on one population set (`name` labels the outputs). `draws` are the shared
    draws of the whole table (`table_draws`)."""
    rows, exclusions = analysis_population(table, populations, splits)
    y, matches = rows.goal_sn.to_numpy(int), rows.match.to_numpy()
    folds = match_folds(matches, seed=fold_seed)
    predictions, coefficients = rung_predictions(rows, folds)
    boot = draws_for(draws, matches)
    point_boot, n_matches = observed(boot), len(np.unique(matches))
    points, rung_draws, metric_rows = {}, {}, []
    for rung in [Rung(v, n) for v in VARIANTS for n in LADDER if Rung(v, n) in predictions]:
        calibration = rung.name not in NO_CALIBRATION
        points[rung] = {k: float(v[0]) for k, v in metric_draws(y, predictions[rung], point_boot, calibration=calibration).items()}
        rung_draws[rung] = metric_draws(y, predictions[rung], boot, calibration=calibration)
        row = {"population": name, "variant": rung.variant, "rung": rung.name, "n": len(rows), "goals": int(y.sum()), "matches": n_matches}
        for metric in METRICS:
            lo, hi = np.percentile(rung_draws[rung][metric], [2.5, 97.5])   # NaN for the slope of a constant
            row.update({metric: points[rung][metric], f"{metric}_lo": lo, f"{metric}_hi": hi})
        metric_rows.append(row)

    contrast_rows, curves = [], []
    for a, b in contrast_pairs():
        for metric in CONTRAST_METRICS:
            d = rung_draws[a][metric] - rung_draws[b][metric]
            result = contrast(points[a][metric] - points[b][metric], d, metric, margin if metric == "log_loss" else None)
            contrast_rows.append({"population": name, "variant_a": a.variant, "a": a.name, "variant_b": b.variant, "b": b.name, **result})
            if metric == "log_loss":
                curves.append(equivalence_curve(d).assign(population=name, variant_a=a.variant, a=a.name, variant_b=b.variant, b=b.name))
    contrasts = pd.DataFrame(contrast_rows)

    def record(pair: tuple[Rung, Rung]) -> dict:
        a, b = pair
        result = contrast(points[a]["log_loss"] - points[b]["log_loss"], rung_draws[a]["log_loss"] - rung_draws[b]["log_loss"], "log_loss", margin)
        return {"a": f"{a.name} ({a.variant})", "b": f"{b.name} ({b.variant})", "log_loss_a": points[a]["log_loss"],
                "log_loss_b": points[b]["log_loss"], **{k: _plain(v) for k, v in result.items()}}

    keys_fold = pd.DataFrame({"match": matches, "fold": folds}).drop_duplicates("match").sort_values("match").reset_index(drop=True)
    summary = {"population": name, "n": len(rows), "goals": int(y.sum()), "matches": n_matches, "n_boot": len(draws.counts),
               "fold_seed": fold_seed, "alpha_per_side": ALPHA, "margin": margin,
               "h2": record(H2), "frame_plus_vendor": record(H2_VENDOR)}
    return LadderResult(pd.DataFrame(metric_rows), contrasts, pd.concat(curves, ignore_index=True), summary,
                        coefficients, keys_fold, exclusions)
