"""Information ladder (H1) and the test of whether the frame adds to positions (H2) (xg-vendor ticket 08, test seam 2).

Pure functions of the evaluation table (`xgpoc.vendor.evaltable`) and the analysis parameters (margin, bootstrap draws,
seeds). The power simulation (`xgpoc.vendor.power`) runs H2 through `h2_draws`, the same code as the analysis.

Population
    the rows of the chosen populations (and splits) without a label exclusion (`label_exclusion`, e.g. a StatsBomb goal
    without any SoccerNet goal), that are not penalties, are not pre-registered frame exclusions (`frame_exclusions`,
    secondary rows whose scored frame does not show the shot) and have a scored frame (`frame_status` ok and a finite
    `frame_logit_0`); the label is `goal_sn`. Every rung is scored on these same rows. Excluded rows are counted, never
    replaced. Penalties are reported apart (`penalty_summary`, spec: penalties reported separately).
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
    its matches' columns of the counts, so every set, contrast and ticket uses the same draws. Recalibrated variant
    (single rungs and stacks, so H2), by the pre-registered `bootstrap` (`evaltable.RECALIBRATED_BOOTSTRAPS`): "refit"
    refits the cross-fitted models in every draw on the resampled matches (`xgpoc.vendor.refit`, same folds, match
    counts as weights) and computes every metric from the refitted out-of-fold predictions; "fixed" reweights the
    observed out-of-fold predictions like the raw variant. Raw variant: nothing is fitted, the stored probabilities are
    reweighted per draw.
    Intervals are percentile intervals.
    TOST at `ALPHA` per side: equivalence at margin d when at most ALPHA of the draws of the difference lie at or
    below -d and at most ALPHA at or above d (the 90% interval inside (-d, d)); equivalence curves over
    `MARGIN_GRID` (0 to 0.01 log-loss). One-sided superiority of a over b: for a loss, p = share of draws >= 0;
    for AUC, share <= 0; superior when p <= ALPHA.
H2 (`H2`)
    log-loss of frame+positions minus positions (freeze_frame), both recalibrated; negative = the frame helps.
    Primary population: confirmatory at the pre-registered margin. Combined population: robustness. frame+vendor
    minus vendor (`H2_VENDOR`) is reported the same way. `h2_draws` is the fixed-prediction version that the frozen
    ticket-08 power simulation ran; the analysis refits per draw unless the pre-registration fixes "fixed".
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import refit
from xgpoc.vendor.draws import TableDraws, table_draws


class Rung(NamedTuple):
    variant: str
    name: str


class CrossFit(NamedTuple):
    """A cross-fitted logistic model: its inputs (n, k) and the observed sample's coefficients per fold (n_folds, 1 + k)."""

    features: np.ndarray
    coefficients: np.ndarray


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
BOOTSTRAP_NOTES = {"refit": "recalibrated variant: cross-fitted models refitted in every match draw; raw variant: stored probabilities reweighted",
                   "fixed": "recalibrated variant: observed out-of-fold predictions reweighted in every match draw (no refit); "
                            "raw variant: stored probabilities reweighted"}
PENALTY_RUNGS = ("vendor", "frame", "freeze_frame")


# ------------------------------------------------------------------ population and folds
def _chosen(table: pd.DataFrame, populations: Sequence[str], splits: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    """(rows of the chosen populations and splits, their label-exclusion mask) of a table with labels joined."""
    if not set(et.LABEL_COLUMNS) <= set(table.columns):
        raise ValueError("the ladder needs a table with labels joined")
    if set(populations) - set(et.POPULATIONS) or set(splits) - set(et.SPLITS):
        raise ValueError(f"unknown population or split: {populations}, {splits}")
    chosen = table[table.population.isin(populations) & table.split.isin(splits)]
    return chosen, chosen.label_exclusion.notna()


def analysis_population(table: pd.DataFrame, populations: Sequence[str], splits: Sequence[str] = et.SPLITS,
                        frame_exclusions: Mapping[str, str] = MappingProxyType({})) -> tuple[pd.DataFrame, dict]:
    """(rows of the analysis, exclusion counts) for a table with labels joined. Rows keep the table order. A row whose
    key is in `frame_exclusions` (the pre-registration's, secondary rows only) is left out like a row without a frame
    and counted apart; each row is counted under the first reason that applies (label, penalty, frame excluded, no frame)."""
    listed = list(frame_exclusions)
    if table.population.eq("primary").to_numpy()[table.key.isin(listed).to_numpy()].any():
        raise ValueError("frame_exclusions may list secondary rows only")
    chosen, excluded = _chosen(table, populations, splits)
    penalty = ~excluded & chosen.is_penalty.astype(bool)
    frame_excluded = ~excluded & ~penalty & chosen.key.isin(listed)
    no_frame = ~excluded & ~penalty & ~frame_excluded & ~(chosen.frame_status.eq(et.OK) & np.isfinite(chosen.frame_logit_0.to_numpy(float)))
    rows = chosen[~excluded & ~penalty & ~frame_excluded & ~no_frame].reset_index(drop=True)
    exclusions = {"chosen": int(len(chosen)), "label_excluded": int(excluded.sum()), "penalty": int(penalty.sum()),
                  "frame_excluded": int(frame_excluded.sum()), "no_frame": int(no_frame.sum()), "analysed": int(len(rows))}
    return rows, exclusions


def penalty_summary(table: pd.DataFrame, populations: Sequence[str], splits: Sequence[str] = et.SPLITS) -> dict:
    """The penalties of a population set, reported apart from H1-H4: count, SoccerNet goals, conversion and, per rung of
    `PENALTY_RUNGS`, the raw mean probability and log-loss over the penalties it scores (the frame: a scored frame; its
    probability is the routed penalty rate). Descriptive, no interval; NaN when the set has no penalty."""
    chosen, excluded = _chosen(table, populations, splits)
    penalties = chosen[~excluded & chosen.is_penalty.astype(bool)]
    y = penalties.goal_sn.to_numpy(float)
    out = {"penalties": int(len(penalties)), "goals": int(y.sum()), "conversion": float(y.mean()) if len(y) else float("nan")}
    raw = raw_rungs(penalties)
    scored = (penalties.frame_status.eq(et.OK) & np.isfinite(penalties.frame_logit_0.to_numpy(float))).to_numpy()
    for name in PENALTY_RUNGS:
        keep = scored if name == "frame" else np.ones(len(penalties), bool)
        p = raw[name][keep]
        out[name] = {"n": int(keep.sum()), "mean_p": float(p.mean()) if len(p) else float("nan"),
                     "log_loss": float(per_shot_log_loss(y[keep], p).mean()) if len(p) else float("nan")}
    return out


def match_folds(matches: np.ndarray, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold of each shot: the sorted unique matches in a seeded random order, fold = position mod `n_folds`."""
    keys, inverse = np.unique(np.asarray(matches), return_inverse=True)
    if len(keys) < n_folds:
        raise ValueError(f"{len(keys)} matches cannot fill {n_folds} folds")
    fold_of_key = np.empty(len(keys), int)
    fold_of_key[np.random.default_rng(seed).permutation(len(keys))] = np.arange(len(keys)) % n_folds
    return fold_of_key[inverse]


def fit_logistic(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> LogisticRegression:
    """sklearn's logistic fit with C = 1e6, as `xgpoc.eval.calibration.PlattScaler`; the power simulation's truth fits
    use it. Its default tolerance stops about 1e-3 short in probability, so the cross-fits use `crossfit` (exact)."""
    return LogisticRegression(C=1e6, max_iter=5000).fit(np.asarray(x, float).reshape(len(y), -1), y, sample_weight=sample_weight)


def frame_feature(frame_logit: np.ndarray) -> np.ndarray:
    """The frame rung's logit as every model sees it: the table logit clipped like any probability rung."""
    return logit(sigmoid(frame_logit))


def crossfit(features: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities of a logistic regression of `y` on `features` (n, k), and its coefficients per fold
    (n_folds, 1 + k: intercept first, folds in `np.unique` order). A shot is scored only by the model of the other
    folds. The fit is sklearn's objective with C = 1e6 solved to convergence (`refit.logistic_newton`), the same fit
    the per-draw refit repeats."""
    x = np.column_stack([np.ones(len(y)), np.asarray(features, float).reshape(len(y), -1)])
    out = np.full(len(y), np.nan)
    coefficients = []
    for fold in np.unique(folds):
        held_out = folds == fold
        start = np.r_[logit(np.mean(y[~held_out])), np.zeros(x.shape[1] - 1)]   # from the training rows only
        beta = refit.logistic_newton(x, y, (~held_out).astype(float)[None], start)[0]
        out[held_out] = sigmoid(x[held_out] @ beta)
        coefficients.append(beta)
    return out, np.array(coefficients)


def raw_rungs(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """The single rungs' probabilities as stored, clipped to [EPS, 1 - EPS]; the frame is sigmoid(frame_logit_0)."""
    raw = {name: rows[column].to_numpy(float) for name, column in SINGLE_RUNGS.items()}
    raw = {name: np.clip(p, EPS, 1 - EPS) for name, p in raw.items() if name != "frame"}
    return {**raw, "frame": sigmoid(frame_feature(rows.frame_logit_0.to_numpy(float)))}


def rung_predictions(rows: pd.DataFrame, folds: np.ndarray) -> tuple[dict[Rung, np.ndarray], dict[Rung, CrossFit]]:
    """Every rung's probabilities over `rows` (both variants) and, per recalibrated rung, its cross-fitted model."""
    y = rows.goal_sn.to_numpy(int)
    raw = raw_rungs(rows)
    predictions = {Rung("raw", name): p for name, p in raw.items()}
    inputs = {name: logit(p) for name, p in raw.items()} | {name: np.column_stack([logit(raw[a]), logit(raw[b])]) for name, (a, b) in STACKS.items()}
    fits = {}
    for name, features in inputs.items():
        rung = Rung("recalibrated", name)
        predictions[rung], coefs = crossfit(features, y, folds)
        fits[rung] = CrossFit(features, coefs)
    return predictions, fits


def refit_predictor(fit: CrossFit, y: np.ndarray, folds: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The out-of-fold probabilities of `fit` refitted in each draw of a block of draw weights (b, n) (`xgpoc.vendor.refit`)."""
    return lambda weights: refit.crossfit_draws(fit.features, y, folds, weights, fit.coefficients)


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
    """Weighted AUC per row of `weights` (b, n) for one prediction vector `p` (n,); ties count one half (equals
    roc_auc_score on the resampled shots)."""
    order = np.argsort(p, kind="stable")
    ordered, goal = p[order], y[order].astype(bool)
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    w = weights[:, order]
    pos = np.add.reduceat(np.where(goal, w, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(goal, 0.0, w), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def _auc_rows(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """`_auc` with one prediction vector per row: `p` and `weights` are (b, n)."""
    n = p.shape[1]
    order = np.argsort(p, axis=1, kind="stable")
    ordered, w = np.take_along_axis(p, order, 1), np.take_along_axis(weights, order, 1)
    goal = y.astype(bool)[order]
    pos, neg = np.where(goal, w, 0.0), np.where(goal, 0.0, w)
    first, last = np.ones(p.shape, bool), np.ones(p.shape, bool)
    first[:, 1:] = last[:, :-1] = ordered[:, 1:] != ordered[:, :-1]   # a tie group starts / ends here
    index = np.arange(n)
    start = np.maximum.accumulate(np.where(first, index, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, index, n - 1)[:, ::-1], axis=1)[:, ::-1]
    cumulative = np.cumsum(neg, axis=1)
    below = np.take_along_axis(cumulative, start, 1) - np.take_along_axis(neg, start, 1)   # negatives below the tie group
    tied = np.take_along_axis(cumulative, end, 1) - below
    return (pos * (below + 0.5 * tied)).sum(1) / (pos.sum(1) * neg.sum(1))


def _calibration(y: np.ndarray, z: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intercept, slope) of the weighted logistic recalibration y ~ a + b * z per row of `weights` (b, n), by Newton;
    `z` is (n,) or one row per draw (b, n)."""
    beta = np.tile([0.0, 1.0], (len(weights), 1))
    for _ in range(NEWTON_ITERATIONS):
        q = sigmoid(beta[:, :1] + beta[:, 1:] * z)
        r, v = weights * (y - q), weights * q * (1 - q)
        g0, g1 = r.sum(1), (r * z).sum(1)
        h00, h01, h11 = v.sum(1), (v * z).sum(1), (v * z * z).sum(1)
        det = h00 * h11 - h01**2
        step = np.column_stack([(h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det])
        beta += step
        if np.abs(step).max() < 1e-9:
            return beta[:, 0], beta[:, 1]
    raise ValueError("calibration Newton did not converge")


def _block_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray, metrics: Sequence[str], calibration: bool) -> dict[str, np.ndarray]:
    """Each metric per row of `weights` (b, n); `p` is (n,) or (b, n), clipped."""
    total = weights.sum(1)
    out = {}
    if "log_loss" in metrics:
        out["log_loss"] = (weights * per_shot_log_loss(y, p)).sum(1) / total
    if "brier" in metrics:
        out["brier"] = (weights * (p - y) ** 2).sum(1) / total
    if "mean_p_minus_rate" in metrics:
        out["mean_p_minus_rate"] = (weights * (p - y)).sum(1) / total
    if "auc" in metrics:
        out["auc"] = _auc(y, p, weights) if p.ndim == 1 else _auc_rows(y, p, weights)
    if {"cal_slope", "cal_intercept"} & set(metrics):
        if calibration:
            out["cal_intercept"], out["cal_slope"] = _calibration(y, logit(p), weights)
        else:
            out["cal_intercept"] = out["cal_slope"] = np.full(len(weights), np.nan)
    return out


def metric_draws(y: np.ndarray, p: np.ndarray | Callable[[np.ndarray], np.ndarray], boot: TableDraws, metrics: Sequence[str] = METRICS,
                 calibration: bool = True) -> dict[str, np.ndarray]:
    """Each metric per draw of `boot` (shot weight = count of its match). `p` is either the fixed probabilities (n,) or
    a function of a block of draw weights (b, n) that returns the probabilities refitted in each of those draws (b, n)
    (`refit_predictor`). `calibration` False gives NaN slope and intercept."""
    y = np.asarray(y, float)
    goals = boot.counts @ np.bincount(boot.match, y, len(boot.keys))
    if (goals == 0).any() or (goals == boot.counts @ np.bincount(boot.match, minlength=len(boot.keys))).any():
        raise ValueError("a bootstrap draw has no goal or only goals; the population is too small for a match bootstrap")
    fixed = None if callable(p) else np.asarray(p, float)
    parts = []
    for i in range(0, len(boot.counts), CHUNK):
        weights = boot.counts[i:i + CHUNK][:, boot.match].astype(float)
        parts.append(_block_metrics(y, np.clip(p(weights) if fixed is None else fixed, EPS, 1 - EPS), weights, metrics, calibration))
    return {k: np.concatenate([part[k] for part in parts]) for k in metrics}


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
    population's own folds, with the cross-fitted predictions held fixed in the draws. This is what the frozen ticket-08
    power simulation ran (`xgpoc.vendor.power`); its point is `analyse`'s, its draws are narrower than `analyse`'s, which
    refits in every draw. `boot` holds the draws with `match` set for these shots (`draws_for`)."""
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
    x `CONTRAST_METRICS`), log-loss equivalence curves of every pair, a summary with the parameters, the H2 and
    frame+vendor records and the penalty summary, recalibration coefficients, folds per match, exclusion counts."""

    metrics: pd.DataFrame
    contrasts: pd.DataFrame
    curves: pd.DataFrame
    summary: dict
    coefficients: dict
    folds: pd.DataFrame
    exclusions: dict


def analyse(table: pd.DataFrame, name: str, populations: Sequence[str], margin: float, draws: TableDraws,
            splits: Sequence[str] = et.SPLITS, fold_seed: int = FOLD_SEED, bootstrap: str = "refit",
            frame_exclusions: Mapping[str, str] = MappingProxyType({})) -> LadderResult:
    """The ladder, all contrasts and H2 on one population set (`name` labels the outputs). `draws` are the shared
    draws of the whole table (`table_draws`); `bootstrap` and `frame_exclusions` are the pre-registration's (module
    docstring)."""
    if bootstrap not in et.RECALIBRATED_BOOTSTRAPS:
        raise ValueError(f"bootstrap must be one of {et.RECALIBRATED_BOOTSTRAPS}")
    rows, exclusions = analysis_population(table, populations, splits, frame_exclusions)
    y, matches = rows.goal_sn.to_numpy(int), rows.match.to_numpy()
    folds = match_folds(matches, seed=fold_seed)
    predictions, fits = rung_predictions(rows, folds)
    boot = draws_for(draws, matches)
    point_boot, n_matches = observed(boot), len(np.unique(matches))
    points, rung_draws, metric_rows = {}, {}, []
    for rung in [Rung(v, n) for v in VARIANTS for n in LADDER if Rung(v, n) in predictions]:
        calibration = rung.name not in NO_CALIBRATION
        points[rung] = {k: float(v[0]) for k, v in metric_draws(y, predictions[rung], point_boot, calibration=calibration).items()}
        sample = refit_predictor(fits[rung], y, folds) if rung in fits and bootstrap == "refit" else predictions[rung]
        rung_draws[rung] = metric_draws(y, sample, boot, calibration=calibration)
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
               "fold_seed": fold_seed, "alpha_per_side": ALPHA, "margin": margin, "bootstrap": BOOTSTRAP_NOTES[bootstrap],
               "h2": record(H2), "frame_plus_vendor": record(H2_VENDOR),
               "penalties": {k: _plain(v) if not isinstance(v, dict) else {m: _plain(x) for m, x in v.items()}
                             for k, v in penalty_summary(table, populations, splits).items()}}
    coefficients = {rung.name: fit.coefficients.round(6).tolist() for rung, fit in fits.items()}
    return LadderResult(pd.DataFrame(metric_rows), contrasts, pd.concat(curves, ignore_index=True), summary,
                        coefficients, keys_fold, exclusions)
