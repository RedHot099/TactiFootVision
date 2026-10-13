"""Statistics of the xg-compare analysis (ticket 02, council C4): the main scale refitted in every draw, the
unadjusted contrast and the adjusted effect, weighted OLS, bootstrap summaries, BH and BY q-values, label metrics
and the within-match label permutation of the dry run.

Stacked draws
    Every per-draw quantity is an array whose first axis is W = 1 + B: row 0 is the observed sample (every match
    once), rows 1..B are the shared match draws of the shot table (`xgpoc.vendor.draws.table_draws`, read through
    `ladder.draws_for`). `stacked_counts` builds the (W, M) match counts; a shot's weight in a row is its match's
    count. A statistic computed on the stack gives its point (row 0) and its bootstrap draws (rows 1..B) at once.
Main scale (`recalibrate`, `main_scale`)
    the paper's shared cross-fitted Platt recalibration (`ladder.match_folds`, `ladder.crossfit`): per method, fold k's
    model is the logistic regression of the goal label on the raw logit over the other folds' shots. Row 0 is the
    observed fit (`ladder.crossfit`, so it equals the paper's coefficients on the paper's population); every draw
    refits it with the match counts as weights (`refit.logistic_newton`, the solver of `refit.crossfit_draws`). The
    main-scale logit of a shot is a_k + b_k z with its fold's coefficients in that row. The scale depends on the
    labels: the C4 design calls it outcome-recalibrated.
Effects (`level_effects`)
    for an outcome y (a per-shot gap, or a per-shot log-loss difference) and a control c (z_D), per level and row:
    the unadjusted contrast, mean(y | in) - mean(y | out), and the adjusted effect, the coefficient of the level
    indicator in the weighted OLS fit y ~ 1 + indicator + c over the level's defined shots. `ols` fits a general
    weighted OLS per row (item 1 and 13's H4 controls, the multivariable fit).
p-values and intervals
    `power.p_values` (the protocol's sides: the share of draws on the wrong side of 0, two-sided doubles the smaller
    tail); `summarise` adds the 95% percentile interval over the finite draws and the number of NaN draws. `q_values`
    gives Benjamini-Hochberg adjusted p-values (Benjamini-Yekutieli with `dependent`): an item is rejected at q when
    its q-value is <= q, the same rejections as `power.bh`.
"""

import numpy as np
import pandas as pd

from xgpoc.compare import power as pw
from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import refit
from xgpoc.vendor.draws import TableDraws

DRAW_CHUNK = 500       # draws per block of the general OLS
XG_POINTS = 100.0      # xG points per unit of probability


# ------------------------------------------------------------------ stacked draws
def stacked_counts(draws: TableDraws) -> np.ndarray:
    """(W, M) match counts: row 0 every match once, then the draws' counts."""
    return np.vstack([np.ones(len(draws.keys)), draws.counts]).astype(float)


# ------------------------------------------------------------------ main scale
def recalibrate(z: np.ndarray, y: np.ndarray, folds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(W, n_folds, 2) Platt coefficients (intercept, slope) of the cross-fitted recalibration of raw logits `z` (n,)
    on labels `y`, per row of shot weights `weights` (W, n) (module docstring). Folds in `np.unique` order."""
    _, observed = ld.crossfit(z, y, folds)
    x = np.column_stack([np.ones(len(z)), z])
    out = np.empty((len(weights), len(observed), 2))
    for i, fold in enumerate(np.unique(folds)):
        held_out = folds == fold
        out[:, i] = refit.logistic_newton(x, y, np.where(held_out, 0.0, weights), observed[i])
    return out


def main_scale(z: np.ndarray, coefficients: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """(W, n) main-scale logits a_k + b_k z of each shot with its fold's coefficients (`recalibrate`) in each row."""
    k = np.searchsorted(np.unique(folds), folds)
    return coefficients[:, k, 0] + coefficients[:, k, 1] * z[None, :]


def percentile_ranks(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(W, n) within-population percentile of `x` (n,) per row of shot weights: 100 (midrank - 1/2) / total weight,
    with ties sharing their midrank (`power.weighted_ranks`)."""
    return 100.0 * (pw.weighted_ranks(x, weights) - 0.5) / weights.sum(1, keepdims=True)


# ------------------------------------------------------------------ effects
def level_effects(y: np.ndarray, control: np.ndarray, inside: np.ndarray, defined: np.ndarray,
                  weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(adjusted, unadjusted), each (W, L): per row of shot weights (W, n) and level (columns of the boolean `inside`
    and `defined`, (n, L)), the coefficient of the indicator in y ~ 1 + indicator + control and mean(y | in) -
    mean(y | out), over the level's defined shots. `y` and `control` are (n,) or (W, n). NaN where a row has no
    weight inside or outside the level, or a singular fit."""
    w = np.asarray(weights, float)
    y, c = np.broadcast_to(y, w.shape), np.broadcast_to(control, w.shape)
    d = defined.astype(float)
    i = (inside & defined).astype(float)
    wc, wy = w * c, w * y
    s11, s12, s13 = w @ d, w @ i, wc @ d
    s23, s33 = wc @ i, (wc * c) @ d
    r1, r2, r3 = wy @ d, wy @ i, (wy * c) @ d
    s22 = s12
    det = s11 * (s22 * s33 - s23**2) - s12 * (s12 * s33 - s23 * s13) + s13 * (s12 * s23 - s22 * s13)
    num = s11 * (r2 * s33 - s23 * r3) - r1 * (s12 * s33 - s23 * s13) + s13 * (s12 * r3 - r2 * s13)
    weight_out = s11 - s12
    ok = (s12 > 0) & (weight_out > 0)
    scale = np.maximum(np.abs(s11 * s22 * s33), 1e-300)
    with np.errstate(invalid="ignore", divide="ignore"):
        adjusted = np.where(ok & (np.abs(det) > 1e-12 * scale), num / det, np.nan)
        unadjusted = np.where(ok, r2 / s12 - (r1 - r2) / weight_out, np.nan)
    return adjusted, unadjusted


def ols(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(W, p) weighted least-squares coefficients of y ~ x per row of shot weights (W, n); `y` is (n,) or (W, n) and
    `x` (n, p) or (W, n, p). Minimum-norm (pseudo-inverse) when a row's design is singular, e.g. a dummy without weight."""
    w = np.asarray(weights, float)
    y = np.broadcast_to(y, w.shape)
    x = np.asarray(x, float)
    out = []
    for start in range(0, len(w), DRAW_CHUNK):
        block = slice(start, start + DRAW_CHUNK)
        xb = np.broadcast_to(x, (len(w),) + x.shape[-2:])[block]
        wb = w[block]
        xtwx = np.einsum("wn,wni,wnj->wij", wb, xb, xb)
        xtwy = np.einsum("wn,wni,wn->wi", wb, xb, y[block])
        out.append(np.einsum("wij,wj->wi", np.linalg.pinv(xtwx, hermitian=True), xtwy))
    return np.concatenate(out)


# ------------------------------------------------------------------ summaries and multiplicity
def interval(draws: np.ndarray) -> tuple[float, float, int]:
    """(lo, hi, NaN draws): the 95% percentile interval of the finite draws (B,)."""
    draws = np.asarray(draws, float)
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return float("nan"), float("nan"), len(draws)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi), int(len(draws) - len(finite))


def summarise(stacked: np.ndarray, side: str | None = None) -> dict:
    """point (row 0), 95% interval and NaN draws of rows 1..B and, when `side` is given, the protocol's bootstrap
    p-value (`power.p_values`; NaN when a draw is NaN)."""
    stacked = np.asarray(stacked, float)
    lo, hi, nan_draws = interval(stacked[1:])
    out = {"point": float(stacked[0]), "lo": lo, "hi": hi, "nan_draws": nan_draws}
    if side is not None:
        out["p"] = float(pw.p_values(stacked[1:, None], [side])[0])
    return out


def q_values(p: np.ndarray, dependent: bool = False) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values over the finite p-values (Benjamini-Yekutieli when `dependent`); NaN stays
    NaN. q_(i) = min over j >= i of min(1, c m p_(j) / j), with c = 1 or the harmonic sum of 1..m."""
    p = np.asarray(p, float)
    tested = np.flatnonzero(np.isfinite(p))
    out = np.full(len(p), np.nan)
    m = len(tested)
    if not m:
        return out
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    order = tested[np.argsort(p[tested], kind="stable")]
    raw = c * m * p[order] / np.arange(1, m + 1)
    out[order] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return out


def xg_points(effect: float, mean_probability: float) -> float:
    """A logit effect in xG points at the category mean: 100 [sigmoid(logit(p) + effect) - p] with p the level's mean
    main-scale vendor probability."""
    return float(XG_POINTS * (sigmoid(logit(np.float64(mean_probability)) + effect) - mean_probability))


# ------------------------------------------------------------------ correlations
def pearson(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(W,) weighted Pearson correlation; `a` and `b` are (n,) or (W, n)."""
    return pw.weighted_pearson(np.broadcast_to(a, weights.shape), np.broadcast_to(b, weights.shape), weights)


# ------------------------------------------------------------------ label metrics
def label_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray, auc: bool = True) -> dict[str, np.ndarray]:
    """Per row of shot weights (W, n), for labels `y` (n,) and probabilities `p` (n,) or (W, n): shots, goals (O),
    expected goals (E), O - E, O / E, mean log-loss and (if `auc`) the weighted AUC (`ladder._auc_rows`; ties count
    one half). NaN where a row has no goal, no non-goal or no expected goal. `_auc_rows` is the ladder's per-draw AUC,
    tested against roc_auc_score (tests/test_vendor_refit.py); ladder.py is in the xg-vendor pre-registration's code
    closure, so it is reused as it is rather than made public."""
    w = np.asarray(weights, float)
    p = np.clip(np.broadcast_to(p, w.shape), EPS, 1 - EPS)
    y = np.asarray(y, float)
    n, observed, expected = w.sum(1), w @ y, (w * p).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"shots": n, "observed": observed, "expected": expected, "o_minus_e": observed - expected,
               "o_over_e": observed / expected, "log_loss": (w * ld.per_shot_log_loss(y, p)).sum(1) / n}
        if auc:
            both = (observed > 0) & (n - observed > 0)
            out["auc"] = np.where(both, ld._auc_rows(y, np.ascontiguousarray(p), w), np.nan)
    return out


# ------------------------------------------------------------------ dry run
def permute_within_matches(labels: pd.DataFrame, matches: np.ndarray, seed: int) -> pd.DataFrame:
    """The label rows shuffled within each match (one `default_rng(seed)`, matches in sorted order); the key column,
    if any, stays in place, so every match keeps its labels' multiset and loses their link to its shots."""
    rng = np.random.default_rng(seed)
    matches = np.asarray(matches)
    order = np.arange(len(labels))
    for match in np.unique(matches):
        rows = np.flatnonzero(matches == match)
        order[rows] = rows[rng.permutation(len(rows))]
    values = labels.drop(columns=["key"], errors="ignore").iloc[order].reset_index(drop=True)
    values.index = labels.index
    return pd.concat([labels[["key"]], values], axis=1) if "key" in labels.columns else values
