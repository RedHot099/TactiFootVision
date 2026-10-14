"""Read-out statistics of xg-stack (ticket 01, council C5).

Scores (`scores`)
    out-of-fold log-loss (probabilities clipped to [1e-6, 1 - 1e-6], `CLIP`), AUC, Brier, and the calibration slope
    and intercept: the logistic regression of y on logit(p) with C = 1e6 (`PlattScaler`, effectively unpenalised).
Paired match bootstrap (`match_weights`, `paired_delta`)
    ΔLL = LL(arm) - LL(R), the mean over shots of the per-shot log-loss difference. 2,000 draws (seed 0) of the 328
    matches with replacement (`testrule.match_weights`: each shot weighted by its match's draw count, the sequence of
    `readability.match_draw_counts`); each draw is the weighted mean of the per-shot differences. Reported: the point,
    the 95% percentile interval, the paired one-sided p of C5, (1 + #{Δ_b >= 0}) / (B + 1), and the one-sided 95%
    bound on the improvement, the 5th percentile of the draws (`bound_one_sided_95`): with 95% confidence ΔLL is at
    least this, so a null reads "no improvement larger than |bound|". The 95th percentile (`upper_one_sided_95`, below
    0 when the improvement is significant one-sided) is reported beside it.
Holm (`holm`) over the confirmatory arms {G, GS} at α = 0.05; `readings` gives each confirmatory arm's "detectable
    improvement" (Holm-rejected) and "practical candidate" (rejected and point ΔLL <= -0.002).
Categories (`CATEGORIES`, `category_table`, descriptive)
    out-of-fold O, E, O/E per arm and ΔLL against R with its 95% interval from the same draws restricted to the
    category's shots, inside detector-defined categories; a shot whose defining input is missing is outside the
    category's population (it is neither in nor out).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.detector.conventions import LENGTH, PITCH
from xgpoc.detector.testrule import match_weights, shot_log_loss
from xgpoc.eval.calibration import EPS, PlattScaler, logit
from xgpoc.eval.metrics import evaluate
from xgpoc.stack import inputs as si

CLIP = EPS
N_BOOT = 2000
BOOT_SEED = 0
ALPHA = 0.05
PRACTICAL = -0.002          # the A3 trigger
INTERVAL = 0.95
ONE_SIDED = 0.95

BOX_LINE = LENGTH - PITCH.penalty_box_length        # 105 - 16.5 = 88.5 m


@dataclass(frozen=True)
class Category:
    """A detector-defined category: its defining column, the rule's text and the rule on the column's finite values."""

    column: str
    rule: str
    inside: Callable[[np.ndarray], np.ndarray]


CATEGORIES = {
    "e3_outside_box": Category(si.col("shot_x", 0), f"shot_x@0 < {BOX_LINE}", lambda x: x < BOX_LINE),
    "e3_angle_lt_15deg": Category(si.col("angle", 0), "angle@0 < 15 degrees", lambda x: x < np.radians(15.0)),
    "e3_dist_lt_6m": Category(si.col("dist", 0), "dist@0 < 6 m", lambda x: x < 6.0),
    "sn_corner_20s": Category("sn_corner_20s", "sn_corner_20s = 1", lambda x: x == 1.0),
    "sn_direct_fk": Category("sn_direct_fk", "sn_direct_fk = 1", lambda x: x == 1.0),
}


def category_masks(table: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (defined, inside) boolean arrays over the population (module docstring)."""
    out = {}
    for name, category in CATEGORIES.items():
        x = table[category.column].to_numpy(float)
        defined = np.isfinite(x)
        inside = np.zeros(len(x), dtype=bool)
        inside[defined] = category.inside(x[defined])
        out[name] = (defined, inside)
    return out


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Log-loss, AUC, Brier, mean p and the calibration slope and intercept of logit(p)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    m = evaluate(y, p)
    platt = PlattScaler().fit(logit(np.clip(p, CLIP, 1 - CLIP)), y)
    return {"n": m["n"], "goals": int(y.sum()), "log_loss": float(shot_log_loss(y, p, CLIP).mean()), "auc": m["auc"],
            "brier": m["brier"], "mean_p": float(p.mean()), "cal_slope": platt.a, "cal_intercept": platt.b}


def bootstrap_weights(games: np.ndarray, draws: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(draws, shots) match-bootstrap weights (module docstring)."""
    return match_weights(np.asarray(games), draws, seed)


def boot_means(weights: np.ndarray, d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(draws,) weighted means of `d` per draw, over the shots of `mask` (NaN where a draw holds none of them)."""
    w = weights if mask is None else weights[:, mask]
    d = d if mask is None else d[mask]
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (w @ d) / total, np.nan)


def summarise(point: float, boot: np.ndarray) -> dict:
    """Point, 95% interval, one-sided 95% bounds, one-sided p (C5) and the draws' SD, over the finite draws."""
    boot = boot[np.isfinite(boot)]
    tail = 100 * (1 - INTERVAL) / 2
    return {"delta_ll": float(point), "lo95": float(np.percentile(boot, tail)), "hi95": float(np.percentile(boot, 100 - tail)),
            "bound_one_sided_95": float(np.percentile(boot, 100 * (1 - ONE_SIDED))),
            "upper_one_sided_95": float(np.percentile(boot, 100 * ONE_SIDED)),
            "p_one_sided": float((1 + np.sum(boot >= 0)) / (len(boot) + 1)), "boot_sd": float(boot.std(ddof=1)),
            "draws": int(len(boot))}


def paired_delta(y: np.ndarray, p_arm: np.ndarray, p_ref: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """ΔLL = LL(arm) - LL(reference) over the shots of `mask` (all by default), with the paired match bootstrap."""
    d = shot_log_loss(y, p_arm, CLIP) - shot_log_loss(y, p_ref, CLIP)
    point = d.mean() if mask is None else d[mask].mean()
    return summarise(point, boot_means(weights, d, mask))


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm's step-down rejections at `alpha` (ties in input order)."""
    order = sorted(p_values, key=lambda k: p_values[k])
    rejected, m = {k: False for k in p_values}, len(p_values)
    for i, k in enumerate(order):
        if p_values[k] > alpha / (m - i):
            break
        rejected[k] = True
    return rejected


def readings(deltas: dict[str, dict], arms: tuple[str, ...]) -> dict[str, dict]:
    """Holm over `arms` on their one-sided p; per arm: detectable improvement and practical candidate (C5)."""
    rejected = holm({a: deltas[a]["p_one_sided"] for a in arms})
    return {a: {"holm_rejected": rejected[a], "detectable_improvement": rejected[a],
                "practical_candidate": bool(rejected[a] and deltas[a]["delta_ll"] <= PRACTICAL)} for a in arms}


def category_table(table: pd.DataFrame, y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray,
                   reference: str = "R") -> pd.DataFrame:
    """Per category and arm (descriptive): n, O, E, O/E, and ΔLL against `reference` with its 95% interval."""
    rows = []
    for name, (defined, inside) in category_masks(table).items():
        observed = int(y[inside].sum())
        for arm, p in probabilities.items():
            expected = float(p[inside].sum())
            row = {"category": name, "rule": CATEGORIES[name].rule, "arm": arm, "n_defined": int(defined.sum()),
                   "n": int(inside.sum()), "observed": observed, "expected": expected,
                   "o_over_e": observed / expected if expected > 0 else np.nan}
            if arm != reference and inside.any():
                delta = paired_delta(y, p, probabilities[reference], weights, inside)
                row |= {k: delta[k] for k in ("delta_ll", "lo95", "hi95")}
            rows.append(row)
    return pd.DataFrame(rows)
