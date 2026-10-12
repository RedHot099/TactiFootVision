"""H3 decoding probes: from a pooled frame embedding to the shot's positional quantities (xg-vendor ticket 09).

A pure function of the evaluation table (`xgpoc.vendor.evaltable`), the embeddings it references and the probe targets
(`xgpoc.vendor.probe_targets`, by `statsbomb_event_id`). Labels are not read, so the probes run on a label-free table.
The method is the prototype's (`s3_probes.py`), ported:

Probe       dual ridge (one eigendecomposition of X X^T serves every alpha) on standardised features, alpha from
            `ALPHAS`, intercept = training mean of the target.
Nesting     outer `GroupKFold(N_OUTER)` by match gives the out-of-fold prediction of every row; inside each outer
            training set an inner `GroupKFold(N_INNER)` by match picks alpha per target by the summed squared error
            on the inner validation rows (ties to the smaller alpha). Standardisation (mean, sd + `STD_EPS`) is fitted
            on the rows a model is trained on: the inner training rows for the inner fits, the outer training rows
            for the outer fit. The outer test rows enter no fit, no standardisation and no alpha choice.
Metric      R^2 of the out-of-fold predictions for continuous targets, AUC of the ridge score for binary ones
            (`probe_targets.BINARY_TARGETS`); Spearman's rho is reported next to R^2.
Control     `CONTROL`: the distance target permuted across the probe's rows (`CONTROL_SEED`); it must not decode.
            `control_gate` is the hard gate: an encoder whose control decodes above the threshold (the pre-registered
            `h3_control_max_r2` in dry-run and confirmatory runs, `CONTROL_MAX_R2` in development runs; point estimate,
            on any of its rows of the subset "all") fails, and every H3 reading of that encoder is invalid.
Encoders    the table's embedding columns; the fold-0 seed-0 G-combo (`emb_g_fold0`) is pre-specified, the fold-1 model
            (`emb_g_fold1`) is the sensitivity check, frozen DINOv2 (`emb_dinov2`) the reference. Every encoder is
            probed on the rows it covers; each G-combo encoder and the reference are also probed on the rows both cover,
            and the difference G-combo minus reference is computed there.
Intervals   match bootstrap with the shared draws (`xgpoc.vendor.draws`): probes are not refitted in draws, each metric
            is recomputed from the out-of-fold predictions with the draw's match weights; differences use the same
            draws for both encoders. 95% percentile intervals; draws without both classes are skipped and counted.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from xgpoc.probe.readability import interval, weighted_auc
from xgpoc.vendor.draws import TableDraws
from xgpoc.vendor.probe_targets import BINARY_TARGETS, TARGET_COLUMNS

ALPHAS = tuple(float(a) for a in np.logspace(0, 6, 13))
N_OUTER, N_INNER = 5, 3
STD_EPS = 1e-6
CONTROL, CONTROL_SOURCE, CONTROL_SEED = "dist_shuffled", "dist", 0
CONTROL_MAX_R2 = 0.0   # development runs; dry-run and confirmatory runs take the pre-registration's h3_control_max_r2
CONTROL_FAILED_EXIT = 3   # exit status of scripts/vendor_probes.py when the control gate fails (outputs written)
PROBE_TARGETS = (CONTROL, *TARGET_COLUMNS)
REFERENCE = "emb_dinov2"
CHUNK = 500   # bootstrap draws per block in the R^2 computation


def metric_name(target: str) -> str:
    """R2 for continuous targets, AUC for binary ones."""
    return "AUC" if target in BINARY_TARGETS else "R2"


# ------------------------------------------------------------------ the probe
def standardise(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Scale `train` and `others` with the mean and sd of `train` alone."""
    mean, sd = train.mean(0), train.std(0) + STD_EPS
    return [(a - mean) / sd for a in (train, *others)]


def ridge_paths(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Dual ridge predictions for every alpha: (len(ALPHAS), n_eval, targets). Inputs already standardised."""
    mean = y_train.mean(0)
    s, u = np.linalg.eigh(x_train @ x_train.T)
    uty = u.T @ (y_train - mean)
    ku = (x_eval @ x_train.T) @ u
    return np.stack([ku @ (uty / (s + a)[:, None]) for a in ALPHAS]) + mean


def outer_folds(groups: np.ndarray) -> np.ndarray:
    """Outer fold of every row (match-grouped, deterministic)."""
    fold = np.empty(len(groups), int)
    for k, (_, test) in enumerate(GroupKFold(N_OUTER).split(groups, groups=groups)):
        fold[test] = k
    return fold


def select_alphas(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Index into ALPHAS per target, by inner match-grouped CV on these rows (standardised inside each inner fold)."""
    sse = np.zeros((len(ALPHAS), y.shape[1]))
    for train, valid in GroupKFold(N_INNER).split(x, groups=groups):
        x_train, x_valid = standardise(x[train], x[valid])
        sse += ((ridge_paths(x_train, y[train], x_valid) - y[valid]) ** 2).sum(1)
    return np.argmin(sse, axis=0)


def probe_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions (n, targets), chosen alpha per outer fold and target (N_OUTER, targets), outer fold per row.
    `y` must be finite; rows with a missing target belong to another call."""
    x, y = np.asarray(x, float), np.asarray(y, float).reshape(len(x), -1)
    if not np.isfinite(y).all():
        raise ValueError("probe targets must be finite on the probe's rows")
    fold = outer_folds(groups)
    pred = np.empty_like(y)
    alphas = np.empty((N_OUTER, y.shape[1]))
    for k in range(N_OUTER):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        chosen = select_alphas(x[train], y[train], groups[train])
        x_train, x_test = standardise(x[train], x[test])
        paths = ridge_paths(x_train, y[train], x_test)
        pred[test] = paths[chosen, :, np.arange(y.shape[1])].T
        alphas[k] = np.asarray(ALPHAS)[chosen]
    return pred, alphas, fold


# ------------------------------------------------------------------ metrics
def weighted_r2(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """R^2 with row weights (B, n); a scalar array without weights."""
    w = np.ones((1, len(y))) if weights is None else weights
    out = []
    for lo in range(0, len(w), CHUNK):
        wc = w[lo:lo + CHUNK]
        total = wc.sum(1)
        mean = wc @ y / total
        sst = wc @ (y ** 2) - total * mean ** 2
        out.append(1 - wc @ ((y - pred) ** 2) / sst)
    res = np.concatenate(out)
    return res[0] if weights is None else res


def metric_with_draws(target: str, y: np.ndarray, pred: np.ndarray, rows: np.ndarray, draws: TableDraws) -> tuple[float, np.ndarray]:
    """Point metric and its (B,) bootstrap draws; `rows` are the table rows of y and pred."""
    if target in BINARY_TARGETS:
        match = draws.match[rows]
        return float(weighted_auc(pred, y, match)), weighted_auc(pred, y, match, draws.counts)
    return float(weighted_r2(y, pred)), weighted_r2(y, pred, draws.weights(rows))


# ------------------------------------------------------------------ runs over the table
@dataclass
class ProbeRun:
    """Out-of-fold predictions of one encoder on one row set, per target: table rows, target values, predictions."""

    encoder: str
    rows: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    pred: dict[str, np.ndarray]
    fold: dict[str, np.ndarray]
    alpha_median: dict[str, float]


def control_target(dist: np.ndarray) -> np.ndarray:
    """The negative control: the distance values permuted across the given rows."""
    return np.random.default_rng(CONTROL_SEED).permutation(dist)


def run_probe(encoder: str, x: np.ndarray, targets: pd.DataFrame, groups: np.ndarray, rows: np.ndarray) -> ProbeRun:
    """Probe every target (and the control) from `x` (one row per entry of `rows`, the table positions).
    Targets missing on some rows are probed on the rows that have them; targets with the same rows share one fit."""
    values = targets.iloc[rows].copy()
    values[CONTROL] = np.nan
    has_source = values[CONTROL_SOURCE].notna().to_numpy()
    values.loc[has_source, CONTROL] = control_target(values.loc[has_source, CONTROL_SOURCE].to_numpy(float))
    run = ProbeRun(encoder, {}, {}, {}, {}, {})
    masks: dict[bytes, list[str]] = {}
    for target in PROBE_TARGETS:
        masks.setdefault(values[target].notna().to_numpy().tobytes(), []).append(target)
    for key, names in masks.items():
        keep = np.frombuffer(key, bool)
        if not keep.any():
            continue
        y = values.loc[keep, names].to_numpy(float)
        pred, alphas, fold = probe_predictions(x[keep], y, groups[rows[keep]])
        for j, name in enumerate(names):
            run.rows[name], run.y[name], run.pred[name], run.fold[name] = rows[keep], y[:, j], pred[:, j], fold
            run.alpha_median[name] = float(np.median(alphas[:, j]))
    return run


def metric_rows(run: ProbeRun, subset: str, coverage: str, draws: TableDraws, groups: np.ndarray) -> tuple[list[dict], dict]:
    """One result row per target of the run, plus the bootstrap draws per target (for differences)."""
    rows, boot = [], {}
    for target in PROBE_TARGETS:
        if target not in run.rows:
            continue
        r, y, pred = run.rows[target], run.y[target], run.pred[target]
        point, draw = metric_with_draws(target, y, pred, r, draws)
        lo, hi, skipped = interval(draw)
        boot[target] = (point, draw)
        binary = target in BINARY_TARGETS
        rows.append({"encoder": run.encoder, "subset": subset, "coverage": coverage, "target": target, "metric": metric_name(target),
                     "n": len(r), "matches": int(len(np.unique(groups[r]))), "positives": int(y.sum()) if binary else None,
                     "value": point, "lo": float(lo), "hi": float(hi), "skipped_draws": skipped,
                     "spearman": None if binary else float(spearmanr(y, pred)[0]), "alpha_median": run.alpha_median[target],
                     "negative_control": target == CONTROL})
    return rows, boot


@dataclass
class ProbeResults:
    """The three outputs of `probe_table`."""

    metrics: pd.DataFrame       # one row per encoder x subset x coverage x target
    differences: pd.DataFrame   # G-combo minus reference on the rows both cover, per subset and target
    predictions: pd.DataFrame   # out-of-fold predictions, long format


class Coverage(NamedTuple):
    """The table rows an encoder has an embedding for (`mask`), each row's position in `matrix` (`lookup`, -1 = none)."""

    mask: np.ndarray
    lookup: np.ndarray
    matrix: np.ndarray


def probe_table(table: pd.DataFrame, targets: pd.DataFrame, embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
                draws: TableDraws, subsets: dict[str, np.ndarray] | None = None, reference: str = REFERENCE) -> ProbeResults:
    """Run every probe of H3 (module docstring).

    `targets`: `probe_targets.targets_for_table` (rows aligned with `table`). `embeddings[column] = (positions, matrix)`
    as returned by `evaltable.load_embeddings`. `subsets[name]`: boolean mask over the table rows (default: all rows)."""
    if reference not in embeddings:
        raise ValueError(f"reference encoder {reference} has no embeddings")
    if len(targets) != len(table):
        raise ValueError("targets are not aligned with the table")
    groups = table.match.to_numpy()
    subsets = subsets or {"all": np.ones(len(table), bool)}
    covered = {}
    for column, (positions, matrix) in embeddings.items():
        if len(positions) != len(matrix):
            raise ValueError(f"{column}: positions and matrix disagree")
        mask = np.zeros(len(table), bool)
        mask[positions] = True
        lookup = np.full(len(table), -1)
        lookup[positions] = np.arange(len(positions))
        covered[column] = Coverage(mask, lookup, matrix)
    cache: dict[tuple[str, bytes], ProbeRun] = {}

    def probe(column: str, mask: np.ndarray) -> ProbeRun | None:
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            return None
        key = (column, rows.tobytes())
        if key not in cache:
            cov = covered[column]
            cache[key] = run_probe(column, np.asarray(cov.matrix[cov.lookup[rows]], float), targets, groups, rows)
        return cache[key]

    metric_out, diff_out, pred_out = [], [], []

    def record(run: ProbeRun, subset: str, coverage: str) -> dict:
        rows, boot = metric_rows(run, subset, coverage, draws, groups)
        metric_out.extend(rows)
        for target, r in run.rows.items():
            pred_out.append(pd.DataFrame({"key": table.key.to_numpy()[r], "encoder": run.encoder, "subset": subset, "coverage": coverage,
                                          "target": target, "y": run.y[target], "prediction": run.pred[target], "outer_fold": run.fold[target]}))
        return boot

    for subset, in_subset in subsets.items():
        own = {}   # encoder -> (run on its own coverage, {target: (point, draws)})
        for column in embeddings:
            run = probe(column, covered[column].mask & in_subset)
            if run is not None:
                own[column] = (run, record(run, subset, "own"))
        for column in embeddings:
            if column == reference:
                continue
            shared = covered[column].mask & covered[reference].mask & in_subset
            runs = {c: probe(c, shared) for c in (column, reference)}
            if runs[column] is None:
                continue
            boots = {}
            for c, run in runs.items():   # reuse the own-coverage numbers when the shared rows are the same rows
                same = c in own and own[c][0] is run
                boots[c] = own[c][1] if same else record(run, subset, f"shared_with_{column}")
            for target in PROBE_TARGETS:
                if target not in boots[column] or target not in boots[reference]:
                    continue
                (g, g_draws), (ref, ref_draws) = boots[column][target], boots[reference][target]
                if not np.array_equal(runs[column].rows[target], runs[reference].rows[target]):
                    raise AssertionError("the two encoders were probed on different rows")
                lo, hi, skipped = interval(g_draws - ref_draws)
                diff_out.append({"encoder": column, "reference": reference, "subset": subset, "target": target, "metric": metric_name(target),
                                 "n": len(runs[column].rows[target]), "encoder_value": g, "reference_value": ref, "difference": g - ref,
                                 "lo": float(lo), "hi": float(hi), "skipped_draws": skipped, "negative_control": target == CONTROL})
    return ProbeResults(pd.DataFrame(metric_out), pd.DataFrame(diff_out),
                        pd.concat(pred_out, ignore_index=True) if pred_out else pd.DataFrame())


def control_gate(metrics: pd.DataFrame, threshold: float, subset: str = "all") -> dict:
    """The gate of the shuffled-target control over `probe_table`'s metrics: per encoder, the largest control R^2 of its
    rows of `subset` (own and shared coverage) and whether it stays at or below `threshold`; `passed` only when every
    encoder passes."""
    control = metrics[metrics.negative_control.astype(bool) & metrics.subset.eq(subset)]
    if control.empty:
        raise ValueError(f"no negative-control rows in the subset {subset!r}")
    encoders = {encoder: {"max_r2": float(rows.value.max()), "passed": bool(rows.value.max() <= threshold)}
                for encoder, rows in control.groupby("encoder", sort=True)}
    return {"threshold": threshold, "subset": subset, "encoders": encoders, "passed": all(e["passed"] for e in encoders.values())}


def apply_thresholds(metrics: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Add the pre-registered threshold of each target and whether the point estimate and the interval's lower end reach
    it (which of the two decides is the pre-registration's rule, not this function's)."""
    unknown = set(thresholds) - set(TARGET_COLUMNS)
    if unknown:
        raise ValueError(f"thresholds for unknown targets: {sorted(unknown)}")
    out = metrics.copy()
    out["threshold"] = out.target.map(thresholds)
    has = out.threshold.notna()
    out["point_meets"] = (out.value >= out.threshold).where(has)
    out["lower_meets"] = (out.lo >= out.threshold).where(has)
    return out
