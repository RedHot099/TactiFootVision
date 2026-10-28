"""Probes of xg-dino-info on the automatic attributes Xa, the header labels and `z_C - z_B` (ticket 12; the pinned
protocol's `probes` families bind, council C3.7).

The fitting is `xgpoc.dino.curves.run` (the host folds, nested layer selection, C2 thresholds, distance-only reference,
shuffled control, C1 routing); this module holds what is new:

    targets        one probe table per offset (`probe_table`): the Xa columns, `set_piece` (any SoccerNet flag), the header
                   labels (caption / GPT provenance: probes only, never a model design) and the OOF `z_C - z_B`. A column is
                   binary when its finite values are all 0 or 1 (AUC of the ridge output), else R².
    rows           a target is probed on the rows where it is present and the automatic t0 distance is finite (the distance
                   spline is the reference). Near-complete targets (present on >= `MAIN_SHARE` of the rows) share the
                   intersection of their present rows, so one eigendecomposition serves them all; the others keep their own.
    weights        the header probes are fitted unweighted; their metrics and match-bootstrap intervals carry the stratum
                   weights of the GPT sample (`metric_draws_w`), unweighted beside, and are reported by outcome.
    families       BH at q = 0.05 over the targets of one protocol family at one encoder and offset, on the one-sided
                   bootstrap p of the gain over the distance-only probe (`bh`).
    z_C - z_B      alone (embedding only) and beyond geometry (`geometry_embedding` against `geometry`, curves.fold_fit).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg

MAIN_SHARE = 0.97
Q = 0.05
HEADER = ("header_gpt", "header_caption")
ZDIFF = "z_C_minus_z_B"
SET_PIECE = ("sn_direct_fk", "sn_indirect_fk_3s", "sn_corner_20s")
OUTCOMES = {"all": None, "goal": 1, "no_goal": 0}
CHUNK = 250


def set_piece(attributes: pd.DataFrame) -> pd.Series:
    """1 when any flag of `SET_PIECE` is 1, 0 when all are 0, missing when a flag is missing and none is 1."""
    flags = attributes[list(SET_PIECE)]
    return pd.Series(np.where(flags.eq(1).any(axis=1), 1.0, np.where(flags.isna().any(axis=1), np.nan, 0.0)), index=flags.index)


def header_targets(gpt: pd.DataFrame, caption: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """`header_gpt` (1 head, 0 foot, unclear / not sampled missing) and `header_caption` (1 header, 0 foot, unknown
    missing) on `index`."""
    out = pd.DataFrame(index=index)
    out["header_gpt"] = gpt.set_index("shot_id").header_gpt.reindex(index)
    out["header_caption"] = caption.set_index("shot_id").header_caption.map({"header": 1.0, "foot": 0.0}).reindex(index)
    return out


def z_difference(probabilities: pd.DataFrame) -> pd.Series:
    """logit p_C - logit p_B of the frozen OOF, by shot id."""
    p = probabilities.set_index("shot_id")
    logit = lambda v: np.log(v / (1.0 - v))     # noqa: E731
    return (logit(p.p_C) - logit(p.p_B)).rename(ZDIFF)


def probe_table(attributes: pd.DataFrame, labels: pd.DataFrame, z_diff: pd.Series, distance: pd.Series,
                columns: Sequence[str], index: pd.Index) -> pd.DataFrame:
    """The probe table on `index` (`distance` first, then `columns`), kinds registered, rows restricted as in the module
    docstring, tagged by `source_of(columns)`. `columns` may name Xa columns, `set_piece`, `HEADER` and `ZDIFF`; a header
    label is probed alone (its provenance is the table's)."""
    full = attributes.assign(set_piece=set_piece(attributes)).join(labels, how="left").join(z_diff, how="left").reindex(index)
    table = pd.concat([distance.reindex(index).rename("distance"), full[list(columns)]], axis=1)
    keep = table.distance.notna()
    table.loc[~keep, list(columns)] = np.nan
    data_cols = [c for c in columns if c != ZDIFF and c not in HEADER]
    share = table.loc[:, data_cols].notna().mean()
    main = [c for c in data_cols if share[c] >= MAIN_SHARE]
    common = table[main].notna().all(axis=1)
    table.loc[~common, main] = np.nan
    tg.register_binary(table, columns)
    return tg.tagged(table, source_of(columns))


def source_of(targets: Sequence[str]) -> str:
    """Provenance of a set of probe targets: `gpt` / `caption` for a header label, else `automatic`."""
    if "header_gpt" in targets and "header_caption" in targets:
        raise ValueError("a probe table holds one label source")
    return "gpt" if "header_gpt" in targets else "caption" if "header_caption" in targets else "automatic"


# ------------------------------------------------------------------ weighted read-out
def weighted_auc_rows(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUC of `score` for y with row weights w (B, n) (ties count one half); NaN where a draw lacks a class."""
    order = np.argsort(score, kind="stable")
    s, pos, w = score[order], y[order] == 1, w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    out = []
    for lo in range(0, len(w), CHUNK):
        wc = w[lo:lo + CHUNK]
        wp, wn = wc * pos, wc * ~pos
        gp, gn = np.add.reduceat(wp, starts, 1), np.add.reduceat(wn, starts, 1)
        below = np.cumsum(gn, 1) - gn
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, (gp * (below + 0.5 * gn)).sum(1) / den, np.nan))
    return np.concatenate(out)


def metric_draws_w(target: str, y: np.ndarray, pred: np.ndarray, rows: np.ndarray, d: pr.Draws,
                   row_weight: np.ndarray | None) -> dict[str, tuple[float, np.ndarray]]:
    """`probes.metric_draws` with extra per-row weights (population-indexed; None: the plain match bootstrap)."""
    if row_weight is None:
        return pr.metric_draws(target, y, pred, rows, d)
    y, pred, rw = np.asarray(y, float), np.asarray(pred, float), np.asarray(row_weight, float)[rows]
    w = d.weights[:, rows] * rw
    if tg.kind(target) == "binary":
        return {"AUC": (float(weighted_auc_rows(pred, y, rw[None])[0]), weighted_auc_rows(pred, y, w))}
    mean = lambda v, wt: (wt @ v) / wt.sum(1)   # noqa: E731
    err = np.abs(y - pred)
    from xgpoc.vendor.probes import weighted_r2
    return {"R2": (float(weighted_r2(y, pred, rw[None])[0]), weighted_r2(y, pred, w)),
            "MAE": (float(mean(err, rw[None])[0]), mean(err, w))}


def target_metrics_w(t: pr.TargetRun, d: pr.Draws, keys: Sequence[str], row_weight: np.ndarray | None = None,
                     mask: np.ndarray | None = None) -> dict:
    """`probes.target_metrics` with row weights; `mask` selects rows of `t.rows`."""
    m = np.ones(len(t.rows), bool) if mask is None else mask
    out = {}
    for key in (k for k in keys if k in t.oof):
        y = t.oof["shuffled_target"] if key == "embedding_shuffled" else t.y
        out[key] = metric_draws_w(t.target, y[m], t.oof[key][m], t.rows[m], d, row_weight)
    return out


def prevalence(t: pr.TargetRun, d: pr.Draws, row_weight: np.ndarray | None, mask: np.ndarray) -> dict:
    """Share of positives of a binary target on the masked rows (weighted by `row_weight`) with its bootstrap interval."""
    rows, y = t.rows[mask], t.y[mask]
    rw = np.ones(len(rows)) if row_weight is None else np.asarray(row_weight, float)[rows]
    w = d.weights[:, rows] * rw
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ y) / w.sum(1)
    return pr.summary(float((rw @ y) / rw.sum()), boot)


# ------------------------------------------------------------------ tests over a family
def one_sided_p(draws: np.ndarray) -> float:
    """(1 + #{draws <= 0}) / (n + 1) over the finite draws: the bootstrap p that a gain is positive."""
    v = np.asarray(draws, float)
    v = v[np.isfinite(v)]
    return float((1 + (v <= 0).sum()) / (len(v) + 1)) if len(v) else float("nan")


def bh(p: Sequence[float], q: float = Q) -> list[bool]:
    """Benjamini-Hochberg rejections at level q (NaN p never rejects)."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    passed = [i for i, j in enumerate(order, 1) if p[j] <= q * i / m]
    reject = np.zeros(len(p), bool)
    if passed:
        reject[order[:max(passed)]] = True
    return reject.tolist()


def geometry_decision(md: dict, rules: pr.Rules = pr.DEFAULT_RULES) -> dict:
    """The C2 gain rule with the geometry design in place of the distance spline (`geometry_embedding` against
    `geometry`, continuous targets)."""
    gain = md["geometry_embedding"]["R2"][0] - md["geometry"]["R2"][0]
    s = pr.summary(gain, md["geometry_embedding"]["R2"][1] - md["geometry"]["R2"][1])
    return {"gain": s, "p": one_sided_p(md["geometry_embedding"]["R2"][1] - md["geometry"]["R2"][1]),
            "encodes_beyond_geometry": bool(s["lo"] > 0 and gain >= rules.gain_min["continuous"])}
