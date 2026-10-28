"""Agreement of three geometry sources on the paired shots (ticket 07; spec §Agreement, councils C1 and C2, which win).

Sources per feature, all frozen before StatsBomb is read:

    detector   the automatic target of the feet_t0 E3 frame (`targets.automatic_targets`), the probes' training target
    probe      the embedding-only ridge probe at the best layer chosen on the 328 matches (ticket 06), fitted on the
               automatic targets only and read on the paired shots (mean of the five fold probes)
    distance   the distance-only reference of the same probe (spline on the detector distance), the floor a probe
               reading must beat

and the reference, StatsBomb (`targets.statsbomb_targets`, evaluation only). Nothing here is fitted to StatsBomb:
every function takes frozen predictions and StatsBomb values and only scores them (`check_sources` refuses a
StatsBomb-tagged table as a prediction source).

Features and tolerances (C2, frozen in the ablation protocol's `agreement` block; |error| <= τ is "right" for both
sources): distance 2.0 m, angle 5° (the targets are in radians; errors are read in degrees), near_opp_m 1.5 m, opp_3m and
def_cone exact counts (τ = 0; a probe's count reading is its prediction rounded to the nearest non-negative integer),
gk_depth_m 1.0 m, gk_lateral_m (absolute) 1.0 m.

Cells (C1/C2): the detector is "missing" (no value, or a goalkeeper feature without a detected goalkeeper: the E3 proxy
is never "wrong"), "right" (|e| <= τ), "wrong" (|e| > τ and above the 75th percentile of its own |e| on the rows with
a detector value) or "between" (> τ, not above the 75th percentile). The probe is "right" when |e| <= τ. The cell
"detector wrong, DINO right" counts a shot only if the probe also beats the distance-only reference there
(|e_probe| < |e_distance|, on the same reading). Permutation null of that count: strata = distance deciles of the
paired shots (a stratum below 100 merged with its lower neighbour), the probe's success flag (right and beating the
reference) permuted within stratum with the detector status fixed, one-sided p = (1 + #{null >= observed}) / (N + 1).

Combination rules (C1, fixed, no weights): "mean" = (detector + probe) / 2 where the detector exists, the probe where it
does not; "fill" = the detector, the probe where the detector is missing. References: the detector alone (rows with a
detector value) and the same two rules with the distance-only reference in place of the probe.

Timing bands (C2): |pairing residual| <= 50 / 50-200 / > 200 ms; a band below 100 shots merges with its lower
neighbour (the lowest band with its upper one).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.dino import curves as cv
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg


@dataclass(frozen=True)
class Feature:
    name: str
    tolerance_key: str                     # key in the protocol's agreement tolerances
    scale: float = 1.0                     # target unit -> error unit (radians -> degrees for the angle)
    count: bool = False
    goalkeeper: bool = False


FEATURES = (
    Feature("distance", "distance_m"),
    Feature("angle", "angle_deg", scale=180.0 / np.pi),
    Feature("near_opp_m", "near_opp_m"),
    Feature("opp_3m", "opp_3m_count", count=True),
    Feature("def_cone", "def_cone_count", count=True),
    Feature("gk_depth_m", "gk_depth_m", goalkeeper=True),
    Feature("gk_lateral_m", "gk_lateral_abs_m", goalkeeper=True),
)
BY_NAME = {f.name: f for f in FEATURES}
# C2 (2026-10-27); the protocol's frozen values must equal these
FROZEN_TOLERANCES = {"distance_m": 2.0, "angle_deg": 5.0, "near_opp_m": 1.5, "opp_3m_count": 0, "def_cone_count": 0,
                     "gk_depth_m": 1.0, "gk_lateral_abs_m": 1.0}
WRONG_QUANTILE = 0.75
N_STRATA, MIN_STRATUM = 10, 100
N_PERMUTATIONS, PERMUTATION_SEED = 10_000, 0
BAND_EDGES_MS = (50.0, 200.0)
BAND_NAMES = ("<=50", "50-200", ">200")
MIN_BAND = 100
DETECTOR_STATUS = ("missing", "right", "between", "wrong")


def tolerances(agreement: dict) -> dict[str, float]:
    """τ per feature from the protocol's frozen `agreement` block, checked against C2."""
    frozen = agreement["tolerances"]
    if {k: float(v) for k, v in frozen.items()} != {k: float(v) for k, v in FROZEN_TOLERANCES.items()}:
        raise ValueError(f"protocol tolerances {frozen} differ from C2 {FROZEN_TOLERANCES}")
    return {f.name: float(frozen[f.tolerance_key]) for f in FEATURES}


def check_sources(*tables: pd.DataFrame) -> None:
    """Prediction tables must not be StatsBomb tables (nothing is fitted to, or predicted from, the reference)."""
    for t in tables:
        if t.attrs.get("source") == "statsbomb":
            raise ValueError("a StatsBomb table is the reference, never a prediction source")


def reading(feature: Feature, values) -> np.ndarray:
    """A source's value in error units; counts rounded to the nearest non-negative integer (exact-count tolerance)."""
    v = np.asarray(values, float) * feature.scale
    return np.where(np.isfinite(v), np.rint(np.clip(v, 0, None)), np.nan) if feature.count else v


# ------------------------------------------------------------------ cells
def detector_status(abs_err: np.ndarray, tau: float, quantile: float = WRONG_QUANTILE) -> tuple[np.ndarray, float]:
    """Status per row (`DETECTOR_STATUS`; NaN error = missing) and the detector's own |e| quantile (C1)."""
    e = np.asarray(abs_err, float)
    present = np.isfinite(e)
    q = float(np.quantile(e[present], quantile)) if present.any() else np.nan
    status = np.full(len(e), "missing", object)
    status[present & (e <= tau)] = "right"
    status[present & (e > tau) & (e <= q)] = "between"
    status[present & (e > tau) & (e > q)] = "wrong"
    return status, q


def cells(det_abs: np.ndarray, probe_abs: np.ndarray, dist_abs: np.ndarray, tau: float) -> tuple[pd.DataFrame, dict]:
    """Per-row cell table and counts on rows with a probe value (all rows passed must have a StatsBomb value).
    `det_abs` NaN = detector missing; errors already in reading units."""
    probe_abs, dist_abs = np.asarray(probe_abs, float), np.asarray(dist_abs, float)
    if not np.isfinite(probe_abs).all():
        raise ValueError("the probe reads every paired shot; pass rows with a StatsBomb value only")
    det, q = detector_status(det_abs, tau)
    probe_right = probe_abs <= tau
    beats = probe_abs < dist_abs                         # NaN reference -> False
    rows = pd.DataFrame({"detector": det, "probe_right": probe_right, "probe_beats_distance": beats})
    rows["counted"] = (det == "wrong") & probe_right & beats
    n = {"rows": len(rows), "detector_q75": q, "tau": tau}
    for s in DETECTOR_STATUS:
        for p, name in ((True, "probe_right"), (False, "probe_wrong")):
            n[f"detector_{s}__{name}"] = int(((det == s) & (probe_right == p)).sum())
        n[f"detector_{s}"] = int((det == s).sum())
    n["detector_wrong__probe_right__beats_distance"] = int(rows.counted.sum())
    n["probe_right"] = int(probe_right.sum())
    return rows, n


def distance_strata(distance: np.ndarray, n: int = N_STRATA, min_size: int = MIN_STRATUM) -> np.ndarray:
    """Decile labels (0..n-1, by rank, ties broken by order) with strata below `min_size` merged into the lower
    neighbour (the lowest into its upper one), relabelled 0..m-1. NaN distance -> -1."""
    d = np.asarray(distance, float)
    lab = np.full(len(d), -1)
    ok = np.flatnonzero(np.isfinite(d))
    order = ok[np.argsort(d[ok], kind="stable")]
    lab[order] = np.arange(len(order)) * n // max(len(order), 1)
    labels = sorted(set(lab[ok]))
    while len(labels) > 1:
        sizes = {g: int((lab == g).sum()) for g in labels}
        small = [g for g in labels if sizes[g] < min_size]
        if not small:
            break
        g = small[-1]
        i = labels.index(g)
        target = labels[i - 1] if i > 0 else labels[1]
        lab[lab == g] = target
        labels.remove(g)
    remap = {g: i for i, g in enumerate(sorted(set(lab[ok])))}
    return np.array([remap.get(v, -1) for v in lab])


def permutation_null(detector_wrong: np.ndarray, probe_success: np.ndarray, strata: np.ndarray,
                     n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> dict:
    """One-sided permutation test of #(detector wrong & probe success): the probe flag permuted within stratum, the
    detector flag fixed. Rows with stratum -1 are dropped."""
    keep = np.asarray(strata) >= 0
    dw, ps, st = np.asarray(detector_wrong, bool)[keep], np.asarray(probe_success, bool)[keep], np.asarray(strata)[keep]
    observed = int((dw & ps).sum())
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, np.int64)
    permutable = 0
    for s in np.unique(st):
        idx = np.flatnonzero(st == s)
        flags = ps[idx]
        if 0 < flags.sum() < len(flags) and 0 < dw[idx].sum() < len(idx):
            permutable += len(idx)
        perm = rng.permuted(np.broadcast_to(flags, (n_perm, len(flags))), axis=1)
        null += (perm & dw[idx]).sum(1)
    return {"observed": observed, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p_one_sided": float((1 + (null >= observed).sum()) / (n_perm + 1)), "n_permutations": n_perm,
            "seed": seed, "permuted_rows": int(keep.sum()), "strata": int(len(np.unique(st))),
            "fraction_permutable": float(permutable / max(keep.sum(), 1))}


# ------------------------------------------------------------------ errors and their agreement
def mean_ci(values: np.ndarray, rows: np.ndarray, d: pr.Draws) -> dict:
    """Mean of `values` (on population rows `rows`) with the 95% match-bootstrap interval."""
    v = np.asarray(values, float)
    w = d.weights[:, rows]
    return pr.summary(float(v.mean()), (w @ v) / w.sum(1))


def spearman_ci(a: np.ndarray, b: np.ndarray, rows: np.ndarray, d: pr.Draws) -> dict:
    return pr.summary(float(cv.weighted_spearman(a, b)), cv.weighted_spearman(a, b, d.weights[:, rows]))


def error_table(feature: Feature, sb: np.ndarray, det: np.ndarray, probe: np.ndarray, d: pr.Draws) -> dict:
    """Detector vs probe against StatsBomb on the rows where all three exist (errors in error units): MAE, mean signed
    error, the paired |e| difference (probe - detector), Spearman of signed and absolute errors, each source's rank
    agreement with StatsBomb, and errors after removing each source's own median shift (descriptive)."""
    y, a, b_raw = (np.asarray(v, float) * feature.scale for v in (sb, det, probe))   # the probe's continuous value
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b_raw)
    rows = np.flatnonzero(m)
    ed, ep = a[m] - y[m], b_raw[m] - y[m]
    out = {"feature": feature.name, "n": int(m.sum())}
    if m.sum() < 3:
        return out
    sd, sp = ed - np.median(ed), ep - np.median(ep)
    out |= {
        "detector_mae": mean_ci(np.abs(ed), rows, d), "probe_mae": mean_ci(np.abs(ep), rows, d),
        "probe_minus_detector_mae": mean_ci(np.abs(ep) - np.abs(ed), rows, d),
        "detector_mean_signed": mean_ci(ed, rows, d), "probe_mean_signed": mean_ci(ep, rows, d),
        "detector_median_signed": float(np.median(ed)), "probe_median_signed": float(np.median(ep)),
        "spearman_signed_errors": spearman_ci(ed, ep, rows, d),
        "spearman_abs_errors": spearman_ci(np.abs(ed), np.abs(ep), rows, d),
        "detector_rank_vs_sb": spearman_ci(a[m], y[m], rows, d), "probe_rank_vs_sb": spearman_ci(b_raw[m], y[m], rows, d),
        "shift_removed_detector_mae": mean_ci(np.abs(sd), rows, d), "shift_removed_probe_mae": mean_ci(np.abs(sp), rows, d),
        "shift_removed_probe_minus_detector_mae": mean_ci(np.abs(sp) - np.abs(sd), rows, d),
    }
    if feature.count:
        rp = reading(feature, probe)[m]
        out |= {"detector_exact_share": float((ed == 0).mean()), "probe_rounded_exact_share": float((rp == y[m]).mean())}
    return out


# ------------------------------------------------------------------ combination rules (fixed, nothing fitted)
def combine(rule: str, detector: np.ndarray, other: np.ndarray) -> np.ndarray:
    """`rule` "mean": (detector + other) / 2, `other` where the detector is missing; "fill": the detector, `other`
    where it is missing; "detector": the detector alone."""
    det, oth = np.asarray(detector, float), np.asarray(other, float)
    if rule == "detector":
        return det.copy()
    if rule == "mean":
        return np.where(np.isfinite(det), (det + oth) / 2, oth)
    if rule == "fill":
        return np.where(np.isfinite(det), det, oth)
    raise ValueError(f"unknown rule {rule}")


def combination_table(feature: Feature, sb: np.ndarray, det: np.ndarray, probe: np.ndarray, dist: np.ndarray,
                      d: pr.Draws) -> list[dict]:
    """MAE against StatsBomb of the rules and references, on (a) rows with a detector value (vs the detector alone)
    and (b) all rows with a StatsBomb value (vs "fill" with the distance-only reference), with paired differences."""
    y, a, p, r = (np.asarray(v, float) * feature.scale for v in (sb, det, probe, dist))
    rows_out = []
    preds = {"detector": combine("detector", a, p), "mean_probe": combine("mean", a, p), "fill_probe": combine("fill", a, p),
             "mean_distance": combine("mean", a, r), "fill_distance": combine("fill", a, r)}
    for scope, base, mask in (("detector_present", "detector", np.isfinite(y) & np.isfinite(a)),
                              ("all", "fill_distance", np.isfinite(y)),
                              ("detector_missing", "fill_distance", np.isfinite(y) & ~np.isfinite(a))):
        m = mask & np.all([np.isfinite(v) for v in preds.values()], axis=0) if scope == "detector_present" \
            else mask & np.isfinite(preds["fill_probe"]) & np.isfinite(preds["fill_distance"])
        rows = np.flatnonzero(m)
        if len(rows) < 3:
            rows_out.append({"feature": feature.name, "scope": scope, "rule": "-", "n": int(len(rows))})
            continue
        names = list(preds) if scope == "detector_present" else ["fill_probe", "fill_distance", "mean_probe", "mean_distance"]
        base_err = np.abs(preds[base][m] - y[m])
        for name in names:
            err = np.abs(preds[name][m] - y[m])
            rows_out.append({"feature": feature.name, "scope": scope, "rule": name, "n": int(len(rows)),
                             "mae": mean_ci(err, rows, d), f"minus_{base}": mean_ci(err - base_err, rows, d)})
    return rows_out


# ------------------------------------------------------------------ timing bands
def timing_bands(abs_residual_ms: np.ndarray, min_shots: int = MIN_BAND) -> tuple[np.ndarray, dict]:
    """Band label per row (NaN residual -> "missing") after the C2 merge rule, and the raw / merged counts."""
    r = np.asarray(abs_residual_ms, float)
    raw = np.full(len(r), "missing", object)
    raw[r <= BAND_EDGES_MS[0]] = BAND_NAMES[0]
    raw[(r > BAND_EDGES_MS[0]) & (r <= BAND_EDGES_MS[1])] = BAND_NAMES[1]
    raw[r > BAND_EDGES_MS[1]] = BAND_NAMES[2]
    groups = [[b] for b in BAND_NAMES]
    while len(groups) > 1:
        sizes = [int(np.isin(raw, g).sum()) for g in groups]
        small = [i for i, s in enumerate(sizes) if s < min_shots]
        if not small:
            break
        i = small[-1]
        j = i - 1 if i > 0 else 1
        lo, hi = min(i, j), max(i, j)
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    label = raw.copy()
    lower = dict(zip(BAND_NAMES, (None, *BAND_EDGES_MS), strict=True))
    upper = dict(zip(BAND_NAMES, (*BAND_EDGES_MS, None), strict=True))
    for g in groups:
        lo, hi = lower[g[0]], upper[g[-1]]
        name = "all" if lo is None and hi is None else f"<={hi:.0f}" if lo is None else f">{lo:.0f}" if hi is None \
            else f"{lo:.0f}-{hi:.0f}"
        label[np.isin(raw, g)] = name
    counts = {"raw": {b: int((raw == b).sum()) for b in [*BAND_NAMES, "missing"]},
              "merged": {str(n): int((label == n).sum()) for n in dict.fromkeys(label)}, "groups": groups}
    return label, counts


def band_errors(feature: Feature, bands: np.ndarray, sb: np.ndarray, sources: dict[str, np.ndarray],
                games: np.ndarray) -> list[dict]:
    """MAE and mean signed error of each source by band (rows with StatsBomb and that source), CIs per band."""
    y = np.asarray(sb, float) * feature.scale
    out = []
    for band in dict.fromkeys(bands):
        for name, v in sources.items():
            v = np.asarray(v, float) * feature.scale
            m = (bands == band) & np.isfinite(y) & np.isfinite(v)
            if m.sum() < 3:
                out.append({"feature": feature.name, "band": band, "source": name, "n": int(m.sum())})
                continue
            d = pr.draws(np.asarray(games)[m])
            e = v[m] - y[m]
            idx = np.arange(int(m.sum()))
            out.append({"feature": feature.name, "band": band, "source": name, "n": int(m.sum()),
                        "mae": mean_ci(np.abs(e), idx, d), "mean_signed": mean_ci(e, idx, d)})
    return out


def flatten(rows: list[dict]) -> pd.DataFrame:
    """Rows with {point, lo, hi} values spread into `<key>`, `<key>_lo`, `<key>_hi` columns."""
    flat = []
    for r in rows:
        f = {}
        for k, v in r.items():
            if isinstance(v, dict) and "point" in v:
                f |= {k: v["point"], f"{k}_lo": v["lo"], f"{k}_hi": v["hi"]}
            else:
                f[k] = v
        flat.append(f)
    return pd.DataFrame(flat)


TARGET_COLUMNS = tuple(f.name for f in FEATURES)
assert set(TARGET_COLUMNS) <= set(tg.TARGETS)
