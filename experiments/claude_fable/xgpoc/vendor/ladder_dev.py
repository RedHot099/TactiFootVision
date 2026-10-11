"""Comparison of the ladder on the DEVELOPMENT table with the prototype's `ladder_metrics.csv` / `ladder_pairs.csv`
(xg-vendor ticket 08). Development only; nothing here is a result.

Raw rungs carry the prototype's predictions unchanged, so their points must agree exactly. Recalibrated rungs and stacks
differ by design: the prototype used sklearn's GroupKFold within each of its populations, the analysis uses seeded
match folds (`ladder.match_folds`) and, for the pooled set, one cross-fit over all primary shots where the prototype
pooled two per-population cross-fits. For the pooled set the first 2,000 of the 5,000 bootstrap draws are the
prototype's 2,000 (same seed and draw call over the same 135 matches); the test and train/valid sets read the shared
draws of the whole table (`xgpoc.vendor.draws`), where the prototype resampled within each set. A cell is within noise when the points and both 95% interval ends differ by at most one bootstrap
standard error of the prototype's cell (its 95% interval width / 3.92).
"""

import pandas as pd

PROTOTYPE_RUNGS = {"r_constant": ("raw", "constant"), "r_dist_angle": ("raw", "dist_angle"), "r_event": ("raw", "event"),
                   "r_freeze_frame": ("raw", "freeze_frame"), "r_vendor": ("raw", "vendor"), "r_frame": ("raw", "frame"),
                   "r_positions_recal": ("recalibrated", "freeze_frame"), "r_vendor_recal": ("recalibrated", "vendor"),
                   "r_frame_recal": ("recalibrated", "frame"), "r_frame+positions": ("recalibrated", "frame+positions"),
                   "r_frame+vendor": ("recalibrated", "frame+vendor")}
PROTOTYPE_POPULATIONS = {"test": "test", "trainvalid": "trainvalid", "pooled_power_projection": "primary"}
METRICS = ("log_loss", "brier", "auc")
Z95_WIDTH = 3.92


def _flag(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("no cell is shared with the prototype (population set names: test, trainvalid, primary)")
    rows["proto_se"] = (rows.proto_hi - rows.proto_lo) / Z95_WIDTH
    worst = pd.concat([(rows.point - rows.proto_point).abs(), (rows.lo - rows.proto_lo).abs(), (rows.hi - rows.proto_hi).abs()], axis=1).max(axis=1)
    rows["max_abs_diff_in_se"] = worst / rows.proto_se.where(rows.proto_se > 0)
    rows["within_noise"] = (worst <= rows.proto_se) | (worst < 1e-12)
    return rows


def compare_metrics(metrics: pd.DataFrame, prototype: pd.DataFrame) -> pd.DataFrame:
    """One row per shared (population, rung, metric) cell: our point and 95% interval next to the prototype's."""
    ours = metrics.set_index(["population", "variant", "rung"])
    rows = []
    for _, ref in prototype[prototype.rung.isin(PROTOTYPE_RUNGS) & prototype.population.isin(PROTOTYPE_POPULATIONS)].iterrows():
        population = PROTOTYPE_POPULATIONS[ref.population]
        key = (population, *PROTOTYPE_RUNGS[ref.rung])
        if key not in ours.index:
            continue
        mine = ours.loc[key]
        for metric in METRICS:
            rows.append({"population": population, "prototype_population": ref.population, "prototype_rung": ref.rung,
                         "variant": key[1], "rung": key[2], "metric": metric, "point": mine[metric], "lo": mine[f"{metric}_lo"],
                         "hi": mine[f"{metric}_hi"], "proto_point": ref[metric], "proto_lo": ref[f"{metric}_lo"], "proto_hi": ref[f"{metric}_hi"]})
    return _flag(pd.DataFrame(rows))


def compare_contrasts(contrasts: pd.DataFrame, prototype: pd.DataFrame) -> pd.DataFrame:
    """One row per shared (population, a - b, metric) contrast: difference and 95% interval next to the prototype's,
    with both smallest equivalence margins (ours on the 1e-4 grid)."""
    ours = contrasts.set_index(["population", "variant_a", "a", "variant_b", "b", "metric"])
    rows = []
    usable = prototype.a.isin(PROTOTYPE_RUNGS) & prototype.b.isin(PROTOTYPE_RUNGS) & prototype.population.isin(PROTOTYPE_POPULATIONS)
    for _, ref in prototype[usable & prototype.metric.isin(METRICS)].iterrows():
        population = PROTOTYPE_POPULATIONS[ref.population]
        key = (population, *PROTOTYPE_RUNGS[ref.a], *PROTOTYPE_RUNGS[ref.b], ref.metric)
        if key not in ours.index:
            continue
        mine = ours.loc[key]
        rows.append({"population": population, "prototype_population": ref.population, "prototype_a": ref.a, "prototype_b": ref.b,
                     "variant": key[1], "a": key[2], "b": key[4], "metric": ref.metric, "point": mine["diff"], "lo": mine.lo95,
                     "hi": mine.hi95, "proto_point": ref["diff"], "proto_lo": ref.lo95, "proto_hi": ref.hi95,
                     "smallest_margin": mine.get("smallest_equivalence_margin"), "proto_smallest_margin": ref.smallest_equivalence_margin})
    return _flag(pd.DataFrame(rows))
