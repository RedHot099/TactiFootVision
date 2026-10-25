"""Detector-defined categories of the xg-temporal read-out (descriptive; spec, "Ewaluacja"). Label-free definitions.

The inputs are xg-stack's E3 columns at t0 (`xgpoc.stack.inputs.e3_table`: the frames replayed with E3's
ball-source rule, checked equal to the cached features, sanitised, angle wrapped to [0, π]), for the population's shots
only. Source (council C9 Q5, `covering_source`): `cache/xgd_full3/E3`, the table xg-stack froze, if the label-free
check covers every shot of P_G (every shot in the replayed table, the replay equal to the cache); otherwise
`cache/xgd_full4_feet_t0/E3`, and the protocol says so. The protocol records the source, the attempts and the
label-free category counts; the real run reads the recorded source. Categories (`CATEGORIES`):
    e3_outside_box       shot_x@0 < 88.5 m                    (xg-stack, `xgpoc.stack.stats.CATEGORIES`)
    e3_angle_lt_15deg    angle@0 < 15 degrees                 (xg-stack)
    e3_dist_lt_6m        dist@0 < 6 m                         (xg-stack)
    e3_gk_depth_ge_4yd   gk_depth@0 >= 3.6576 m (4 yd, 105 m pitch frame): the goalkeeper off his line, xg-compare
                         item 6a in the detector's metres. Caveat: E3's goalkeeper depth has Spearman ρ 0.04
                         [-0.02; 0.09] with StatsBomb's (`.scratch/xg-compare/feature-ideas.md`), so this category is
                         not evidence that the goalkeeper is measured.
A shot whose defining input is missing is outside the category's population (neither in nor out). Per category and
arm: n, O, E, O/E and ΔLL against R with its 95% interval from the same match draws restricted to the category's shots.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.detector import records
from xgpoc.stack import inputs as si
from xgpoc.stack import stats as ss

E3 = si.E3
E3_SOURCES = (si.E3, si.EXPERIMENT / "cache/xgd_full4_feet_t0/E3")
GK_DEPTH_M = 4 * 0.9144
CATEGORIES = {name: ss.CATEGORIES[name] for name in ("e3_outside_box", "e3_angle_lt_15deg", "e3_dist_lt_6m")} | {
    "e3_gk_depth_ge_4yd": ss.Category(si.col("gk_depth", 0), f"gk_depth@0 >= {GK_DEPTH_M:.4f} m (4 yd)",
                                      lambda x: x >= GK_DEPTH_M)}
CAVEATS = {"e3_gk_depth_ge_4yd": "E3 goalkeeper depth: Spearman 0.04 [-0.02; 0.09] with StatsBomb's; not evidence that "
                                 "the goalkeeper is measured"}


def definitions() -> dict:
    return {name: {"column": c.column, "rule": c.rule, **({"caveat": CAVEATS[name]} if name in CAVEATS else {})}
            for name, c in CATEGORIES.items()}


def e3_inputs(shot_ids, e3=E3) -> tuple[pd.DataFrame, dict]:
    """(the categories' columns for `shot_ids`, index shot_id in their order; the E3 files' hashes), label-free."""
    ids = sorted(shot_ids)
    e3 = Path(e3)
    hashes = {name: {"path": str(e3 / name), "sha256": records.verified_file(e3 / name)}
              for name in ("frames.parquet", "features.parquet")}
    frames = pd.read_parquet(e3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(e3 / "features.parquet", filters=[("shot_id", "in", ids)])
    table, _ = si.e3_table(frames, features)
    columns = sorted({c.column for c in CATEGORIES.values()})
    if not table.index.equals(pd.Index(ids, name="shot_id")):
        raise ValueError("E3 does not cover the population exactly")
    return table[columns].loc[list(shot_ids)], hashes


def covering_source(shot_ids, sources=E3_SOURCES) -> dict:
    """The first E3 source (C9 Q5 order) whose replayed table covers every shot of `shot_ids`, label-free: its path
    (relative to the experiment), hashes, the attempts, and per category the number of shots with the input defined and
    inside. Raises when no source covers the population."""
    attempts = []
    for source in sources:
        try:
            table, hashes = e3_inputs(shot_ids, source)
        except (ValueError, FileNotFoundError) as e:
            attempts.append({"source": str(source), "covers": False, "reason": str(e)})
            continue
        attempts.append({"source": str(source), "covers": True})
        counts = {name: {"defined": int(defined.sum()), "inside": int(inside.sum())}
                  for name, (defined, inside) in masks(table).items()}
        return {"source": str(Path(source).relative_to(si.EXPERIMENT)), "fallback_used": source != sources[0],
                "shots": len(table), "hashes": hashes, "attempts": attempts, "category_counts": counts}
    raise ValueError(f"no E3 source covers the population: {attempts}")


def masks(table: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (defined, inside) over the rows of `table`."""
    out = {}
    for name, category in CATEGORIES.items():
        x = table[category.column].to_numpy(float)
        defined = np.isfinite(x)
        inside = np.zeros(len(x), dtype=bool)
        inside[defined] = category.inside(x[defined])
        out[name] = (defined, inside)
    return out


def category_table(table: pd.DataFrame, y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray,
                   reference: str = "R") -> pd.DataFrame:
    """Per category and arm (module docstring)."""
    rows = []
    for name, (defined, inside) in masks(table).items():
        observed = int(y[inside].sum())
        for arm, p in probabilities.items():
            expected = float(p[inside].sum())
            row = {"category": name, "rule": CATEGORIES[name].rule, "arm": arm, "n_defined": int(defined.sum()),
                   "n": int(inside.sum()), "observed": observed, "expected": expected,
                   "o_over_e": observed / expected if expected > 0 else np.nan}
            if arm != reference and inside.any():
                delta = ss.paired_delta(y, p, probabilities[reference], weights, inside)
                row |= {k: delta[k] for k in ("delta_ll", "lo95", "hi95")}
            rows.append(row)
    return pd.DataFrame(rows)
