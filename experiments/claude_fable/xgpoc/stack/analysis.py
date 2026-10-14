"""The xg-stack read-out on one outcome vector (council C5): the run of `--real`, and of control 3 on permuted labels.

`analyse` fits every arm out of fold (`fit.out_of_fold` on designs built once, `fit.fold_designs`), scores each arm,
gives ΔLL against R with the paired match bootstrap, Holm over {G, GS} with the readings, the GBM sensitivity of G
and GS (reported, never a claim) and the descriptive category table.
"""

import numpy as np
import pandas as pd

from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss


def all_designs(table: pd.DataFrame, folds: np.ndarray) -> dict[str, list[sf.FoldDesign]]:
    return {name: sf.fold_designs(table, folds, arm) for name, arm in sf.ARMS.items()}


def analyse(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, weights: np.ndarray,
            designs: dict[str, list[sf.FoldDesign]] | None = None, with_gbm: bool = True) -> dict:
    """{"probabilities", "scores", "deltas", "readings", "gbm", "categories"} of the arms on outcomes `y` (module
    docstring). `weights` are the match-bootstrap weights of the population's matches."""
    designs = all_designs(table, folds) if designs is None else designs
    p = {name: sf.out_of_fold(designs[name], y) for name in sf.ARMS}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in sf.ARMS if name != "R"}
    out = {"probabilities": p, "scores": {name: ss.scores(y, q) for name, q in p.items()}, "deltas": deltas,
           "readings": ss.readings(deltas, sf.CONFIRMATORY), "gbm": {}}
    if with_gbm:
        for name in sf.GBM_ARMS:
            q = sf.out_of_fold_gbm(table, folds, y, name)
            p[f"gbm_{name}"] = q
            out["gbm"][name] = {"scores": ss.scores(y, q), "delta_vs_R": ss.paired_delta(y, q, p["R"], weights),
                                "delta_vs_logistic": ss.paired_delta(y, q, p[name], weights)}
    out["categories"] = ss.category_table(table, y, {name: p[name] for name in sf.ARMS}, weights)
    return out
