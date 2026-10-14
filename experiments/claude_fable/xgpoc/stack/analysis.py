"""The xg-stack read-out on one outcome vector (councils C5 and C6): the run of `--real`, and of control 3 on permuted
labels.

`analyse` fits every arm of the frozen design out of fold (`fit.out_of_fold` on designs built once, `fit.all_designs`),
scores each arm, gives ΔLL against R with the paired match bootstrap, Holm over {G, GS} with the readings, the chosen
C per arm and fold, the GBM sensitivity of G and GS (reported, never a claim) and the descriptive category table. Q is
descriptive with a bound only: its readings are its interval and one-sided bound, never a test.
"""

import numpy as np
import pandas as pd

from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss


def analyse(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, weights: np.ndarray, spec: sf.Spec,
            designs: dict[str, list[sf.FoldDesign]] | None = None, with_gbm: bool = True) -> dict:
    """{"probabilities", "scores", "deltas", "readings", "chosen_c", "gbm", "categories"} of the arms on outcomes `y`
    (module docstring). `weights` are the match-bootstrap weights of the population's matches."""
    designs = sf.all_designs(table, folds, spec) if designs is None else designs
    chosen = {name: [] for name in sf.ARM_NAMES}
    p = {name: sf.out_of_fold(designs[name], y, spec.model, chosen[name]) for name in sf.ARM_NAMES}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in sf.ARM_NAMES if name != "R"}
    gbm_p = {name: sf.out_of_fold_gbm(table, folds, y, spec.gbm_columns[name]) for name in sf.GBM_ARMS} if with_gbm else {}
    gbm = {name: {"scores": ss.scores(y, q), "delta_vs_R": ss.paired_delta(y, q, p["R"], weights),
                  "delta_vs_logistic": ss.paired_delta(y, q, p[name], weights)} for name, q in gbm_p.items()}
    return {"probabilities": p | {f"gbm_{name}": q for name, q in gbm_p.items()},
            "scores": {name: ss.scores(y, q) for name, q in p.items()}, "deltas": deltas,
            "readings": ss.readings(deltas, sf.CONFIRMATORY), "chosen_c": chosen, "gbm": gbm,
            "categories": ss.category_table(table, y, p, weights)}
