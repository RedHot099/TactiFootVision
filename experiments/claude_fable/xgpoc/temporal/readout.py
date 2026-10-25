"""The read-out of arms R, S and T (ticket 01; spec, "Ewaluacja"): the same code serves the synthetic tracer here
and the single real run of ticket 04.

Model: the L2 logistic regression of `xgpoc.stack.fit` with `INNER_CV`: C chosen inside each outer training fold by
    match-grouped inner CV over `xgpoc.stack.fit.C_GRID` (the other host folds), z_g_combo unpenalised (λ = 0), so R
    is the unpenalised recalibration of z and every arm nests it.
Contrasts: ΔLL = LL(arm) - LL(reference) out of fold, with the paired match bootstrap of `xgpoc.stack.stats` (2,000
    draws, seed 0) and Holm over the confirmatory pair {T - R, T - S} (`CONFIRMATORY`) at α = 0.05; S - R is
    descriptive.
Synthetic outcomes (`synthetic_outcomes`): y* ~ Bernoulli(sigmoid(z_g_combo)), drawn from `default_rng(seed)`, one
    vector per replicate in order. No real label is involved.
"""

import numpy as np

from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss

MODEL = sf.INNER_CV
CONFIRMATORY = {"T-R": ("T", "R"), "T-S": ("T", "S")}
DESCRIPTIVE = {"S-R": ("S", "R")}


def out_of_fold(designs: dict[str, list[sf.FoldDesign]], y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """(arm -> out-of-fold probabilities, arm -> chosen C per outer fold; None where only z is fitted)."""
    probabilities, chosen = {}, {}
    for arm, folds in designs.items():
        chosen[arm] = []
        probabilities[arm] = sf.out_of_fold(folds, y, MODEL, chosen[arm])
    return probabilities, chosen


def contrasts(y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    """ΔLL with the match bootstrap of every contrast, and the Holm readings of the confirmatory pair."""
    pairs = CONFIRMATORY | DESCRIPTIVE
    deltas = {name: ss.paired_delta(y, probabilities[a], probabilities[b], weights) for name, (a, b) in pairs.items()}
    return {"deltas": deltas, "readings": ss.readings(deltas, tuple(CONFIRMATORY)),
            "scores": {arm: ss.scores(y, p) for arm, p in probabilities.items()}}


def synthetic_outcomes(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(len(z)) < sigmoid(np.asarray(z, float))).astype(int)
