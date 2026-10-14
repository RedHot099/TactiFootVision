"""Council C6's choice of G's design, on simulated outcomes only (ticket 01).

For each plant of `controls.PLANTS` (dist; dist + angle) at a true ΔLL of -0.002, the outcomes y* ~ Bernoulli(
sigmoid(a + b z + β s)) are drawn once per replicate from the plant's own seed, and every candidate of
`fit.CANDIDATES` is fitted on the same y* (common random numbers). (a, b) is the unpenalised logistic regression of
the real labels on [1, z] over every shot (`fit.unpenalised_reference`), one truth for every candidate. The criterion
is G's Holm power averaged over the two plants; the highest wins, and equal averages go to the simpler design in the
order (b), (c), (b)+(c) (`choose`). The selection seeds are distinct from the final controls' seeds.
"""

import numpy as np
import pandas as pd

from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf

SEEDS = {"dist": 101, "dist_angle": 102}
TARGET = sc.POSITIVE_TARGET
PREDICTED_RECOVERY_B = (-0.0017, -0.0015)       # (b)'s mean recovered ΔLL of G on the dist plant, written first (C6)


def choose(averaged: dict[str, float]) -> str:
    """The candidate with the highest averaged power; equal averages go to the earliest of `fit.SIMPLICITY`."""
    best = max(averaged.values())
    return next(name for name in sf.SIMPLICITY if averaged[name] == best)


def run(table: pd.DataFrame, folds: np.ndarray, weights: np.ndarray, a: float, b: float, n: int = sc.N_REPLICATES,
        seeds: dict[str, int] = SEEDS, log: sc.Log = sc.quiet) -> tuple[dict[tuple[str, str], pd.DataFrame], dict]:
    """(replicate rows per (candidate, plant), summary with the per-plant results, the averaged powers and the
    chosen candidate)."""
    z = table[sf.Z].to_numpy(float)
    frames, plants = {}, {}
    specs = {name: sf.spec(name, table) for name in sf.SIMPLICITY}
    designs = {name: sf.all_designs(table, folds, s, ("R", *sf.CONFIRMATORY)) for name, s in specs.items()}
    for kind in sc.PLANTS:
        s = sc.plant(table, kind)
        beta = sc.solve_beta(a, b, z, s, TARGET)
        p_true = sigmoid(a + b * z + beta * s)
        plants[kind] = {"beta": beta, "true_delta": sc.true_delta(beta, a, b, z, s), "seed": seeds[kind]}
        for name in sf.SIMPLICITY:
            log(f"selection: plant {kind}, candidate {name}")
            frames[name, kind] = sc.simulate(designs[name], weights, specs[name].model, sc.fixed(p_true), n, seeds[kind], log)
    candidates = {name: {kind: sc.power_summary(frames[name, kind]) for kind in sc.PLANTS} for name in sf.SIMPLICITY}
    averaged = {name: float(np.mean([candidates[name][kind]["power_G"] for kind in sc.PLANTS])) for name in sf.SIMPLICITY}
    recovery_b = candidates["b"]["dist"]["mean_recovered_delta_G"]
    lo, hi = PREDICTED_RECOVERY_B
    summary = {"truth": {"a": a, "b": b}, "target": TARGET, "replicates": n, "plants": plants, "candidates": candidates,
               "averaged_power_G": averaged, "chosen": choose(averaged),
               "predicted_recovery_b": list(PREDICTED_RECOVERY_B), "observed_recovery_b": recovery_b,
               "prediction_held": bool(lo <= recovery_b <= hi),
               "columns": {name: {arm: s.arms[arm].columns for arm in ("G", "GS")} for name, s in specs.items()}}
    return frames, summary
