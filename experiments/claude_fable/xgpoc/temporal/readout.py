"""The read-out of arms R, S and T (ticket 01; spec, "Ewaluacja"): the same code serves the synthetic tracer here
and the single real run of ticket 04.

Model (`MODEL`): the L2 logistic regression of `xgpoc.stack.fit` (Newton's method on the standardised design, λ = 1 / C)
    with C chosen inside each outer training fold by match-grouped inner CV (the other four host folds, summed inner
    held-out log-loss, ties to the smaller C) over `C_GRID`, z_g_combo unpenalised (λ = 0), so R is the unpenalised
    recalibration of z and every arm nests it. `C_GRID` is council C8 Q1 (b): (1e-5, 3e-5, 1e-4, 3e-4) below
    `xgpoc.stack.fit.C_GRID`, frozen before any label, with no fallback to R (as C goes to 0 an arm tends to R by
    itself). `xgpoc.stack` keeps its own grid: other protocols pin it.
Contrasts: ΔLL = LL(arm) - LL(reference) out of fold, with the paired match bootstrap of `xgpoc.stack.stats` (2,000
    draws, seed 0) and Holm over the confirmatory pair {T - R, T - S} (`CONFIRMATORY`) at α = 0.05; S - R is
    descriptive.
Synthetic outcomes (`synthetic_outcomes`): y* ~ Bernoulli(sigmoid(z_g_combo)), drawn from `default_rng(seed)`, one
    vector per replicate in order. No real label is involved.
Real read-out rules (ticket 04; the protocol records them through `xgpoc.temporal.protocol.frozen`)
    claims (`claims`)        a Holm-rejected contrast is a "detectable improvement", with point ΔLL <= -0.002 a
                             "practical candidate" unless the readability veto fires; "motion helps" only when T - S is
                             rejected; an arm whose inner CV chose the grid's smallest C in >= `NEAR_REFERENCE_FOLDS`
                             outer folds is "arm ≈ R" (`near_reference`, C8).
    veto (`readability_veto`) among the non-goal shots, the AUC of logit(p_T) - logit(p_S) for on_target; its lower
                             bound, the 5th percentile of 2,000 match-bootstrap draws (seed 0) over the non-goal shots'
                             matches (`xgpoc.probe.readability.weighted_auc`), above `VETO_AUC` blocks every
                             "practical candidate".
    leak curve (`leak_rules`) start -1600, o_max -1000 ... +400: the method's positive control is S - R at +400 with
                             hi95 < 0; the jump rule takes the per-step trend g = (Δ(-600) - Δ(-1000)) / 2 and flags a
                             leak signal when Δ(o) - (Δ(-600) + g (o + 600) / 200) <= `LEAK_EXCESS` at o = -400 or
                             -200, for T - R or T - S (descriptive, disclosed beside every claim).
"""

import numpy as np

from xgpoc.eval.calibration import logit, sigmoid
from xgpoc.probe import readability
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss

C_GRID = (1e-5, 3e-5, 1e-4, 3e-4, *sf.C_GRID)
MODEL = sf.Model("inner_cv_c8", C_GRID, True)
CONFIRMATORY = {"T-R": ("T", "R"), "T-S": ("T", "S")}
DESCRIPTIVE = {"S-R": ("S", "R")}
VETO_AUC = 0.55
VETO_BOUND_PERCENTILE = 5.0
NEAR_REFERENCE_FOLDS = 3
LEAK_EXCESS = -0.001
LEAK_TREND = (-1000, -600)
LEAK_TESTED = (-400, -200)
METHOD_POSITIVE_O_MAX = 400


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


# ------------------------------------------------------------------ the real read-out's rules (ticket 04)
def near_reference(chosen: list) -> bool:
    """The inner CV chose the grid's smallest C in at least `NEAR_REFERENCE_FOLDS` outer folds."""
    return sum(c == C_GRID[0] for c in chosen if c is not None) >= NEAR_REFERENCE_FOLDS


def readability_veto(goal: np.ndarray, on_target: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, games: np.ndarray) -> dict:
    """The readability veto (module docstring): AUC point, 95% interval, one-sided lower bound and the verdict."""
    keep = np.asarray(goal) == 0
    score = logit(np.clip(p_t[keep], ss.CLIP, 1 - ss.CLIP)) - logit(np.clip(p_s[keep], ss.CLIP, 1 - ss.CLIP))
    keys, counts = readability.match_draw_counts(np.asarray(games)[keep], ss.N_BOOT, ss.BOOT_SEED)
    match = np.searchsorted(keys, np.asarray(games)[keep])
    target = np.asarray(on_target)[keep].astype(int)
    point = float(readability.weighted_auc(score, target, match))
    draws = readability.weighted_auc(score, target, match, counts)
    draws = draws[np.isfinite(draws)]
    lower = float(np.percentile(draws, VETO_BOUND_PERCENTILE))
    return {"n": int(keep.sum()), "on_target": int(target.sum()), "auc": point,
            "lo95": float(np.percentile(draws, 2.5)), "hi95": float(np.percentile(draws, 97.5)),
            "lower_one_sided_95": lower, "threshold": VETO_AUC, "vetoed": bool(lower > VETO_AUC), "draws": int(len(draws))}


def claims(result: dict, veto: dict, chosen: dict[str, list]) -> dict:
    """The claims the read-out allows (module docstring)."""
    readings = result["readings"]
    out = {}
    for name in CONFIRMATORY:
        practical = readings[name]["practical_candidate"]
        out[name] = {"detectable_improvement": readings[name]["detectable_improvement"],
                     "practical_candidate": bool(practical and not veto["vetoed"]),
                     "practical_blocked_by_veto": bool(practical and veto["vetoed"])}
    return out | {"motion_helps": readings["T-S"]["holm_rejected"],
                  "arm_near_reference": {arm: near_reference(c) for arm, c in chosen.items() if arm != "R"}}


def leak_rules(curve: dict[int, dict]) -> dict:
    """The method's positive control and the jump rule on the leak curve: o_max -> contrast -> paired_delta."""
    positive = curve[METHOD_POSITIVE_O_MAX]["S-R"]
    out = {"method_positive": {"contrast": "S-R", "o_max": METHOD_POSITIVE_O_MAX, "delta_ll": positive["delta_ll"],
                               "hi95": positive["hi95"], "passed": bool(positive["hi95"] < 0)}, "jump": {}}
    early, late = LEAK_TREND
    for name in CONFIRMATORY:
        step = (curve[late][name]["delta_ll"] - curve[early][name]["delta_ll"]) / ((late - early) / 200)
        excess = {str(o): curve[o][name]["delta_ll"] - (curve[late][name]["delta_ll"] + step * (o - late) / 200)
                  for o in LEAK_TESTED}
        out["jump"][name] = {"trend_per_step": step, "excess": excess,
                             "signal": any(e <= LEAK_EXCESS for e in excess.values())}
    out["leak_signal"] = any(j["signal"] for j in out["jump"].values())
    return out
