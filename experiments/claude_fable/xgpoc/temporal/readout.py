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
                             "practical candidate" unless the readability veto fires or the leak curve shows a jump for
                             that contrast (then "ambiguous: leak or late pre-contact motion", C9 Q3); "motion helps"
                             only when T - S is rejected; an arm whose inner CV chose the grid's smallest C in >= `NEAR_REFERENCE_FOLDS`
                             outer folds is "arm ≈ R" (`near_reference`, C8).
    veto (`readability_veto`) among the non-goal shots, the AUC of logit(p_T) - logit(p_S) for on_target; its lower
                             bound, the 5th percentile of 2,000 match-bootstrap draws (seed 0) over the non-goal shots'
                             matches (`xgpoc.probe.readability.weighted_auc`), above `VETO_AUC` blocks every
                             "practical candidate".
    leak curve (`leak_rules`) start -1600, o_max -1000 ... +400, every point with the same match draws (the main
                             population's weights restricted to the point's shots): the method's positive control is
                             S - R at +400 with hi95 < 0; the jump rule (C9 Q3) fits the least-squares line through
                             o_max -1000, -800, -600 (`LEAK_TREND`) and takes the excess e(o) = Δ(o) - line(o) at -400
                             and -200, with its match-bootstrap interval (the line refitted in every draw); a jump is
                             e(o) <= `LEAK_EXCESS` at either point, per confirmatory contrast, and blocks the
                             "practical candidate" wording of that contrast.
"""

import numpy as np

from xgpoc.detector.testrule import shot_log_loss
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
LEAK_TREND = (-1000, -800, -600)
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


AMBIGUOUS = "ambiguous: leak or late pre-contact motion"


def claims(result: dict, veto: dict, chosen: dict[str, list], leak: dict) -> dict:
    """The claims the read-out allows (module docstring)."""
    readings = result["readings"]
    out = {}
    for name in CONFIRMATORY:
        practical = readings[name]["practical_candidate"]
        jump = leak["jump"][name]["signal"]
        out[name] = {"detectable_improvement": readings[name]["detectable_improvement"],
                     "practical_candidate": bool(practical and not veto["vetoed"] and not jump),
                     "practical_blocked_by_veto": bool(practical and veto["vetoed"]),
                     "practical_blocked_by_leak_jump": bool(practical and jump),
                     "wording": AMBIGUOUS if practical and jump else None}
    return out | {"motion_helps": readings["T-S"]["holm_rejected"],
                  "arm_near_reference": {arm: near_reference(c) for arm, c in chosen.items() if arm != "R"}}


def curve_point(y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, dict]:
    """contrast -> `paired_delta` with its draws (`draws`, one per row of `weights`) for one leak-curve point."""
    out = {}
    for name, (a, b) in (CONFIRMATORY | DESCRIPTIVE).items():
        d = shot_log_loss(y, probabilities[a], ss.CLIP) - shot_log_loss(y, probabilities[b], ss.CLIP)
        out[name] = ss.summarise(d.mean(), boot := ss.boot_means(weights, d)) | {"draws": boot}
    return out


def trend_weights(at: int, through: tuple[int, ...] = LEAK_TREND) -> np.ndarray:
    """c with line(at) = Σ c_i Δ(through_i) for the least-squares line through the points."""
    x = np.column_stack([np.ones(len(through)), np.asarray(through, float)])
    return np.array([1.0, float(at)]) @ np.linalg.pinv(x)


def leak_rules(curve: dict[int, dict]) -> dict:
    """The method's positive control and the jump rule on the leak curve (module docstring): o_max -> contrast ->
    `curve_point` entry (delta_ll, hi95, draws)."""
    positive = curve[METHOD_POSITIVE_O_MAX]["S-R"]
    out = {"method_positive": {"contrast": "S-R", "o_max": METHOD_POSITIVE_O_MAX, "delta_ll": positive["delta_ll"],
                               "hi95": positive["hi95"], "passed": bool(positive["hi95"] < 0)},
           "trend": list(LEAK_TREND), "jump": {}}
    for name in CONFIRMATORY:
        excess = {}
        for o in LEAK_TESTED:
            c = trend_weights(o)
            point = curve[o][name]["delta_ll"] - sum(ci * curve[t][name]["delta_ll"] for ci, t in zip(c, LEAK_TREND, strict=True))
            draws = curve[o][name]["draws"] - sum(ci * curve[t][name]["draws"] for ci, t in zip(c, LEAK_TREND, strict=True))
            draws = draws[np.isfinite(draws)]
            excess[str(o)] = {"excess": float(point), "lo95": float(np.percentile(draws, 2.5)),
                              "hi95": float(np.percentile(draws, 97.5))}
        out["jump"][name] = {"excess": excess, "signal": any(e["excess"] <= LEAK_EXCESS for e in excess.values())}
    out["leak_signal"] = any(j["signal"] for j in out["jump"].values())
    return out
