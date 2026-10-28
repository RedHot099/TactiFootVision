"""The read-out of arms R, S and T (ticket 01; spec, "Ewaluacja"): the same code serves the synthetic tracer here
and the single real run of ticket 04.

Model (`MODEL`): the L2 logistic regression of `xgpoc.stack.fit` (Newton's method on the standardised design, λ = 1 / C)
    with C chosen inside each outer training fold by match-grouped inner CV (the other four host folds, summed inner
    held-out log-loss, ties to the smaller C) over `C_GRID`, z_g_combo unpenalised (λ = 0), so R is the unpenalised
    recalibration of z and every arm nests it. `C_GRID` is council C8 Q1 (b): (1e-5, 3e-5, 1e-4, 3e-4) below
    `xgpoc.stack.fit.C_GRID`, frozen before any label, with no fallback to R (as C goes to 0 an arm tends to R by
    itself). `xgpoc.stack` keeps its own grid: other protocols pin it.
T's model (`T_MODEL`, the pre-registered remedy of council C9 Q2, built after the first label-free control run missed
    the positive gate by shrinkage; `xgpoc.temporal.controls.REMEDY`; kept by council C10 Q4 (a) for C10's smaller
    motion block, `xgpoc.temporal.features.MOTION_K` columns): T's design is S's columns (the static block, z first)
    followed by the motion block. Per outer fold T keeps the C that S's inner CV chose there (C_S) for the static
    block, z unpenalised, and chooses a separate C_M for the motion block over `C_GRID` by the same inner match-grouped
    CV (the static block at C_S in every inner fit; ties to the smaller C_M). The block penalty is implemented exactly by
    rescaling the motion columns by sqrt(C_M / C_S) under the common penalty 1 / C_S (`rescaled`): a coefficient b of
    an original motion column is w sqrt(C_M / C_S) with penalty w² / C_S = b² / C_M. T still nests S (`check_nested`:
    its first columns are S's design, fold by fold). T's chosen C per fold is C_M; its static C is S's.
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
                             outer folds is "arm ≈ R" (`near_reference`, C8; for T both its static C, S's, and its
                             motion C_M at the floor in the same fold).
    veto (`readability_veto`) among the non-goal shots, the AUC of logit(p_T) - logit(p_S) for on_target; its lower
                             bound, the 5th percentile of 2,000 match-bootstrap draws (seed 0) over the non-goal shots'
                             matches (`xgpoc.probe.readability.weighted_auc`), above `VETO_AUC` blocks every
                             "practical candidate".
    leak curve (`leak_rules`) start -1600, o_max -1000 ... +400, every point on the shots common to every window
                             (`protocol.common_populations`, the arms refitted there) with the same match draws (the
                             main population's weights restricted to those shots): the method's positive control is
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
T_MODEL = "inner_cv_c8_motion_block_c9q2"      # T's variant (module docstring), every other arm `MODEL`
STATIC, MOTION = "S", "T"                      # T's static block is S's design
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
    """(arm -> out-of-fold probabilities, arm -> chosen C per outer fold; None where only z is fitted). Every arm under
    `MODEL` but T, which is fitted under `T_MODEL` after S (module docstring); its chosen C is the motion block's."""
    probabilities, chosen = {}, {}
    for arm, folds in designs.items():
        chosen[arm] = []
        if arm == MOTION:
            if STATIC not in chosen:
                raise ValueError(f"arm {MOTION} needs arm {STATIC} fitted before it")
            probabilities[arm] = motion_block_out_of_fold(folds, designs[STATIC], chosen[STATIC], y, chosen[arm])
        else:
            probabilities[arm] = sf.out_of_fold(folds, y, MODEL, chosen[arm])
    return probabilities, chosen


# ------------------------------------------------------------------ T's model: a separate motion-block C (C9 Q2)
def check_nested(t: sf.FoldDesign, s: sf.FoldDesign) -> int:
    """The width of the static block: T's design must be S's (same rows, same leading columns) plus more columns, in
    the outer fold and in every inner fold."""
    for a, b in ((t, s), *zip(t.inner, s.inner, strict=True)):
        n = b.x_train.shape[1]
        if not (a.x_train.shape[1] > n and np.array_equal(a.train, b.train) and np.array_equal(a.test, b.test)
                and np.array_equal(a.x_train[:, :n], b.x_train) and np.array_equal(a.x_test[:, :n], b.x_test)):
            raise ValueError("T's design does not start with S's: the motion block's penalty needs T ⊃ S")
    return s.x_train.shape[1]


def motion_scale(c_static: float, c_motion: float) -> float:
    return float(np.sqrt(c_motion / c_static))


def rescaled(x: np.ndarray, n_static: int, scale: float) -> np.ndarray:
    """`x` with its columns from `n_static` on (the motion block) multiplied by `scale`."""
    out = np.array(x, float, copy=True)
    out[:, n_static:] *= scale
    return out


def fit_blocks(x: np.ndarray, y: np.ndarray, n_static: int, c_static: float, c_motion: float) -> tuple[np.ndarray, float]:
    """(coefficients on the rescaled design, the scale): z unpenalised, the static block at 1 / C_S, the motion block
    at 1 / C_M, by rescaling the motion columns under the common penalty 1 / C_S (module docstring)."""
    scale = motion_scale(c_static, c_motion)
    return sf.logistic_fit(rescaled(x, n_static, scale), y, MODEL.penalties(x.shape[1], c_static)), scale


def predict_blocks(beta: np.ndarray, scale: float, x: np.ndarray, n_static: int) -> np.ndarray:
    return sf.predict(beta, rescaled(x, n_static, scale))


def motion_c(design: sf.FoldDesign, y: np.ndarray, n_static: int, c_static: float) -> float:
    """C_M of one outer fold: the `C_GRID` value with the smallest summed inner held-out log-loss with the static block
    at `c_static` (ties: the smaller C), as `xgpoc.stack.fit.chosen_c` does for one common C."""
    if not design.inner:
        raise ValueError("the motion block's C needs the inner designs")
    losses = []
    for c in MODEL.grid:
        total = 0.0
        for d in design.inner:
            beta, scale = fit_blocks(d.x_train, y[d.train], n_static, c_static, c)
            total += sf.mean_log_loss(y[d.test], predict_blocks(beta, scale, d.x_test, n_static)) * len(d.test)
        losses.append(total)
    return float(MODEL.grid[int(np.argmin(losses))])


def motion_block_out_of_fold(t_folds: list[sf.FoldDesign], s_folds: list[sf.FoldDesign], c_static: list, y: np.ndarray,
                             chosen: list | None = None) -> np.ndarray:
    """(n,) out-of-fold probabilities of T under `T_MODEL`: per outer fold the static block at S's chosen C there
    (`c_static`, one per fold in the folds' order) and the motion block at its own inner-CV C_M, appended to
    `chosen`."""
    y = np.asarray(y, float)
    p = np.full(len(y), np.nan)
    if len(c_static) != len(t_folds) or len(s_folds) != len(t_folds) or None in c_static:
        raise ValueError("T needs S's chosen C in every outer fold")
    for t, s, c_s in zip(t_folds, s_folds, c_static, strict=True):
        n_static = check_nested(t, s)
        c_m = motion_c(t, y, n_static, c_s)
        beta, scale = fit_blocks(t.x_train, y[t.train], n_static, c_s, c_m)
        p[t.test] = predict_blocks(beta, scale, t.x_test, n_static)
        if chosen is not None:
            chosen.append(c_m)
    if np.isnan(p).any():
        raise ValueError("the folds do not cover every shot")
    return p


def contrasts(y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    """ΔLL with the match bootstrap of every contrast, and the Holm readings of the confirmatory pair."""
    pairs = CONFIRMATORY | DESCRIPTIVE
    deltas = {name: ss.paired_delta(y, probabilities[a], probabilities[b], weights) for name, (a, b) in pairs.items()}
    return {"deltas": deltas, "readings": ss.readings(deltas, tuple(CONFIRMATORY)),
            "scores": {arm: ss.scores(y, p) for arm, p in probabilities.items()}}


def synthetic_outcomes(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(len(z)) < sigmoid(np.asarray(z, float))).astype(int)


# ------------------------------------------------------------------ the real read-out's rules (ticket 04)
def near_reference(chosen: dict[str, list], arm: str) -> bool:
    """The inner CV chose the grid's smallest C in at least `NEAR_REFERENCE_FOLDS` outer folds; for T in a fold only
    when its static C (S's) and its motion C_M are both the smallest."""
    floor = [c == C_GRID[0] for c in chosen[arm]]
    if arm == MOTION:
        floor = [m and s == C_GRID[0] for m, s in zip(floor, chosen[STATIC], strict=True)]
    return sum(floor) >= NEAR_REFERENCE_FOLDS


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
                  "arm_near_reference": {arm: near_reference(chosen, arm) for arm in chosen if arm != "R"}}


def curve_point(y: np.ndarray, probabilities: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, dict]:
    """contrast -> `paired_delta` with its bootstrap means (`boot`, one per row of `weights`) for one leak-curve
    point."""
    out = {}
    for name, (a, b) in (CONFIRMATORY | DESCRIPTIVE).items():
        d = shot_log_loss(y, probabilities[a], ss.CLIP) - shot_log_loss(y, probabilities[b], ss.CLIP)
        out[name] = ss.summarise(d.mean(), boot := ss.boot_means(weights, d)) | {"boot": boot}
    return out


def trend_weights(at: int, through: tuple[int, ...] = LEAK_TREND) -> np.ndarray:
    """c with line(at) = Σ c_i Δ(through_i) for the least-squares line through the points."""
    x = np.column_stack([np.ones(len(through)), np.asarray(through, float)])
    return np.array([1.0, float(at)]) @ np.linalg.pinv(x)


def leak_rules(curve: dict[int, dict]) -> dict:
    """The method's positive control and the jump rule on the leak curve (module docstring): o_max -> contrast ->
    `curve_point` entry (delta_ll, hi95, boot)."""
    positive = curve[METHOD_POSITIVE_O_MAX]["S-R"]
    out = {"method_positive": {"contrast": "S-R", "o_max": METHOD_POSITIVE_O_MAX, "delta_ll": positive["delta_ll"],
                               "hi95": positive["hi95"], "passed": bool(positive["hi95"] < 0)},
           "trend": list(LEAK_TREND), "jump": {}}
    for name in CONFIRMATORY:
        excess = {}
        for o in LEAK_TESTED:
            c = trend_weights(o)
            point = curve[o][name]["delta_ll"] - sum(ci * curve[t][name]["delta_ll"] for ci, t in zip(c, LEAK_TREND, strict=True))
            draws = curve[o][name]["boot"] - sum(ci * curve[t][name]["boot"] for ci, t in zip(c, LEAK_TREND, strict=True))
            draws = draws[np.isfinite(draws)]
            excess[str(o)] = {"excess": float(point), "lo95": float(np.percentile(draws, 2.5)),
                              "hi95": float(np.percentile(draws, 97.5))}
        out["jump"][name] = {"excess": excess, "signal": any(e["excess"] <= LEAK_EXCESS for e in excess.values())}
    out["leak_signal"] = any(j["signal"] for j in out["jump"].values())
    return out
