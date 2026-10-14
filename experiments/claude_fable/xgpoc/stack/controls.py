"""Controls 2-4 of xg-stack (ticket 01, council C5); control 1 is `provenance.control`. All run before the real run.

2. Synthetic nulls (`synthetic_nulls`): 200 outcome vectors y* ~ Bernoulli(p_R), p_R = sigmoid(a + b z) with (a, b)
   R fitted on the real labels of every shot (`fit.reference_fit`); the features carry no information beyond z by
   construction. Each replicate fits R, G and GS out of fold and applies Holm over {G, GS}. Pass: the share of
   replicates with a Holm "detectable improvement" (any arm) is <= 0.09.
3. Permutation (`permutation`): the full read-out (`analysis.analyse`) on the real labels permuted within matches
   (one `default_rng(PERMUTATION_SEED)`, matches in sorted order, `compare.stats.permute_within_matches`). Pass: no
   Holm rejection.
4. Positive control (`positive_control`): 200 vectors y* ~ Bernoulli(sigmoid(a + b z + β d)), d the E3 t0 distance
   standardised over the population with missing values set to 0. β is the negative root (`solve_beta`) of
   true ΔLL(β) = -0.002, where true ΔLL(β) = E[LL(p_β)] - E[LL(q_β)] over the population with y ~ p_β, and q_β is the
   β = 0 model refitted to p_β (`projection`: the unpenalised logistic regression of the probabilities p_β on [1, z]). Reported: β,
   its true ΔLL, the empirical power of G (share of replicates where G is Holm-detectable) and the mean recovered
   out-of-fold ΔLL of G. Pass: that mean lies in [-0.003, -0.001].
   Diagnostic, never a criterion: the oracle ΔLL of each replicate, LL(p_true) - LL(R out of fold) on the same y*,
   i.e. what an arm that knew the generating model would reach; the gap between it and G's recovered ΔLL is the cost
   of estimating G's coefficients, which tells a design limit from a code fault.

Each replicate's outcomes are drawn from the control's own generator, in replicate order: y* = (rng.random(n) < p).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from xgpoc.compare.stats import permute_within_matches
from xgpoc.detector.testrule import shot_log_loss
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import analysis as sa
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import stats as ss

N_NULL = 200
N_POSITIVE = 200
NULL_SEED = 1
POSITIVE_SEED = 2
PERMUTATION_SEED = 0
NULL_MAX_DETECTABLE_RATE = 0.09
POSITIVE_TARGET = -0.002
POSITIVE_WINDOW = (-0.003, -0.001)
BETA_BRACKET = (-5.0, 0.0)
SIGNAL = si.col("dist", 0)


def confirmatory(designs: dict[str, list[sf.FoldDesign]], y: np.ndarray, weights: np.ndarray,
                 p_true: np.ndarray | None = None) -> dict:
    """One replicate: ΔLL of G and GS against R, their one-sided p and Holm readings, flat; with `p_true`, also the
    oracle ΔLL (module docstring)."""
    p = {name: sf.out_of_fold(designs[name], y) for name in ("R", *sf.CONFIRMATORY)}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in sf.CONFIRMATORY}
    read = ss.readings(deltas, sf.CONFIRMATORY)
    row = {}
    for name in sf.CONFIRMATORY:
        row |= {f"delta_{name}": deltas[name]["delta_ll"], f"p_{name}": deltas[name]["p_one_sided"],
                f"hi95_{name}": deltas[name]["hi95"], f"detectable_{name}": read[name]["detectable_improvement"],
                f"practical_{name}": read[name]["practical_candidate"]}
    if p_true is not None:
        row["delta_oracle"] = float(np.mean(shot_log_loss(y, p_true, ss.CLIP) - shot_log_loss(y, p["R"], ss.CLIP)))
    return row | {"detectable_any": any(read[name]["detectable_improvement"] for name in sf.CONFIRMATORY)}


def simulate(designs: dict[str, list[sf.FoldDesign]], weights: np.ndarray, p_true: np.ndarray, n: int, seed: int,
             log: Callable[[str], None] = lambda _: None) -> pd.DataFrame:
    """`n` replicates of `confirmatory` on y* ~ Bernoulli(p_true), one row each."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = (rng.random(len(p_true)) < p_true).astype(int)
        rows.append({"replicate": i, "goals": int(y.sum())} | confirmatory(designs, y, weights, p_true))
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{n}")
    return pd.DataFrame(rows)


def synthetic_nulls(designs, weights, p_r: np.ndarray, n: int = N_NULL, seed: int = NULL_SEED, log=lambda _: None) -> tuple[pd.DataFrame, dict]:
    frame = simulate(designs, weights, p_r, n, seed, log)
    rate = float(frame.detectable_any.mean())
    summary = {"replicates": n, "seed": seed, "detectable_rate": rate, "diagnostic_mean_oracle_delta": float(frame.delta_oracle.mean()),
               "detectable_rate_by_arm": {a: float(frame[f"detectable_{a}"].mean()) for a in sf.CONFIRMATORY},
               "mean_delta": {a: float(frame[f"delta_{a}"].mean()) for a in sf.CONFIRMATORY},
               "criterion": f"detectable rate <= {NULL_MAX_DETECTABLE_RATE}", "passed": rate <= NULL_MAX_DETECTABLE_RATE}
    return frame, summary


def permutation(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, weights: np.ndarray, designs=None,
                seed: int = PERMUTATION_SEED) -> tuple[dict, dict]:
    """(the read-out on labels permuted within matches, summary with the pass criterion)."""
    permuted = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), seed).goal.to_numpy()
    result = sa.analyse(table, folds, permuted, weights, designs)
    rejected = [a for a, r in result["readings"].items() if r["holm_rejected"]]
    summary = {"seed": seed, "holm_rejected": rejected, "deltas": {a: d["delta_ll"] for a, d in result["deltas"].items()},
               "p_one_sided": {a: d["p_one_sided"] for a, d in result["deltas"].items()},
               "criterion": "no Holm rejection", "passed": not rejected}
    return result, summary


# ------------------------------------------------------------------ positive control
def signal(table: pd.DataFrame) -> np.ndarray:
    """d: the E3 t0 distance standardised over the population, missing values 0."""
    x = table[SIGNAL].to_numpy(float)
    finite = np.isfinite(x)
    return np.where(finite, (x - x[finite].mean()) / x[finite].std(), 0.0)


def expected_log_loss(p: np.ndarray, q: np.ndarray) -> float:
    """Mean over shots of E[log-loss of q] when y ~ Bernoulli(p)."""
    q = np.clip(q, ss.CLIP, 1 - ss.CLIP)
    return float(-np.mean(p * np.log(q) + (1 - p) * np.log(1 - q)))


def true_delta(beta: float, a: float, b: float, z: np.ndarray, d: np.ndarray) -> float:
    """True ΔLL of the generating model against its β = 0 refit (module docstring, control 4)."""
    p = sigmoid(a + b * z + beta * d)
    return expected_log_loss(p, p) - expected_log_loss(p, projection(z, p))


def projection(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The unpenalised logistic regression of the probabilities `p` on [1, z], fitted as a goal row of weight p and a
    non-goal row of weight 1 - p per shot; its probabilities."""
    x = np.concatenate([z, z]).reshape(-1, 1)
    target = np.r_[np.ones(len(z)), np.zeros(len(z))]
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10).fit(x, target, sample_weight=np.r_[p, 1 - p])
    return model.predict_proba(z.reshape(-1, 1))[:, 1]


def solve_beta(a: float, b: float, z: np.ndarray, d: np.ndarray, target: float = POSITIVE_TARGET) -> float:
    """The negative β with true ΔLL(β) = `target` (brentq on `BETA_BRACKET`)."""
    return float(brentq(lambda beta: true_delta(beta, a, b, z, d) - target, *BETA_BRACKET, xtol=1e-12))


def positive_control(designs, weights, table: pd.DataFrame, a: float, b: float, n: int = N_POSITIVE,
                     seed: int = POSITIVE_SEED, log=lambda _: None) -> tuple[pd.DataFrame, dict]:
    z, d = table[sf.Z].to_numpy(float), signal(table)
    beta = solve_beta(a, b, z, d)
    frame = simulate(designs, weights, sigmoid(a + b * z + beta * d), n, seed, log)
    mean = float(frame.delta_G.mean())
    lo, hi = POSITIVE_WINDOW
    summary = {"replicates": n, "seed": seed, "a": a, "b": b, "beta": beta, "true_delta": true_delta(beta, a, b, z, d),
               "power_G": float(frame.detectable_G.mean()), "power_GS": float(frame.detectable_GS.mean()),
               "power_any": float(frame.detectable_any.mean()), "mean_recovered_delta_G": mean,
               "sd_recovered_delta_G": float(frame.delta_G.std(ddof=1)), "mean_recovered_delta_GS": float(frame.delta_GS.mean()),
               "diagnostic_mean_oracle_delta": float(frame.delta_oracle.mean()),
               "criterion": f"mean recovered ΔLL of G in [{lo}, {hi}]", "passed": lo <= mean <= hi}
    return frame, summary
