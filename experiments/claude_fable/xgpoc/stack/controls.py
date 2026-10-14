"""Controls 2-4 and the simulation reports of xg-stack (ticket 01; councils C5 and C6). Control 1 is
`provenance.control`. All run on simulated outcomes, or on real labels permuted within matches, before the real run.

2. Synthetic nulls (`synthetic_nulls`): y* ~ Bernoulli(p_R), p_R = sigmoid(a + b z) with (a, b) R fitted on the real
   labels of every shot under the design's model (`fit.reference_fit`). Each replicate fits R, G and GS out of fold
   and applies Holm over {G, GS}. Pass: the share of replicates with a Holm "detectable improvement" (any arm) <= 0.09.
3. Permutation (`permutation`): the full read-out (`analysis.analyse`) on the real labels permuted within matches
   (`compare.stats.permute_within_matches`). Pass: no Holm rejection.
4. Positive control (`positive_control`): y* ~ Bernoulli(sigmoid(a + b z + β s)) for a plant `s` (`plant`):
       dist        the E3 t0 distance standardised over the population, missing values 0 (C5);
       dist_angle  (d - e) standardised, d as above and e the wrapped E3 t0 angle standardised the same way: farther and
                   narrower shots score less, through both inputs (C6).
   β is the negative root (`solve_beta`) of true ΔLL(β) = target, where true ΔLL(β) = E[LL(p_β)] - E[LL(q_β)] over the
   population with y ~ p_β, and q_β is the β = 0 model refitted to p_β (`projection`). Reported: β, its true ΔLL, the
   power of G (share of replicates where G is Holm-detectable) with its Monte Carlo SE sqrt(p (1 - p) / n), and the
   mean recovered out-of-fold ΔLL of G. The gate (C6) is the dist plant at -0.002: mean recovered ΔLL of G in
   [-0.003, -0.001]. The oracle ΔLL of each replicate, LL(p_true) - LL(R out of fold) on the same y*, and its one-sided
   p from the same match draws, give the ceiling (`oracle_power`: share with p <= 0.05).
Power curve (`power_curve`): the dist plant at each true ΔLL of the grid; the smallest detectable effect
   (`smallest_detectable`) is where G's Holm power first reaches 0.8, linearly interpolated on the grid.
Clustered nulls (`clustered_nulls`): y* ~ Bernoulli(sigmoid(a + b z + u_m)) with a match random intercept
   u_m ~ N(0, τ²), τ frozen, drawn per replicate. No arm can use u_m, so the arms' true gain is nil. Reported per
   confirmatory arm: the coverage of the one-sided 95% bound (5th percentile of the match-bootstrap ΔLL) of the
   procedure's expected ΔLL (the replicates' mean, `coverage`), and its coverage of 0. The "no improvement larger than
   |bound|" wording is allowed only if the coverage of the expected ΔLL is >= 0.95 - 2 Monte Carlo SE.

Each simulation draws its replicates from its own `default_rng(seed)`, in replicate order: y* = (rng.random(n) < p)
after the replicate's match intercepts, if any.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from xgpoc.compare.stats import permute_within_matches
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import analysis as sa
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import stats as ss

N_REPLICATES = 200
NULL_MAX_DETECTABLE_RATE = 0.09
POSITIVE_TARGET = -0.002
POSITIVE_WINDOW = (-0.003, -0.001)
CURVE = (-0.002, -0.003, -0.004)
DETECTABLE_POWER = 0.8
ORACLE_ALPHA = 0.05
CLUSTER_SD = 0.25
COVERAGE_LEVEL = 0.95
BETA_BRACKET = (-5.0, 0.0)
PLANTS = ("dist", "dist_angle")

Designs = dict[str, list[sf.FoldDesign]]
Log = Callable[[str], None]


def quiet(_: str) -> None:
    return None


def mc_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / n))


def confirmatory(designs: Designs, y: np.ndarray, weights: np.ndarray, model: sf.Model, p_true: np.ndarray | None = None) -> dict:
    """One replicate: ΔLL of G and GS against R, their one-sided p, bounds and Holm readings, flat; with `p_true`,
    the oracle ΔLL and its one-sided p (module docstring)."""
    p = {name: sf.out_of_fold(designs[name], y, model) for name in ("R", *sf.CONFIRMATORY)}
    deltas = {name: ss.paired_delta(y, p[name], p["R"], weights) for name in sf.CONFIRMATORY}
    read = ss.readings(deltas, sf.CONFIRMATORY)
    row = {}
    for name in sf.CONFIRMATORY:
        row |= {f"delta_{name}": deltas[name]["delta_ll"], f"p_{name}": deltas[name]["p_one_sided"],
                f"bound_{name}": deltas[name]["bound_one_sided_95"], f"hi95_{name}": deltas[name]["hi95"],
                f"detectable_{name}": read[name]["detectable_improvement"], f"practical_{name}": read[name]["practical_candidate"]}
    if p_true is not None:
        oracle = ss.paired_delta(y, p_true, p["R"], weights)
        row |= {"delta_oracle": oracle["delta_ll"], "p_oracle": oracle["p_one_sided"]}
    return row | {"detectable_any": any(read[name]["detectable_improvement"] for name in sf.CONFIRMATORY)}


def simulate(designs: Designs, weights: np.ndarray, model: sf.Model, truth: Callable[[np.random.Generator], np.ndarray],
             n: int, seed: int, log: Log = quiet) -> pd.DataFrame:
    """`n` replicates of `confirmatory` on y* ~ Bernoulli(truth(rng)), one row each (`truth` draws any per-replicate
    random effect from the same generator before the outcomes)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p_true = truth(rng)
        y = (rng.random(len(p_true)) < p_true).astype(int)
        rows.append({"replicate": i, "goals": int(y.sum())} | confirmatory(designs, y, weights, model, p_true))
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{n}")
    return pd.DataFrame(rows)


def fixed(p: np.ndarray) -> Callable[[np.random.Generator], np.ndarray]:
    return lambda _rng: p


def power_summary(frame: pd.DataFrame) -> dict:
    """Holm power of G and GS with their Monte Carlo SE, the recovered ΔLL and the oracle ceiling of a simulation."""
    n = len(frame)
    out = {"replicates": n}
    for name in sf.CONFIRMATORY:
        power = float(frame[f"detectable_{name}"].mean())
        out |= {f"power_{name}": power, f"power_{name}_mc_se": mc_se(power, n),
                f"mean_recovered_delta_{name}": float(frame[f"delta_{name}"].mean()),
                f"sd_recovered_delta_{name}": float(frame[f"delta_{name}"].std(ddof=1))}
    if "delta_oracle" in frame:
        oracle = float((frame.p_oracle <= ORACLE_ALPHA).mean())
        out |= {"mean_oracle_delta": float(frame.delta_oracle.mean()), "oracle_power": oracle,
                "oracle_power_mc_se": mc_se(oracle, n)}
    return out


# ------------------------------------------------------------------ control 2 and 3
def synthetic_nulls(designs: Designs, weights: np.ndarray, model: sf.Model, p_r: np.ndarray, n: int, seed: int,
                    log: Log = quiet) -> tuple[pd.DataFrame, dict]:
    """Control 2 (module docstring): (one row per replicate, summary with the pass criterion)."""
    frame = simulate(designs, weights, model, fixed(p_r), n, seed, log)
    rate = float(frame.detectable_any.mean())
    summary = {"seed": seed, "detectable_rate": rate, "detectable_rate_mc_se": mc_se(rate, n),
               "detectable_rate_by_arm": {a: float(frame[f"detectable_{a}"].mean()) for a in sf.CONFIRMATORY},
               "mean_delta": {a: float(frame[f"delta_{a}"].mean()) for a in sf.CONFIRMATORY},
               "coverage": {a: coverage(frame, a) for a in sf.CONFIRMATORY}, "replicates": n,
               "criterion": f"detectable rate <= {NULL_MAX_DETECTABLE_RATE}", "passed": rate <= NULL_MAX_DETECTABLE_RATE}
    return frame, summary


def permutation(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, weights: np.ndarray, spec: sf.Spec, seed: int,
                designs: Designs | None = None) -> tuple[dict, dict]:
    """Control 3 (module docstring): (the read-out on labels permuted within matches, summary with the pass criterion)."""
    permuted = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), seed).goal.to_numpy()
    result = sa.analyse(table, folds, permuted, weights, spec, designs)
    rejected = [a for a, r in result["readings"].items() if r["holm_rejected"]]
    summary = {"seed": seed, "holm_rejected": rejected, "deltas": {a: d["delta_ll"] for a, d in result["deltas"].items()},
               "p_one_sided": {a: d["p_one_sided"] for a, d in result["deltas"].items()},
               "criterion": "no Holm rejection", "passed": not rejected}
    return result, summary


# ------------------------------------------------------------------ plants and β
def standardised(x: np.ndarray) -> np.ndarray:
    """x standardised over its finite values, missing values 0."""
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    return np.where(finite, (x - x[finite].mean()) / x[finite].std(), 0.0)


def plant(table: pd.DataFrame, kind: str) -> np.ndarray:
    """The planted signal s (module docstring, control 4)."""
    d = standardised(table[si.col("dist", 0)].to_numpy())
    if kind == "dist":
        return d
    if kind == "dist_angle":
        return standardised(d - standardised(table[si.col("angle", 0)].to_numpy()))
    raise ValueError(f"unknown plant {kind!r}")


def expected_log_loss(p: np.ndarray, q: np.ndarray) -> float:
    """Mean over shots of E[log-loss of q] when y ~ Bernoulli(p)."""
    q = np.clip(q, ss.CLIP, 1 - ss.CLIP)
    return float(-np.mean(p * np.log(q) + (1 - p) * np.log(1 - q)))


def projection(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The unpenalised logistic regression of the probabilities `p` on [1, z] (a goal row of weight p and a non-goal
    row of weight 1 - p per shot, i.e. the expected log-likelihood), by Newton's method; its probabilities."""
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(2)
    for _ in range(sf.NEWTON_ITERATIONS):
        q = sigmoid(x @ beta)
        step = np.linalg.solve((x.T * (q * (1 - q))) @ x, x.T @ (p - q))
        beta += step
        if np.abs(x @ step).max() < sf.NEWTON_TOLERANCE:
            return sigmoid(x @ beta)
    raise RuntimeError("the projection did not converge")


def true_delta(beta: float, a: float, b: float, z: np.ndarray, s: np.ndarray) -> float:
    """True ΔLL of the generating model against its β = 0 refit (module docstring, control 4)."""
    p = sigmoid(a + b * z + beta * s)
    return expected_log_loss(p, p) - expected_log_loss(p, projection(z, p))


def solve_beta(a: float, b: float, z: np.ndarray, s: np.ndarray, target: float = POSITIVE_TARGET) -> float:
    """The negative β with true ΔLL(β) = `target` (brentq on `BETA_BRACKET`)."""
    return float(brentq(lambda beta: true_delta(beta, a, b, z, s) - target, *BETA_BRACKET, xtol=1e-12))


def positive_control(designs: Designs, weights: np.ndarray, model: sf.Model, table: pd.DataFrame, a: float, b: float,
                     kind: str, n: int, seed: int, target: float = POSITIVE_TARGET, log: Log = quiet) -> tuple[pd.DataFrame, dict]:
    """Control 4 for one plant and target (module docstring): (one row per replicate, summary with β, power, recovery
    and, for the dist plant at -0.002, the pass criterion)."""
    z, s = table[sf.Z].to_numpy(float), plant(table, kind)
    beta = solve_beta(a, b, z, s, target)
    frame = simulate(designs, weights, model, fixed(sigmoid(a + b * z + beta * s)), n, seed, log)
    summary = {"plant": kind, "target": target, "seed": seed, "a": a, "b": b, "beta": beta,
               "true_delta": true_delta(beta, a, b, z, s)} | power_summary(frame)
    if kind == "dist" and target == POSITIVE_TARGET:
        lo, hi = POSITIVE_WINDOW
        mean = summary["mean_recovered_delta_G"]
        summary |= {"criterion": f"mean recovered ΔLL of G in [{lo}, {hi}]", "passed": bool(lo <= mean <= hi)}
    return frame, summary


def smallest_detectable(curve: list[dict], power: float = DETECTABLE_POWER) -> float | None:
    """The true ΔLL at which G's Holm power first reaches `power` along the grid (weakest effect first), linearly
    interpolated between grid points; None when the grid never reaches it."""
    points = sorted(curve, key=lambda c: -c["target"])
    if points[0]["power_G"] >= power:
        return float(points[0]["target"])
    for weak, strong in zip(points, points[1:], strict=False):
        if strong["power_G"] >= power:
            t = (power - weak["power_G"]) / (strong["power_G"] - weak["power_G"])
            return float(weak["target"] + t * (strong["target"] - weak["target"]))
    return None


# ------------------------------------------------------------------ clustered nulls
def coverage(frame: pd.DataFrame, arm: str) -> dict:
    """Coverage of the one-sided 95% bound of `arm`: of the replicates' mean ΔLL (the procedure's expected ΔLL) and
    of 0, with the Monte Carlo SE and the wording rule (module docstring)."""
    n, expected = len(frame), float(frame[f"delta_{arm}"].mean())
    covered = float((frame[f"bound_{arm}"] <= expected).mean())
    floor = COVERAGE_LEVEL - 2 * mc_se(COVERAGE_LEVEL, n)
    return {"expected_delta": expected, "coverage_of_expected": covered, "coverage_mc_se": mc_se(covered, n),
            "coverage_of_zero": float((frame[f"bound_{arm}"] <= 0).mean()), "floor": floor,
            "wording_allowed": bool(covered >= floor)}


def clustered_nulls(designs: Designs, weights: np.ndarray, model: sf.Model, table: pd.DataFrame, a: float, b: float,
                    n: int, seed: int, sd: float = CLUSTER_SD, log: Log = quiet) -> tuple[pd.DataFrame, dict]:
    """Clustered null simulations (module docstring): (one row per replicate, coverage per confirmatory arm)."""
    z = table[sf.Z].to_numpy(float)
    matches, index = np.unique(table.game.to_numpy(), return_inverse=True)

    def truth(rng: np.random.Generator) -> np.ndarray:
        return sigmoid(a + b * z + rng.normal(0.0, sd, len(matches))[index])
    frame = simulate(designs, weights, model, truth, n, seed, log)
    summary = {"seed": seed, "sd": sd, "replicates": n, "detectable_rate": float(frame.detectable_any.mean()),
               "coverage": {name: coverage(frame, name) for name in sf.CONFIRMATORY}}
    return frame, summary
