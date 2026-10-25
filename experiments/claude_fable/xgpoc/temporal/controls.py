"""The controls of xg-temporal (ticket 03; spec, "Ewaluacja"; council C8). Pattern: `xgpoc.stack.controls`.

Every control but the permutation runs on simulated outcomes only, before the protocol is committed; no label is read,
so every truth is built on z = z_g_combo itself (a = 0, b = 1: R's model holds every null's truth exactly).

Truths (`Truths`, built once per population from the pooled PCA of `pooled_scores`)
    plain        y* ~ Bernoulli(sigmoid(z)), as `fixed(p_r)` in xg-stack.
    rich         y* ~ Bernoulli(sigmoid(z + a u)) (C8 Q3 (b)): u = standardised(P w), P the PCA-K scores of the o_max
                 frame (imputed as in the arms) from one PCA fitted on every usable window frame of the population
                 pooled (label-free; the arms' PCA is per fold, a disclosed mismatch), w ~ N(0, I_K) drawn from
                 `default_rng(SEEDS["rich_projection"])`. a is the negative root of oracle S - R ΔLL(a) = scale, computed
                 analytically (`xgpoc.stack.controls.true_delta` with (a, b) = (0, 1): E[LL(p)] - E[LL(q_R)] over the
                 population with y ~ p, q_R the unpenalised logistic projection of p on [1, z]); no outcome is sampled.
                 S has a static signal to find and T nothing beyond S. Gating scale -0.004 (C8), descriptive -0.002.
    positive     y* ~ Bernoulli(sigmoid(z + β m)), m a motion plant, residualised by least squares on S's pooled
                 inputs [1, z, pc01..pcK@o_max, the missing indicators] and standardised, so S cannot use it linearly:
                     motion (gating, council C9 Q1 (b); `motion_plant`)  a fixed-seed Gaussian combination
                                  (`default_rng(SEEDS["motion_plant"])`) of every motion column of the pooled PCA
                                  (vs_start, vs_mid, slope per component, as the arm T has them), each standardised
                                  and residualised;
                     pc01 (reported only; `pc01_plant`)  the pooled PCA's first component at o_max minus at the window
                                  start (pc01_vs_start@o_max).
                 β is the negative root of oracle T - S ΔLL(β) = target, E[LL(p_β)] - E[LL(q_S)] with q_S the
                 unpenalised logistic projection of p_β on [1, S's pooled inputs] (`true_delta_beyond`); the oracle
                 against R (projection on [1, z]) is reported beside it. The control checks the pipeline, not the
                 detectability of real motion (the fold PCAs drift from the pooled one; C9).
    clustered    y* ~ Bernoulli(sigmoid(z + u_m)), a match random intercept u_m ~ N(0, CLUSTER_SD²) drawn per replicate
                 before the outcomes: no arm can use it.
Each simulation draws its replicates from its own `default_rng(seed)` in replicate order (`draw`): the truth's random
effect, if any, then y* = (rng.random(n) < p). The seeds (`SEEDS`) are fixed here, distinct from the tracer's (0), the
PCA's and the bootstrap's (0) and xg-stack's (200-206).

One replicate (`replicate_row`): R, S and T out of fold under `readout.MODEL`, ΔLL T - R, T - S and S - R with the
paired match bootstrap, Holm over {T - R, T - S} (`readout.CONFIRMATORY`), the oracle ΔLL LL(p_true) - LL(R) with its
one-sided p, and the share of outer folds whose inner CV chose the grid's smallest C (S and T).

Gates (`GATES`; a failed gate is fixed in the code, never by loosening the criterion)
    provenance   every leak assertion of `provenance.control` passes.
    plain_nulls  N_REPLICATES plain nulls: the share with a Holm rejection of either confirmatory contrast <= 0.09.
    rich_nulls   N_REPLICATES rich nulls at scale -0.004: the share with a Holm rejection of T - S <= 0.09 (C8: the
                 failure event is a T - S rejection only).
    positive     the motion plant at true T - S ΔLL -0.002: the mean recovered out-of-fold T - S ΔLL in
                 [-0.003, -0.001]; Holm power of T - S (and T - R, reported only) with Monte Carlo SE
                 sqrt(p (1 - p) / n). A miss with the recovered T - S above -0.001 (shrinkage) triggers the
                 pre-registered remedy (`REMEDY`, C9 Q2), built only then; any other miss is fixed in the code.
    permutation  after the protocol commit, inside the label seal: the read-out on the real labels permuted within
                 matches (`permutation`); no Holm rejection.
Reports (not gating)
    positive at -0.004 (the same plant), the pc01 plant at -0.002, the power curve of T - S at -0.002 / -0.003 / -0.004 (the -0.002 and -0.004
    points are the positive controls) with the smallest detectable effect at power 0.8 (`smallest_detectable`), the
    rich nulls at -0.002, and the clustered nulls: per contrast the coverage of the one-sided 95% bound of the
    procedure's expected ΔLL (`coverage`, as xg-stack); "no improvement larger than |bound|" is allowed for a contrast
    only if its coverage is >= 0.95 - 2 Monte Carlo SE. If CPU time binds, the rich nulls at -0.002 drop first, never a
    gating set (C8).

Parallel fits (`Fitter`): the outcomes are drawn in the parent in replicate order and only the fits run in worker
processes (spawned, one BLAS thread each, the designs sent once per worker), so the results do not depend on `jobs`.
"""

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import brentq

from xgpoc.compare.stats import permute_within_matches
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgpoc.temporal import features as tf
from xgpoc.temporal import readout as tr

N_REPLICATES = 200
NULL_MAX_REJECTION_RATE = 0.09
POSITIVE_TARGET = -0.002
POSITIVE_REPORTED = -0.004
POSITIVE_WINDOW = (-0.003, -0.001)
CURVE = (-0.002, -0.003, -0.004)
RICH_SCALE = -0.004                     # C8 Q3: oracle S - R ΔLL of the gating rich null
RICH_SCALE_EXTRA = -0.002               # C8: reported only
DETECTABLE_POWER = sc.DETECTABLE_POWER  # 0.8
ORACLE_ALPHA = sc.ORACLE_ALPHA          # 0.05
CLUSTER_SD = sc.CLUSTER_SD              # 0.25 logit
COVERAGE_LEVEL = sc.COVERAGE_LEVEL      # 0.95
BETA_BRACKET = sc.BETA_BRACKET          # (-5, 0)
PLANT_COMPONENT = 1                     # the pc01 plant's PCA component
MOTION_CONTRAST = "T-S"
SEEDS = {"plain_nulls": 30301, "rich_projection": 30302, "rich_nulls": 30303, "rich_nulls_-0.002": 30304,
         "positive_-0.002": 30305, "positive_-0.004": 30306, "curve_-0.003": 30307, "clustered_nulls": 30308,
         "permutation": 30309, "motion_plant": 30310, "positive_pc01_-0.002": 30311}
REMEDY_SEEDS = {name: seed + 100 for name, seed in SEEDS.items() if name not in ("rich_projection", "motion_plant")}
TRACER_SEED = 0                         # scripts/xgt_run.py --synthetic's default
GATES = ("provenance", "plain_nulls", "rich_nulls", "positive", "permutation")
LABEL_FREE_GATES = GATES[:-1]
SIMULATIONS = ("plain_nulls", "rich_nulls", "positive_-0.002", "positive_-0.004", "curve_-0.003", "clustered_nulls",
               "positive_pc01_-0.002", "rich_nulls_-0.002")     # run order; the last is the first to drop when CPU time binds (C8)
CONTRASTS = (*tr.CONFIRMATORY, *tr.DESCRIPTIVE)

Designs = dict[str, list[sf.FoldDesign]]
Log = Callable[[str], None]
Truth = Callable[[np.random.Generator], np.ndarray]


def quiet(_: str) -> None:
    return None


mc_se = sc.mc_se


REMEDY = {
    "source": "council C9 Q2 (b), pre-registered; built only on a miss, not before",
    "trigger": f"the positive gate fails with the mean recovered T-S ΔLL above {POSITIVE_WINDOW[1]} (shrinkage); a miss "
               f"below {POSITIVE_WINDOW[0]} is a code bug, fixed in the code",
    "change": "T keeps, per outer fold, the C that S's inner CV chose for the static block (z unpenalised, the S "
              "columns at 1 / C_S) and chooses a separate C_M for the motion block by the same inner match-grouped CV "
              "over the C8 grid, implemented exactly by rescaling the motion columns by sqrt(C_M / C_S) under the "
              "common penalty 1 / C_S, so T still nests S; S, R, the grid and every criterion unchanged",
    "runs": "once; then every label-free gate (provenance, plain nulls, rich nulls, positive) and every report rerun on "
            "the fresh seeds below (the truths' projection seeds rich_projection and motion_plant unchanged)",
    "seeds": REMEDY_SEEDS,
    "second_miss": "a second positive-control miss stops the phase and goes to the council (D9)",
    "refreeze": "the protocol is re-frozen after the code change, with the reason, before any label",
}


def settings() -> dict:
    """The controls' frozen settings (module docstring), as the protocol records them."""
    lo, hi = POSITIVE_WINDOW
    return {
        "replicates": N_REPLICATES, "seeds": dict(SEEDS), "simulations": list(SIMULATIONS),
        "seeds_fresh": {"tracer": TRACER_SEED, "pca": tf.PCA_SEED, "bootstrap": ss.BOOT_SEED,
                        "distinct": not ({TRACER_SEED, tf.PCA_SEED, ss.BOOT_SEED, *range(200, 207)}
                                         & {*SEEDS.values(), *REMEDY_SEEDS.values()})
                        and not set(SEEDS.values()) & set(REMEDY_SEEDS.values())},
        "gates": list(GATES),
        "provenance": "every leak assertion of xgpoc.temporal.provenance.control on the population and the arms' columns",
        "plain_nulls": {"truth": "y* ~ Bernoulli(sigmoid(z_g_combo))",
                        "pass": f"share of replicates with a Holm rejection of T-R or T-S <= {NULL_MAX_REJECTION_RATE}"},
        "rich_nulls": {"truth": "y* ~ Bernoulli(sigmoid(z_g_combo + a u)), u = standardised(P w), P the o_max PCA-32 "
                                "scores (imputed as in the arms) of one PCA pooled over the population's usable window "
                                "frames, w ~ N(0, I_32) from default_rng(seeds.rich_projection); a the negative root of "
                                "the analytic oracle S-R ΔLL(a) = scale (projection on [1, z])",
                       "scale": RICH_SCALE, "extra_scale_reported": RICH_SCALE_EXTRA,
                       "pass": f"share of replicates with a Holm rejection of T-S <= {NULL_MAX_REJECTION_RATE}"},
        "positive": {"plant": "council C9 Q1 (b): a Gaussian combination, w ~ N(0, I) from "
                              "default_rng(seeds.motion_plant), of every motion column of the pooled PCA (vs_start, "
                              "vs_mid, slope of pc01..pc32, as arm T has them), each standardised and residualised by "
                              "least squares on S's pooled inputs [1, z, pc01..pc32@o_max, the missing indicators] "
                              "(constant and linearly dependent columns dropped); the combination standardised",
                     "second_plant": {"plant": f"pc0{PLANT_COMPONENT}_vs_start of the pooled PCA, residualised the same "
                                               "way and standardised", "target": POSITIVE_TARGET,
                                      "simulation": "positive_pc01_-0.002", "gating": False},
                     "truth": "y* ~ Bernoulli(sigmoid(z_g_combo + β m)), β the negative root of the analytic oracle T-S "
                              "ΔLL(β) = target (E[LL(p)] - E[LL(q_S)], q_S the unpenalised logistic projection of p on "
                              "[1, S's pooled inputs]); the oracle against R reported beside it",
                     "targets": [POSITIVE_TARGET, POSITIVE_REPORTED], "gated_contrast": MOTION_CONTRAST,
                     "pass": f"at {POSITIVE_TARGET}: mean recovered {MOTION_CONTRAST} ΔLL in [{lo}, {hi}]",
                     "reported": "Holm power of T-S and T-R with Monte Carlo SE sqrt(p (1 - p) / n), the oracle ΔLL "
                                 "and its unadjusted power, the grid-floor share of S and T"},
        "remedy": REMEDY,
        "power_curve": {"contrast": MOTION_CONTRAST, "targets": list(CURVE), "seeds": "positive_-0.002, curve_-0.003, "
                        "positive_-0.004", "smallest_detectable": f"Holm power >= {DETECTABLE_POWER}, linear "
                                                                  "interpolation along the grid"},
        "clustered_nulls": {"truth": f"y* ~ Bernoulli(sigmoid(z_g_combo + u_m)), u_m ~ N(0, {CLUSTER_SD}²) per match "
                                     "and replicate", "coverage": "share of replicates whose one-sided 95% bound is <= "
                                                                  "the replicates' mean ΔLL",
                            "wording": f"'no improvement larger than |bound|' only at coverage >= {COVERAGE_LEVEL} - 2 "
                                       "Monte Carlo SE", "gating": False},
        "permutation": {"labels": "real goal labels permuted within matches inside the label seal, after the protocol "
                                  "commit (compare.stats.permute_within_matches)", "pass": "no Holm rejection",
                        "seed": SEEDS["permutation"]},
        "monte_carlo_se": "sqrt(p (1 - p) / n) for every share",
        "time_budget": "if CPU time binds, rich_nulls_-0.002 drops first, never a gating set (C8)",
        "parallel": "outcomes drawn in the parent in replicate order; fits in spawned workers with one BLAS thread "
                    "each; results do not depend on the number of workers",
    }


# ------------------------------------------------------------------ pooled PCA and plants
def pooled_scores(data: tf.DesignData, seed: int = tf.PCA_SEED) -> np.ndarray:
    """(shots, window offsets, k) PCA-k scores of every window frame (imputed as in the arms) from one PCA fitted on
    every usable window frame of the population pooled (label-free)."""
    shots = np.arange(len(data.z))
    pca = tf.FramePCA.fit(data.embeddings[tf.training_frames(data.position, shots)], data.k, seed)
    take, _ = data.imputation
    return tf.window_scores(pca, data.embeddings, data.position, take, shots)


def standardised(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def independent_columns(x: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """The columns of `x` (in their order) that a pivoted QR keeps as linearly independent of [1, earlier kept]."""
    x1 = np.column_stack([np.ones(len(x)), x])
    _, r, pivot = qr(x1, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    keep = np.sort(pivot[:int((diagonal > tolerance * diagonal[0]).sum())])
    return x[:, keep[keep > 0] - 1]


def static_inputs(data: tf.DesignData, scores: np.ndarray) -> np.ndarray:
    """S's pooled inputs without the intercept: z, the o_max scores and the missing indicators, standardised, with
    constant and linearly dependent columns dropped (the leading run of a cut makes several indicators equal), so the
    unpenalised projection on them is well defined."""
    _, missing = data.imputation
    x = np.column_stack([data.z, scores[:, -1, :], np.asarray(missing, float)])
    x = x[:, x.std(axis=0) > 1e-12]
    return independent_columns((x - x.mean(axis=0)) / x.std(axis=0))


def rich_signal(scores: np.ndarray, seed: int = SEEDS["rich_projection"]) -> np.ndarray:
    """u = standardised(P w): P the o_max scores, w ~ N(0, I_k) from `default_rng(seed)`."""
    w = np.random.default_rng(seed).normal(size=scores.shape[2])
    return standardised(scores[:, -1, :] @ w)


def residualised(raw: np.ndarray, x_static: np.ndarray) -> np.ndarray:
    """The least-squares residual of each column of `raw` on [1, x_static]."""
    x = np.column_stack([np.ones(len(raw)), x_static])
    return raw - x @ np.linalg.lstsq(x, raw, rcond=None)[0]


def pc01_plant(data: tf.DesignData, scores: np.ndarray, component: int = PLANT_COMPONENT) -> np.ndarray:
    """The reported plant: the pooled component's o_max - start score, residualised on [1, S's pooled inputs],
    standardised."""
    raw = tf.motion_scores(scores, data.window, "vs_start")[:, component - 1]
    return standardised(residualised(raw, static_inputs(data, scores)))


def motion_columns(data: tf.DesignData, scores: np.ndarray) -> np.ndarray:
    """Every motion column of the pooled PCA (T's kinds, components in order), standardised, constant ones dropped."""
    raw = np.column_stack([tf.motion_scores(scores, data.window, kind) for kind in tf.motion_kinds(data.window)])
    raw = raw[:, raw.std(axis=0) > 1e-12]
    return (raw - raw.mean(axis=0)) / raw.std(axis=0)


def motion_plant(data: tf.DesignData, scores: np.ndarray, seed: int = SEEDS["motion_plant"]) -> np.ndarray:
    """The gating plant (C9 Q1): standardised(R w), R the motion columns each residualised on [1, S's pooled inputs],
    w ~ N(0, I) from `default_rng(seed)`."""
    columns = residualised(motion_columns(data, scores), static_inputs(data, scores))
    w = np.random.default_rng(seed).normal(size=columns.shape[1])
    return standardised(columns @ w)


def projection_on(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The unpenalised logistic regression of the probabilities `p` on [1, x]; its probabilities."""
    beta = sf.logistic_fit(x, p, np.zeros(x.shape[1]))
    return sf.predict(beta, x)


def true_delta_beyond(beta: float, z: np.ndarray, s: np.ndarray, x: np.ndarray) -> float:
    """E[LL(p_β)] - E[LL(q)] with p_β = sigmoid(z + β s) and q its projection on [1, x] (module docstring)."""
    p = sigmoid(np.asarray(z, float) + beta * s)
    return sc.expected_log_loss(p, p) - sc.expected_log_loss(p, projection_on(x, p))


def solve_beyond(z: np.ndarray, s: np.ndarray, x: np.ndarray, target: float) -> float:
    """The negative β with `true_delta_beyond` = target (brentq on `BETA_BRACKET`)."""
    return float(brentq(lambda b: true_delta_beyond(b, z, s, x) - target, *BETA_BRACKET, xtol=1e-12))


@dataclass(frozen=True)
class Truths:
    """The label-free truths of one population (module docstring)."""

    z: np.ndarray
    games: np.ndarray
    u: np.ndarray                       # rich-null signal
    m: np.ndarray                       # gating motion plant
    m_pc01: np.ndarray                  # reported pc01 plant
    x_static: np.ndarray                # S's pooled inputs

    @classmethod
    def build(cls, data: tf.DesignData, games: np.ndarray) -> "Truths":
        scores = pooled_scores(data)
        return cls(np.asarray(data.z, float), np.asarray(games), rich_signal(scores), motion_plant(data, scores),
                   pc01_plant(data, scores),
                   static_inputs(data, scores))

    @property
    def x_reference(self) -> np.ndarray:
        return self.z.reshape(-1, 1)

    def rich(self, scale: float) -> tuple[np.ndarray, dict]:
        a = sc.solve_beta(0.0, 1.0, self.z, self.u, scale)
        return sigmoid(self.z + a * self.u), {"scale": scale, "a": a, "oracle_S-R": sc.true_delta(a, 0.0, 1.0, self.z, self.u),
                                              "w_seed": SEEDS["rich_projection"]}

    def positive(self, target: float, plant: str = "motion") -> tuple[np.ndarray, dict]:
        m = {"motion": self.m, "pc01": self.m_pc01}[plant]
        beta = solve_beyond(self.z, m, self.x_static, target)
        detail = {"w_seed": SEEDS["motion_plant"]} if plant == "motion" else {"component": PLANT_COMPONENT}
        return sigmoid(self.z + beta * m), {
            "plant": plant, **detail, "target": target, "beta": beta,
            "oracle_T-S": true_delta_beyond(beta, self.z, m, self.x_static),
            "oracle_T-R": true_delta_beyond(beta, self.z, m, self.x_reference),
            "plant_corr_static_max": float(np.abs(self.x_static.T @ m / len(m)).max())}

    def clustered(self, sd: float = CLUSTER_SD) -> Truth:
        matches, index = np.unique(self.games, return_inverse=True)
        return lambda rng: sigmoid(self.z + rng.normal(0.0, sd, len(matches))[index])


def fixed(p: np.ndarray) -> Truth:
    return lambda _rng: p


# ------------------------------------------------------------------ fits, serial or in worker processes
_DESIGNS: Designs | None = None


def _start_worker(designs: Designs) -> None:
    from threadpoolctl import threadpool_limits

    global _DESIGNS
    threadpool_limits(1)
    _DESIGNS = designs


def _fit(y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    return tr.out_of_fold(_DESIGNS, y)


class Fitter:
    """Out-of-fold fits of every arm for many outcome vectors, in `jobs` spawned worker processes (one BLAS thread
    each) or in this process when jobs == 1; a context manager that owns its pool."""

    def __init__(self, designs: Designs, jobs: int = 1):
        self.designs, self.jobs, self.pool = designs, max(1, int(jobs)), None

    def __enter__(self) -> "Fitter":
        if self.jobs > 1:
            self.pool = ProcessPoolExecutor(self.jobs, mp_context=get_context("spawn"), initializer=_start_worker,
                                            initargs=(self.designs,))
        return self

    def __exit__(self, *exc) -> None:
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None

    def map(self, ys: Iterable[np.ndarray]) -> Iterable[tuple[dict[str, np.ndarray], dict[str, list]]]:
        if self.pool is None:
            return (tr.out_of_fold(self.designs, y) for y in ys)
        return self.pool.map(_fit, ys)


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


# ------------------------------------------------------------------ replicates
def draw(truth: Truth, n: int, seed: int, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` (p_true, y*) pairs from `default_rng(seed)` in replicate order (module docstring)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = truth(rng)
        if len(p) != size:
            raise ValueError("the truth does not cover the population")
        out.append((p, (rng.random(size) < p).astype(int)))
    return out


def grid_floor_share(chosen: list) -> float | None:
    """Share of outer folds whose inner CV chose the grid's smallest C (None for an arm with z alone)."""
    values = [c for c in chosen if c is not None]
    return float(np.mean([c == tr.C_GRID[0] for c in values])) if values else None


def replicate_row(y: np.ndarray, p_true: np.ndarray, probabilities: dict[str, np.ndarray], chosen: dict[str, list],
                  weights: np.ndarray) -> dict:
    """One replicate, flat (module docstring)."""
    result = tr.contrasts(y, probabilities, weights)
    row = {"events": int(y.sum())}
    for name, d in result["deltas"].items():
        row |= {f"delta_{name}": d["delta_ll"], f"p_{name}": d["p_one_sided"], f"bound_{name}": d["bound_one_sided_95"],
                f"hi95_{name}": d["hi95"]}
    for name, r in result["readings"].items():
        row |= {f"holm_{name}": r["holm_rejected"], f"practical_{name}": r["practical_candidate"]}
    oracle = ss.paired_delta(y, p_true, probabilities["R"], weights)
    row |= {"delta_oracle": oracle["delta_ll"], "p_oracle": oracle["p_one_sided"],
            "holm_any": any(result["readings"][n]["holm_rejected"] for n in tr.CONFIRMATORY)}
    return row | {f"floor_share_{arm}": grid_floor_share(chosen[arm]) for arm in ("S", "T")}


def simulate(fitter: Fitter, weights: np.ndarray, truth: Truth, n: int, seed: int, log: Log = quiet) -> pd.DataFrame:
    """`n` replicates of `replicate_row` on y* ~ Bernoulli(truth(rng)), one row each, in replicate order."""
    pairs = draw(truth, n, seed, weights.shape[1])
    rows, clock = [], time.time()
    for i, ((p_true, y), (probabilities, chosen)) in enumerate(zip(pairs, fitter.map([y for _, y in pairs]), strict=True)):
        rows.append({"replicate": i} | replicate_row(y, p_true, probabilities, chosen, weights))
        if (i + 1) % 25 == 0 or i + 1 == n:
            log(f"  {i + 1}/{n} ({time.time() - clock:.0f} s)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summaries
def rate(frame: pd.DataFrame, column: str) -> dict:
    value = float(frame[column].mean())
    return {"rate": value, "mc_se": mc_se(value, len(frame))}


def mean_deltas(frame: pd.DataFrame) -> dict:
    return {name: float(frame[f"delta_{name}"].mean()) for name in CONTRASTS}


def floor_shares(frame: pd.DataFrame) -> dict:
    return {arm: float(frame[f"floor_share_{arm}"].mean()) for arm in ("S", "T")}


def coverage(frame: pd.DataFrame, name: str) -> dict:
    return sc.coverage(frame, name)


def null_summary(frame: pd.DataFrame, seed: int, gate: str | None) -> dict:
    """A null simulation: Holm rejection rates (any, per contrast), mean ΔLL, bound coverage, grid-floor shares; with
    `gate` ("any" or a contrast) the pass criterion on that rejection rate."""
    out = {"seed": seed, "replicates": len(frame), "holm_rejection_any": rate(frame, "holm_any"),
           "holm_rejection": {name: rate(frame, f"holm_{name}") for name in tr.CONFIRMATORY},
           "mean_delta": mean_deltas(frame), "coverage": {name: coverage(frame, name) for name in CONTRASTS},
           "grid_floor_share": floor_shares(frame)}
    if gate is not None:
        observed = out["holm_rejection_any"] if gate == "any" else out["holm_rejection"][gate]
        what = "either confirmatory contrast" if gate == "any" else gate
        out |= {"criterion": f"share of replicates with a Holm rejection of {what} <= {NULL_MAX_REJECTION_RATE}",
                "passed": bool(observed["rate"] <= NULL_MAX_REJECTION_RATE)}
    return out


def power_summary(frame: pd.DataFrame) -> dict:
    """Holm power of each confirmatory contrast with its Monte Carlo SE, the recovered ΔLL and the oracle ceiling."""
    n, out = len(frame), {"replicates": len(frame)}
    for name in tr.CONFIRMATORY:
        power = float(frame[f"holm_{name}"].mean())
        out |= {f"power_{name}": power, f"power_{name}_mc_se": mc_se(power, n)}
    for name in CONTRASTS:
        out |= {f"mean_recovered_delta_{name}": float(frame[f"delta_{name}"].mean()),
                f"sd_recovered_delta_{name}": float(frame[f"delta_{name}"].std(ddof=1)) if n > 1 else None}
    oracle = float((frame.p_oracle <= ORACLE_ALPHA).mean())
    return out | {"mean_oracle_delta": float(frame.delta_oracle.mean()), "oracle_power": oracle,
                  "oracle_power_mc_se": mc_se(oracle, n), "grid_floor_share": floor_shares(frame)}


def positive_summary(frame: pd.DataFrame, seed: int, plant: dict) -> dict:
    out = {"seed": seed, **plant, **power_summary(frame)}
    if plant["target"] == POSITIVE_TARGET and plant.get("plant", "motion") == "motion":
        lo, hi = POSITIVE_WINDOW
        mean = out[f"mean_recovered_delta_{MOTION_CONTRAST}"]
        out |= {"criterion": f"mean recovered {MOTION_CONTRAST} ΔLL in [{lo}, {hi}]", "passed": bool(lo <= mean <= hi)}
    return out


def smallest_detectable(curve: list[dict], contrast: str = MOTION_CONTRAST, power: float = DETECTABLE_POWER) -> float | None:
    """The true ΔLL at which `contrast`'s Holm power first reaches `power` along the grid (weakest effect first),
    linearly interpolated; None when the grid never reaches it."""
    renamed = [{"target": c["target"], "power_G": c[f"power_{contrast}"]} for c in curve]
    return sc.smallest_detectable(renamed, power)


def clustered_summary(frame: pd.DataFrame, seed: int, sd: float = CLUSTER_SD) -> dict:
    return {"seed": seed, "sd": sd, "replicates": len(frame), "holm_rejection_any": rate(frame, "holm_any"),
            "mean_delta": mean_deltas(frame), "coverage": {name: coverage(frame, name) for name in CONTRASTS}}


# ------------------------------------------------------------------ the label-free controls
def run_label_free(fitter: Fitter, weights: np.ndarray, truths: Truths, provenance: dict, n: int = N_REPLICATES,
                   seeds: dict | None = None, simulations: tuple[str, ...] = SIMULATIONS,
                   log: Log = quiet) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every label-free gate and report (module docstring) in `SIMULATIONS` order: (replicate frames by simulation,
    summary with `gates`, `reports`, `timing_seconds` and `label_free_passed`). `simulations` may leave out
    reports, never a gate (`run_label_free` refuses)."""
    seeds = SEEDS if seeds is None else seeds
    if missing := [s for s in ("plain_nulls", "rich_nulls", "positive_-0.002") if s not in simulations]:
        raise ValueError(f"the gating simulations {missing} cannot be left out")
    frames, plants, timing = {}, {}, {}

    def run(name: str, truth: Truth) -> pd.DataFrame:
        log(f"{name}: {n} replicates, seed {seeds[name]}")
        clock = time.time()
        frames[name] = simulate(fitter, weights, truth, n, seeds[name], log)
        timing[name] = round(time.time() - clock, 1)
        return frames[name]

    gates = {"provenance": {"passed": bool(provenance["passed"]), "criterion": "every leak assertion passes",
                            "checks": provenance["checks"]}}
    reports = {}
    curve = {}
    for name in simulations:
        if name == "plain_nulls":
            gates["plain_nulls"] = null_summary(run(name, fixed(sigmoid(truths.z))), seeds[name], "any")
        elif name.startswith("rich_nulls"):
            scale = RICH_SCALE if name == "rich_nulls" else float(name.rsplit("_", 1)[1])
            p, plants[name] = truths.rich(scale)
            summary = null_summary(run(name, fixed(p)), seeds[name], MOTION_CONTRAST if name == "rich_nulls" else None)
            (gates if name == "rich_nulls" else reports)[name] = summary | {"truth": plants[name]}
        elif name.startswith(("positive_", "curve_")):
            target = float(name.rsplit("_", 1)[1])
            p, plants[name] = truths.positive(target, "pc01" if name.startswith("positive_pc01") else "motion")
            summary = positive_summary(run(name, fixed(p)), seeds[name], plants[name])
            if not name.startswith("positive_pc01"):
                curve[target] = summary
            if name == "positive_-0.002":
                gates["positive"] = summary
            else:
                reports[name] = summary
        elif name == "clustered_nulls":
            reports[name] = clustered_summary(run(name, truths.clustered()), seeds[name])
        else:
            raise ValueError(f"unknown simulation {name!r}")
    points = [curve[t] for t in sorted(curve, reverse=True)]
    reports["power_curve"] = {"contrast": MOTION_CONTRAST, "targets": [p["target"] for p in points],
                              "power": [p[f"power_{MOTION_CONTRAST}"] for p in points],
                              "power_mc_se": [p[f"power_{MOTION_CONTRAST}_mc_se"] for p in points],
                              "power_T-R": [p["power_T-R"] for p in points],
                              "smallest_detectable": smallest_detectable(points) if points else None,
                              "power_level": DETECTABLE_POWER}
    passed = all(gates[g]["passed"] for g in LABEL_FREE_GATES)
    return frames, {"gates": gates, "reports": reports, "timing_seconds": timing, "replicates": n, "seeds": dict(seeds),
                    "label_free_passed": bool(passed)}


# ------------------------------------------------------------------ the permutation (after the protocol commit)
def permutation(designs: Designs, weights: np.ndarray, y: np.ndarray, games: np.ndarray,
                seed: int = SEEDS["permutation"]) -> dict:
    """The read-out on `y` permuted within matches (`compare.stats.permute_within_matches`): ΔLL, one-sided p and the
    Holm readings only (neither the labels nor the probabilities are returned); passes without a Holm rejection."""
    permuted = permute_within_matches(pd.DataFrame({"y": np.asarray(y, int)}), np.asarray(games), seed).y.to_numpy()
    probabilities, chosen = tr.out_of_fold(designs, permuted)
    result = tr.contrasts(permuted, probabilities, weights)
    rejected = [n for n, r in result["readings"].items() if r["holm_rejected"]]
    return {"seed": seed, "deltas": {n: {k: d[k] for k in ("delta_ll", "lo95", "hi95", "p_one_sided")}
                                     for n, d in result["deltas"].items()},
            "holm_rejected": rejected, "chosen_c": chosen, "criterion": "no Holm rejection", "passed": not rejected}
