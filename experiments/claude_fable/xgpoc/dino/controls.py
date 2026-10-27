"""Label-free control simulations of the xg-dino ablation (tickets 01 and 01b, councils C1 and C2). Pattern:
`xgpoc.stack.controls` and `xgpoc.temporal.controls`; no label is read, so every truth is built from z_g_combo and the
geometry design.

Truths (`Truths.build`, on the primary block's B design fitted on every shot, its linearly independent columns;
label-free)
    g      the geometry-only logit: the least-squares projection of z on [1, B's design], shifted so that the mean
           probability of the clustered null generator is the population's goal rate (`GOAL_RATE`, 902 / 7,062: a
           count, not a fit)
    r      the image-specific signal: z minus that projection, standardised (B cannot use it linearly; C can)
    null   y* ~ Bernoulli(sigmoid(g + u_m)): the image adds nothing over geometry, oracle C-B = 0
    plant  y* ~ Bernoulli(sigmoid(g + β r + u_m)), β > 0 the root of oracle C-B ΔLL(β) = -0.002 under the actual
           clustered generator (C2): with u_m ~ N(0, 0.25²) (`stack.controls.CLUSTER_SD`) drawn per replicate, the
           expected log-loss of any predictor q is that of the marginal p̄ = E_u[sigmoid(g + β r + u)] (Gauss-Hermite,
           `marginal`), so oracle C-B = E[LL(q_C)] - E[LL(q_B)] with y ~ p̄ and q the unpenalised logistic projections
           of p̄ on [1, z, B's design] and on [1, B's design] (`fit.logistic_fit` with fractional y). The unclustered
           oracle of the same β is reported beside it.
    No arm can use u_m. A true C-B of +0.002 cannot be built: C nests B, so its oracle is <= 0 (disclosed, no control).
One replicate (`replicate_row`): A, B and C out of fold on the primary designs, the C-B and C-A LL contrasts with the
    paired match bootstrap, Holm over their one-sided tests and the C1 decision table (`ablation.decision`); the raw
    flags (one-sided 95% upper bound < 0, Holm rejection, TOST pass) beside the final wording.
Summaries (`summary`): mean and SD of the recovered C-B (its known overfit shift under the null, C2: no correction),
    the mean half-width of its 90% interval, the rates of each final C-B wording (adds / equivalent / inconclusive),
    of each raw flag and of "geometry adds", with Monte Carlo SE sqrt(p (1 - p) / n) and the 95% Wilson interval;
    coverage of the oracle and of the procedure's expected C-B by the 95% and 90% intervals and by the one-sided 95%
    upper bound; the grid-edge shares of B and C. Gates (`gates`, ticket 01b): false "adds" at the null <= 0.05 and
    false equivalence (final wording) at the plant <= 0.05.
Replicates are drawn from `default_rng(seed)` per scenario in replicate order; a chunk [start, stop) draws the first
stop outcome vectors and fits the last stop - start, so chunks give the same rows as one run. Fits run in forked
workers with one BLAS thread each, so results do not depend on the number of workers.
"""

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from xgpoc.dino import ablation as ab
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss

GOAL_RATE = 902 / 7062
CLUSTER_SD = sc.CLUSTER_SD
PLANT_TARGET = -ab.DELTA
SEEDS = {"null": 30521, "plant": 30522}             # C2: fresh seeds (r200 used 30501, 30502)
SCENARIOS = ("null", "plant")
ARMS = ("A", "B", "C")
BETA_BRACKET = (0.0, 5.0)
GATE_RATE = 0.05
HERMITE_NODES = 40
WILSON_Z = 1.959963984540054

Log = Callable[[str], None]


def independent_columns(x: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """The columns of `x` (in order) a pivoted QR keeps as linearly independent of [1, earlier kept] (the real B design
    has duplicated missing indicators, which make the unpenalised projection singular)."""
    x1 = np.column_stack([np.ones(len(x)), x])
    _, r, pivot = qr(x1, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    keep = np.sort(pivot[:int((diagonal > tolerance * diagonal[0]).sum())])
    return x[:, keep[keep > 0] - 1]


def projection_on(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The unpenalised logistic projection of the probabilities `p` on [1, x]; its probabilities."""
    return sf.predict(sf.logistic_fit(x, p, np.zeros(x.shape[1])), x)


def marginal(eta: np.ndarray, sd: float = CLUSTER_SD) -> np.ndarray:
    """E_u[sigmoid(eta + u)], u ~ N(0, sd²), by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(HERMITE_NODES)
    return (sigmoid(np.asarray(eta, float)[:, None] + np.sqrt(2.0) * sd * x[None, :]) @ w) / np.sqrt(np.pi)


@dataclass(frozen=True)
class Truths:
    g: np.ndarray
    r: np.ndarray
    x_b: np.ndarray
    x_c: np.ndarray
    games: np.ndarray

    @classmethod
    def build(cls, block: ab.Block, table: pd.DataFrame, goal_rate: float = GOAL_RATE) -> "Truths":
        arm = block.arms["B"]
        x_b = independent_columns(sf.Preprocessor.fit(arm, table)(table))
        z = table[ab.Z].to_numpy(float)
        x1 = np.column_stack([np.ones(len(z)), x_b])
        fitted = x1 @ np.linalg.lstsq(x1, z, rcond=None)[0]
        shift = brentq(lambda a: marginal(a + fitted).mean() - goal_rate, -20, 20, xtol=1e-12)
        r = sc.standardised(z - fitted)
        x_c = independent_columns(np.column_stack([sc.standardised(z), x_b]))
        return cls(shift + fitted, r, x_b, x_c, table.game.to_numpy())

    def oracle(self, beta: float, clustered: bool = True) -> float:
        eta = self.g + beta * self.r
        p = marginal(eta) if clustered else sigmoid(eta)
        return sc.expected_log_loss(p, projection_on(self.x_c, p)) - sc.expected_log_loss(p, projection_on(self.x_b, p))

    def beta(self, target: float = PLANT_TARGET) -> float:
        return float(brentq(lambda b: self.oracle(b) - target, *BETA_BRACKET, xtol=1e-10))

    def clustered(self, beta: float, sd: float = CLUSTER_SD) -> Callable[[np.random.Generator], np.ndarray]:
        matches, index = np.unique(self.games, return_inverse=True)
        return lambda rng: sigmoid(self.g + beta * self.r + rng.normal(0.0, sd, len(matches))[index])


def draw(truth: Callable[[np.random.Generator], np.ndarray], stop: int, seed: int, start: int = 0) -> list[np.ndarray]:
    """Outcome vectors start..stop-1 of the sequence of `default_rng(seed)` (the earlier ones are drawn and dropped)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(stop):
        p = truth(rng)
        y = (rng.random(len(p)) < p).astype(int)
        if i >= start:
            out.append(y)
    return out


def replicate_row(block: ab.Block, y: np.ndarray) -> dict:
    """One replicate (module docstring), flat."""
    p, chosen = ab.fit_block(block, y, ARMS)
    reads = {c: ab.ll_contrast(y, ab.ll_differences(y, p[a], p[b]), block.weights)[0]
             for c, (a, b) in ab.SUPERIORITY.items()}
    holm = ss.holm({c: r["p_one_sided"] for c, r in reads.items()}, ab.HOLM_ALPHA)
    decided = ab.decision(reads["C-B"], reads["C-A"], holm)
    row = {"goals": int(y.sum()), "wording_C-B": decided["C-B"].split(" ")[0], "geometry_adds": decided["C-A"].startswith("geometry")}
    row |= {f"raw_{k}": v for k, v in decided["flags"]["C-B"].items()}
    for c, r in reads.items():
        row |= {f"{c}_{k}": r[k] for k in ("delta", "lo95", "hi95", "lo90", "hi90", "upper_one_sided_95", "p_one_sided")}
        row |= {f"{c}_equivalent": r["tost"]["equivalent"], f"{c}_smallest_margin": r["tost"]["smallest_margin"],
                f"{c}_holm": holm[c]}
    for arm in ("B", "C"):
        edges = [e for e in ab.grid_edges(chosen[arm]) if e is not None]
        row[f"grid_edge_share_{arm}"] = float(np.mean(edges))
    return row


_BLOCK: ab.Block | None = None


def _start(block: ab.Block) -> None:
    global _BLOCK
    threadpool_limits(1)
    _BLOCK = block


def _row(y: np.ndarray) -> dict:
    return replicate_row(_BLOCK, y)


def simulate(block: ab.Block, truth, n: int, seed: int, jobs: int = 1, log: Log | None = None,
             start: int = 0) -> pd.DataFrame:
    """Replicates start..n-1 (all by default), one row each, `replicate` their index in the scenario's sequence."""
    ys = draw(truth, n, seed, start)
    clock = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=get_context("fork"), initializer=_start, initargs=(block,)) as pool:
            rows = []
            for i, row in enumerate(pool.map(_row, ys)):
                rows.append(row)
                if log and ((i + 1) % 100 == 0 or i + 1 == len(ys)):
                    log(f"  {i + 1}/{len(ys)} ({time.time() - clock:.0f} s)")
    else:
        with threadpool_limits(1):
            rows = [replicate_row(block, y) for y in ys]
    frame = pd.DataFrame(rows)
    frame.insert(0, "replicate", np.arange(start, n))
    return frame


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    p, d = k / n, 1 + z**2 / n
    centre, half = (p + z**2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(centre - half), float(centre + half)


def rate(frame: pd.DataFrame, mask) -> dict:
    mask = np.asarray(mask, bool)
    value, n = float(mask.mean()), len(mask)
    lo, hi = wilson(int(mask.sum()), n)
    return {"rate": value, "mc_se": sc.mc_se(value, n), "wilson95": [lo, hi], "count": int(mask.sum())}


def summary(frame: pd.DataFrame, oracle: float) -> dict:
    """The scenario's summary (module docstring)."""
    d = frame["C-B_delta"]
    expected = float(d.mean())
    out = {"replicates": len(frame), "oracle_C-B": oracle, "mean_C-B": expected, "sd_C-B": float(d.std(ddof=1)),
           "mean_half_width_90_C-B": float(((frame["C-B_hi90"] - frame["C-B_lo90"]) / 2).mean()),
           "mean_C-A": float(frame["C-A_delta"].mean())}
    for wording in ("DINOv2", "equivalent", "inconclusive"):
        out[f"rate_{wording}"] = rate(frame, frame["wording_C-B"] == wording)
    out["rate_geometry_adds"] = rate(frame, frame.geometry_adds)
    out["raw"] = {k: rate(frame, frame[f"raw_{k}"]) for k in ("upper_below_0", "holm_rejected", "tost_pass")}
    for target, value in (("oracle", oracle), ("expected", expected)):
        out[f"coverage_{target}"] = {
            "interval_95": rate(frame, (frame["C-B_lo95"] <= value) & (value <= frame["C-B_hi95"])),
            "interval_90": rate(frame, (frame["C-B_lo90"] <= value) & (value <= frame["C-B_hi90"])),
            "upper_one_sided_95": rate(frame, frame["C-B_upper_one_sided_95"] >= value)}
    out["grid_edge_share"] = {arm: float(frame[f"grid_edge_share_{arm}"].mean()) for arm in ("B", "C")}
    return out


def gates(scenarios: dict) -> dict:
    """Ticket 01b: false "adds" at the null and false equivalence at the plant, each <= `GATE_RATE`."""
    out = {"false_adds_null": scenarios["null"]["rate_DINOv2"]["rate"],
           "false_equivalence_plant": scenarios["plant"]["rate_equivalent"]["rate"], "limit": GATE_RATE}
    return out | {"passed": bool(out["false_adds_null"] <= GATE_RATE and out["false_equivalence_plant"] <= GATE_RATE)}


def truth_record(truths: Truths, beta: float) -> dict:
    return {"goal_rate": GOAL_RATE, "cluster_sd": CLUSTER_SD, "beta_plant": beta,
            "oracle_plant_clustered": truths.oracle(beta), "oracle_plant_unclustered": truths.oracle(beta, False),
            "mean_p_null": float(marginal(truths.g).mean()),
            "corr_r_geometry_max": float(np.abs(truths.x_b.T @ truths.r / len(truths.r)).max()),
            "plus_0.002": "not constructible: C nests B, its oracle C-B is <= 0", "hermite_nodes": HERMITE_NODES}


def chunk(prepared: ab.Prepared, table: pd.DataFrame, n: int, start: int, stop: int, jobs: int = 1,
          log: Log | None = None) -> tuple[dict, pd.DataFrame]:
    """Replicates [start, stop) of both scenarios of an n-replicate run: (truth record, rows)."""
    block = prepared.blocks["primary"]
    truths = Truths.build(block, table)
    beta = truths.beta()
    frames = []
    for name, b in (("null", 0.0), ("plant", beta)):
        frames.append(simulate(block, truths.clustered(b), stop, SEEDS[name], jobs, log, start).assign(scenario=name))
    return truth_record(truths, beta) | {"replicates": n}, pd.concat(frames, ignore_index=True)


def finish(truth: dict, frame: pd.DataFrame) -> dict:
    """The summary of a complete run: every replicate 0..n-1 of both scenarios exactly once."""
    n = truth["replicates"]
    for name in SCENARIOS:
        if sorted(frame.replicate[frame.scenario == name]) != list(range(n)):
            raise ValueError(f"scenario {name}: the chunks do not cover replicates 0..{n - 1} exactly once")
    oracle = {"null": 0.0, "plant": truth["oracle_plant_clustered"]}
    scenarios = {name: summary(frame[frame.scenario == name], oracle[name]) | {"beta": 0.0 if name == "null" else truth["beta_plant"]}
                 for name in SCENARIOS}
    return {"truth": truth, "seeds": SEEDS, "scenarios": scenarios, "gates": gates(scenarios)}


def run(prepared: ab.Prepared, table: pd.DataFrame, n: int, jobs: int = 1, log: Log | None = None) -> tuple[dict, pd.DataFrame]:
    """Both scenarios in one chunk: (summary, one row per replicate)."""
    truth, frame = chunk(prepared, table, n, 0, n, jobs, log)
    return finish(truth, frame), frame
