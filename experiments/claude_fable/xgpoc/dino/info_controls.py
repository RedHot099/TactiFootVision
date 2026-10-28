"""Label-free control simulations of the xg-dino-info redundancy contract (ticket 11, C3.8), on the pattern of
`xgpoc.dino.controls` (C2): the truths are built from z_g_combo, the geometry design of B and the joint content design
Xc; no label is read.

Truths (`Truths.build`; designs fitted on every shot, linearly independent columns, `controls.independent_columns`)
    f_B    the least-squares projection of z on [1, B's design]; f_BX the same on [1, B, Xc]
    e_X    (f_BX - f_B) / s: the image signal Xc can carry beyond geometry; e_⊥ = (z - f_BX) / s: the rest; s = SD(z - f_B)
    η      c + f_B + a e_X + b e_⊥, c chosen so that the mean probability of the clustered generator is the population's
           goal rate (`controls.GOAL_RATE`); y* ~ Bernoulli(sigmoid(η + u_m)), u_m ~ N(0, 0.25²) per match and replicate
    oracle G = E LL(q_B) - E LL(q_C), G_X = E LL(q_BX) - E LL(q_CX), R = G - G_X, with y ~ the clustered marginal p̄ and q
           the unpenalised logistic projections of p̄ on [1, B], [1, z, B], [1, B, Xc], [1, z, B, Xc] (C2's construction)
Scenarios (`SCENARIOS`), each with oracle G = `G_TARGET` (xg-dino's observed C - B, 0.0097):
    null      a = 0: Xc carries none of the image signal, oracle R <= 0 for every content block at once (e_⊥ is orthogonal
              to B and Xc, so to every block); fits B, C and B+X, C+X for Xc and every family block
    plant25   (a, b) with oracle R = 0.25 G: the boundary of "explains part"; fits Xc
    plant50   (a, b) with oracle R = 0.50 G: power of "explains part" (boundary of "explains most"); fits Xc
One replicate (`replicate_row`): the arms out of fold on the primary designs and `info_protocol.redundancy` (Holm over the
    family where it is fitted, Xc single); verdicts and the raw statistics per block.
Summaries (`summary`): raw verdict rates per block with Monte Carlo SE and the 95% Wilson interval, the family-wise rate
    of a false "explains part" or "explains most" (null), mean recovered G, R, gain. Gates (`gates`): false "explains part
    / most" <= 0.05 at the null (Xc and family-wise) and at the plant25 boundary (Xc); the boundary rate of "does not
    explain" decides whether that verdict is available (`info_protocol.boundary_gate`).
Replicates are drawn per scenario from `default_rng(seed)` in order; a chunk [start, stop) gives the same rows as one
run (`controls.draw`). Fits run in forked workers with one BLAS thread each.
"""

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from xgpoc.dino import ablation as ab
from xgpoc.dino import controls as dc
from xgpoc.dino import info_protocol as ip
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf

G_TARGET = 0.0097
GOAL_RATE = dc.GOAL_RATE
CLUSTER_SD = dc.CLUSTER_SD
SCENARIOS = {"null": None, "plant25": 0.25, "plant50": 0.5}
SEEDS = {"null": 30611, "plant25": 30612, "plant50": 30613}
FITTED = {"null": (ip.PRIMARY, *ip.FAMILY), "plant25": (ip.PRIMARY,), "plant50": (ip.PRIMARY,)}
GATE_RATE = 0.05
S_BRACKET = (1e-4, 6.0)
THETA_BRACKET = (0.0, 1.5)
FALSE = ("explains part", "explains most")

Log = Callable[[str], None]


def lstsq_fit(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(z)), x])
    return x1 @ np.linalg.lstsq(x1, z, rcond=None)[0]


@dataclass(frozen=True)
class Truths:
    f_b: np.ndarray
    e_x: np.ndarray
    e_perp: np.ndarray
    designs: dict[str, np.ndarray]
    games: np.ndarray

    @classmethod
    def build(cls, prepared: ip.Prepared, table: pd.DataFrame) -> "Truths":
        b_arm = prepared.block.arms["B"]
        x_b = dc.independent_columns(sf.Preprocessor.fit(b_arm, table)(table))
        xc = [c for c in prepared.columns[ip.PRIMARY] if c not in sf.constant_columns(table, prepared.columns[ip.PRIMARY])]
        x_arm = sf.Arm("Xc", "truth", tuple(xc))
        x_x = sf.Preprocessor.fit(x_arm, table)(table)
        z = table[ab.Z].to_numpy(float)
        x_bx = dc.independent_columns(np.column_stack([x_b, x_x]))
        f_b, f_bx = lstsq_fit(x_b, z), lstsq_fit(x_bx, z)
        s = float(np.std(z - f_b))
        zs = sc.standardised(z)
        designs = {"B": x_b, "C": dc.independent_columns(np.column_stack([zs, x_b])), "BX": x_bx,
                   "CX": dc.independent_columns(np.column_stack([zs, x_bx]))}
        return cls(f_b, (f_bx - f_b) / s, (z - f_bx) / s, designs, table.game.to_numpy())

    def eta(self, a: float, b: float) -> np.ndarray:
        raw = self.f_b + a * self.e_x + b * self.e_perp
        shift = brentq(lambda c: dc.marginal(c + raw).mean() - GOAL_RATE, -20, 20, xtol=1e-12)
        return shift + raw

    def oracle(self, a: float, b: float) -> dict:
        p = dc.marginal(self.eta(a, b))
        ll = {k: sc.expected_log_loss(p, dc.projection_on(x, p)) for k, x in self.designs.items()}
        g, gx = ll["B"] - ll["C"], ll["BX"] - ll["CX"]
        return {"G": g, "G_X": gx, "R": g - gx, "gain": ll["BX"] - ll["B"]}

    def scale_for(self, theta: float, target: float = G_TARGET) -> float:
        """s with oracle G(s sin θ, s cos θ) = target."""
        return float(brentq(lambda s: self.oracle(s * np.sin(theta), s * np.cos(theta))["G"] - target, *S_BRACKET,
                            xtol=1e-6))

    def solve(self, fraction: float | None) -> tuple[float, float]:
        """(a, b) of a scenario (module docstring)."""
        if fraction is None:
            return 0.0, self.scale_for(0.0)

        def excess(theta: float) -> float:
            s = self.scale_for(theta)
            o = self.oracle(s * np.sin(theta), s * np.cos(theta))
            return o["R"] / o["G"] - fraction
        theta = float(brentq(excess, *THETA_BRACKET, xtol=1e-5))
        s = self.scale_for(theta)
        return s * np.sin(theta), s * np.cos(theta)

    def sampler(self, a: float, b: float, sd: float = CLUSTER_SD) -> Callable[[np.random.Generator], np.ndarray]:
        eta = self.eta(a, b)
        matches, index = np.unique(self.games, return_inverse=True)
        return lambda rng: sigmoid(eta + rng.normal(0.0, sd, len(matches))[index])


def solve_all(truths: Truths) -> dict:
    """Every scenario's (a, b) and its oracle statistics."""
    out = {}
    for name, fraction in SCENARIOS.items():
        a, b = truths.solve(fraction)
        out[name] = {"fraction": fraction, "a": a, "b": b, "oracle": truths.oracle(a, b)}
    return {"scenarios": out, "g_target": G_TARGET, "goal_rate": GOAL_RATE, "cluster_sd": CLUSTER_SD,
            "design_ranks": {k: int(v.shape[1]) for k, v in truths.designs.items()}}


def arms_for(blocks) -> tuple[str, ...]:
    return ("B", "C", *(f"{s}+{b}" for b in blocks for s in ("B", "C")))


def replicate_row(prepared: ip.Prepared, y: np.ndarray, blocks) -> dict:
    """One replicate (module docstring), flat."""
    p, chosen = ab.fit_block(prepared.block, y, arms_for(blocks))
    read = ip.redundancy(y, p, prepared.block.weights, blocks=blocks)["blocks"]
    row = {"goals": int(y.sum())}
    for b, r in read.items():
        row |= {f"{b}_verdict": r["raw_verdict"], f"{b}_gain_rejected": r["gain_rejected"], f"{b}_G": r["G"]["delta"],
                f"{b}_R": r["R"]["delta"], f"{b}_gain": r["gain"]["delta"], f"{b}_gain_p": r["gain"]["p_one_sided"],
                f"{b}_gain_upper": r["gain"]["upper_one_sided_95"], f"{b}_part_lower": r["part"]["lo90"],
                f"{b}_part_upper": r["part"]["upper_one_sided_95"], f"{b}_most_lower": r["most"]["lo90"],
                f"{b}_c_bx": r["C-BX"]["wording"].split(" ")[0]}
    if set(ip.FAMILY) <= set(read):
        row["family_false_part"] = any(read[b]["verdict"] in FALSE for b in ip.FAMILY)
    for arm in ("B", "C"):
        row[f"grid_edge_share_{arm}"] = float(np.mean([e for e in ab.grid_edges(chosen[arm]) if e is not None]))
    return row


_STATE: tuple | None = None


def _start(prepared: ip.Prepared, blocks) -> None:
    global _STATE
    threadpool_limits(1)
    _STATE = (prepared, blocks)


def _row(y: np.ndarray) -> dict:
    return replicate_row(_STATE[0], y, _STATE[1])


def simulate(prepared: ip.Prepared, truth, blocks, n: int, seed: int, jobs: int = 1, log: Log | None = None,
             start: int = 0) -> pd.DataFrame:
    """Replicates start..n-1 of a scenario, one row each."""
    ys = dc.draw(truth, n, seed, start)
    clock = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=get_context("fork"), initializer=_start,
                                 initargs=(prepared, blocks)) as pool:
            rows = []
            for i, row in enumerate(pool.map(_row, ys)):
                rows.append(row)
                if log and ((i + 1) % 50 == 0 or i + 1 == len(ys)):
                    log(f"  {i + 1}/{len(ys)} ({time.time() - clock:.0f} s)")
    else:
        with threadpool_limits(1):
            rows = [replicate_row(prepared, y, blocks) for y in ys]
    frame = pd.DataFrame(rows)
    frame.insert(0, "replicate", np.arange(start, n))
    return frame


def chunk(prepared: ip.Prepared, truths: Truths, solved: dict, start: int, stop: int, jobs: int = 1,
          log: Log | None = None) -> pd.DataFrame:
    """Replicates [start, stop) of every scenario."""
    frames = []
    for name in SCENARIOS:
        s = solved["scenarios"][name]
        if log:
            log(f"{name}: replicates [{start}, {stop})")
        frames.append(simulate(prepared, truths.sampler(s["a"], s["b"]), FITTED[name], stop, SEEDS[name], jobs, log,
                               start).assign(scenario=name))
    return pd.concat(frames, ignore_index=True)


def summary(frame: pd.DataFrame, name: str, oracle: dict) -> dict:
    out = {"replicates": len(frame), "oracle": oracle, "blocks": {}}
    for b in FITTED[name]:
        v = frame[f"{b}_verdict"]
        out["blocks"][b] = {"rates": {k: dc.rate(frame, v == k) for k in ip.VERDICTS},
                            "false_part_or_most": dc.rate(frame, v.isin(FALSE)),
                            "gain_rejected": dc.rate(frame, frame[f"{b}_gain_rejected"].astype(bool)),
                            "mean_G": float(frame[f"{b}_G"].mean()), "mean_R": float(frame[f"{b}_R"].mean()),
                            "mean_gain": float(frame[f"{b}_gain"].mean()),
                            "mean_part_lower": float(frame[f"{b}_part_lower"].mean())}
    if "family_false_part" in frame and frame.family_false_part.notna().all():
        out["family_false_part_or_most"] = dc.rate(frame, frame.family_false_part.astype(bool))
    out["grid_edge_share"] = {a: float(frame[f"grid_edge_share_{a}"].mean()) for a in ("B", "C")}
    return out


def gates(scenarios: dict) -> dict:
    null = scenarios["null"]
    boundary = scenarios["plant25"]["blocks"][ip.PRIMARY]["rates"]
    out = {"null_false_part_Xc": null["blocks"][ip.PRIMARY]["false_part_or_most"]["rate"],
           "null_false_part_family": null["family_false_part_or_most"]["rate"], "limit": GATE_RATE,
           "boundary_false_part_Xc": boundary["explains part"]["rate"] + boundary["explains most"]["rate"],
           "boundary_false_not_explain_Xc": boundary["does not explain"]["rate"]}
    return out | {"passed": bool(out["null_false_part_Xc"] <= GATE_RATE and out["null_false_part_family"] <= GATE_RATE
                                 and out["boundary_false_part_Xc"] <= GATE_RATE),
                  "not_explain_available": bool(out["boundary_false_not_explain_Xc"] <= GATE_RATE)}


def finish(solved: dict, frame: pd.DataFrame, n: int) -> dict:
    for name in SCENARIOS:
        if sorted(frame.replicate[frame.scenario == name]) != list(range(n)):
            raise ValueError(f"scenario {name}: the chunks do not cover replicates 0..{n - 1} exactly once")
    scenarios = {name: summary(frame[frame.scenario == name], name, solved["scenarios"][name]["oracle"]) |
                 {k: solved["scenarios"][name][k] for k in ("fraction", "a", "b")} for name in SCENARIOS}
    return {"truth": solved, "seeds": SEEDS, "fitted": {k: list(v) for k, v in FITTED.items()}, "scenarios": scenarios,
            "gates": gates(scenarios), "replicates": n}
