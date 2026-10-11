"""Power simulation that picks the H2 equivalence margin (xg-vendor ticket 08, decision D5).

The design and the rule are fixed in `research/xg_vendor_delta_rule.md` (Polish), written before the first run and
pinned here by `RULE_SHA256`; `scripts/vendor_power.py` refuses another note. Development data only.

Basis
    the primary population of the development evaluation table (`ladder.analysis_population`): positions
    probability (freeze-frame rung), frame logit, match, SoccerNet goal label.
Truth (labels are drawn from it)
    development   logistic regression of the label on logit(positions) and the frame logit, fitted on the basis
    null          the same without the frame logit: the frame adds nothing beyond positions
One simulated study
    draw the basis matches with replacement (each copy is its own match), keep their shots' predictions, draw new
    labels from the truth, and run H2 with `ladder.h2_draws` (the analysis code: match-grouped cross-fitted Platt and
    stack, seeded folds, count-weighted match bootstrap, TOST at 5% per side).
    weaker frame     Gaussian noise N(0, sigma^2) on the frame logit, fresh per study; sigma makes the frame alone,
                     Platt-recalibrated, `WEAKER_FRAME_LOG_LOSS` worse in expected log-loss against the truth on the basis
    combined         the same matches, each with round(rate * n_match) extra shots drawn with replacement from its own
                     shots, rate = `EXTRA_SHOTS` / basis shots (about 50 recovered goals + 650 StatsBomb-only shots);
                     the secondary shots are assumed to look like the primary shots of their match
Power at a margin = share of studies whose TOST concludes equivalence. The rule (`choose_margin`): the smallest margin in
`MARGINS` whose decision power, the minimum over `DECISION_SCENARIOS`, is at least `POWER_TARGET`; otherwise none.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.eval.calibration import EPS, logit
from xgpoc.vendor import ladder as ld

RULE_NOTE = Path("research/xg_vendor_delta_rule.md")   # relative to experiments/claude_fable
RULE_SHA256 = "766ee11b3ba5ea33d7a6e9dce8bce691fbc9468815cfe5f7b6b320032e5ff30a"
MARGINS = (0.002, 0.003, 0.005)
POWER_TARGET = 0.80
WEAKER_FRAME_LOG_LOSS = 0.002
EXTRA_SHOTS = 50 + 650
N_SIMS = 1000
SIM_SEED = 0
NOISE_DRAWS = 64          # common noise draws behind the expected log-loss of the weaker frame (one draw varies by ~0.001)
SIGMA_TOLERANCE = 1e-4
TRUTHS = ("development", "null")
FRAMES = ("archived", "weaker")
POPULATIONS = ("primary", "combined")


@dataclass(frozen=True)
class Scenario:
    population: str
    truth: str
    frame: str

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS or self.truth not in TRUTHS or self.frame not in FRAMES:
            raise ValueError(f"unknown scenario {self}")

    @property
    def name(self) -> str:
        return f"{self.population}/{self.truth}/{self.frame}"


SCENARIOS = tuple(Scenario(p, t, f) for p in POPULATIONS for t in TRUTHS for f in FRAMES)   # index = generator stream
DECISION_SCENARIOS = (Scenario("primary", "development", "archived"), Scenario("primary", "development", "weaker"))


def rule_note_sha256(root: Path) -> str:
    """sha256 of the rule note; raises when it is not the pinned note."""
    digest = hashlib.sha256((Path(root) / RULE_NOTE).read_bytes()).hexdigest()
    if digest != RULE_SHA256:
        raise ValueError(f"{RULE_NOTE} has sha256 {digest}, the rule pinned {RULE_SHA256}")
    return digest


# ------------------------------------------------------------------ basis and truth
@dataclass
class Basis:
    """Arrays over the basis shots; `groups` holds the row indices of each match (sorted match keys)."""

    y: np.ndarray
    positions: np.ndarray
    frame_logit: np.ndarray
    matches: np.ndarray

    @property
    def groups(self) -> list[np.ndarray]:
        _, inverse = np.unique(self.matches, return_inverse=True)
        return [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]


def basis_from_table(table: pd.DataFrame) -> Basis:
    rows, _ = ld.analysis_population(table, ("primary",))
    return Basis(rows.goal_sn.to_numpy(int), np.clip(rows.pred_freeze_frame.to_numpy(float), EPS, 1 - EPS),
                 rows.frame_logit_0.to_numpy(float), rows.match.to_numpy())


def _features(positions: np.ndarray, frame_logit: np.ndarray | None) -> np.ndarray:
    columns = [logit(positions)] if frame_logit is None else [ld.frame_feature(frame_logit), logit(positions)]
    return np.column_stack(columns)


def truth_probabilities(basis: Basis, truth: str) -> tuple[np.ndarray, list[float]]:
    """(q, coefficients: intercept, then frame and positions or positions alone) of the truth fitted on the basis."""
    if truth not in TRUTHS:
        raise ValueError(f"unknown truth {truth!r}")
    x = _features(basis.positions, basis.frame_logit if truth == "development" else None)
    model = ld.fit_logistic(x, basis.y)
    return model.predict_proba(x)[:, 1], np.r_[model.intercept_, model.coef_[0]].tolist()


def expected_recalibrated_log_loss(q: np.ndarray, z: np.ndarray) -> float:
    """Expected log-loss against the truth q of the best Platt recalibration of logit z (fitted on soft labels)."""
    model = ld.fit_logistic(np.r_[z, z], np.r_[np.ones(len(z)), np.zeros(len(z))], sample_weight=np.r_[q, 1 - q])
    p = np.clip(model.predict_proba(z.reshape(-1, 1))[:, 1], EPS, 1 - EPS)
    return float(-np.mean(q * np.log(p) + (1 - q) * np.log(1 - p)))


def weaker_frame_sigma(q: np.ndarray, frame_logit: np.ndarray, target: float, rng: np.random.Generator,
                       n_noise: int = NOISE_DRAWS) -> float:
    """Noise SD on the frame logit that raises the frame's expected recalibrated log-loss by `target` (bisection,
    the same `n_noise` noise draws at every sigma)."""
    z = ld.frame_feature(frame_logit)
    base = expected_recalibrated_log_loss(q, z)
    noise = rng.standard_normal((n_noise, len(z)))

    def excess(sigma: float) -> float:
        return float(np.mean([expected_recalibrated_log_loss(q, z + sigma * e) for e in noise])) - base - target

    lo, hi = 0.0, 0.5
    while excess(hi) < 0:
        lo, hi = hi, 2 * hi
        if hi > 64:
            raise ValueError("no noise level reaches the target log-loss")
    while hi - lo > SIGMA_TOLERANCE:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if excess(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


# ------------------------------------------------------------------ simulation
def simulate_study(rng: np.random.Generator, basis: Basis, groups: list[np.ndarray], q: np.ndarray, scenario: Scenario,
                   sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(basis rows, match copy index, y, frame logit) of one simulated study; positions are `basis.positions[rows]`."""
    rate = EXTRA_SHOTS / len(basis.y)
    rows, copies = [], []
    for k, g in enumerate(rng.integers(0, len(groups), len(groups))):
        r = groups[g]
        if scenario.population == "combined":
            r = np.r_[r, rng.choice(r, int(np.floor(rate * len(r) + 0.5)), replace=True)]
        rows.append(r)
        copies.append(np.full(len(r), k))
    rows, copies = np.concatenate(rows), np.concatenate(copies)
    y = (rng.random(len(rows)) < q[rows]).astype(int)
    frame = basis.frame_logit[rows] + (sigma * rng.standard_normal(len(rows)) if scenario.frame == "weaker" else 0.0)
    return rows, copies, y, frame


def study_record(point: float, draws: np.ndarray, margins: tuple[float, ...] = MARGINS, alpha: float = ld.ALPHA) -> dict:
    """What one study concludes: H2 point, bootstrap SD and 90% interval, superiority, TOST verdict per margin."""
    lo90, hi90 = np.percentile(draws, [5, 95])
    curve = ld.equivalence_curve(draws, np.asarray(margins, float), alpha)
    return {"diff": point, "boot_sd": float(np.std(draws, ddof=1)), "lo90": lo90, "hi90": hi90,
            "superior": ld.superiority_p(draws) <= alpha, **{f"equivalent_{m}": bool(e) for m, e in zip(margins, curve.equivalent, strict=True)}}


def run_scenario(basis: Basis, scenario: Scenario, q: np.ndarray, sigma: float, n_sims: int, n_boot: int,
                 fold_seed: int = ld.FOLD_SEED, boot_seed: int = ld.BOOT_SEED, seed: int = SIM_SEED) -> pd.DataFrame:
    """One row per simulated study (`study_record`). Generator stream: default_rng([seed, scenario index])."""
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario)])
    groups = basis.groups
    boot = ld.bootstrap(np.arange(len(groups)), n_boot, boot_seed)   # match copy k is draw key k in every study
    records = []
    for sim in range(n_sims):
        rows, copies, y, frame = simulate_study(rng, basis, groups, q, scenario, sigma)
        point, draws = ld.h2_draws(y, basis.positions[rows], frame, copies, boot._replace(match=copies), fold_seed)
        records.append({"scenario": scenario.name, "sim": sim, "n": len(y), "goals": int(y.sum()), **study_record(point, draws)})
    return pd.DataFrame(records)


def power_table(studies: pd.DataFrame, margins: tuple[float, ...] = MARGINS) -> pd.DataFrame:
    """Per scenario and margin: TOST power with its Monte Carlo SE, superiority power, and the H2 difference's mean,
    sampling SD across studies and mean bootstrap SD."""
    rows = []
    for name, s in studies.groupby("scenario", sort=False):
        for m in margins:
            power = float(s[f"equivalent_{m}"].mean())
            rows.append({"scenario": name, "margin": m, "power_tost": power, "mc_se": float(np.sqrt(power * (1 - power) / len(s))),
                         "power_superiority": float(s.superior.mean()), "mean_diff": float(s["diff"].mean()),
                         "sd_diff": float(s["diff"].std(ddof=1)), "mean_boot_sd": float(s.boot_sd.mean()),
                         "mean_n": float(s.n.mean()), "n_sims": len(s)})
    return pd.DataFrame(rows)


def choose_margin(power: pd.DataFrame, margins: tuple[float, ...] = MARGINS, target: float = POWER_TARGET) -> dict:
    """The rule: decision power = minimum over `DECISION_SCENARIOS` per margin; the smallest margin reaching `target`."""
    names = [s.name for s in DECISION_SCENARIOS]
    if any(power[power.margin.eq(m)].scenario.isin(names).sum() != len(names) for m in margins):
        raise ValueError("the power table lacks a decision scenario")
    decision = {m: float(power[power.scenario.isin(names) & power.margin.eq(m)].power_tost.min()) for m in margins}
    chosen = next((m for m in sorted(margins) if decision[m] >= target), None)
    return {"decision_scenarios": names, "decision_power": {str(m): p for m, p in decision.items()}, "target": target, "chosen_margin": chosen}
