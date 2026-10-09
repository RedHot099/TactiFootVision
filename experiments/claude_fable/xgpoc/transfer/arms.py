"""Training sets of the leave-one-league-out transfer arms (xg-credibility ticket 12), without torch.

`verify_plan` re-derives the split plan from the index and refuses to go on unless it is the
plan the owner approved (plan.csv sha256, manifest, source hashes, protocol hash). `arm_sets`
cuts the shots of one (league, half-set, arm) into the four roles of the plan. `completeness`
checks the finished runs against the plan without reading any prediction value other than
which shots are present.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xgpoc.bundle import sha256_file
from xgpoc.transfer import plan as split_plan

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]  # experiments/claude_fable
PLAN_DIR = EXPERIMENT_DIR / "research" / "xgc_transfer_plan"
PROTOCOL = EXPERIMENT_DIR / "research" / "xgc_transfer_protocol.md"
PLAN_SHA256 = "33229bbfd38dcbbdcd2ba33524cdaf2ab5ae766485f6ba7da08e3dda40b7bc88"  # approved 2026-10-09 (commit 07dc33f)
PROTOCOL_SHA256 = "e5ae46624703c4a2c6b520aa70d446f6f07a127b906913fdb41dba2daa077c0e"  # research/xgc_transfer_protocol.md
SEEDS = (0, 1)
ARM_SIZES = {"train": 225, "select": 30, "calibration": 45}
TRAINING_ROLES = ("train", "select", "calibration")
PREDICTED_ROLES = ("select", "calibration", "test")


@dataclass(frozen=True)
class FitKey:
    league: str
    half_set: str
    arm: str
    seed: int

    @property
    def path(self) -> Path:
        return Path(self.league) / self.half_set / self.arm / f"seed{self.seed}"


def fit_grid(seeds: Iterable[int] = SEEDS, leagues: Iterable[str] = split_plan.DOMESTIC_LEAGUES) -> list[FitKey]:
    """Every run, seed-major, so a partial campaign still holds complete seeds first."""
    leagues = tuple(leagues)
    return [FitKey(league, h, arm, seed) for seed in seeds for league in leagues for h in split_plan.HALF_SETS
            for arm in split_plan.ARMS]


def verify_plan(index: pd.DataFrame, plan_dir: Path = PLAN_DIR, protocol: Path = PROTOCOL, *, expected_plan: str = PLAN_SHA256,
                expected_protocol: str = PROTOCOL_SHA256) -> tuple[pd.DataFrame, dict]:
    """The approved plan and the hashes every completion record carries. Raises ValueError
    unless the plan rebuilt from `index`, the plan.csv on disk, the manifest, the split-plan
    code and the protocol all agree with the approved hashes."""
    rebuilt = split_plan.build_plan(index, seed=0).to_csv(index=False)
    manifest = json.loads((plan_dir / "manifest.json").read_text())
    hashes = {"rebuilt": hashlib.sha256(rebuilt.encode()).hexdigest(), "plan.csv": sha256_file(plan_dir / "plan.csv"), "manifest": manifest["plan_sha256"]}
    wrong = {k: v for k, v in hashes.items() if v != expected_plan}
    if wrong:
        raise ValueError(f"split plan is not the approved one ({expected_plan}): {wrong}")
    code = {p: sha256_file(EXPERIMENT_DIR / p) for p in manifest["sources"]}
    if code != manifest["sources"]:
        raise ValueError(f"split-plan code changed since the manifest was written: {code} vs {manifest['sources']}")
    protocol_sha = sha256_file(protocol)
    if protocol_sha != expected_protocol:
        raise ValueError(f"protocol {protocol} hashes to {protocol_sha}, not the approved {expected_protocol}")
    plan = pd.read_csv(plan_dir / "plan.csv")
    return plan, {"plan_sha256": expected_plan, "protocol_sha256": protocol_sha, "split_plan_code": code,
                  "index_sha256": manifest["index_sha256"]}


def arm_sets(shots: pd.DataFrame, plan: pd.DataFrame, key: FitKey) -> dict[str, pd.DataFrame]:
    """Shots of the matches in each role of (league, half_set, arm). `shots` keeps its row order,
    which fixes batch composition. Raises ValueError when a match has no shots, roles overlap, or
    the role sizes differ from ARM_SIZES."""
    rows = plan[(plan.league == key.league) & (plan.half_set == key.half_set) & (plan.arm == key.arm)]
    games = {role: set(rows.game[rows.role == role]) for role in (*TRAINING_ROLES, "test")}
    if sum(map(len, games.values())) != len(rows):
        raise ValueError(f"{key}: a match holds two roles")
    sizes = {role: len(games[role]) for role in ARM_SIZES}
    if sizes != ARM_SIZES:
        raise ValueError(f"{key}: role sizes {sizes}, expected {ARM_SIZES}")
    missing = set(rows.game) - set(shots.game)
    if missing:
        raise ValueError(f"{key}: {len(missing)} planned matches have no shots, e.g. {sorted(missing)[0]}")
    return {role: shots[shots.game.isin(g)].reset_index(drop=True) for role, g in games.items()}


def arm_constants(sets: dict[str, pd.DataFrame]) -> dict:
    """What calibration later takes from the whole 300-match arm (protocol 5): goal rate and
    penalty counts over train, select and calibration matches. The test half-set is excluded."""
    arm = pd.concat([sets[r] for r in TRAINING_ROLES])
    pens = arm[arm.is_penalty.astype(bool)]
    return {"arm_shots": len(arm), "arm_goals": int(arm.goal.sum()), "arm_goal_rate": float(arm.goal.mean()),
            "arm_penalties": len(pens), "arm_penalty_goals": int(pens.goal.sum())}


def completeness(out: Path, shots: pd.DataFrame, plan: pd.DataFrame) -> list[str]:
    """Problems found among the runs under `out`; an empty list means all 40 runs of the full
    grid are complete: each has a completion record of the approved plan and protocol whose file
    hashes match, and its select, calibration and test predictions cover exactly the planned
    shots. Reads shot ids and file bytes only, never a prediction value."""
    problems = []
    for key in fit_grid():
        run = out / key.path
        if not (run / "complete.json").exists():
            problems.append(f"{key.path}: no complete.json")
            continue
        record = json.loads((run / "complete.json").read_text())
        if (record.get("plan_sha256"), record.get("protocol_sha256")) != (PLAN_SHA256, PROTOCOL_SHA256):
            problems.append(f"{key.path}: completion record names another plan or protocol")
        for name, expected in record["files"].items():
            if not (run / name).exists() or sha256_file(run / name) != expected:
                problems.append(f"{key.path}: {name} missing or changed since the run finished")
        arm = plan[(plan.league == key.league) & (plan.half_set == key.half_set) & (plan.arm == key.arm)]
        for role in PREDICTED_ROLES:
            expected_ids = set(shots.shot_id[shots.game.isin(arm.game[arm.role == role])])
            file = run / f"{role}_predictions.csv"
            if not file.exists():
                problems.append(f"{key.path}: no {role}_predictions.csv")
                continue
            predicted = pd.read_csv(file, usecols=["shot_id"]).shot_id
            if predicted.duplicated().any() or set(predicted) != expected_ids:
                problems.append(f"{key.path}: {role} predictions cover {len(set(predicted) & expected_ids)} "
                                f"of {len(expected_ids)} shots")
    return problems
