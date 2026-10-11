"""The plan record of the retraining split and the check the retrain runs before it starts (xg-vendor ticket 05).

`plan_record` is the content of `results/xgv_split_plan/plan.json`: counts, the gate allocation, the gate tolerance, the
archived predictions the gate compares with, and the hashes of the inputs, the sources and plan.csv. `verify_plan`
re-derives the plan from the index and the match map and refuses to go on unless every artefact on disk is the plan the
owner approved (`PLAN_SHA256`); ticket 10's runner calls it before it trains anything. `gate_passes` applies the
tolerance of the plan to the point estimates.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from xgpoc.paths import INDEX
from xgpoc.vendor import split_plan as sp
from xgpoc.vendor.statsbomb_fetch import file_sha256

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]   # experiments/claude_fable
PLAN_DIR = EXPERIMENT_DIR / "results/xgv_split_plan"
FOLDS = EXPERIMENT_DIR / "cache/folds_xgv_home5.csv"
MATCH_MAP = EXPERIMENT_DIR / "results/xgv_pairing/match_map.csv"
PAIRING_COMPLETE = EXPERIMENT_DIR / "results/xgv_pairing/complete.json"
# The archived ten-model G-combo (5 folds x 2 seeds, mean probability) on the 2,229 official-test shots: shot_id, goal, p.
# Its log-loss 0.32420 and AUC 0.79233 are the "archived" numbers of results/xgc_g_combo/gate.json.
ARCHIVED_G_COMBO = EXPERIMENT_DIR / "results/v12_cv_ensemble/g_combo_test_predictions.csv"
SOURCES = ("xgpoc/vendor/split_plan.py", "scripts/vendor_split_plan.py")
VERSION = "xgv_split_plan_v1"
EXPECTED_COUNTS = {"evaluation": 135, "gate": 37, "training": 328}
PLAN_SHA256 = "a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a"   # seed 0, written 2026-10-11, before any retrain (as xgpoc.transfer.arms.PLAN_SHA256)


def gate_shots(index: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """The index shots of the gate matches."""
    return index[index.game.isin(plan.game[plan.role == "gate"])]


def archived_on_gate(index: pd.DataFrame, plan: pd.DataFrame, archived: pd.DataFrame) -> dict:
    """Shots and archived G-combo metrics on the gate matches. Raises ValueError when a gate shot has no archived
    prediction or the stored goal label differs from the index."""
    shots = gate_shots(index, plan)[["shot_id", "goal"]]
    joined = shots.merge(archived[["shot_id", "goal", "p"]], on="shot_id", how="left", suffixes=("", "_archived"))
    if joined.p.isna().any():
        raise ValueError(f"{int(joined.p.isna().sum())} gate shots have no archived prediction")
    if (joined.goal.astype(int) != joined.goal_archived.astype(int)).any():
        raise ValueError("Archived goal labels differ from the index on the gate shots")
    y = joined.goal.astype(int).to_numpy()
    return {"shots": len(joined), "goals": int(y.sum()), "log_loss": float(log_loss(y, joined.p)),
            "auc": float(roc_auc_score(y, joined.p))}


def role_counts(plan: pd.DataFrame, index: pd.DataFrame) -> dict:
    """Matches and shots per role, matches per role and official split, matches per training fold."""
    shots = index.groupby("game").size().rename("shots")
    rows = plan.join(shots, on="game")
    by_role = rows.groupby("role").agg(matches=("game", "size"), shots=("shots", "sum"))
    return {"matches": by_role.matches.to_dict(), "shots": {k: int(v) for k, v in by_role.shots.items()},
            "by_official_split": {role: group.official_split.value_counts().sort_index().to_dict()
                                  for role, group in plan.groupby("role")},
            "training_fold_matches": {int(k): int(v) for k, v in rows[rows.role == "training"].fold.value_counts().sort_index().items()},
            "training_fold_shots": {int(k): int(v) for k, v in rows[rows.role == "training"].groupby("fold").shots.sum().items()}}


def gate_allocation(plan: pd.DataFrame, n_gate: int) -> dict:
    """Per (league, season) stratum of the non-overlap official-test matches: matches available and matches in the gate."""
    pool = plan[(plan.official_split == "test") & (plan.role != "evaluation")]
    strata = sp.stratum_of(pool)
    return {s: {"pool": int((strata == s).sum()), "gate": int(((strata == s) & (pool.role == "gate")).sum())}
            for s in sorted(strata.unique())} | {"_total": {"pool": len(pool), "gate": n_gate}}


def plan_record(plan: pd.DataFrame, index: pd.DataFrame, archived: pd.DataFrame, *, seed: int, n_gate: int, n_folds: int,
                inputs: Mapping[str, dict], folds: dict, sources: Mapping[str, str]) -> dict:
    """The content of plan.json. `inputs` and `folds` hold {"path", "sha256"} entries; `sources` maps relative path -> sha256."""
    return {
        "version": VERSION,
        "seed": seed,
        "n_gate": n_gate,
        "n_folds": n_folds,
        "counts": role_counts(plan, index),
        "gate_allocation": gate_allocation(plan, n_gate),
        "rules": {
            "evaluation": f"match map reason == {sp.OVERLAP_REASON!r}",
            "gate": "official-test matches that are not evaluation matches, drawn stratified by league and season: "
                    "floor(n_gate * n_s / N) per stratum, the remainder one each by largest remainder "
                    "(n_gate * n_s mod N), ties to the larger stratum then the name; inside a stratum the sorted matches are "
                    "permuted by default_rng([seed, crc32(stratum)])",
            "training": "every other match; fold = GroupKFold(n_folds) over the home team of the matches sorted by name "
                        "(as scripts/run_cv.py fold_table); -1 outside training",
        },
        "gate_tolerance": {
            **sp.GATE_TOLERANCE,
            "statistic": "point estimates on the gate matches, retrained ten-model G-combo minus archived G-combo",
            "decision": "pass if log_loss(retrained) - log_loss(archived) <= tolerance.log_loss and "
                        "auc(archived) - auc(retrained) <= tolerance.auc; outside the tolerance the study stops at the retrain",
            "bootstrap": "paired match bootstrap interval of both differences, reported, not gated",
        },
        "archived_g_combo": {"predictions": inputs["archived_predictions"], "columns": ["shot_id", "goal", "p"], "gate": archived_on_gate(index, plan, archived)},
        "plan_csv_sha256": hashlib.sha256(sp.plan_text(plan).encode()).hexdigest(),
        "folds_file": dict(folds),
        "inputs": dict(inputs),
        "sources": dict(sources),
    }


def gate_passes(delta_log_loss: float, delta_auc: float, tolerance: Mapping[str, float] = sp.GATE_TOLERANCE) -> dict:
    """The gate decision from point estimates, each retrained minus archived: log-loss may be at most `tolerance
    ["log_loss"]` higher and AUC at most `tolerance["auc"]` lower (a difference exactly at the tolerance passes)."""
    ll, auc = bool(delta_log_loss <= tolerance["log_loss"]), bool(-delta_auc <= tolerance["auc"])
    return {"log_loss": ll, "auc": auc, "passed": ll and auc}


def verify_plan(index_path: Path = INDEX, match_map_path: Path = MATCH_MAP, plan_dir: Path = PLAN_DIR, folds_path: Path = FOLDS,
                archived_path: Path = ARCHIVED_G_COMBO, *, expected_plan: str = PLAN_SHA256,
                expected_counts: Mapping[str, int] = EXPECTED_COUNTS) -> tuple[pd.DataFrame, dict]:
    """(plan, plan.json record) of the approved plan. Raises ValueError unless the plan rebuilt from the index and the
    match map, plan.csv on disk, plan.json, the folds file, the archived predictions and the source files all agree with
    `expected_plan` (the sha256 of plan.csv) and with the completion record, and the role counts are `expected_counts`."""
    complete = json.loads((plan_dir / "complete.json").read_text())
    for name, expected in complete["files"].items():
        if file_sha256(plan_dir / name) != expected:
            raise ValueError(f"{plan_dir / name} changed since the completion record")
    if unlisted := sorted({p.name for p in plan_dir.iterdir()} - set(complete["files"]) - {"complete.json"}):
        raise ValueError(f"{plan_dir} holds files that are not in the completion record: {unlisted}")
    record = json.loads((plan_dir / "plan.json").read_text())
    for key, path in (("index", index_path), ("match_map", match_map_path), ("archived_predictions", archived_path)):
        if (found := file_sha256(path)) != record["inputs"][key]["sha256"]:
            raise ValueError(f"{key} {path} hashes to {found}, the plan was built from {record['inputs'][key]['sha256']}")
    if (found := file_sha256(folds_path)) != record["folds_file"]["sha256"]:
        raise ValueError(f"folds file {folds_path} hashes to {found}, the plan wrote {record['folds_file']['sha256']}")
    if (found := {p: file_sha256(EXPERIMENT_DIR / p) for p in record["sources"]}) != record["sources"]:
        raise ValueError(f"split-plan code changed since the plan was written: {found} vs {record['sources']}")

    rebuilt = sp.build_plan(pd.read_parquet(index_path), pd.read_csv(match_map_path), record["seed"], record["n_gate"], record["n_folds"])
    hashes = {"rebuilt": hashlib.sha256(sp.plan_text(rebuilt).encode()).hexdigest(), "plan.csv": file_sha256(plan_dir / "plan.csv"),
              "plan.json": record["plan_csv_sha256"]}
    if wrong := {k: v for k, v in hashes.items() if v != expected_plan}:
        raise ValueError(f"split plan is not the approved one ({expected_plan}): {wrong}")
    plan = pd.read_csv(plan_dir / "plan.csv")
    if not pd.read_csv(folds_path).equals(sp.training_folds(plan)):
        raise ValueError(f"{folds_path} is not the fold assignment of the plan's training matches")
    if (counts := plan.role.value_counts().to_dict()) != dict(expected_counts):
        raise ValueError(f"role counts {counts} differ from {dict(expected_counts)}")
    return plan, record
