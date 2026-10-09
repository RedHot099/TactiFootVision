"""Train the 40 models of the leave-one-league-out transfer test (xg-credibility ticket 12).

    ./run.sh python scripts/run_xgc_transfer.py                     # all 40 runs -> results/xgc_transfer
    ./run.sh python scripts/run_xgc_transfer.py --check             # completeness + select-set sanity table only
    ./run.sh python scripts/run_xgc_transfer.py --smoke --out results/xgc_transfer_smoke --max-fits 1

Protocol: research/xgc_transfer_protocol.md, plan research/xgc_transfer_plan (hashes asserted in
`xgpoc.transfer.arms.verify_plan` before the first training). One G-combo model per
(league, half-set, arm, seed) with the recipe of ticket 04 (`run_xgc_g.recipe`). The arm's
"select" matches choose the epoch; "calibration" and "test" matches are only predicted.

Each run writes into <out>/<league>/<half_set>/<arm>/seed<s>/ (no weights are kept, protocol 4):
    select_predictions.csv, calibration_predictions.csv, test_predictions.csv   shot_id, goal, logit, p
        (p = sigmoid(logit): the model's raw output, no temperature; Platt is ticket 13's job)
    training.json    history (select-set loss/AUC per epoch), best epoch, time_limit_hit, compute, arm constants
    complete.json    identity (recipe, data, plan, protocol) + source hashes + file hashes

Resumable: a run with a valid complete.json is skipped; a different recorded identity is an error.
A run that raises is retried once; a second failure is written to <out>/failures.json, the queue goes on
and the cell is reported missing by --check (protocol 12). --check also writes <out>/deviations.json
(runs that hit the 45-minute limit, missing cells). Training needs --gpu-approved (owner approval) and
refuses to start while another python process holds the GPU. No test-set metric is computed here.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import run_poc as rp
import torch
from run_xgc_g import (
    MIN_FREE_GIB,
    OVERLAY,
    frame_fingerprint,
    own_sources,
    recipe,
    verify_finished,
)
from xgpoc.bundle import canonical_digest, code_version, sha256_file
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig, config_dict
from xgpoc.models.g_bundle import g_identity, g_spec
from xgpoc.models.g_fit import fit_g
from xgpoc.paths import INDEX
from xgpoc.transfer.arms import (
    PLAN_DIR,
    PREDICTED_ROLES,
    PROTOCOL,
    SEEDS,
    FitKey,
    arm_constants,
    arm_sets,
    completeness,
    fit_grid,
    verify_plan,
)

AUX_WEIGHT = 0.5  # on-target auxiliary loss weight of the G-combo recipe (run_xgc_g's --aux-weight default)
OUT = Path("results/xgc_transfer")
TABLE = "training_table.csv"
FAILURES = "failures.json"
DEVIATIONS = "deviations.json"
ATTEMPTS = 2  # one retry of a failed fit, then the cell is reported missing


@dataclass(frozen=True)
class FitContext:
    """What is the same for all 40 fits of one launch."""

    cfg: TrainConfig  # recipe config; the seed is set per fit
    recipe_kw: dict  # backbone, blocks, mask, crop scale, backbone lr multiplier
    sources: dict  # relative path -> sha256 of every source file loaded
    shared_identity: dict  # plan, protocol, index, overlay, frame hashes

    def cfg_for(self, key: FitKey) -> TrainConfig:
        return TrainConfig(**{**config_dict(self.cfg), "seed": key.seed})


def load_shots(index: Path) -> pd.DataFrame:
    """Every shot of the index. Unlike the usual loader this refuses to drop shots with missing frames."""
    shots = rp.add_targets(load_index(index, require_frames=False))
    absent = [p for p in shots.frame_path.unique() if not Path(p).exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} frames of the index are missing, e.g. {absent[0]}")
    return shots


def fit_identity(key: FitKey, ctx: FitContext, sets: dict[str, pd.DataFrame]) -> dict:
    """Everything that makes one fit the same fit: change any of it and a finished fit is not resumed."""
    spec = g_spec(ctx.recipe_kw["backbone"], ctx.recipe_kw["n_blocks"], n_out=2, mask=ctx.recipe_kw["mask"])
    members = {role: sorted(df.game.unique()) for role, df in sets.items()}
    return {**ctx.shared_identity, "league": key.league, "half_set": key.half_set, "arm": key.arm, "seed": key.seed,
            "spec": spec, "backbone": g_identity(spec), "cfg": config_dict(ctx.cfg_for(key)),
            "rrc": list(ctx.recipe_kw["rrc"]), "backbone_lr_mult": ctx.recipe_kw["backbone_lr_mult"],
            "matches": {role: len(games) for role, games in members.items()}, "matches_sha256": canonical_digest(members)}


def fit_arm(key: FitKey, sets: dict[str, pd.DataFrame], out: Path, identity: dict, ctx: FitContext) -> None:
    """Fit one model on sets['train'], select the epoch on sets['select'], predict select,
    calibration and test, and write the directory with its completion record last."""
    print(f"=== {key.path} ({time.strftime('%H:%M')}) " + " ".join(f"{r} {len(d)}" for r, d in sets.items()), flush=True)
    frames = pd.concat(sets.values())
    fingerprint = frame_fingerprint(frames)
    cfg = ctx.cfg_for(key)
    result = fit_g(sets["train"], sets["select"], {role: sets[role] for role in PREDICTED_ROLES}, cfg, ctx.recipe_kw,
                   identity["spec"], aux_col=rp.ON_TARGET)
    del result.model  # no weights are kept (protocol 4)
    gc.collect()
    torch.cuda.empty_cache()
    if ctx.sources != own_sources():
        raise ValueError("source files changed during training; restart with stable code")
    if fingerprint != frame_fingerprint(frames):
        raise ValueError("input frames changed during training")

    out.mkdir(parents=True, exist_ok=True)
    files = []
    for role in PREDICTED_ROLES:
        logit, df = result.logits[role], sets[role]
        pd.DataFrame({"shot_id": df.shot_id, "goal": df.goal.astype(int), "logit": logit, "p": 1 / (1 + np.exp(-logit))}).to_csv(
            out / f"{role}_predictions.csv", index=False)
        files.append(f"{role}_predictions.csv")
    best = result.fit["history"][result.fit["best_epoch"]]  # select-set row of the chosen epoch, for the sanity table
    # patience (5) equals the epoch cap, so training stops early only through the minutes limit
    (out / "training.json").write_text(json.dumps({
        "history": result.fit["history"], "best_epoch": result.fit["best_epoch"], "best": best, "compute": result.compute,
        "time_limit_hit": result.compute["epochs_run"] < cfg.epochs, "max_minutes": cfg.max_minutes,
        "arm_constants": arm_constants(sets)}, indent=2) + "\n")
    files.append("training.json")
    record = {"identity": identity, "plan_sha256": identity["plan_sha256"], "protocol_sha256": identity["protocol_sha256"],
              "split_plan_code": identity["split_plan_code"], "code_sha256": ctx.sources, "commit": code_version(),
              "files": {f: sha256_file(out / f) for f in files}}
    (out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"--- {key.path} done in {result.compute['train_minutes']:.1f} min, best epoch {result.fit['best_epoch']}, "
          f"select LL(cal) {best['valid_log_loss_cal']:.4f} AUC {best['valid_auc']:.4f}", flush=True)


def foreign_gpu_processes(nvidia_smi_csv: str, own_pid: int) -> list[str]:
    """Lines of `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` that are
    python processes other than this one (desktop and browser GPU processes do not count)."""
    found = []
    for line in nvidia_smi_csv.splitlines():
        pid, _, name = line.partition(",")
        if pid.strip().isdigit() and int(pid) != own_pid and "python" in name.lower():
            found.append(line.strip())
    return found


def check_gpu_free() -> None:
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout
    busy = foreign_gpu_processes(query, os.getpid())
    if busy:
        raise SystemExit(f"another python process is using the GPU, refusing to train: {busy}")  # not retried


def validate_seeds(seeds: list[int], smoke: bool) -> list[int]:
    if not smoke and not set(seeds) <= set(SEEDS):
        raise ValueError(f"seeds must be a subset of {SEEDS} (the protocol grid); got {seeds}")
    return seeds


def read_failures(out: Path) -> dict:
    file = out / FAILURES
    return json.loads(file.read_text()) if file.exists() else {}


def fit_queue(keys: list[FitKey], out: Path, shots: pd.DataFrame, plan: pd.DataFrame,
              identity_of: Callable[[FitKey, dict], dict], fit: Callable[..., None], max_fits: int | None = None) -> int:
    """Fit every arm without a valid completion record; returns the number fitted.

    A fit that raises is tried once more (protocol 12); after ATTEMPTS failures it is written to
    FAILURES, skipped from then on and left to `--check` to report as missing. The queue goes on."""
    fitted = 0
    failures = read_failures(out)
    for key in keys:
        sets = arm_sets(shots, plan, key)
        identity = identity_of(key, sets)
        if verify_finished(out / key.path, identity):
            print("verified resume", key.path, flush=True)
            continue
        if str(key.path) in failures:
            print("skipping, failed twice earlier:", key.path, flush=True)
            continue
        if max_fits is not None and fitted >= max_fits:
            break
        if shutil.disk_usage(Path(".")).free < MIN_FREE_GIB * 2**30:
            raise OSError(f"Less than {MIN_FREE_GIB} GiB free; refusing to train")
        for attempt in range(1, ATTEMPTS + 1):
            try:
                fit(key, sets, out / key.path, identity)
                fitted += 1
                break
            except Exception as error:  # noqa: BLE001 - a failed cell must not stop the other 39
                print(f"!!! {key.path} attempt {attempt}/{ATTEMPTS} failed: {error!r}", flush=True)
                traceback.print_exc()
                if attempt == ATTEMPTS:
                    failures[str(key.path)] = {"attempts": attempt, "error": repr(error)}
                    out.mkdir(parents=True, exist_ok=True)
                    (out / FAILURES).write_text(json.dumps(failures, indent=2) + "\n")
    return fitted


def sanity_table(out: Path) -> pd.DataFrame:
    """Select-set loss and AUC of every run at its selected epoch, from training.json only."""
    rows = []
    for key in fit_grid():
        file = out / key.path / "training.json"
        if file.exists():
            t = json.loads(file.read_text())
            rows.append({"league": key.league, "half_set": key.half_set, "arm": key.arm, "seed": key.seed,
                         "best_epoch": t["best_epoch"], "epochs_run": t["compute"]["epochs_run"],
                         "time_limit_hit": t["time_limit_hit"],
                         "select_ll_cal": t["best"]["valid_log_loss_cal"], "select_ll_raw": t["best"]["valid_log_loss"],
                         "select_auc": t["best"]["valid_auc"], "temperature": t["best"]["temperature"],
                         "train_minutes": t["compute"]["train_minutes"]})
    return pd.DataFrame(rows)


def check(out: Path, shots: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Completeness over the full grid plus the sanity table; writes TABLE and DEVIATIONS into `out`."""
    problems = completeness(out, shots, plan)
    table = sanity_table(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / TABLE, index=False)
    limit_hit = [] if table.empty else [f"{r.league}/{r.half_set}/{r.arm}/seed{r.seed}" for r in table.itertuples() if r.time_limit_hit]
    (out / DEVIATIONS).write_text(json.dumps({"time_limit_hit": limit_hit, "failed_twice": read_failures(out),
                                              "problems": problems}, indent=2) + "\n")
    return table, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--max-fits", type=int, help="stop after fitting this many arms")
    ap.add_argument("--check", action="store_true", help="completeness check and sanity table only; no training")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 64 shots per role (pipeline check only)")
    ap.add_argument("--gpu-approved", action="store_true", help="owner approval to use the GPU; required to train on one")
    a = ap.parse_args()
    validate_seeds(a.seeds, a.smoke)

    shots = load_shots(a.index)
    plan, hashes = verify_plan(shots, PLAN_DIR, PROTOCOL)
    print(f"plan {hashes['plan_sha256'][:12]} protocol {hashes['protocol_sha256'][:12]} verified", flush=True)
    if a.check:
        table, problems = check(a.out, shots, plan)
        print(table.to_string(index=False))
        print(f"completeness: {len(table)} runs with training metrics; {len(problems)} problems {problems[:5]}")
        raise SystemExit(1 if problems or read_failures(a.out) else 0)

    if torch.cuda.is_available():
        if not a.gpu_approved:
            raise SystemExit("refusing to train on the GPU without --gpu-approved (owner approval)")
        check_gpu_free()
    shared_identity = {**hashes, "index": {"path": str(a.index), "sha256": sha256_file(a.index)},
                       "overlay_sha256": sha256_file(OVERLAY), "smoke": a.smoke,
                       "frames": frame_fingerprint(shots[shots.game.isin(plan.game)])}
    if shared_identity["index"]["sha256"] != hashes["index_sha256"]:
        raise ValueError("index differs from the one the plan was built on")
    sources = {**own_sources(), "scripts/run_xgc_transfer.py": sha256_file(Path(__file__))}
    cfg, recipe_kw = recipe(AUX_WEIGHT, a.smoke)
    ctx = FitContext(cfg, recipe_kw, sources, shared_identity)

    def fit(key: FitKey, sets: dict, out: Path, identity: dict) -> None:
        if torch.cuda.is_available():
            check_gpu_free()
        if a.smoke:
            sets = {role: df.head(64) for role, df in sets.items()}
        fit_arm(key, sets, out, identity, ctx)

    started = time.time()
    n = fit_queue(fit_grid(a.seeds), a.out, shots, plan, lambda key, sets: fit_identity(key, ctx, sets), fit, a.max_fits)
    print(f"XGC TRANSFER QUEUE DONE ({time.strftime('%H:%M')}): fitted {n} arms in {(time.time() - started) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
