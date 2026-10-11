"""Retrain G-combo on the 328 training matches of the frozen split plan (xg-vendor ticket 10; the approved GPU job).

    ./run.sh python scripts/vendor_retrain_g.py                 # 5 folds x seeds 0, 1 -> results/xgv_g_combo
    ./run.sh python scripts/vendor_retrain_g.py --smoke --index cache/dummy/index.parquet --out results/xgv_g_smoke \
        --folds 0 --seeds 0                                      # pipeline check on a synthetic plan, 64 shots per role
    ./run.sh python scripts/vendor_verify_g.py                  # afterwards: replay, ensemble calibration, gate

The recipe, the fit (`xgpoc.models.g_fit.fit_g`), the bundle, the completion record and the resume rules are those of
`scripts/run_xgc_g.py` (its `recipe`, `own_sources`, `frame_fingerprint` and `verify_finished` are used as they are).
What differs is the split. `xgpoc.vendor.split_check.verify_plan` runs first and refuses every plan but the approved one;
the pool is the plan's role "training" with the plan's fold (328 matches, the fold file `cache/folds_xgv_home5.csv`), and
the frames every fold model scores besides its held-out fold are the 37 gate matches (file role "gate", where the archived
runs wrote "test"). The evaluation matches are never read here: they are scored in tickets 11 and 13.

Calibration constants of a fold model (the penalty rate in its bundle) come from the penalties of the whole training pool:
the archived runs took the train split of a pool that had a test split, and this pool has none.

Each (fold, seed) writes into <out>/fold<k>/seed<s>/ the files of run_xgc_g.py (bundle.pt, <name>_{valid,gate}_predictions.csv,
<name>_metrics.json with the gate metrics of the single fold model, training.json, complete.json). Resumable as there.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import run_poc as rp
import torch
from run_v12 import STAGES
from run_xgc_g import (
    MIN_FREE_GIB,
    OVERLAY,
    frame_fingerprint,
    own_sources,
    recipe,
    verify_finished,
)
from xgpoc.bundle import code_version, make_provenance, save_bundle, sha256_file
from xgpoc.data.index import load_index
from xgpoc.eval.calibration import penalty_rate
from xgpoc.eval.metrics import evaluate
from xgpoc.models.common import config_dict
from xgpoc.models.g_bundle import g_identity, g_spec
from xgpoc.models.g_fit import fit_g
from xgpoc.paths import INDEX
from xgpoc.vendor import split_check

STAGE = "g_combo"
AUX_WEIGHT = 0.5
OUT = Path("results/xgv_g_combo")
SMOKE_INDEX = Path("cache/dummy/index.parquet")
BUSY_MIB = 1024  # a compute process holding more GPU memory than this is a job; pytest's CUDA contexts hold ~200 MiB


def foreign_gpu_jobs(nvidia_smi_csv: str, own_pid: int, busy_mib: int = BUSY_MIB) -> list[str]:
    """Lines of `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` of other processes that
    hold at least `busy_mib` MiB: another training job, not a test-suite context or the desktop."""
    found = []
    for line in nvidia_smi_csv.splitlines():
        pid, _, mib = (part.strip() for part in line.partition(","))
        if pid.isdigit() and int(pid) != own_pid and mib.isdigit() and int(mib) >= busy_mib:
            found.append(line.strip())
    return found


def check_gpu_free() -> None:
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout
    if busy := foreign_gpu_jobs(query, os.getpid()):
        raise SystemExit(f"another job is using the GPU, refusing to train: {busy}")


def smoke_plan(index: pd.DataFrame) -> pd.DataFrame:
    """A plan for the dummy index, for the pipeline check only: the games of the official test split are the gate, the
    others train with fold = rank mod 5. It is not the frozen plan and carries no evaluation matches."""
    games = (index.drop_duplicates("game").sort_values("game")[["game", "split"]]).reset_index(drop=True)
    training = games.split != "test"
    games["role"] = games.split.map(lambda s: "gate" if s == "test" else "training")
    games["fold"] = -1
    games.loc[training, "fold"] = [i % 5 for i in range(int(training.sum()))]
    return games.rename(columns={"split": "official_split"})


def plan_split(df: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pool with the plan's fold column, gate shots) of the shots `df`. Raises ValueError if a shot sits in a match that
    the plan does not list or an evaluation match, so that no evaluation shot can reach training or the gate."""
    role = plan.set_index("game").role
    unknown = sorted(set(df.game) - set(role.index))
    if unknown:
        raise ValueError(f"{len(unknown)} index matches are not in the plan, e.g. {unknown[0]}")
    shots = df[df.game.map(role) != "evaluation"]
    pool = shots[shots.game.map(role) == "training"].merge(plan.loc[plan.role == "training", ["game", "fold"]], on="game")
    gate = shots[shots.game.map(role) == "gate"].reset_index(drop=True)
    return pool, gate


def run_trial(k: int, seed: int, pool: pd.DataFrame, gate: pd.DataFrame, out: Path, *, name: str, base_identity: dict,
              sources: dict, smoke: bool, pixels: dict) -> None:
    frames = pd.concat([pool, gate])  # every frame any fold can read; `pixels` is its fingerprint
    cfg, kw = recipe(AUX_WEIGHT, smoke)
    cfg.seed = seed
    spec = g_spec(kw["backbone"], kw["n_blocks"], n_out=2, mask=kw["mask"])
    backbone = g_identity(spec)
    pen = pool[pool.is_penalty]
    constants = {"penalty_rate": penalty_rate(int(pen.goal.sum()), len(pen)), "pooled_goal_rate": float(pool.goal.mean())}
    identity = {**base_identity, "fold": k, "seed": seed, "name": name, "spec": spec, "backbone": backbone,
                "cfg": config_dict(cfg), "rrc": list(kw["rrc"]), "backbone_lr_mult": kw["backbone_lr_mult"],
                "constants": constants, "smoke": smoke}
    if verify_finished(out, identity):
        print("verified resume", out, flush=True)
        return
    if shutil.disk_usage(Path(".")).free < MIN_FREE_GIB * 2**30:
        raise OSError(f"Less than {MIN_FREE_GIB} GiB free; refusing to train")
    tr, va = (pool[pool.fold != k].reset_index(drop=True), pool[pool.fold == k].reset_index(drop=True))
    te = gate
    if smoke:
        tr, va, te = tr.head(64), va.head(64), te.head(64)
    print(f"=== {out} ({time.strftime('%H:%M')}) train {len(tr)} oof {len(va)} gate {len(te)} ===", flush=True)
    trained = fit_g(tr, va, {"valid": va, "gate": te}, cfg, kw, spec, aux_col=rp.ON_TARGET)
    model, fit, logits, compute = trained.model, trained.fit, trained.logits, trained.compute
    temp = fit["temperature"]  # fitted on the held-out fold's logits of the selected (EMA) weights
    if sources != own_sources():
        raise ValueError("source files changed during training; restart with stable code")
    if pixels != frame_fingerprint(frames):
        raise ValueError("input frames changed during training")

    out.mkdir(parents=True, exist_ok=True)
    files = []
    for role, df in (("valid", va), ("gate", te)):
        p = temp(logits[role])
        pd.DataFrame({"shot_id": df.shot_id, "goal": df.goal.astype(int), "logit": logits[role], "p": p}).to_csv(
            out / f"{name}_{role}_predictions.csv", index=False)
        files.append(f"{name}_{role}_predictions.csv")
    metrics = {**evaluate(te.goal.to_numpy(), temp(logits["gate"])), "model": name, "params": compute["trainable_parameters"],
               "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"], "temperature": temp.t, "scored": "gate"}
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    (out / "training.json").write_text(json.dumps({"history": fit["history"], "best_epoch": fit["best_epoch"],
                                                   "compute": compute, "temperature": temp.t}, indent=2) + "\n")
    files += [f"{name}_metrics.json", "training.json"]
    commit = code_version()
    provenance = make_provenance(sources, index_sha256=base_identity["index"]["sha256"],
                                 folds_sha256=base_identity["folds"]["sha256"],
                                 overlay_sha256=base_identity["overlay_sha256"])
    saved = save_bundle(out / "bundle.pt", model, spec=spec, backbone=backbone, commit=commit, provenance=provenance,
                        calibration={"temperature": temp.t, "penalty_rate": constants["penalty_rate"]},
                        recipe={"stage": STAGE, "name": name, "cfg": config_dict(cfg), "rrc": list(kw["rrc"]),
                                "backbone_lr_mult": kw["backbone_lr_mult"], "fold": k, "seed": seed, "best_epoch": fit["best_epoch"],
                                "ema": "selected-epoch EMA weights"})
    files.append("bundle.pt")
    record = {"identity": identity, "code_sha256": sources, "commit": commit, "bundle": saved,
              "files": {f: sha256_file(out / f) for f in files}}
    (out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"--- {out} done in {compute['train_minutes']:.1f} min, gate LL {metrics['log_loss']:.4f} AUC {metrics['auc']:.4f}, "
          f"bundle {saved['bytes'] / 2**20:.0f} MiB", flush=True)
    del model, fit, trained
    gc.collect()
    torch.cuda.empty_cache()


def load_frames(index: Path, smoke: bool) -> pd.DataFrame:
    df = rp.add_targets(load_index(index))
    if smoke and rp.ON_TARGET not in df:  # the dummy index has no shot label; the auxiliary target only has to exist
        df = df.assign(**{rp.ON_TARGET: df.goal.astype(float)})
    return df


def approved_split(index: Path, smoke: bool, folds_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """(pool, gate, plan identity, folds file). Not smoke: the frozen plan, refused unless `verify_plan` accepts it and the
    shot counts of the loaded index are those the plan was built on. Smoke: `smoke_plan` of the dummy index."""
    df = load_frames(index, smoke)
    if smoke:
        plan = smoke_plan(df)
        folds_file = folds_dir / "folds_smoke.csv"
        folds_file.parent.mkdir(parents=True, exist_ok=True)
        folds_file.write_text(plan.loc[plan.role == "training", ["game", "fold"]].to_csv(index=False))
        identity = {"sha256": sha256_file(folds_file), "seed": None}
    else:
        plan, record = split_check.verify_plan(index_path=index)
        folds_file = split_check.FOLDS
        identity = {"sha256": split_check.PLAN_SHA256, "seed": record["seed"]}
    pool, gate = plan_split(df, plan)
    if not smoke:
        expected = record["counts"]["shots"]
        if (len(pool), len(gate)) != (expected["training"], expected["gate"]):
            raise ValueError(f"loaded {len(pool)} training and {len(gate)} gate shots, the plan has {expected} "
                             "(frames missing from disk?)")
    return pool, gate, identity, folds_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--name", default=STAGES[STAGE][1], help="model name used in file names")
    ap.add_argument("--smoke", action="store_true", help="synthetic plan on the dummy index, 2 epochs on 64 shots per role (pipeline check only)")
    a = ap.parse_args()
    if not a.smoke and a.index != INDEX:
        raise SystemExit("the retrain runs on the approved index; --index is for --smoke only")
    check_gpu_free()

    pool, gate, plan_identity, folds_file = approved_split(a.index, a.smoke, a.out / "smoke_plan")
    sources = own_sources()
    print(f"index {a.index}: pool {len(pool)} shots / {pool.game.nunique()} games, gate {len(gate)} shots / {gate.game.nunique()} games",
          flush=True)
    pixels = frame_fingerprint(pd.concat([pool, gate]))
    base = {"index": {"path": str(a.index), "sha256": sha256_file(a.index)},
            "folds": {"path": str(folds_file), "sha256": sha256_file(folds_file)},
            "plan": plan_identity, "overlay_sha256": sha256_file(OVERLAY), "frames": pixels}
    for k in a.folds:
        for seed in a.seeds:
            run_trial(k, seed, pool, gate, a.out / f"fold{k}" / f"seed{seed}", name=a.name, base_identity=base,
                      sources=sources, smoke=a.smoke, pixels=pixels)
    print(f"XGV G RETRAIN QUEUE DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
