"""Train approach 1 of the xg-detector study (ticket 03): A1 or A2, 5 folds x seed 0 on the plan's 328 training matches.

    ./run.sh python scripts/xgd_approach1.py --variant A1             # -> results/xgd_a1/fold<k>/seed0/
    ./run.sh python scripts/xgd_approach1.py --variant A2             # -> results/xgd_a2/fold<k>/seed0/
    ./run.sh python scripts/xgd_approach1.py --variant A1 --replay    # afterwards: each bundle reloaded, held-out fold rescored
    ./run.sh python scripts/xgd_approach1.py --variant A1 --seeds 1   # ticket 06: seed 1 of the 10-model A1 ensemble
    ./run.sh python scripts/xgd_approach1.py --variant A1 --seeds 0 1 --replay --replay-file replay_ensemble.json
    ./run.sh python scripts/xgd_approach1.py --variant A2 --smoke --out /tmp/xgd_smoke --folds 0   # pipeline check

The model, its inputs and the frozen hyperparameters are `xgpoc.detector.approach1` and ticket 03; the recipe is
G-combo's (`run_xgc_g.recipe`, on-target weight 0.5: DINOv2-L last 4 blocks, overlay mask, random resized crop 0.85-1,
flip, photometric jitter, EMA 0.998, 5 epochs, batch 8, epoch and temperature on the held-out fold). The split is the
xg-vendor plan (`approach1.verify_plan` refuses any other) with its home-team folds; only the training matches are
read, and only their held-out folds are scored: the gate and evaluation matches belong to the single test (ticket 06).
The inputs are the sanitised D3 table (`scripts/xgd_sanitise.py`), checked against its completion record. Everything
after the GPU check runs inside `no_statsbomb_reads()` and the held-out label seal (`seal.development_labels_only`: no label
of a gate or evaluation shot reaches this process).

Each (fold, seed) writes <out>/fold<k>/seed<s>/: bundle.pt (trained tensors, EMA weights of the selected epoch; spec with
the A1 standardiser), <name>_valid_predictions.csv (shot_id, goal, logit, p = temperature-scaled; the held-out fold),
<name>_metrics.json (held-out-fold metrics), training.json and complete.json (identity, code, file hashes; written last).
A finished run with the same identity is skipped; a different identity is an error. `--replay` writes <out>/replay.json:
every bundle rebuilt from disk scores its held-out fold in the stored row order and batch size, and the logits must
agree within `REPLAY_TOLERANCE` (`--replay-file` names another record, e.g. ticket 06's replay of all ten A1 runs).
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
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
from xgpoc.bundle import code_version, make_provenance, save_bundle, sha256_file
from xgpoc.data.index import load_index
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.eval.calibration import penalty_rate
from xgpoc.eval.metrics import evaluate
from xgpoc.models.common import TemperatureScaler, config_dict
from xgpoc.models.g_bundle import g_identity
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
SANITISED = ROOT / "cache/xgd_sanitised/D3/sanitised.parquet"
AUX_WEIGHT = 0.5
BUSY_MIB = 1024                 # another process holding this much GPU memory is a job (pytest contexts hold ~200 MiB)
REPLAY_TOLERANCE = 1e-4         # max |logit difference| of a replay in identical batch order


def check_gpu_free() -> None:
    """Refuse to start while another process holds `BUSY_MIB` of GPU memory (one GPU job at a time)."""
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout
    busy = [line for line in query.splitlines() if (parts := [p.strip() for p in line.split(",")])[0].isdigit()
            and int(parts[0]) != os.getpid() and parts[1].isdigit() and int(parts[1]) >= BUSY_MIB]
    if busy:
        raise SystemExit(f"another job is using the GPU, refusing to start: {busy}")


def development_pool(index: Path) -> pd.DataFrame:
    """The training matches' shots of the approved plan with their fold, refused unless the counts are the plan's."""
    plan = a1.verify_plan(index)
    pool = a1.training_pool(rp.add_targets(load_index(index)), plan)
    expected = json.loads((a1.PLAN_DIR / "plan.json").read_text())["counts"]["shots"]["training"]
    if len(pool) != expected:
        raise ValueError(f"loaded {len(pool)} training shots, the plan has {expected} (frames missing from disk?)")
    return pool


def variant_inputs(sanitised: pd.DataFrame, pool: pd.DataFrame, variant: str) -> tuple[pd.DataFrame | None, dict | None]:
    """(A1 table indexed by shot_id, None) or (None, A2 points per shot) for the pool shots; both offsets of every pool
    shot must be in the sanitised table."""
    rows = sanitised[sanitised.shot_id.isin(pool.shot_id)]
    if len(rows) != len(a1.OFFSETS) * len(pool) or set(rows.offset_ms) != set(a1.OFFSETS):
        raise ValueError(f"the sanitised table has {len(rows)} rows for {len(pool)} pool shots x {len(a1.OFFSETS)} offsets")
    if variant == "A1":
        return a1.a1_table(rows).loc[pool.shot_id], None
    return None, a1.frame_points(rows)


def run_trial(k: int, seed: int, variant: str, pool: pd.DataFrame, out: Path, *, table: pd.DataFrame | None,
              points: dict | None, base_identity: dict, sources: dict, pixels: dict, smoke: bool) -> None:
    cfg, kw = recipe(AUX_WEIGHT, smoke)
    cfg.seed = seed
    name = a1.run_name(variant)
    tr, va = pool[pool.fold != k].reset_index(drop=True), pool[pool.fold == k].reset_index(drop=True)
    standardiser = a1.Standardiser.fit(table.loc[tr.shot_id]) if variant == "A1" else None
    spec = a1.make_spec(variant, kw["backbone"], kw["n_blocks"], kw["mask"], standardiser)
    backbone = g_identity(spec)
    pen = pool[pool.is_penalty]
    constants = {"penalty_rate": penalty_rate(int(pen.goal.sum()), len(pen)), "pooled_goal_rate": float(pool.goal.mean())}
    identity = {**base_identity, "fold": k, "seed": seed, "variant": variant, "name": name, "spec": spec, "backbone": backbone,
                "cfg": config_dict(cfg), "rrc": list(kw["rrc"]), "backbone_lr_mult": kw["backbone_lr_mult"],
                "constants": constants, "smoke": smoke}
    if verify_finished(out, identity):
        print("verified resume", out, flush=True)
        return
    if shutil.disk_usage(ROOT / "results").free < MIN_FREE_GIB * 2**30:
        raise OSError(f"Less than {MIN_FREE_GIB} GiB free; refusing to train")
    if smoke:
        tr, va = tr.head(64), va.head(64)
    geometry = a1.geometry_inputs(table, standardiser) if variant == "A1" else None
    print(f"=== {out} ({time.strftime('%H:%M')}) {variant} train {len(tr)} held-out {len(va)} ===", flush=True)
    model, fit, compute = a1.fit(tr, va, cfg, kw, spec, geometry=geometry, points=points, aux_col=rp.ON_TARGET)
    logits, temp = fit["valid_logits"], fit["temperature"]
    if sources != own_sources():
        raise ValueError("source files changed during training; restart with stable code")
    if pixels != frame_fingerprint(pool):
        raise ValueError("input frames changed during training")

    out.mkdir(parents=True, exist_ok=True)
    p = temp(logits)
    pd.DataFrame({"shot_id": va.shot_id, "goal": va.goal.astype(int), "logit": logits, "p": p}).to_csv(
        out / f"{name}_valid_predictions.csv", index=False)
    metrics = {**evaluate(va.goal.to_numpy(), p), "model": name, "params": compute["trainable_parameters"],
               "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"], "temperature": temp.t, "scored": "held-out fold"}
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    (out / "training.json").write_text(json.dumps({"history": fit["history"], "best_epoch": fit["best_epoch"],
                                                   "compute": compute, "temperature": temp.t}, indent=2) + "\n")
    files = [f"{name}_valid_predictions.csv", f"{name}_metrics.json", "training.json"]
    commit = code_version()
    provenance = make_provenance(sources, index_sha256=base_identity["index"]["sha256"],
                                 folds_sha256=base_identity["folds"]["sha256"], overlay_sha256=base_identity["overlay_sha256"])
    saved = save_bundle(out / "bundle.pt", model, spec=spec, backbone=backbone, commit=commit, provenance=provenance,
                        calibration={"temperature": temp.t, "penalty_rate": constants["penalty_rate"]},
                        recipe={"stage": "g_combo", "variant": variant, "name": name, "cfg": config_dict(cfg),
                                "rrc": list(kw["rrc"]), "backbone_lr_mult": kw["backbone_lr_mult"], "fold": k, "seed": seed,
                                "best_epoch": fit["best_epoch"], "ema": "selected-epoch EMA weights"})
    files.append("bundle.pt")
    record = {"identity": identity, "code_sha256": sources, "commit": commit, "bundle": saved,
              "files": {f: sha256_file(out / f) for f in files}}
    (out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"--- {out} done in {compute['train_minutes']:.1f} min, held-out LL {metrics['log_loss']:.4f} AUC {metrics['auc']:.4f}, "
          f"best epoch {fit['best_epoch']}, branch {compute['branch_parameters']} parameters", flush=True)
    del model, fit
    gc.collect()
    torch.cuda.empty_cache()


def replay(out: Path, variant: str, pool: pd.DataFrame, table: pd.DataFrame | None, points: dict | None,
           folds: list[int], seeds: list[int]) -> dict:
    """Rebuild every bundle from disk and rescore its held-out fold in the stored row order and batch size."""
    rows = []
    for k in folds:
        for seed in seeds:
            run = out / f"fold{k}" / f"seed{seed}"
            record = json.loads((run / "complete.json").read_text())
            for filename, digest in record["files"].items():
                if sha256_file(run / filename) != digest:
                    raise ValueError(f"corrupt file {run / filename}")
            stored = pd.read_csv(run / f"{a1.run_name(variant)}_valid_predictions.csv")
            bundle = a1.load(run / "bundle.pt")
            geometry = a1.geometry_inputs(table, a1.standardiser_of(bundle.spec)) if variant == "A1" else None
            frames = pool.set_index("shot_id").loc[stored.shot_id].reset_index()
            cfg = record["identity"]["cfg"]
            logits = a1.score(bundle.model, frames, a1.dataset_args(bundle.spec, geometry, points), cfg["batch_size"],
                              cfg["device"], cfg["num_workers"])
            p = TemperatureScaler()
            p.t = bundle.calibration["temperature"]
            rows.append({"fold": k, "seed": seed, "shots": len(stored),
                         "max_abs_logit": float(np.max(np.abs(logits - stored.logit.to_numpy()))),
                         "max_abs_p": float(np.max(np.abs(p(logits) - stored.p.to_numpy()))),
                         "bundle_sha256": record["bundle"]["sha256"]})
            print(rows[-1], flush=True)
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
    worst = max(r["max_abs_logit"] for r in rows)
    return {"variant": variant, "tolerance": REPLAY_TOLERANCE, "passed": bool(worst <= REPLAY_TOLERANCE),
            "max_abs_logit": worst, "runs": rows, "batch_order": "stored row order of each held-out fold, the recorded batch size",
            "git": code_version(), "finished": datetime.now(UTC).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=a1.VARIANTS, required=True)
    ap.add_argument("--out", type=Path, help="default results/xgd_<variant>")
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--sanitised", type=Path, default=SANITISED)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 64 shots per role (pipeline check only)")
    ap.add_argument("--replay", action="store_true", help="rescore the finished runs from their bundles")
    ap.add_argument("--replay-file", default="replay.json", help="file name of the replay record under --out")
    a = ap.parse_args()
    out = a.out or a1.results_dir(a.variant)
    check_gpu_free()

    with no_statsbomb_reads(), seal.development_labels_only(a1.PLAN_DIR / "plan.csv", a.index):
        pool = development_pool(a.index)
        try:
            sanitised_sha256 = records.verified_file(a.sanitised)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        table, points = variant_inputs(pd.read_parquet(a.sanitised), pool, a.variant)
        print(f"pool {len(pool)} shots / {pool.game.nunique()} matches, variant {a.variant}", flush=True)
        if a.replay:
            result = replay(out, a.variant, pool, table, points, a.folds, a.seeds)
            (out / a.replay_file).write_text(json.dumps(result, indent=2) + "\n")
            print(f"replay {'PASSED' if result['passed'] else 'FAILED'}: max |logit difference| {result['max_abs_logit']:.2e}")
            return
        sources = own_sources()
        pixels = frame_fingerprint(pool)
        base = {"index": {"path": str(a.index), "sha256": sha256_file(a.index)},
                "folds": {"path": str(a1.FOLDS), "sha256": sha256_file(a1.FOLDS)},
                "plan": {"sha256": a1.PLAN_SHA256}, "overlay_sha256": sha256_file(OVERLAY), "frames": pixels,
                "sanitised": {"path": str(a.sanitised), "sha256": sanitised_sha256}}
        for k in a.folds:
            for seed in a.seeds:
                run_trial(k, seed, a.variant, pool, out / f"fold{k}" / f"seed{seed}", table=table, points=points,
                          base_identity=base, sources=sources, pixels=pixels, smoke=a.smoke)
    print(f"XGD APPROACH 1 {a.variant} DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
