"""Retrain G-combo with every fold model's trained weights saved (xg-credibility ticket 04).

    ./run.sh python scripts/run_xgc_g.py                      # 5 folds x seeds 0, 1 -> results/xgc_g_combo
    ./run.sh python scripts/run_xgc_g.py --smoke --out results/xgc_g_smoke --folds 0 --seeds 0

The recipe is `run_v12.STAGES["g_combo"]` (DINOv2-L, last 4 blocks + head, overlay mask,
random resized crop, on-target auxiliary head at weight 0.5, EMA 0.998, 5 epochs, epoch
selection on calibrated validation log-loss) on the archived home-team five-fold
assignment of `run_cv.py`; the body of `run_poc.run_g` is repeated here only because that
function discards the model. The head-free control is the same command with
`--aux-weight 0 --name <control name> --out <other dir>`.

Each (fold, seed) writes into <out>/fold<k>/seed<s>/:
    bundle.pt                        EMA weights of the trained tensors + calibration + spec + backbone hash
    <name>_{valid,test}_predictions.csv   shot_id, goal, logit, p (temperature-scaled); valid = the held-out fold
    <name>_metrics.json              test metrics of the single fold model (as the archived runs)
    training.json                    history, best epoch, compute
    complete.json                    identity (recipe, data, backbone) + source hashes + file hashes; its presence
                                     with matching hashes is what makes a run count as finished

Resumable: a run with a valid complete.json is skipped; one whose recorded identity differs
from the requested one is an error, never silently overwritten.
"""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import run_poc as rp
import torch
from run_cv import fold_table
from run_v12 import STAGES
from torch.utils.data import DataLoader
from xgpoc.bundle import code_version, make_provenance, save_bundle, sha256_file
from xgpoc.data.datasets import ImageDataset
from xgpoc.data.index import load_index
from xgpoc.eval.calibration import penalty_rate
from xgpoc.eval.metrics import evaluate
from xgpoc.models.common import (
    TrainConfig,
    _predict_logits,
    config_dict,
    n_trainable,
    seed_all,
    train_torch,
)
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.models.g_bundle import g_identity, g_spec
from xgpoc.paths import INDEX

STAGE = "g_combo"
MIN_FREE_GIB = 6  # the ticket keeps >= 5 GiB free throughout
OVERLAY = Path("configs/overlay_mask.png")
FOLDS_FILE = Path("cache/folds_v12_home5.csv")


def own_sources() -> dict[str, str]:
    """sha256 of every source file of this experiment that is loaded in this process."""
    root = Path.cwd().resolve()
    files = {Path(m.__file__) for m in sys.modules.values() if getattr(m, "__file__", None)}
    files = {f for f in files if f.is_absolute() and f.suffix == ".py" and f.is_relative_to(root)}
    return {str(f.relative_to(root)): sha256_file(f) for f in sorted(files)}


def frame_fingerprint(df: pd.DataFrame) -> dict:
    """Content hash of every input frame the models see (path + bytes), as one digest."""
    entries = [f"{p}\t{sha256_file(Path(p))}" for p in sorted(df.frame_path.unique())]
    return {"count": len(entries), "sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}


def fold_split(index: Path):
    """(CV pool with fold column, test) exactly as run_cv.py builds them."""
    df = rp.add_targets(load_index(index))
    folds = fold_table(df)
    pool = df[df.split != "test"].merge(folds[["game", "fold"]], on="game")
    return pool, df[df.split == "test"].reset_index(drop=True)


def recipe(aux_weight: float, smoke: bool) -> tuple[TrainConfig, dict]:
    """The archived g_combo recipe (config as run_g adjusts it) with the auxiliary weight overridable."""
    _, _, cfg_over, kw = STAGES[STAGE]
    cfg = TrainConfig(**{**cfg_over, "aux_weight": aux_weight})
    cfg = TrainConfig(**{**config_dict(cfg), "lr": kw["lr"], "batch_size": min(cfg.batch_size, 8 if "_l" in kw["backbone"] else 16)})
    if smoke:
        cfg.epochs, cfg.patience, cfg.warmup_epochs = 2, 2, 0.5
    return cfg, kw


def verify_finished(out: Path, identity: dict) -> bool:
    done = out / "complete.json"
    if not done.exists():
        return False
    record = json.loads(done.read_text())
    if record["identity"] != identity:
        raise ValueError(f"{out} holds a different run than requested (identity changed); use a new --out")
    for filename, expected in record["files"].items():
        if sha256_file(out / filename) != expected:
            raise ValueError(f"corrupt file {out / filename}")
    return True


def run_trial(k: int, seed: int, pool: pd.DataFrame, test: pd.DataFrame, out: Path, *, name: str, aux_weight: float,
              base_identity: dict, sources: dict, smoke: bool, pixels: dict) -> None:
    frames = pd.concat([pool, test])  # every frame any fold can read; `pixels` is its fingerprint
    cfg, kw = recipe(aux_weight, smoke)
    cfg.seed = seed
    spec = g_spec(kw["backbone"], kw["n_blocks"], n_out=2, mask=kw["mask"])
    backbone = g_identity(spec)
    pen = pool[(pool.split == "train") & pool.is_penalty]  # as the archived calibration: train-split penalties
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
    te = test
    if smoke:
        tr, va, te = tr.head(64), va.head(64), te.head(64)
    print(f"=== {out} ({time.strftime('%H:%M')}) train {len(tr)} oof {len(va)} test {len(te)} ===", flush=True)
    started = time.time()
    seed_all(cfg.seed)
    model = FinetuneViTXG(kw["backbone"], n_blocks=kw["n_blocks"], n_out=2)
    inp = dict(size_wh=tuple(spec["input"]["size_wh"]), mask=spec["input"]["mask"])
    fit = train_torch(model, ImageDataset(tr, train=True, **inp, rrc_scale=tuple(kw["rrc"]), aux_col=rp.ON_TARGET),
                      ImageDataset(va, **inp), cfg, backbone_lr_mult=kw["backbone_lr_mult"])
    if fit["best_epoch"] < 0:
        raise ValueError("no epoch was selected (non-finite validation score); refusing to save a bundle")
    temp = fit["temperature"]  # fitted on the held-out fold's logits of the selected (EMA) weights
    datasets = {"valid": ImageDataset(va, **inp), "test": ImageDataset(te, **inp)}
    logits = {role: _predict_logits(model, DataLoader(ds, cfg.batch_size, num_workers=cfg.num_workers), cfg.device)
              for role, ds in datasets.items()}
    compute = {"train_minutes": (time.time() - started) / 60, "epochs_run": len(fit["history"]),
               "trainable_parameters": n_trainable(model),
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    if sources != own_sources():
        raise ValueError("source files changed during training; restart with stable code")
    if pixels != frame_fingerprint(frames):
        raise ValueError("input frames changed during training")

    out.mkdir(parents=True, exist_ok=True)
    files = []
    for role, df in (("valid", va), ("test", te)):
        p = temp(logits[role])
        pd.DataFrame({"shot_id": df.shot_id, "goal": df.goal.astype(int), "logit": logits[role], "p": p}).to_csv(
            out / f"{name}_{role}_predictions.csv", index=False)
        files.append(f"{name}_{role}_predictions.csv")
    metrics = {**evaluate(te.goal.to_numpy(), temp(logits["test"])), "model": name, "params": compute["trainable_parameters"],
               "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"], "temperature": temp.t}
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
    print(f"--- {out} done in {compute['train_minutes']:.1f} min, test LL {metrics['log_loss']:.4f} AUC {metrics['auc']:.4f}, "
          f"bundle {saved['bytes'] / 2**20:.0f} MiB", flush=True)
    del model, fit, datasets
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/xgc_g_combo"))
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of the on-target auxiliary loss; 0 = head-free control")
    ap.add_argument("--name", default=STAGES[STAGE][1], help="model name used in file names")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 64 shots per role (pipeline check only)")
    a = ap.parse_args()

    pool, test = fold_split(a.index)
    sources = own_sources()
    print(f"index {a.index}: pool {len(pool)} shots / {pool.game.nunique()} games, test {len(test)}", flush=True)
    base = {"index": {"path": str(a.index), "sha256": sha256_file(a.index)},
            "folds": {"path": str(FOLDS_FILE), "sha256": sha256_file(FOLDS_FILE)},
            "overlay_sha256": sha256_file(OVERLAY)}
    pixels = frame_fingerprint(pd.concat([pool, test]))
    base["frames"] = pixels
    for k in a.folds:
        for seed in a.seeds:
            run_trial(k, seed, pool, test, a.out / f"fold{k}" / f"seed{seed}", name=a.name, aux_weight=a.aux_weight,
                      base_identity=base, sources=sources, smoke=a.smoke, pixels=pixels)
    print(f"XGC G QUEUE DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
