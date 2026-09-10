"""Round v1.4: G, G + local views and G + control views under one frame admission rule.

Same roles, cohort, selection and calibration as run_v13.py. New: the input frame of
every shot is decided by the contact admission rule (admissions.csv from
contact_timing_report.py for reviewed shots, nominal t0 for the rest), and two jointly
fine-tuned multi-view variants. Evaluate labels are never read here.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, penalty_rate, route_penalties
from xgpoc.eval.metrics import evaluate
from xgpoc.models.common import TrainConfig, _predict_logits, n_trainable, seed_all, train_torch
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.paths import INDEX
from xgpoc.v13.contact import AdmissionStatus
from xgpoc.v13.inputs import OFFSETS, PRIMARY_POSITIONAL, input_files_fingerprint, unique_map
from xgpoc.v13.protocol import digest, prepare
from xgpoc.v13.provenance import snapshot, verify_sources
from xgpoc.v14.models import MultiViewViTXG
from xgpoc.v14.views import MultiViewDataset

VARIANTS = ("g_base", "g_views", "g_views_control")
SEEDS = [0, 1, 2]
ADMISSIONS = Path("cache/v13/contact_timing/admissions.csv")
BOXES = PRIMARY_POSITIONAL / "tokens.npz"
SOURCE_PATHS = [Path(__file__), *Path("xgpoc").rglob("*.py")]
SOURCE_SNAPSHOT = snapshot(SOURCE_PATHS)


def apply_admissions(pool: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """`frame_path` becomes the admitted input frame; `frame_path_t0` keeps the detection
    frame. Unreviewed shots are nominal. Unsafe shots get the earliest frame, flagged, and
    are excluded from fit by the caller; nothing is dropped from the other roles."""
    if admissions.shot_id.duplicated().any():
        raise ValueError("duplicate shots in admissions")
    outside = admissions[~admissions.shot_id.isin(pool.shot_id)]
    if (outside.admission != AdmissionStatus.NOMINAL.value).any():
        raise ValueError(f"{len(outside)} admitted shots are not in the pool")
    admissions = admissions[admissions.shot_id.isin(pool.shot_id)]  # nominal rows of the historical test split
    out = pool.copy()
    out["frame_path_t0"] = out.frame_path
    out["admission"] = AdmissionStatus.NOMINAL.value
    out["gap_ms"] = np.nan
    table = admissions.set_index("shot_id")
    for shot_id, row in table.iterrows():
        i = out.index[out.shot_id == shot_id][0]
        status = AdmissionStatus(row.admission)
        out.loc[i, ["admission", "gap_ms"]] = status.value, row.gap_ms
        if status is AdmissionStatus.SAFE:
            if not (str(row.frame_path) and Path(row.frame_path).is_file()):
                raise ValueError(f"admitted frame missing for {shot_id}")
            out.loc[i, "frame_path"] = row.frame_path
        elif status is AdmissionStatus.UNSAFE:
            earliest = next(c for c in (f"frame_path_{o}" for o in OFFSETS) if str(out.loc[i].get(c, "") or "") not in ("", "nan"))
            out.loc[i, "frame_path"] = out.loc[i, earliest]
    return out


def construct(spec, pretrained=True):
    if spec["variant"] == "g_base":
        return FinetuneViTXG(**spec["kwargs"], pretrained=pretrained)
    return MultiViewViTXG(**spec["kwargs"], pretrained=pretrained)


def inputs(df, spec, boxes, train=False):
    common = {"train": train, "rrc_scale": (.85, 1.) if train else None, "aux_col": "on_target" if train else None}
    if spec["variant"] == "g_base":
        return ImageDataset(df, size_wh=(448, 252), mask="overlay", path_col="frame_path", **common)
    return MultiViewDataset(df, boxes, control=spec["variant"] == "g_views_control", box_frame_col="frame_path_t0",
                            **common)


def predict(model, ds, cfg):
    return _predict_logits(model, DataLoader(ds, cfg.batch_size, num_workers=cfg.num_workers), cfg.device)


def pixel_identity(pool):
    # frame_path already holds the admitted frame, so the fingerprint covers the admission outcome.
    return {"frames": input_files_fingerprint(pool), "overlay_sha256": digest(Path("configs/overlay_mask.png"))}


def run_trial(pool, manifest, out, variant, seed, boxes, sources, smoke=False):
    verify_sources(SOURCE_SNAPSHOT)
    pixels = pixel_identity(pool)
    kwargs = {"backbone": "dinov2_l", "n_blocks": 4, "n_out": 2}
    spec = {"kind": "g_multiview" if variant != "g_base" else "g", "variant": variant, "kwargs": kwargs}
    cfg = TrainConfig(epochs=5, patience=5, lr=3e-4, batch_size=8, warmup_epochs=1, max_minutes=90, seed=seed,
                      ema_decay=.998, aux_weight=.5)
    if smoke:
        cfg.epochs, cfg.patience, cfg.warmup_epochs = 2, 2, .5
    identity = {"protocol": manifest, "spec": spec, "cfg": asdict(cfg), "smoke": smoke,
                "code_sha256": SOURCE_SNAPSHOT, "pixels": pixels, "sources": sources,
                "admission_counts": pool.admission.value_counts().to_dict()}
    done = out / "complete.json"
    if done.exists():
        previous = json.loads(done.read_text())
        if previous["identity"] != identity:
            raise ValueError(f"Stale result at {out}")
        for filename, expected in previous["files"].items():
            if digest(out / filename) != expected:
                raise ValueError(f"Corrupt run file {out / filename}")
        print("verified resume", out, flush=True)
        return
    if shutil.disk_usage(out.parent if out.parent.exists() else Path(".")).free < 8 * 2**30:
        raise OSError("Less than 8 GiB free; refusing training")
    parts = {r: pool[pool.role == r].copy().reset_index(drop=True) for r in manifest["roles"]}
    if smoke:
        parts = {r: d.head(64) for r, d in parts.items()}
    parts["evaluate"] = parts["evaluate"].assign(goal=False, on_target=0.)
    fit_rows = parts["fit"][parts["fit"].admission != AdmissionStatus.UNSAFE.value].reset_index(drop=True)
    seed_all(seed)
    model = construct(spec)
    datasets = {r: inputs(d, spec, boxes, train=r == "fit") for r, d in parts.items()}
    datasets["fit"] = inputs(fit_rows, spec, boxes, train=True)
    for role, ds in datasets.items():
        expected = fit_rows if role == "fit" else parts[role]
        if ds.df.shot_id.tolist() != expected.shot_id.tolist():
            raise ValueError(f"{variant} changed {role} cohort")
    selection = parts["select"][~parts["select"].is_penalty].reset_index(drop=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    fit = train_torch(model, datasets["fit"], inputs(selection, spec, boxes), cfg, backbone_lr_mult=.02)
    compute = {"train_minutes": (time.time() - started) / 60, "epochs_run": len(fit["history"]),
               "updates": len(fit["history"]) * (len(datasets["fit"]) // cfg.batch_size),
               "trainable_parameters": n_trainable(model),
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    logits = {r: predict(model, datasets[r], cfg) for r in ("select", "calibrate", "evaluate")}
    verify_sources(SOURCE_SNAPSHOT)
    if pixels != pixel_identity(pool):
        raise ValueError("Input pixels changed during training")
    cal = parts["calibrate"]
    nonpen = ~cal.is_penalty.to_numpy()
    platt = PlattScaler().fit(logits["calibrate"][nonpen], cal.goal.to_numpy()[nonpen])
    pen = parts["fit"][parts["fit"].is_penalty]
    pp = penalty_rate(int(pen.goal.sum()), len(pen))
    out.mkdir(parents=True, exist_ok=True)
    source_dir = out / "sources"
    for filename in SOURCE_SNAPSHOT:
        target = source_dir / Path(filename).relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
    filenames = []
    for role, z in logits.items():
        d = parts[role]
        probability = route_penalties(platt(z), d.is_penalty.to_numpy(), pp)
        pred = pd.DataFrame({"shot_id": d.shot_id, "logit": z, "p": probability, "admission": d.admission})
        if role != "evaluate":
            pred["goal"] = d.goal.astype(int)
        filename = f"{role}_predictions.csv"
        pred.to_csv(out / filename, index=False)
        filenames.append(filename)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    bundle = {"version": 1, "spec": spec, "state_dict": state, "platt": {"a": platt.a, "b": platt.b},
              "penalty_rate": pp, "identity": identity,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    temporary = out / "bundle.tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, out / "bundle.pt")
    filenames.append("bundle.pt")
    metrics = {"best_epoch": fit["best_epoch"], "history": fit["history"], "compute": compute,
               "fit_rows": len(fit_rows), "fit_excluded_unsafe": int(len(parts["fit"]) - len(fit_rows)),
               "selection_log_loss": evaluate(parts["select"].goal, route_penalties(
                   fit["temperature"](logits["select"]), parts["select"].is_penalty.to_numpy(), pp))["log_loss"],
               "calibration": {"a": platt.a, "b": platt.b, "penalty_rate": pp}}
    (out / "training.json").write_text(json.dumps(metrics, indent=2) + "\n")
    filenames.append("training.json")
    done.write_text(json.dumps({"identity": identity, "files": {f: digest(out / f) for f in filenames}}, indent=2) + "\n")
    print("frozen", out, "selection LL", metrics["selection_log_loss"], "compute", compute, flush=True)
    del model, state, bundle, fit, datasets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pool(index: Path, out: Path, admissions_path: Path = ADMISSIONS):
    pool, manifest = prepare(index, out)
    pool["on_target"] = pool.label.eq("Shots on target").astype(float)
    pool = apply_admissions(pool, pd.read_csv(admissions_path))
    with np.load(BOXES) as data:
        boxes = unique_map(data["shot_id"].astype(str), data["boxes"], set(pool.shot_id))
    sources = {"admissions": {"path": str(admissions_path), "sha256": digest(admissions_path)},
               "boxes": {"path": str(BOXES), "sha256": digest(BOXES)}}
    return pool, manifest, boxes, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("results/v14"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--admissions", type=Path, default=ADMISSIONS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if any((a.out / f).exists() for f in ("frozen.json", "evaluated.json")):
        raise ValueError("Candidates are frozen. This protocol cannot train more candidates.")
    pool, manifest, boxes, sources = load_pool(a.index, a.out, a.admissions)
    suite_identity = {"code_sha256": SOURCE_SNAPSHOT, "pixels": pixel_identity(pool), "sources": sources}
    identity_path = a.out / "suite_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != suite_identity:
        raise ValueError("Suite source code, pixels or admissions changed; use a new output directory")
    if not identity_path.exists():
        identity_path.write_text(json.dumps(suite_identity, indent=2) + "\n")
    suite = {v: SEEDS for v in VARIANTS}
    suite_path = a.out / "suite.json"
    if suite_path.exists() and json.loads(suite_path.read_text()) != suite:
        raise ValueError("Declared suite changed")
    if not suite_path.exists():
        suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    print(manifest["counts"], pool.admission.value_counts().to_dict(), flush=True)
    for variant in a.variants:
        for seed in a.seeds:
            run_trial(pool, manifest, a.out / variant / f"seed{seed}", variant, seed, boxes, sources, smoke=a.smoke)


if __name__ == "__main__":
    main()
