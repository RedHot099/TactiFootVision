"""Train with independent selection and calibration; never score evaluate labels."""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, penalty_rate, route_penalties
from xgpoc.eval.metrics import evaluate
from xgpoc.models.common import TrainConfig, _predict_logits, seed_all, train_torch
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.paths import INDEX
from xgpoc.v13.protocol import digest, prepare
from xgpoc.v13.provenance import snapshot, verify_sources
from xgpoc.v13.inputs import VARIANTS, input_files_fingerprint


SOURCE_PATHS = [Path(__file__), *Path("xgpoc").rglob("*.py")]
SOURCE_SNAPSHOT = snapshot(SOURCE_PATHS)


def construct(spec, pretrained=True):
    if spec["kind"] == "g":
        return FinetuneViTXG(**spec["kwargs"], pretrained=pretrained)
    from xgpoc.v13.models import ContextHead
    return ContextHead(**spec["kwargs"])


def inputs(df, spec, train=False, cache=None):
    if spec["kind"] == "g":
        return ImageDataset(df, size_wh=(448, 252), mask="overlay", train=train,
                            rrc_scale=(.85, 1.) if train else None, aux_col="on_target" if train else None,
                            path_col=spec["path_col"])
    from xgpoc.v13.inputs import ContextDataset
    return ContextDataset(df, cache, spec["variant"])


def predict(model, ds, cfg):
    return _predict_logits(model, DataLoader(ds, cfg.batch_size, num_workers=cfg.num_workers), cfg.device)


def run_trial(pool, manifest, out, variant, seed, cache, smoke=False):
    verify_sources(SOURCE_SNAPSHOT)
    pixel_identity = {"frames": input_files_fingerprint(pool),
                      "overlay_sha256": digest(Path("configs/overlay_mask.png"))}
    if variant.startswith("g_"):
        path_col = "frame_path_-400" if variant == "g_pre400" else "frame_path"
        spec = {"kind": "g", "variant": variant, "path_col": path_col,
                "kwargs": {"backbone": "dinov2_l", "n_blocks": 4, "n_out": 2}}
        cfg = TrainConfig(epochs=5, patience=5, lr=3e-4, batch_size=8, warmup_epochs=1,
                          max_minutes=60, seed=seed, ema_decay=.998, aux_weight=.5)
    else:
        from xgpoc.v13.models import variant_kwargs
        spec = {"kind": "head", "variant": variant, "kwargs": variant_kwargs(variant, cache)}
        cfg = TrainConfig(epochs=25, patience=25, lr=1e-4, batch_size=32, warmup_epochs=2,
                          max_minutes=20, seed=seed, ema_decay=.999)
    if smoke:
        cfg.epochs, cfg.patience, cfg.warmup_epochs = 2, 2, .5
    identity = {"protocol": manifest, "spec": spec, "cfg": asdict(cfg), "smoke": smoke,
                "code_sha256": SOURCE_SNAPSHOT,
                "input_fingerprint": cache.metadata["fingerprint"] if cache is not None else None,
                "pixels": pixel_identity}
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
    # Blind evaluation inputs cannot participate in selection or calibration.
    parts["evaluate"] = parts["evaluate"].assign(goal=False, on_target=0.)
    seed_all(seed)
    model = construct(spec)
    datasets = {r: inputs(d, spec, train=r == "fit", cache=cache) for r, d in parts.items()}
    for role, ds in datasets.items():
        if ds.df.shot_id.tolist() != parts[role].shot_id.tolist():
            raise ValueError(f"{variant} changed {role} cohort")
    selection = parts["select"][~parts["select"].is_penalty].reset_index(drop=True)
    fit = train_torch(model, datasets["fit"], inputs(selection, spec, cache=cache), cfg,
                      backbone_lr_mult=.02 if spec["kind"] == "g" else None)
    logits = {r: predict(model, datasets[r], cfg) for r in ("select", "calibrate", "evaluate")}
    verify_sources(SOURCE_SNAPSHOT)
    if pixel_identity != {"frames": input_files_fingerprint(pool),
                           "overlay_sha256": digest(Path("configs/overlay_mask.png"))}:
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
        pred = pd.DataFrame({"shot_id": d.shot_id, "logit": z, "p": probability})
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
    metrics = {"best_epoch": fit["best_epoch"], "history": fit["history"],
               "selection_log_loss": evaluate(parts["select"].goal, route_penalties(
                   fit["temperature"](logits["select"]), parts["select"].is_penalty.to_numpy(), pp))["log_loss"],
               "calibration": {"a": platt.a, "b": platt.b, "penalty_rate": pp}}
    (out / "training.json").write_text(json.dumps(metrics, indent=2) + "\n")
    filenames.append("training.json")
    done.write_text(json.dumps({"identity": identity, "files": {f: digest(out / f) for f in filenames}}, indent=2) + "\n")
    print("frozen", out, "selection LL", metrics["selection_log_loss"], flush=True)
    del model, state, bundle, fit, datasets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("results/v13"))
    ap.add_argument("--variants", nargs="+", default=["g_base"],
                    choices=["g_base", "g_pre400", *VARIANTS])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if any((a.out / f).exists() for f in ("frozen.json", "evaluated.json")):
        raise ValueError("Candidates are frozen. This protocol cannot train more candidates.")
    pool, manifest = prepare(a.index, a.out)
    suite_identity = {"code_sha256": SOURCE_SNAPSHOT,
                      "pixels": {"frames": input_files_fingerprint(pool),
                                 "overlay_sha256": digest(Path("configs/overlay_mask.png"))}}
    identity_path = a.out / "suite_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != suite_identity:
        raise ValueError("Suite source code or pixels changed; use a new output directory")
    if not identity_path.exists():
        identity_path.write_text(json.dumps(suite_identity, indent=2) + "\n")
    suite = {v: [0, 1] if v.startswith("g_") else [0, 1, 2]
             for v in ("g_base", "g_pre400", *VARIANTS)}
    suite_path = a.out / "suite.json"
    if suite_path.exists() and json.loads(suite_path.read_text()) != suite:
        raise ValueError("Declared suite changed")
    if not suite_path.exists():
        suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    pool["on_target"] = pool.label.eq("Shots on target").astype(float)
    print(manifest["counts"], flush=True)
    if a.prepare:
        return
    cache = None
    if any(not v.startswith("g_") for v in a.variants):
        from xgpoc.v13.inputs import load_inputs
        cache = load_inputs(pool)
    for variant in a.variants:
        for seed in a.seeds:
            run_trial(pool, manifest, a.out / variant / f"seed{seed}", variant, seed, cache, smoke=a.smoke)


if __name__ == "__main__":
    main()
