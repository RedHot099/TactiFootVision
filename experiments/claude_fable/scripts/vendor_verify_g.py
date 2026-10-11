"""Replay, ensemble calibration, gate and manifest of the retrained G-combo (xg-vendor ticket 10); the CPU/GPU-light
counterpart of `scripts/verify_xgc_g.py` for the runs of `scripts/vendor_retrain_g.py`.

    ./run.sh python scripts/vendor_verify_g.py                          # results/xgv_g_combo, everything
    ./run.sh python scripts/vendor_verify_g.py --smoke --root results/xgv_g_smoke --index cache/dummy/index.parquet

1. Replay. Each bundle is loaded in this fresh process, then scores its own held-out fold and the gate shots in the
   training-time batch order. |dp| against the stored predictions must stay <= 1e-5.
2. Ensemble calibration (`ensemble.json`). Platt on the mean logit of the ten fold models over the training pool's
   out-of-fold predictions, prior correction to the pool's goal rate, penalty routing: `g_bundle.fit_ensemble_calibration`,
   the same steps as the archived G-combo. Nothing is fitted on the gate.
3. Gate (`gate.json`). The ten models on the gate shots against the archived G-combo's stored predictions on the same
   shots: point estimates decide with the plan's tolerance, the paired match-bootstrap interval is reported
   (`xgpoc.vendor.retrain_gate`). A failed gate or a replay above tolerance ends with a non-zero exit; nothing is retuned.
`manifest.tsv` lists every bundle. A smoke run stops after the replay and the manifest.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from report_complexity import configurations, deploy, mean_logit
from vendor_retrain_g import approved_split
from xgpoc.bundle import canonical_digest, check_data_files, read_bundle, sha256_file
from xgpoc.eval.calibration import logit, penalty_rate, sigmoid
from xgpoc.models.g_bundle import (
    ensemble_probability,
    fit_ensemble_calibration,
    g_outputs,
    load_g,
)
from xgpoc.paths import INDEX
from xgpoc.vendor import split_check
from xgpoc.vendor.retrain_gate import gate_report
from xgpoc.vendor.split_plan import GATE_TOLERANCE

ROOT = Path("results/xgv_g_combo")
REPLAY_TOL = 1e-5


def load_runs(records: dict[Path, dict], name: str) -> dict[tuple[int, int], dict[str, pd.DataFrame]]:
    """(fold, seed) -> {'valid': out-of-fold frame, 'test': gate frame} (shot_id, z = logit of the stored p), in the shape
    `report_complexity.deploy` reads, for exactly the runs that have a completion record."""
    runs = {}
    for run, rec in records.items():
        key = rec["identity"]["fold"], rec["identity"]["seed"]
        runs[key] = {role: pd.read_csv(run / f"{name}_{file}_predictions.csv").drop_duplicates("shot_id")
                     .assign(z=lambda d: logit(d.p.to_numpy()))[["shot_id", "z"]] for role, file in (("valid", "valid"), ("test", "gate"))}
    return runs


def replay(run: Path, pool: pd.DataFrame, gate: pd.DataFrame, name: str, smoke: bool) -> dict:
    """Max |dlogit| and |dp| between the stored training-time predictions and a fresh load."""
    k = int(run.parent.name.removeprefix("fold"))
    b = load_g(run / "bundle.pt")
    t = b.calibration["temperature"]
    worst = {"logit": 0.0, "p": 0.0}
    for role, df in (("valid", pool[pool.fold == k].reset_index(drop=True)), ("gate", gate)):
        stored = pd.read_csv(run / f"{name}_{role}_predictions.csv")
        df = df.head(len(stored)) if smoke else df
        if df.shot_id.tolist() != stored.shot_id.tolist():
            raise ValueError(f"{run}: {role} rows differ from the stored predictions")
        z, _, embedding = g_outputs(b.model, df, b.spec, batch_size=b.recipe["cfg"]["batch_size"])  # training batch composition
        if embedding.shape != (len(df), 5120) or not np.isfinite(embedding).all():
            raise ValueError(f"{run}: unexpected embedding {embedding.shape}")
        worst["logit"] = max(worst["logit"], float(np.abs(z - stored.logit.to_numpy()).max()))
        worst["p"] = max(worst["p"], float(np.abs(sigmoid(z / t) - stored.p.to_numpy()).max()))
    print(f"replay {run}: max |dlogit| {worst['logit']:.2e}, max |dp| {worst['p']:.2e}", flush=True)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="the runs were smoke runs (first 64 rows per role): replay and manifest only")
    a = ap.parse_args()

    records = {p.parent: json.loads(p.read_text()) for p in sorted(a.root.glob("fold*/seed*/complete.json"))}
    if not records:
        raise SystemExit(f"no finished runs under {a.root}")
    name = next(iter(records.values()))["identity"]["name"]
    for run, rec in records.items():
        for filename, expected in rec["files"].items():
            if sha256_file(run / filename) != expected:
                raise SystemExit(f"{run / filename} changed since it was written")

    pool, gate, plan_identity, folds_file = approved_split(a.index, a.smoke, a.root / "smoke_plan")
    rows = []
    for run, rec in records.items():
        r = replay(run, pool, gate, name, a.smoke) if not a.no_replay else {"logit": np.nan, "p": np.nan}
        bundle = read_bundle(run / "bundle.pt")
        prov = bundle["provenance"]
        ident = rec["identity"]
        if ident["plan"] != plan_identity:
            raise SystemExit(f"{run}: trained on a different plan than the one verified now")
        if prov["code"]["digest"] != canonical_digest(rec["code_sha256"]) or prov["data"] != {
                "index_sha256": ident["index"]["sha256"], "folds_sha256": ident["folds"]["sha256"],
                "overlay_sha256": ident["overlay_sha256"]}:
            raise SystemExit(f"{run}: bundle provenance disagrees with its completion record")
        check_data_files(prov, index_sha256=a.index, folds_sha256=folds_file)
        rows.append({"fold": ident["fold"], "seed": ident["seed"], "path": str(run / "bundle.pt"),
                     "bytes": rec["bundle"]["bytes"], "sha256": rec["files"]["bundle.pt"],
                     "n_params": rec["bundle"]["n_params"], "temperature": bundle["calibration"]["temperature"],
                     "penalty_rate": bundle["calibration"]["penalty_rate"],
                     "commit": bundle["commit"]["sha"], "tree_dirty": bundle["commit"]["dirty"],
                     "backbone_sha256": bundle["backbone"]["sha256"],
                     "backbone_file": bundle["backbone"]["file"], "backbone_revision": bundle["backbone"]["revision"],
                     "best_epoch": bundle["recipe"]["best_epoch"], "code_digest": prov["code"]["digest"],
                     "index_sha256": prov["data"]["index_sha256"], "folds_sha256": prov["data"]["folds_sha256"],
                     "overlay_sha256": prov["data"]["overlay_sha256"], "replay_max_abs_dlogit": r["logit"],
                     "replay_max_abs_dp": r["p"]})
    manifest = pd.DataFrame(rows).sort_values(["fold", "seed"])
    manifest.to_csv(a.root / "manifest.tsv", sep="\t", index=False)
    replay_ok = True
    if not a.no_replay:
        worst = manifest.replay_max_abs_dp.max()
        replay_ok = bool(worst <= REPLAY_TOL)
        print(f"replay worst |dp| {worst:.2e} (tolerance {REPLAY_TOL:g})", flush=True)
    if a.smoke:
        raise SystemExit(0 if replay_ok else "smoke replay error exceeds tolerance")

    # ---- ensemble calibration on the training pool's out-of-fold predictions
    shots = pd.concat([pool, gate]).drop_duplicates("shot_id").set_index("shot_id")
    labels, is_pen = shots.goal.astype(bool), shots.is_penalty.astype(bool)
    pooled_rate = float(pool.goal.mean())
    pen_pool = pool[pool.is_penalty]
    p_pen = penalty_rate(int(pen_pool.goal.sum()), len(pen_pool))
    if not np.allclose(manifest.penalty_rate, p_pen):
        raise SystemExit("bundles were calibrated with a different penalty rate than this plan implies")
    runs = load_runs(records, name)
    keys = configurations(runs)["g10"][0][1]
    if len(keys) != 10:
        raise SystemExit(f"{len(keys)} fold models found, expected 10")
    oof = mean_logit([runs[k]["valid"] for k in keys])
    if set(oof.index) != set(pool.shot_id):
        raise SystemExit("the out-of-fold predictions do not cover the training pool exactly")
    p = deploy(runs, keys, labels, is_pen, pooled_rate, p_pen)
    calibration = fit_ensemble_calibration(oof.to_numpy(), labels.loc[oof.index].to_numpy(), pooled_rate, p_pen)
    temps = [float(manifest[(manifest.fold == f) & (manifest.seed == s)].temperature.iloc[0]) for f, s in keys]
    raw = {k: pd.read_csv(a.root / f"fold{k[0]}" / f"seed{k[1]}" / f"{name}_gate_predictions.csv").set_index("shot_id").logit
           for k in keys}
    q = ensemble_probability([raw[k].loc[p.index].to_numpy() for k in keys], temps, calibration, is_pen.loc[p.index].to_numpy())
    calibration["check_max_abs_dp_vs_gate_predictions"] = float(np.abs(q - p.to_numpy()).max())
    members = [{"fold": f, "seed": s, "bundle": str(a.root / f"fold{f}" / f"seed{s}" / "bundle.pt"), "temperature": t,
                "sha256": manifest[(manifest.fold == f) & (manifest.seed == s)].sha256.iloc[0]}
               for (f, s), t in zip(keys, temps, strict=True)]
    (a.root / "ensemble.json").write_text(json.dumps({"name": name, "plan": plan_identity, "members": members, "calibration": calibration}, indent=2) + "\n")
    print("ensemble calibration", calibration, flush=True)

    # ---- gate against the archived G-combo's stored predictions on the same shots
    archived = pd.read_csv(split_check.ARCHIVED_G_COMBO).drop_duplicates("shot_id").set_index("shot_id").p
    if set(gate.shot_id) - set(archived.index) or set(p.index) != set(gate.shot_id):
        raise SystemExit("the gate shots are not covered exactly by both the retrained and the archived predictions")
    g = gate.drop_duplicates("shot_id").set_index("shot_id").loc[p.index]
    report = gate_report(g.goal.astype(int).to_numpy(), p.to_numpy(), archived.loc[p.index].to_numpy(), g.game.to_numpy())
    report |= {"n_models": len(keys), "plan": plan_identity, "tolerance": GATE_TOLERANCE,
               "max_abs_dp_vs_archived": float((p - archived.loc[p.index]).abs().max()),
               "replay_max_abs_dp": None if a.no_replay else float(manifest.replay_max_abs_dp.max())}
    (a.root / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    problems = []
    if not replay_ok:
        problems.append(f"replay error {manifest.replay_max_abs_dp.max():.2e} exceeds {REPLAY_TOL:g}")
    if not report["passed"]:
        problems.append("gate failed: stop here, nothing is retuned")
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
