"""Replay, reproduction gate, ensemble calibration and manifest for the retrained G-combo.

    ./run.sh python scripts/verify_xgc_g.py                       # results/xgc_g_combo, everything
    ./run.sh python scripts/verify_xgc_g.py --no-replay           # gate + manifest from stored predictions only

1. Replay. Each bundle is loaded in this fresh process from disk (pretrained backbone rebuilt,
   weight hash checked), then scores its own held-out fold and the test split in the
   training-time batch order. |dp| against the stored predictions must stay <= 1e-5.
2. Gate. The ten fold models are combined exactly as the archived G-combo was
   (`report_complexity.deploy`: Platt on the pooled out-of-fold logits, prior correction,
   penalty routing) and scored on the official test split against the archived log-loss and AUC.
3. Calibration and manifest. `ensemble.json` holds what is needed to score new inputs with the
   ten bundles (`models.g_bundle.ensemble_probability`); `manifest.tsv` lists every bundle with
   size, sha256, commit, provenance digests and replay error. Nothing here is fitted on the test split.
   A bundle's recorded index/folds/overlay hashes are checked against the files on disk first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from report_complexity import configurations, deploy, mean_logit, scores
from run_xgc_g import FOLDS_FILE, OVERLAY, fold_split
from xgpoc.bundle import canonical_digest, check_data_files, read_bundle, sha256_file
from xgpoc.eval.calibration import logit, penalty_rate, sigmoid
from xgpoc.models.g_bundle import (
    ensemble_probability,
    fit_ensemble_calibration,
    g_outputs,
    load_g,
)
from xgpoc.paths import INDEX

ARCHIVED = {"log_loss": 0.3241997, "auc": 0.7923346}  # results_reports/ensemble_vs_g_2026-10/ladder_vs_g10.csv, rung g10
GATE = {"log_loss": 0.002, "auc": 0.005}
REPLAY_TOL = 1e-5
ARCHIVED_G10 = Path("results/v12_cv_ensemble/g_combo_test_predictions.csv")


def load_runs(records: dict[Path, dict], name: str) -> dict[tuple[int, int], dict[str, pd.DataFrame]]:
    """(fold, seed) -> {'valid': OOF frame, 'test': test frame} (shot_id, z = logit of the stored p),
    for exactly the runs that have a completion record."""
    runs = {}
    for run, rec in records.items():
        key = rec["identity"]["fold"], rec["identity"]["seed"]
        runs[key] = {role: pd.read_csv(run / f"{name}_{role}_predictions.csv").drop_duplicates("shot_id")
                     .assign(z=lambda d: logit(d.p.to_numpy()))[["shot_id", "z"]] for role in ("valid", "test")}
    return runs


def replay(run: Path, pool: pd.DataFrame, test: pd.DataFrame, name: str, smoke: bool) -> dict:
    """Max |dlogit| and |dp| between the stored training-time predictions and a fresh load."""
    k = int(run.parent.name.removeprefix("fold"))
    b = load_g(run / "bundle.pt")
    t = b.calibration["temperature"]
    worst = {"logit": 0.0, "p": 0.0}
    for role, df in (("valid", pool[pool.fold == k].reset_index(drop=True)), ("test", test)):
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
    ap.add_argument("--root", type=Path, default=Path("results/xgc_g_combo"))
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--name", default=None, help="model name in file names (default: read from the first complete.json)")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="not the reproduction target (e.g. the head-free control): report test metrics, no pass/fail gate")
    ap.add_argument("--smoke", action="store_true", help="the runs were smoke runs (first 64 rows per role)")
    a = ap.parse_args()

    records = {p.parent: json.loads(p.read_text()) for p in sorted(a.root.glob("fold*/seed*/complete.json"))}
    if not records:
        raise SystemExit(f"no finished runs under {a.root}")
    name = a.name or next(iter(records.values()))["identity"]["name"]
    for run, rec in records.items():
        for filename, expected in rec["files"].items():
            if sha256_file(run / filename) != expected:
                raise SystemExit(f"{run / filename} changed since it was written")

    pool, test = fold_split(a.index)
    rows = []
    for run, rec in records.items():
        r = replay(run, pool, test, name, a.smoke) if not a.no_replay else {"logit": np.nan, "p": np.nan}
        bundle = read_bundle(run / "bundle.pt")
        prov = bundle["provenance"]
        ident = rec["identity"]
        if prov["code"]["digest"] != canonical_digest(rec["code_sha256"]) or prov["data"] != {
                "index_sha256": ident["index"]["sha256"], "folds_sha256": ident["folds"]["sha256"],
                "overlay_sha256": ident["overlay_sha256"]}:
            raise SystemExit(f"{run}: bundle provenance disagrees with its completion record")
        check_data_files(prov, index_sha256=a.index, folds_sha256=FOLDS_FILE, overlay_sha256=OVERLAY)
        rows.append({"fold": rec["identity"]["fold"], "seed": rec["identity"]["seed"], "path": str(run / "bundle.pt"),
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
    if not a.no_replay:
        print(f"replay worst |dp| {manifest.replay_max_abs_dp.max():.2e} (tolerance {REPLAY_TOL:g})", flush=True)

    # ---- gate: the ten fold models as the archived G10 deployment
    idx = pd.read_parquet(a.index).drop_duplicates("shot_id")
    idx["goal"], idx["is_penalty"] = idx.goal.astype(bool), idx.is_penalty.astype(bool)
    labels, is_pen = idx.set_index("shot_id").goal, idx.set_index("shot_id").is_penalty
    pooled_rate = float(idx[idx.split != "test"].goal.mean())
    pen_train = idx[(idx.split == "train") & idx.is_penalty]
    p_pen = penalty_rate(int(pen_train.goal.sum()), len(pen_train))
    runs = load_runs(records, name)
    keys = configurations(runs)["g10"][0][1]
    if len(keys) != 10 and not a.smoke:
        raise SystemExit(f"{len(keys)} fold models found, expected 10")
    p = deploy(runs, keys, labels, is_pen, pooled_rate, p_pen)
    auc, ll, brier = scores(labels.loc[p.index].to_numpy().astype(float), p.to_numpy())
    got = {"log_loss": float(ll), "auc": float(auc)}
    verdict = {m: {"retrained": got[m], "archived": ARCHIVED[m], "delta": got[m] - ARCHIVED[m], "tolerance": GATE[m],
                   "pass": abs(got[m] - ARCHIVED[m]) <= GATE[m]} for m in ARCHIVED}
    if not np.allclose(manifest.penalty_rate, p_pen):
        raise SystemExit("bundles were calibrated with a different penalty rate than this index implies")
    ref = pd.read_csv(ARCHIVED_G10).drop_duplicates("shot_id").set_index("shot_id").p
    ref_dp = float((p.loc[ref.index] - ref).abs().max()) if set(ref.index) <= set(p.index) else None
    gate = {"n_models": len(keys), "n_test": len(p), "brier": float(brier), "metrics": verdict,
            "max_abs_dp_vs_archived_g10": ref_dp,
            "passed": None if a.control else all(v["pass"] for v in verdict.values())}
    (a.root / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2), flush=True)

    # ---- ensemble calibration (fitted on out-of-fold logits only) and a check that it reproduces the gate predictions
    oof = mean_logit([runs[k]["valid"] for k in keys])
    y = labels.loc[oof.index].to_numpy()
    calibration = fit_ensemble_calibration(oof.to_numpy(), y, pooled_rate, p_pen)
    temps = [float(manifest[(manifest.fold == f) & (manifest.seed == s)].temperature.iloc[0]) for f, s in keys]
    raw = {k: pd.read_csv(a.root / f"fold{k[0]}" / f"seed{k[1]}" / f"{name}_test_predictions.csv").set_index("shot_id").logit
           for k in keys}
    ids = p.index
    q = ensemble_probability([raw[k].loc[ids].to_numpy() for k in keys], temps, calibration, is_pen.loc[ids].to_numpy())
    calibration["check_max_abs_dp_vs_gate_predictions"] = float(np.abs(q - p.to_numpy()).max())
    members = [{"fold": f, "seed": s, "bundle": str(a.root / f"fold{f}" / f"seed{s}" / "bundle.pt"), "temperature": t,
                "sha256": manifest[(manifest.fold == f) & (manifest.seed == s)].sha256.iloc[0]}
               for (f, s), t in zip(keys, temps, strict=True)]
    (a.root / "ensemble.json").write_text(json.dumps({"name": name, "members": members, "calibration": calibration}, indent=2) + "\n")
    print("ensemble calibration", calibration, flush=True)
    problems = []
    if not a.no_replay and manifest.replay_max_abs_dp.max() > REPLAY_TOL:
        problems.append(f"replay error {manifest.replay_max_abs_dp.max():.2e} exceeds {REPLAY_TOL:g}")
    if gate["passed"] is False:
        problems.append("reproduction gate failed: stop here, nothing is retuned")
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
