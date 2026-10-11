"""Freeze the positional comparator (xg-vendor ticket 06) and write `results/xgv_comparator/`.

Reads the verified StatsBomb data version (ticket 01) and the frozen pairing (ticket 02, `results/xgv_pairing`, checked
against its completion record and against the same StatsBomb version). Computes the comparator features of every
StatsBomb shot (`xgpoc.vendor.comparator_features`), trains the three rungs once on the matches outside the 135-match
overlap with match-grouped CV tuning (`xgpoc.vendor.comparator`), enforces kill rule K1 and publishes:

    predictions.parquet     label-free predictions for every evaluation shot (primary pairs, recovered goals,
                            StatsBomb-only shots; penalties among them get the pool's penalty rate) by StatsBomb
                            event id: soccernet_shot_id (primary), population,
                            prediction_source, p_<rung> and logit_<rung> for constant, dist_angle, event, freeze_frame
    features.parquet        the comparator features of every StatsBomb shot (no labels)
    models/<rung>.joblib    each fitted rung (features, pool categories, estimator), saved separately
    grid_<rung>.csv         every setting searched: best iteration and pooled CV log-loss
    pool_cv.parquet         training pool: fold, label, vendor xG and each rung's out-of-fold probability
    pool_metrics.csv        pool metrics of the vendor and each rung, overall and per competition and season
    report.json             pool and population counts, selected settings, grid edges, K1, penalty rate, hashes
    complete.json           input hashes, source hashes, code commit, summary and the sha256 of every file above

If K1 fails, the report goes to `xgv_comparator_k1_failed.json` next to the folder and nothing is published. A finished
folder is verified instead of rebuilt; changed inputs or sources are an error (use another --out).

    ./run.sh python scripts/vendor_comparator.py [--out DIR] [--jobs 8]
"""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd
import pyarrow
import sklearn
from xgpoc.bundle import code_version
from xgpoc.paths import STATSBOMB
from xgpoc.vendor import comparator as cmp
from xgpoc.vendor import comparator_features as cf
from xgpoc.vendor import pairing_inputs
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
VERSION = "xgv_comparator_v1"
OUT = ROOT / "results/xgv_comparator"
PAIRING = ROOT / "results/xgv_pairing"
PAIRING_FILES = ("match_map.csv", "pairs.parquet", "sb_unpaired.parquet", "recovered_goals.csv")
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/comparator.py", ROOT / "xgpoc/vendor/comparator_features.py")
# The expected sizes on the real data (the prototype's); a different number is reported in the summary, not hidden.
EXPECTED = {"overlap_matches": 135, "pool_matches": 1491, "pool_shots": 36626}
PROTOTYPE = {"freeze_frame_cv_log_loss": 0.2478, "event_cv_log_loss": 0.2552, "vendor_log_loss": 0.2473}


def json_text(value) -> str:
    """The one JSON form of every record here: sorted keys, indented, no NaN."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def counts(values: pd.Series) -> dict:
    """Value counts as a JSON-ready dict."""
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


def verify_pairing(root: Path, statsbomb_sha: str) -> dict:
    """The pairing's completion record, after checking the files used here against it and its StatsBomb version
    against the one read here."""
    record = json.loads((root / "complete.json").read_text())
    if record["identity"]["inputs"]["statsbomb"]["complete_json_sha256"] != statsbomb_sha:
        raise ValueError(f"{root} was built from another StatsBomb version")
    for name in PAIRING_FILES:
        if file_sha256(root / name) != record["files"][name]:
            raise ValueError(f"{root / name} differs from the hash in the pairing's completion record")
    return {"root": str(root), "complete_json_sha256": file_sha256(root / "complete.json"),
            "files": {name: record["files"][name] for name in PAIRING_FILES}}


def identity_of(statsbomb: dict, pairing: dict) -> dict:
    """What the folder is a function of: data version, pairing, source files."""
    return {"version": VERSION, "inputs": {"statsbomb": statsbomb, "pairing": pairing},
            "sources": {str(p): file_sha256(p) for p in SOURCE_FILES}}


def verify_complete(out: Path, identity: dict) -> dict:
    """The completion record of a finished folder, after checking its identity and every file hash (as vendor_pair.py)."""
    record = json.loads((out / "complete.json").read_text())
    if record["identity"] != identity:
        raise ValueError(f"Inputs or sources changed since {out / 'complete.json'}; use a new --out")
    present = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - {"complete.json"}
    if unlisted := sorted(present - set(record["files"])):
        raise ValueError(f"{len(unlisted)} files are not in the completion record, e.g. {unlisted[0]}")
    for name, expected in record["files"].items():
        if file_sha256(out / name) != expected:
            raise ValueError(f"Output changed: {name}")
    return record


def read_inputs(pairing_root: Path):
    """(shots, context, matches, pairing frames, identity), every input checked against its completion record."""
    _, statsbomb = pairing_inputs.read_statsbomb(STATSBOMB)   # verifies the table hashes against the data version
    shots = pd.read_parquet(STATSBOMB / "tables/shots.parquet")
    context = pd.read_parquet(STATSBOMB / "tables/context.parquet")
    matches = pd.read_parquet(STATSBOMB / "tables/matches.parquet")
    pairing = verify_pairing(pairing_root, statsbomb["complete_json_sha256"])
    frames = {"match_map": pd.read_csv(pairing_root / "match_map.csv"), "pairs": pd.read_parquet(pairing_root / "pairs.parquet"),
              "sb_unpaired": pd.read_parquet(pairing_root / "sb_unpaired.parquet"),
              "recovered_goals": pd.read_csv(pairing_root / "recovered_goals.csv")}
    return shots, context, matches, frames, identity_of(statsbomb, pairing)


def build(shots, context, matches, frames, jobs: int) -> dict:
    """Everything the folder holds, or K1Failed with the report attached."""
    overlap = set(frames["match_map"].sb_match_id.dropna().astype(int))
    evaluation = cmp.evaluation_shots(frames["pairs"], frames["sb_unpaired"])
    recovered = set(frames["recovered_goals"].loc[frames["recovered_goals"].recovered, "sb_event_id"])
    if recovered != set(evaluation.statsbomb_event_id[evaluation.population.eq("recovered_goal")]):
        raise ValueError("recovered_goals.csv and the recovered_goal population of sb_unpaired disagree")
    features = cf.feature_table(shots, context)
    pool = cmp.training_pool(features, shots, overlap, evaluation.statsbomb_match_id)

    folds = cmp.match_folds(pool.match_id)
    fits = {name: cmp.fit_rung(name, pool, folds, n_jobs=jobs) for name in cmp.RUNGS}
    for name, fit in fits.items():
        print(f"{name}: CV log-loss {fit.cv_log_loss:.5f}, {fit.config}, iterations {fit.iterations}, edges {fit.grid_edges}", flush=True)
    cv = cmp.pool_cv_table(pool, fits, folds, matches)
    overall = cmp.pool_metrics(cv)
    by_season = cmp.pool_metrics(cv, ["competition", "season"])
    vendor_ll = float(overall.loc[overall.rung.eq("vendor"), "log_loss"].iloc[0])
    k1 = cmp.k1(fits["freeze_frame"].cv_log_loss, vendor_ll)
    pens = cmp.penalty_rate(shots, overlap)
    constant = float(pool.goal.mean())
    report = {"pool": {"shots": len(pool), "goals": int(pool.goal.sum()), "matches": int(pool.match_id.nunique()),
                       "excluded_overlap_matches": len(overlap), "freeze_frame_share": float((pool.ff_n > 0).mean()),
                       "folds": cmp.N_FOLDS, "shots_per_fold": pd.Series(folds).value_counts().sort_index().tolist()},
              "rungs": {name: fit.summary() for name, fit in fits.items()} | {"constant": {"goal_rate": constant}},
              "k1": k1, "penalty_rate": pens, "prototype": PROTOTYPE,
              "post_contact_exclusions": {f"{where}.{name}": why for (where, name), why in cf.POST_CONTACT.items()},
              "evaluation": {"shots": len(evaluation), "by_population": counts(evaluation.population),
                             "matches": int(evaluation.statsbomb_match_id.nunique())}}
    cmp.check_k1(k1, report)
    predictions = cmp.predict({n: f.model for n, f in fits.items()}, features, evaluation, constant, pens["rate"], pool.match_id)
    report["evaluation"]["by_prediction_source"] = counts(predictions.prediction_source)
    return {"report": report, "predictions": predictions, "features": features, "fits": fits, "cv": cv,
            "metrics": pd.concat([overall.assign(competition="(all)", season="(all)"), by_season], ignore_index=True)}


def publish(out: Path, result: dict, identity: dict) -> dict:
    """Write `result` into a staging folder, hash every file, write the completion record, then swap it into `out`."""
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "models").mkdir(parents=True)
    result["predictions"].to_parquet(stage / "predictions.parquet", index=False, compression="zstd")
    result["features"].to_parquet(stage / "features.parquet", index=False, compression="zstd")
    result["cv"].to_parquet(stage / "pool_cv.parquet", index=False, compression="zstd")
    metrics = result["metrics"]
    metrics[["competition", "season", *[c for c in metrics.columns if c not in ("competition", "season")]]].to_csv(
        stage / "pool_metrics.csv", index=False)
    for name, fit in result["fits"].items():
        joblib.dump(fit.model, stage / "models" / f"{name}.joblib")
        fit.grid.to_csv(stage / f"grid_{name}.csv", index=False)
    report = result["report"] | {"predictions_sha256": file_sha256(stage / "predictions.parquet")}
    (stage / "report.json").write_text(json_text(report))
    files = {str(p.relative_to(stage)): file_sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    if {str(p): file_sha256(p) for p in SOURCE_FILES} != identity["sources"]:
        raise ValueError("A source changed during the run")
    pool = report["pool"]
    summary = {"pool_matches": pool["matches"], "pool_shots": pool["shots"], "overlap_matches": pool["excluded_overlap_matches"],
               "k1": report["k1"], "cv_log_loss": {n: r["cv_log_loss"] for n, r in report["rungs"].items() if "cv_log_loss" in r},
               "grid_edges": {n: r["grid_edges"] for n, r in report["rungs"].items() if "grid_edges" in r},
               "evaluation_shots": report["evaluation"]["by_population"], "predictions_sha256": report["predictions_sha256"]}
    summary["differs_from_expected"] = {k: [summary[k], v] for k, v in EXPECTED.items() if summary[k] != v}
    runtime = {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__}
    (stage / "complete.json").write_text(json_text({"identity": identity, "code": code_version(ROOT), "summary": summary,
                                                    "runtime": runtime, "files": files}))
    shutil.rmtree(out, ignore_errors=True)
    stage.rename(out)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pairing", type=Path, default=PAIRING)
    parser.add_argument("--jobs", type=int, default=8, help="parallel single-thread fits (CPU only)")
    args = parser.parse_args(argv)

    shots, context, matches, frames, identity = read_inputs(args.pairing)
    if (args.out / "complete.json").exists():
        record = verify_complete(args.out, identity)
        print(f"Verified {args.out}; nothing rebuilt\n{json_text(record['summary'])}")
        return
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and holds no completion record; refusing to replace it. Remove it or use another --out")
    try:
        result = build(shots, context, matches, frames, args.jobs)
    except cmp.K1Failed as exc:
        failed = args.out.parent / (args.out.name + "_k1_failed.json")
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(json_text(exc.report))
        sys.exit(f"STOPPED, nothing published: kill rule K1 fired. Report: {failed}")
    print(json_text(publish(args.out, result, identity)))


if __name__ == "__main__":
    main()
