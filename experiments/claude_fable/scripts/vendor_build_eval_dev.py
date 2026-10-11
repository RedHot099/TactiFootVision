"""Build the DEVELOPMENT evaluation table of the vendor comparison (xg-vendor ticket 07).

Joins the prototype's per-shot rungs (`ladder_shots.parquet`, `pairs.parquet`) with the archived per-shot G-combo
scores and embeddings (`results/xgc_offset_scores`), writes the table without labels, joins the labels in the
logged step (`xgpoc.vendor.evaltable.join_labels`), validates both, checks that the table reproduces the prototype's
ladder (`ladder_metrics.csv`) and writes a completion record. Development data only: nothing here is a result.

    ./run.sh python scripts/vendor_build_eval_dev.py [--prototype DIR] [--archive DIR] [--out DIR]
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
from xgpoc.bundle import code_version, sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import evaltable_dev as dev

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
PROTOTYPE = Path("/home/kuba/projects/tactifoot_vision/.scratch/xg-vendor/prototype")
ARCHIVE = ROOT / "results/xgc_offset_scores"
OUT = ROOT / "results/xgv_eval_dev"
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/evaltable.py", ROOT / "xgpoc/vendor/evaltable_dev.py")
LABEL_FREE, JOINED, LOG = "table_labels_absent", "table", "label_join_log.jsonl"
TOLERANCE = 1e-6   # reproduction of the prototype's ladder: the CSV holds full floats, so this is far above rounding
EMBEDDINGS = {"emb_g_fold0": ("embeddings_g_combo.npy", 0), "emb_g_fold1": ("embeddings_g_combo.npy", 1), "emb_dinov2": ("embeddings_dinov2_frozen.npy", None)}


def embedding_sets(archive: Path) -> dict:
    """Sidecar description of the archived embedding files; their hashes are checked against the archive's completion record."""
    recorded = json.loads((archive / "complete.json").read_text())["files"]
    sets = {}
    for column, (name, fold) in EMBEDDINGS.items():
        digest = sha256_file(archive / name)
        if digest != recorded[name]:
            raise ValueError(f"{name} does not match the archive's completion record")
        rows, dim = np.load(archive / name, mmap_mode="r").shape
        model = "frozen DINOv2-L, CLS + mean patch token, offset 0" if fold is None else (
            f"archived G-combo, seed-0 model of fold {fold}, offset 0; only train/valid shots of fold {fold} reference it")
        sets[column] = {"file": str(archive / name), "sha256": digest, "dim": int(dim), "rows": int(rows), "model": model}
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prototype", type=Path, default=PROTOTYPE)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        sys.exit(f"{args.out} is complete; delete it to rebuild")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} holds files without a completion record; delete it to rebuild")

    inputs = {name: args.prototype / name for name in ("ladder_shots.parquet", "pairs.parquet", "ladder_metrics.csv")}
    inputs["archive/predictions.parquet"] = args.archive / "predictions.parquet"
    ladder, pairs = pd.read_parquet(inputs["ladder_shots.parquet"]), pd.read_parquet(inputs["pairs.parquet"])
    archive = pd.read_parquet(inputs["archive/predictions.parquet"], columns=["shot_id", "offset_ms", "encoder", "frame_status", "xg_logit_raw",
                                                                              "fold_source", "embedding_ref"])
    sets = embedding_sets(args.archive)
    table, labels = dev.build_development_tables(ladder, pairs, archive, sets)
    sources = {name: sha256_file(path) for name, path in inputs.items()}
    meta = et.write_table(table, args.out, LABEL_FREE, kind="development", embedding_sets=sets, unavailable_columns=("frame_logit_m400",),
                          notes=dev.NOTES, sources=sources)
    joined_meta = et.join_labels(args.out, LABEL_FREE, labels, JOINED, args.out / LOG, now=datetime.now(UTC))
    joined, _ = et.load_table(args.out, JOINED)   # reads and validates

    comparison = dev.prototype_comparison(joined, pd.read_csv(inputs["ladder_metrics.csv"]))
    comparison.to_csv(args.out / "ladder_comparison.csv", index=False)
    worst = comparison.loc[comparison.abs_diff.idxmax()]
    passed = bool(worst.abs_diff <= TOLERANCE)
    print(comparison[comparison.metric.eq("log_loss")].round(4).to_string(index=False))
    print(f"max |table - prototype| over {len(comparison)} cells: {worst.abs_diff:.2e} ({worst.population}/{worst.rung}/{worst.metric})")
    if not passed:
        sys.exit("the development table does not reproduce the prototype's ladder; no completion record written")

    names = sorted(p.name for p in args.out.iterdir() if p.is_file())
    files = {name: sha256_file(args.out / name) for name in names}
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "identity": {"table_kind": "development", "schema_version": et.SCHEMA_VERSION, "inputs": sources,
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": {"rows": meta["rows"], "by_population_split": meta["by_population_split"],
                          "goals": int(labels.goal_sn.sum()), "label_free_sha256": meta["table_sha256"],
                          "joined_sha256": joined_meta["table_sha256"], "unavailable_columns": meta["unavailable_columns"],
                          "embedding_references": {c: int((table[c] >= 0).sum()) for c in et.EMBEDDING_COLUMNS},
                          "reproduction": {"cells": len(comparison), "max_abs_diff": float(worst.abs_diff), "tolerance": TOLERANCE, "passed": passed}},
              "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
              "files": files}
    (args.out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
