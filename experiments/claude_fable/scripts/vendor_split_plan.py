"""Write the retraining split plan of the vendor comparison (xg-vendor ticket 05) to `results/xgv_split_plan/`.

Reads the index v1.2 and the frozen match map of ticket 02 (its completion record must list that match map's hash),
builds the pure plan of `xgpoc.vendor.split_plan`, checks the expected role counts and publishes:

    plan.csv       game, league, season, home, official_split, role (evaluation | gate | training), fold
    plan.json      seed, counts, gate allocation, gate tolerance, archived G-combo predictions, input/source hashes
    complete.json  hash of every file above, the code commit

and the fold file of the retrain, `cache/folds_xgv_home5.csv` (game, home, fold of the training matches, the layout of
`cache/folds_v12_home5.csv`). The plan's sha256 is what `xgpoc.vendor.split_check.verify_plan` pins. A finished folder is
verified instead of rebuilt; a different plan is never written over it (remove the folder on purpose).

    ./run.sh python scripts/vendor_split_plan.py [--out DIR] [--seed N]
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from xgpoc.bundle import code_version
from xgpoc.paths import INDEX
from xgpoc.vendor import split_check as sc
from xgpoc.vendor import split_plan as sp
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def entry(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def write_folds(plan: pd.DataFrame, path: Path) -> None:
    """The retrain's fold file; a different file already there is an error, not overwritten."""
    text = sp.training_folds(plan).to_csv(index=False)
    if path.exists() and path.read_text() != text:
        sys.exit(f"{path} exists and differs from this plan's folds; remove it on purpose")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def publish(out: Path, plan: pd.DataFrame, record: dict) -> None:
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    (stage / "plan.csv").write_text(sp.plan_text(plan))
    (stage / "plan.json").write_text(json_text(record))
    files = {p.name: file_sha256(p) for p in sorted(stage.iterdir())}
    if {s: file_sha256(ROOT / s) for s in sc.SOURCES} != record["sources"]:
        raise ValueError("A source changed during the run")
    (stage / "complete.json").write_text(json_text({"code": code_version(ROOT), "summary": record["counts"]["matches"], "files": files}))
    shutil.rmtree(out, ignore_errors=True)
    stage.rename(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=sc.PLAN_DIR)
    parser.add_argument("--seed", type=int, default=sp.SEED)
    args = parser.parse_args(argv)

    pairing = json.loads(sc.PAIRING_COMPLETE.read_text())
    if pairing["files"]["match_map.csv"] != file_sha256(sc.MATCH_MAP):
        sys.exit(f"{sc.MATCH_MAP} is not the match map of the completion record {sc.PAIRING_COMPLETE}")
    index, match_map, archived = pd.read_parquet(INDEX), pd.read_csv(sc.MATCH_MAP), pd.read_csv(sc.ARCHIVED_G_COMBO)
    plan = sp.build_plan(index, match_map, args.seed, sp.N_GATE, sp.N_FOLDS)
    if (counts := plan.role.value_counts().to_dict()) != sc.EXPECTED_COUNTS:
        sys.exit(f"STOPPED, nothing written. Role counts {counts}, expected {sc.EXPECTED_COUNTS}")
    plan_sha = hashlib.sha256(sp.plan_text(plan).encode()).hexdigest()

    if (args.out / "complete.json").exists():
        sc.verify_plan(INDEX, sc.MATCH_MAP, args.out, sc.FOLDS, sc.ARCHIVED_G_COMBO, expected_plan=plan_sha,
                       expected_counts=sc.EXPECTED_COUNTS)
        print(f"Verified {args.out}; plan {plan_sha}; nothing rebuilt")
        return
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and holds no completion record; refusing to replace it. Remove it or use another --out")
    write_folds(plan, sc.FOLDS)
    inputs = {"index": entry(INDEX), "match_map": entry(sc.MATCH_MAP), "pairing_complete": entry(sc.PAIRING_COMPLETE),
              "archived_predictions": entry(sc.ARCHIVED_G_COMBO)}
    record = sc.plan_record(plan, index, archived, seed=args.seed, n_gate=sp.N_GATE, n_folds=sp.N_FOLDS, inputs=inputs,
                            folds=entry(sc.FOLDS), sources={s: file_sha256(ROOT / s) for s in sc.SOURCES})
    publish(args.out, plan, record)
    print(json_text({"plan_sha256": plan_sha, "counts": record["counts"], "gate_allocation": record["gate_allocation"],
                     "archived_on_gate": record["archived_g_combo"]["gate"]}))


if __name__ == "__main__":
    main()
