"""Information ladder (H1) and H2 on an evaluation table (xg-vendor ticket 08).

    ./run.sh python scripts/vendor_ladder.py --table-dir DIR --margin DELTA --out DIR [--table-name table] [--prereg FILE]
        [--n-boot 5000] [--fold-seed 0] [--boot-seed 0] [--split-groups] [--prototype DIR]

Opens the table through `xgpoc.vendor.evaltable.load_table` (validated; a confirmatory table needs `--prereg` and at least
`ladder.MIN_CONFIRMATORY_BOOT` draws), runs `xgpoc.vendor.ladder.analyse` on the primary and the combined population
(the combined set is skipped when it holds the same rows as the primary one), and writes:
    metrics.csv              every rung x variant: n, goals, matches, metrics with 95% match-bootstrap intervals
    contrasts.csv            every rung pair of one variant x (log-loss, Brier, AUC): difference, intervals, superiority,
                             TOST at the margin and the smallest equivalence margin (log-loss)
    equivalence_curves.csv   TOST p-value per margin (0-0.01 log-loss) for every pair
    h2.json                  H2 and frame+vendor per population set, exclusion counts, parameters
    coefficients.json        per-fold Platt and stack coefficients;  folds.csv  match -> fold per set
    prototype_comparison_*.csv   with --prototype (development only)
    complete.json            completion record (table hash and kind, parameters, source hashes, files)
Every output carries the table kind; development numbers are not results. `--split-groups` adds the prototype's
populations (primary test matches, primary train/valid matches) for the development comparison.
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import code_version, sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import ladder_dev
from xgpoc.vendor.draws import table_draws

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/ladder.py", ROOT / "xgpoc/vendor/ladder_dev.py", ROOT / "xgpoc/vendor/evaltable.py",
                ROOT / "xgpoc/vendor/draws.py", ROOT / "xgpoc/probe/readability.py", ROOT / "xgpoc/eval/calibration.py")
SPLIT_GROUPS = {"test": ("test",), "trainvalid": ("train", "valid")}


def population_sets(table: pd.DataFrame, split_groups: bool) -> tuple[list[tuple[str, tuple, tuple]], list[str]]:
    """(name, populations, splits) of each set to analyse, and notes on skipped sets."""
    sets, notes = [("primary", ld.POPULATION_SETS["primary"], et.SPLITS)], []
    primary_rows = ld.analysis_population(table, ld.POPULATION_SETS["primary"])[0]
    combined_rows = ld.analysis_population(table, ld.POPULATION_SETS["combined"])[0]
    if len(combined_rows) == len(primary_rows):
        notes.append("combined population skipped: the table holds no recovered-goal or StatsBomb-only rows that enter the analysis")
    else:
        sets.append(("combined", ld.POPULATION_SETS["combined"], et.SPLITS))
    if split_groups:
        sets += [(name, ld.POPULATION_SETS["primary"], splits) for name, splits in SPLIT_GROUPS.items()]
    return sets, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--table-name", default="table")
    parser.add_argument("--prereg", type=Path, help="pre-registration record (required for a confirmatory table)")
    parser.add_argument("--margin", type=float, required=True, help="H2 equivalence margin in log-loss (pre-registered)")
    parser.add_argument("--n-boot", type=int, default=ld.MIN_CONFIRMATORY_BOOT)
    parser.add_argument("--fold-seed", type=int, default=ld.FOLD_SEED)
    parser.add_argument("--boot-seed", type=int, default=ld.BOOT_SEED)
    parser.add_argument("--split-groups", action="store_true", help="also analyse the primary test and train/valid matches separately")
    parser.add_argument("--prototype", type=Path, help="prototype folder with ladder_metrics.csv / ladder_pairs.csv (development only)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty; delete it to rerun")

    prereg = et.load_preregistration(args.prereg) if args.prereg else None
    table, meta = et.load_table(args.table_dir, args.table_name, prereg)
    kind = meta["table_kind"]
    if meta["labels_state"] != "joined":
        sys.exit("the table has no labels")
    if kind == "confirmatory" and (args.n_boot < ld.MIN_CONFIRMATORY_BOOT or args.prototype or args.split_groups):
        sys.exit(f"a confirmatory run needs at least {ld.MIN_CONFIRMATORY_BOOT} draws and no development options")

    sets, notes = population_sets(table, args.split_groups)
    draws = table_draws(table, args.n_boot, args.boot_seed)   # shared by every set and contrast (and tickets 09-15)
    results = {}
    for name, populations, splits in sets:
        print(f"{name}: ladder with {args.n_boot} draws", flush=True)
        results[name] = ld.analyse(table, name, populations, args.margin, draws, splits, args.fold_seed)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = {"table_kind": kind}
    metrics = pd.concat([r.metrics for r in results.values()]).assign(**tag)
    contrasts = pd.concat([r.contrasts for r in results.values()]).assign(**tag)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)
    pd.concat([r.curves for r in results.values()]).assign(**tag).to_csv(args.out / "equivalence_curves.csv", index=False)
    pd.concat([r.folds.assign(population=n) for n, r in results.items()]).to_csv(args.out / "folds.csv", index=False)
    h2 = {"table_kind": kind, "confirmatory_population": "primary", "notes": notes,
          "sets": {n: {**r.summary, "exclusions": r.exclusions} for n, r in results.items()}}
    (args.out / "h2.json").write_text(json.dumps(h2, indent=2, allow_nan=False) + "\n")
    (args.out / "coefficients.json").write_text(json.dumps({"table_kind": kind, **{n: r.coefficients for n, r in results.items()}}, indent=1) + "\n")

    comparison = {}
    if args.prototype:
        proto_metrics = pd.read_csv(args.prototype / "ladder_metrics.csv")
        proto_pairs = pd.read_csv(args.prototype / "ladder_pairs.csv")
        by_metric = ladder_dev.compare_metrics(metrics, proto_metrics)
        by_contrast = ladder_dev.compare_contrasts(contrasts, proto_pairs)
        by_metric.to_csv(args.out / "prototype_comparison_metrics.csv", index=False)
        by_contrast.to_csv(args.out / "prototype_comparison_contrasts.csv", index=False)
        comparison = {"inputs": {p.name: sha256_file(p) for p in (args.prototype / "ladder_metrics.csv", args.prototype / "ladder_pairs.csv")},
                      "metric_cells": len(by_metric), "metric_cells_within_noise": int(by_metric.within_noise.sum()),
                      "contrast_cells": len(by_contrast), "contrast_cells_within_noise": int(by_contrast.within_noise.sum()),
                      "max_abs_diff_in_se": {"metrics": float(by_metric.max_abs_diff_in_se.max()),
                                             "contrasts": float(by_contrast.max_abs_diff_in_se.max())}}
        print(f"prototype comparison: {comparison}")

    primary = results["primary"].summary["h2"]
    print(f"H2 ({kind}): {primary['diff']:+.4f} [{primary['lo95']:+.4f}, {primary['hi95']:+.4f}], TOST p at {args.margin} = {primary['p_tost']:.4f}, "
          f"superiority p = {primary['p_superiority']:.4f}, smallest margin {primary['smallest_equivalence_margin']}")
    names = sorted(p.name for p in args.out.iterdir() if p.is_file())
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "identity": {"table_kind": kind, "table": {"dir": str(args.table_dir), "name": args.table_name, "sha256": meta["table_sha256"],
                                                         "label_join": meta["label_join"]},
                           "prereg_sha256": prereg.sha256 if prereg else None,
                           "parameters": {"margin": args.margin, "n_boot": args.n_boot, "fold_seed": args.fold_seed, "boot_seed": args.boot_seed,
                                          "n_folds": ld.N_FOLDS, "alpha_per_side": ld.ALPHA, "sets": [s[0] for s in sets]},
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": {"h2": {n: r.summary["h2"] for n, r in results.items()}, "notes": notes, "prototype_comparison": comparison},
              "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
              "files": {name: sha256_file(args.out / name) for name in names}}
    (args.out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
