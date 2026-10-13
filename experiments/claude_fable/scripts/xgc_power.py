"""Expected power of the frozen xg-compare design (xg-compare ticket 01; CPU, about 4 minutes with 8 workers). Label-free.

    ./run.sh python scripts/xgc_power.py [--table results/xgc_compare/table] [--out results/xgc_compare/power]

The design is `xgpoc.compare.protocol` and the simulation is `xgpoc.compare.power`, whose docstring gives the basis,
the permutation, the planted effects and the scenarios. Inputs are checked against their completion records: the shot
table (no label file is opened) and the paper's frozen recalibration of the primary population
(`results/xgv_confirmatory/ladder/coefficients.json`, `folds.csv`). Writes family.parquet, scan.parquet,
scan_levels.parquet (per study and scenario), power.json (the summary that goes into the protocol) and complete.json,
last.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgpoc.compare import power as pw
from xgpoc.compare import protocol as pr
from xgpoc.compare import records as xr
from xgpoc.compare import table as tb
from xgpoc.detector import records

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "results/xgv_confirmatory/ladder"
_LIMITS = []


def single_thread() -> None:
    """Worker initialiser: one BLAS thread per worker process (the workers share the machine's cores)."""
    _LIMITS.append(threadpool_limits(1))


def verified(path: Path) -> str:
    try:
        return records.verified_file(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=ROOT / "results/xgc_compare/table")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgc_compare/power")
    parser.add_argument("--sims", type=int, default=pw.N_SIMS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started = datetime.now(UTC)

    inputs = {"table": {"path": str(args.table / tb.TABLE_FILE), "sha256": verified(args.table / tb.TABLE_FILE)},
              "coefficients": {"path": str(LADDER / "coefficients.json"), "sha256": verified(LADDER / "coefficients.json")},
              "folds": {"path": str(LADDER / "folds.csv"), "sha256": verified(LADDER / "folds.csv")}}
    table, _ = tb.read(args.table)
    coefficients = json.loads((LADDER / "coefficients.json").read_text())["primary"]
    folds = pd.read_csv(LADDER / "folds.csv")
    basis = pw.basis_from_table(table, coefficients, folds[folds.population.eq("primary")])

    chunks = [list(c) for c in np.array_split(np.arange(args.sims), max(1, args.workers * 4))]
    with ProcessPoolExecutor(args.workers, initializer=single_thread) as pool:
        parts = list(pool.map(partial(pw.simulate, basis), chunks))
    family, scan, scan_levels = (pd.concat([p[i] for p in parts], ignore_index=True) for i in range(3))
    summary = pw.summarise(family, scan, scan_levels)
    summary["parameters"] = {"n_sims": args.sims, "sim_seed": pw.SIM_SEED, "n_boot": pr.N_BOOT, "boot_seed": pr.BOOT_SEED,
                             "alpha_unadjusted": pw.ALPHA, "truth_for_labels": "sigmoid of the main-scale vendor logit",
                             "scale": "paper's frozen out-of-fold Platt recalibration, held fixed across draws and halves"}

    args.out.mkdir(parents=True, exist_ok=True)
    family.to_parquet(args.out / "family.parquet", index=False)
    scan.to_parquet(args.out / "scan.parquet", index=False)
    scan_levels.to_parquet(args.out / "scan_levels.parquet", index=False)
    (args.out / "power.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    complete = {"inputs": inputs, "code": xr.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                        "workers": args.workers},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps({i: {k: round(v, 3) for k, v in r.items() if k.startswith(("power_bh", "null_raw"))} for i, r in summary["items"].items()}))
    print(json.dumps(summary["scan"]))


if __name__ == "__main__":
    main()
