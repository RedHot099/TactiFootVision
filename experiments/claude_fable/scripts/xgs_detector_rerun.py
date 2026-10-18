"""xg-stack's read-out re-run with the E3 features of detector-data's retrained detector (development only, CPU).

    ./run.sh --no-sync python scripts/xgs_detector_rerun.py [--e3 cache/xgd_full4/E3] [--out results/xgs_stack_dd2c]

The frozen C6 protocol (`research/xgs_stack_protocol_c6.json`, sha256 pinned in `xgs_run.py`) gives the arms and the
chosen model; its code digest is not required to match, since the extractor gained a detector option after the freeze.
Instead the run first rebuilds the population from the frozen E3 cache (`cache/xgd_full3/E3`) and refuses unless the
read-out reproduces `results/xgs_stack/real/summary.json` exactly (scores, deltas, chosen C); then it swaps only the
E3 cache (`--e3`) and repeats the same read-out (`analysis.analyse`, same folds, bootstrap weights and spec). Population
and outcomes are xg-stack's: 7,062 non-penalty shots of the 328 training matches, labels inside the held-out seal. No
StatsBomb read; the 172 held-out matches are not touched (they are spent for detector-input variants).

Writes <out>/summary.json, probabilities.csv, categories.csv (as `xgs_run.py --real`), paired.json (the new arms
against the frozen run's arms, shot-level log-loss differences, match bootstrap) and complete.json, last.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgs_run import OUT, PROTOCOL_SHA256, read_out_files, write_json
from xgpoc.bundle import sha256_file
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import analysis as sa
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack import records
from xgpoc.stack import stats as ss

ROOT = Path(__file__).resolve().parents[1]
FROZEN_REAL = OUT / "real"


def read_out(population: si.Population, y: np.ndarray, frozen: dict) -> dict:
    spec = sf.spec(frozen["design"]["chosen"], population.table)
    if json.loads(json.dumps(pr.design(spec))) != frozen["design"]:
        raise SystemExit("the population's design differs from the protocol's (columns or arms)")
    return sa.analyse(population.table, population.folds, y, ss.bootstrap_weights(population.games), spec)


def summary_of(result: dict) -> dict:
    return json.loads(json.dumps({k: result[k] for k in ("scores", "deltas", "readings", "chosen_c", "gbm")}, default=float))


def shot_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(new: dict, old: dict, y: np.ndarray, games: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Each arm with the new E3 minus the same arm with the frozen E3: mean shot log-loss difference, 95% match bootstrap."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(games)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])      # (n_boot, matches)
    out = {}
    for arm in new:
        d = shot_log_loss(new[arm], y) - shot_log_loss(old[arm], y)
        per_match, n_match = np.bincount(codes, d, len(uniq)), np.bincount(codes, minlength=len(uniq))
        boot = (counts @ per_match) / (counts @ n_match)
        out[arm] = {"delta_ll": float(d.mean()), "lo95": float(np.quantile(boot, 0.025)),
                    "hi95": float(np.quantile(boot, 0.975)), "draws": n_boot}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--e3", type=Path, default=ROOT / "cache/xgd_full4/E3")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgs_stack_dd2c")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    frozen = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)

    with no_statsbomb_reads():
        reference = si.build_population()
        y = si.load_outcomes(reference).aligned(reference)
        again = read_out(reference, y, frozen)
        if summary_of(again) != json.loads((FROZEN_REAL / "summary.json").read_text()):
            raise SystemExit("the frozen E3 read-out does not reproduce results/xgs_stack/real/summary.json")
        print("frozen read-out reproduced", flush=True)

        population = si.build_population(e3=args.e3 if args.e3.is_absolute() else ROOT / args.e3)
        if not population.table.index.equals(reference.table.index) or (population.folds != reference.folds).any():
            raise SystemExit("the new E3 population differs in shots or folds")
        result = read_out(population, y, frozen)
        args.out.mkdir(parents=True)
        read_out_files(args.out, population, result)
        pairs = paired(result["probabilities"], again["probabilities"], y, population.games)
        write_json(args.out / "paired.json", pairs)

    complete = {"protocol_sha256": PROTOCOL_SHA256, "mode": "detector re-run (development)",
                "reference": {"summary": str(FROZEN_REAL / "summary.json"),
                              "sha256": sha256_file(FROZEN_REAL / "summary.json"), "reproduced": True},
                "readings": result["readings"], "inputs": population.inputs, "counts": population.counts,
                "reference_counts": reference.counts, "code": records.code_hashes(Path(__file__)),
                "git": detector_records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": {str(p.relative_to(args.out)): sha256_file(p) for p in sorted(args.out.rglob("*")) if p.is_file()}}
    write_json(args.out / "complete.json", complete)
    print(json.dumps({"deltas_vs_R": result["deltas"], "new_minus_frozen": pairs, "readings": result["readings"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
