"""xg-temporal read-out of arms R, S and T (ticket 01 tracer; the real run is ticket 04, after the protocol of ticket
03). Only synthetic outcomes are implemented: y* ~ Bernoulli(sigmoid(z_g_combo)), no label is read. CPU only.

    ./run.sh --no-sync python scripts/xgt_run.py --synthetic --source dinov2 [--start -1600 --o-max -600] [--k 32]
        [--replicates 1] [--seed 0] [--control] [--out results/xgt_synthetic_<source>_<start>_<o_max>]

Steps: verify the plan and the source (`xgpoc.temporal.inputs`), build the population and the window, run the leak
assertions (`xgpoc.temporal.provenance`; the run stops when one fails), read the window embeddings through a memmap,
build the fold-wise designs of R, S and T once (`xgpoc.temporal.features`), then per replicate draw y*, fit every arm
out of fold and read ΔLL T - R, T - S (Holm) and S - R with the match bootstrap (`xgpoc.temporal.readout`).

Writes to --out (a new folder): population.csv (shot_id, game, fold, frames, complete), replicates.csv (one row per
replicate), summary.json (population, columns, provenance, chosen C, ΔLL with intervals, timing and peak memory) and
complete.json (input, code and output hashes, git state), last.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.paths import INDEX
from xgpoc.stack import stats as ss
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import readout as tr
from xgpoc.temporal import records as tr_records

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", action="store_true", help="synthetic outcomes y* ~ Bernoulli(sigmoid(z)); required")
    parser.add_argument("--source", choices=sorted(ti.SOURCES), required=True)
    parser.add_argument("--start", type=int, default=ti.START)
    parser.add_argument("--o-max", type=int, default=ti.O_MAX)
    parser.add_argument("--control", action="store_true", help="a leakage-control row (o_max > -600), never a candidate")
    parser.add_argument("--k", type=int, default=tf.K)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.k < 1 or args.replicates < 1:
        parser.error("--k and --replicates must be at least 1")
    if not args.synthetic:
        parser.error("only --synthetic is implemented: the real read-out is ticket 04, after the frozen protocol")
    if args.out is None:
        suffix = "_control" if args.control else ""
        args.out = ROOT / f"results/xgt_synthetic_{args.source}_{args.start}_{args.o_max}{suffix}"
    return args


def replicate_row(i: int, y: np.ndarray, result: dict) -> dict:
    row = {"replicate": i, "events": int(y.sum())}
    for name, d in result["deltas"].items():
        row |= {f"delta_{name}": d["delta_ll"], f"lo95_{name}": d["lo95"], f"hi95_{name}": d["hi95"],
                f"p_{name}": d["p_one_sided"]}
    for name, r in result["readings"].items():
        row |= {f"holm_{name}": r["holm_rejected"]}
    return row


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started, clock, stages = datetime.now(UTC), time.time(), {}

    def stage(name: str, since: float) -> float:
        stages[name] = round(time.time() - since, 1)
        return time.time()

    window = ti.Window(args.start, args.o_max, args.control)
    plan = a1.verify_plan(INDEX)
    population, files = ti.build_population(ti.SOURCES[args.source], window, plan)
    columns = tf.arm_columns(window, args.k)
    provenance = tp.control(population, columns, plan)
    if not provenance["passed"]:
        raise SystemExit(f"provenance failed: {[c for c in provenance['checks'] if not c['passed']]}")
    t = stage("population_and_provenance", clock)
    embeddings, position = ti.load_embeddings(files, population.refs)
    t = stage("embeddings_read", t)
    data = tf.DesignData(window, population.z, population.folds, position, embeddings, args.k)
    designs, pca = tf.fold_designs(data)
    t = stage("designs", t)
    weights = ss.bootstrap_weights(population.games)
    rng = np.random.default_rng(args.seed)
    rows, results, chosen = [], [], {}
    for i in range(args.replicates):
        y = tr.synthetic_outcomes(population.z, rng)
        probabilities, chosen[str(i)] = tr.out_of_fold(designs, y)
        result = tr.contrasts(y, probabilities, weights)
        rows.append(replicate_row(i, y, result))
        results.append(result)
        print(f"replicate {i}: " + ", ".join(f"{n} {d['delta_ll']:+.5f} [{d['lo95']:+.5f}; {d['hi95']:+.5f}]"
                                              for n, d in result["deltas"].items()), flush=True)
    stage("fits", t)
    args.out.mkdir(parents=True)
    population.table.reset_index()[["shot_id", "game", "fold", "frames", "complete"]].to_csv(args.out / "population.csv",
                                                                                             index=False)
    pd.DataFrame(rows).to_csv(args.out / "replicates.csv", index=False, float_format="%.6g")
    summary = {"label": "synthetic tracer: y* ~ Bernoulli(sigmoid(z_g_combo)); no label read",
               "source": args.source, "window": {"start": window.start, "o_max": window.o_max, "control": window.control,
                                                 "offsets": list(window.offsets), "middle": window.middle},
               "k": args.k, "seed": args.seed, "replicates": args.replicates,
               "population": population.counts, "population_csv_sha256": sha256_file(args.out / "population.csv"),
               "columns": {arm: {"n": len(c), "names": c} for arm, c in columns.items()},
               "provenance": provenance, "pca": pca, "chosen_c": chosen,
               "first_replicate": {k: results[0][k] for k in ("deltas", "readings", "scores")},
               "mean_delta": {n: float(np.mean([r[f"delta_{n}"] for r in rows])) for n in results[0]["deltas"]},
               "holm_rejection_rate": {n: float(np.mean([r[f"holm_{n}"] for r in rows])) for n in tr.CONFIRMATORY},
               "run": {"stages_seconds": stages, "seconds": round(time.time() - clock, 1),
                       "peak_rss_gib": round(tr_records.peak_rss_gib(), 2), "device": "cpu"}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=tr_records.json_default) + "\n")
    complete = {"inputs": population.inputs, "code": tr_records.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"),
                        "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1) + "\n")
    print(json.dumps({"population": {k: population.counts[k] for k in ("shots", "matches", "complete_shots")},
                      "mean_delta": summary["mean_delta"], "holm_rejection_rate": summary["holm_rejection_rate"],
                      "run": summary["run"]}, indent=1))


if __name__ == "__main__":
    main()
