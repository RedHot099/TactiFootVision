"""The A/B/C ablation of xg-dino (ticket 01): protocol, permuted-label dry run and the real run. CPU only.

    ./run.sh --no-sync python scripts/xgdino_ablation.py --protocol   # research/xgdino_ablation_protocol.json
    ./run.sh --no-sync python scripts/xgdino_ablation.py --dryrun     # results/xgdino_ablation/dryrun
    ./run.sh --no-sync python scripts/xgdino_ablation.py --controls --replicates 40 --jobs 8
                                                                       # results/xgdino_ablation/controls_r40
    ./run.sh --no-sync python scripts/xgdino_ablation.py --real       # results/xgdino_ablation/real (needs the pin)

--protocol  label-free: builds the inputs (`xgpoc.dino.ablation.load_inputs`) and writes the protocol: the frozen part
            (`ablation.protocol`), the sha256 of every input, the label-free counts, the import closure's hashes
            (`xgpoc.dino.records.package_hashes`) and the git state. Refuses an existing file. Prints its sha256; the
            pin below is set only when the orchestrator says so.
--dryrun    refuses unless the protocol file's frozen part, inputs and package digest equal the current ones; reads the
            index `goal` inside the held-out label seal only to permute it within matches (seed 0), runs the whole
            read-out on the permuted labels and writes summary.json, probabilities_<block>.csv (no outcome),
            equivalence_curves.csv, reliability.csv and complete.json last. A diagnostic, not the validation.
--controls  refuses like --dryrun; no label is read: the label-free simulations of `xgpoc.dino.controls` (clustered
            null and image-specific plant at C-B -0.002) on the primary designs; writes controls.json, replicates.csv
            and complete.json last.
--real      refuses unless PROTOCOL_SHA256 is pinned and matches the file, the frozen part, inputs and package digest
            match, and the dry run's complete.json was written under the same protocol; then the read-out on the real
            labels. Every output folder must be new.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.compare.stats import permute_within_matches
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.dino import controls as dc
from xgpoc.dino import records
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/xgdino_ablation_protocol.json"
OUT = ROOT / "results/xgdino_ablation"
PERMUTATION_SEED = 0
PROTOCOL_SHA256: str | None = None


def json_equal(a, b) -> bool:
    return json.loads(json.dumps(a, default=float)) == json.loads(json.dumps(b, default=float))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def checked_protocol(path: Path, inputs: ab.Inputs, pinned: str | None, need_pin: bool) -> tuple[dict, str]:
    """The protocol file, refused unless (with `need_pin`) its sha256 is the pinned one, and its frozen part, inputs,
    counts and package digest are the current ones."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --protocol first")
    digest = sha256_file(path)
    if need_pin and (pinned is None or digest != pinned):
        raise SystemExit("PROTOCOL_SHA256 is not pinned or does not match the protocol file: --real refuses")
    record = json.loads(path.read_text())
    if not json_equal(record["frozen"], ab.protocol(inputs)):
        raise SystemExit("the protocol's frozen part differs from xgpoc.dino.ablation.protocol()")
    if not json_equal(record["inputs"], inputs.hashes) or not json_equal(record["counts"], inputs.counts):
        raise SystemExit("the inputs or counts differ from the protocol's")
    if record["code"]["digest"] != records.package_hashes()["digest"]:
        raise SystemExit("the code differs from the code the protocol pins")
    return record, digest


def check_dryrun(folder: Path, protocol_sha256: str) -> None:
    path = folder / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --dryrun under the pinned protocol first")
    record = json.loads(path.read_text())
    if record["protocol_sha256"] != protocol_sha256:
        raise SystemExit("the dry run ran under another protocol")


def write_read_out(out: Path, result: dict) -> None:
    write_json(out / "summary.json", result["summary"])
    for name, frame in result["probabilities"].items():
        frame.to_csv(out / f"probabilities_{name.replace(':', '_')}.csv", index=False, float_format="%.9g")
    result["curves"].to_csv(out / "equivalence_curves.csv", index=False, float_format="%.6g")
    result["reliability"].to_csv(out / "reliability.csv", index=False, float_format="%.9g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", action="store_true")
    mode.add_argument("--dryrun", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--replicates", type=int, default=40, help="replicates per scenario (--controls)")
    parser.add_argument("--jobs", type=int, default=8, help="worker processes (--controls)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dryrun-dir", type=Path, default=OUT / "dryrun", help="the dry run --real checks")
    args = parser.parse_args()
    started, clock = datetime.now(UTC), time.perf_counter()
    if args.protocol:
        out = args.out or PROTOCOL
        if out.exists():
            raise SystemExit(f"{out} exists; the protocol is written once")
        inputs = ab.load_inputs()
        record = {"purpose": "xg-dino ticket 01: the frozen A/B/C ablation protocol, written before any real fit",
                  "frozen": ab.protocol(inputs), "inputs": inputs.hashes, "counts": inputs.counts,
                  "code": records.package_hashes(), "git": detector_records.git_state(),
                  "written": started.isoformat(timespec="seconds")}
        out.write_text(json.dumps(record, indent=2, allow_nan=False, default=float) + "\n")
        print(f"inputs {time.perf_counter() - clock:.0f} s; protocol sha256 {sha256_file(out)}")
        return

    name = "dryrun" if args.dryrun else "real" if args.real else f"controls_r{args.replicates}"
    out = args.out or OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder")
    inputs = ab.load_inputs()
    t_inputs = time.perf_counter() - clock
    protocol, protocol_sha = checked_protocol(PROTOCOL, inputs, PROTOCOL_SHA256, need_pin=args.real)
    if args.real:
        check_dryrun(args.dryrun_dir, protocol_sha)
    if args.controls:
        run_controls(out, inputs, protocol, protocol_sha, started, t_inputs, args.replicates, args.jobs)
        return
    with no_statsbomb_reads():
        y = si.load_outcomes(inputs.primary).aligned(inputs.primary)
    goals = {"primary": int(y.sum()), "no_v3": int(y[inputs.subsets["no_v3"]].sum())}
    if goals != ab.EXPECTED_GOALS:
        raise SystemExit(f"goal counts {goals} are not the spec's {ab.EXPECTED_GOALS}")
    if args.dryrun:
        y = permute_within_matches(pd.DataFrame({"goal": y}), inputs.primary.games, PERMUTATION_SEED).goal.to_numpy()
    clock = time.perf_counter()
    prepared = ab.prepare(inputs)
    t_prepare = time.perf_counter() - clock
    clock = time.perf_counter()
    result = ab.read_out(prepared, y, log=print)
    t_read = time.perf_counter() - clock
    out.mkdir(parents=True)
    write_read_out(out, result)
    extra = {"dry_run": ab.dry_run_check(result["summary"]), "permutation_seed": PERMUTATION_SEED} if args.dryrun else {}
    extra |= {"decision": result["summary"]["decision"], "leak": result["summary"].get("leak")}
    timings = {"inputs": round(t_inputs, 1), "designs": round(t_prepare, 1), "read_out": round(t_read, 1)}
    complete(out, name, protocol, protocol_sha, inputs, started, timings, extra)


def complete(out: Path, name: str, protocol: dict, protocol_sha: str, inputs: ab.Inputs, started: datetime,
             timings: dict, extra: dict) -> None:
    record = {"protocol_sha256": protocol_sha, "pinned": PROTOCOL_SHA256, "mode": name, **extra,
              "inputs": inputs.hashes, "counts": inputs.counts,
              "code": {"package_digest": protocol["code"]["digest"], "script_sha256": sha256_file(Path(__file__))},
              "git": detector_records.git_state(), "timings_s": timings,
              "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
              "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "complete.json", record)
    print(json.dumps({"mode": name, "timings_s": timings} | extra, indent=2, default=float))


def run_controls(out: Path, inputs: ab.Inputs, protocol: dict, protocol_sha: str, started: datetime, t_inputs: float,
                 replicates: int, jobs: int) -> None:
    """The label-free control simulations (no label is read)."""
    clock = time.perf_counter()
    with no_statsbomb_reads():
        prepared = ab.prepare(inputs, primary_arms=dc.ARMS, full=False)
        t_prepare = time.perf_counter() - clock
        clock = time.perf_counter()
        summary, frame = dc.run(prepared, inputs.primary.table, replicates, jobs, log=print)
    t_run = time.perf_counter() - clock
    out.mkdir(parents=True)
    write_json(out / "controls.json", summary)
    frame.to_csv(out / "replicates.csv", index=False, float_format="%.9g")
    timings = {"inputs": round(t_inputs, 1), "designs": round(t_prepare, 1), "simulations": round(t_run, 1),
               "jobs": jobs, "seconds_per_replicate_per_worker": round(t_run * jobs / (2 * replicates), 1)}
    half = summary["scenarios"]["null"]["mean_half_width_90_C-B"]
    complete(out, f"controls_r{replicates}", protocol, protocol_sha, inputs, started, timings,
             {"expected_c_minus_b_half_width_90": half})


if __name__ == "__main__":
    main()
