"""The A/B/C ablation of xg-dino (tickets 01 and 01b): label-free controls, protocol, permuted-label dry run and the
real run. CPU only.

    ./run.sh --no-sync python scripts/xgdino_ablation.py --controls --replicates 2000 --start 0 --stop 500 --jobs 12
        ... --start 500 --stop 1000, ... (chunks; the chunk that completes [0, 2000) writes the summary)
                                                                       # results/xgdino_ablation/controls_r2000
    ./run.sh --no-sync python scripts/xgdino_ablation.py --protocol   # research/xgdino_ablation_protocol.json
    ./run.sh --no-sync python scripts/xgdino_ablation.py --dryrun     # results/xgdino_ablation/dryrun
    ./run.sh --no-sync python scripts/xgdino_ablation.py --real --check-only   # every gate of --real, no label read
    ./run.sh --no-sync python scripts/xgdino_ablation.py --real       # results/xgdino_ablation/real

--controls  label-free, no protocol needed (the protocol embeds their result): the simulations of `xgpoc.dino.controls`
            (clustered null and image-specific plant at oracle C-B -0.002 under the clustered generator) on the
            primary designs, replicates [start, stop) of each scenario into parts/rows_<start>_<stop>.csv; the chunk
            that completes [0, replicates) writes controls.json, replicates.csv and complete.json last (with the
            package digest). Chunks must not overlap; a chunk refuses a folder written under other code.
--protocol  label-free: refuses unless the controls' complete.json matches its files and the current code and every
            gate passed; writes the protocol: the frozen part (`ablation.protocol`, its validation section from the
            controls), the sha256 of every input, the label-free counts, the import closure's hashes
            (`xgpoc.dino.records.package_hashes`) and the git state. Refuses an existing file. Prints its sha256.
--dryrun    refuses unless the protocol file's frozen part, inputs and package digest equal the current ones; reads the
            index `goal` inside the held-out label seal only to permute it within matches (seed 0), runs the whole
            read-out on the permuted labels and writes summary.json, probabilities_<block>.csv (no outcome),
            equivalence_curves.csv, reliability.csv and complete.json last. A diagnostic, not the validation.
--real      refuses unless PROTOCOL_SHA256 is pinned and matches the file, the frozen part (with the controls record it
            names), inputs and package digest match, and the dry run's complete.json was written under the same
            protocol; then the read-out on the real labels. `--check-only` stops after every gate, before any label.
--controls-finish  assembles a controls folder whose chunks were written under earlier code (its package_digest.txt):
            refused unless every chunk is present and replicates 0 and replicates - 1 of both scenarios, refitted with
            the current code, equal the stored rows; the record keeps the earlier digest and the replay.
Every output folder must be new (a controls folder may be continued by its chunks).
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
CONTROLS = OUT / "controls_r2000"
PERMUTATION_SEED = 0
PROTOCOL_SHA256: str | None = "fd142d152a1bc546b023a5f3e3e199888b967e5b2296393c12ca40a4bf75b6b0"


def json_equal(a, b) -> bool:
    return json.loads(json.dumps(a, default=float)) == json.loads(json.dumps(b, default=float))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def controls_record(folder: Path, digest: str) -> tuple[dict, str]:
    """The controls' completion record and its sha256, refused unless it matches its files, was written under the
    package digest `digest` and every gate passed."""
    path = folder / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --controls first")
    record = json.loads(path.read_text())
    for name, sha in record["files"].items():
        if not (folder / name).exists() or sha256_file(folder / name) != sha:
            raise SystemExit(f"{folder / name} does not match the controls' completion record")
    if record["code"]["package_digest"] != digest:
        raise SystemExit("the controls ran under other code")
    if not record["controls"]["gates"]["passed"]:
        raise SystemExit(f"a control gate failed: {record['controls']['gates']}; stop for the council")
    return record, sha256_file(path)


def frozen_part(inputs: ab.Inputs, folder: Path, digest: str) -> dict:
    record, sha = controls_record(folder, digest)
    return ab.protocol(inputs, ab.validation(record, sha, str(folder.relative_to(ROOT))))


def checked_protocol(path: Path, inputs: ab.Inputs, pinned: str | None, need_pin: bool) -> tuple[dict, str]:
    """The protocol file, refused unless (with `need_pin`) its sha256 is the pinned one, its package digest is the
    current code's, and its frozen part (rebuilt with the controls record it names), inputs and counts are the
    current ones."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --protocol first")
    digest = sha256_file(path)
    if need_pin and (pinned is None or digest != pinned):
        raise SystemExit("PROTOCOL_SHA256 is not pinned or does not match the protocol file: --real refuses")
    record = json.loads(path.read_text())
    code = records.package_hashes()["digest"]
    if record["code"]["digest"] != code:
        raise SystemExit("the code differs from the code the protocol pins")
    folder = ROOT / record["frozen"]["validation"]["controls"]["path"]
    if not json_equal(record["frozen"], frozen_part(inputs, folder, code)):
        raise SystemExit("the protocol's frozen part differs from xgpoc.dino.ablation.protocol()")
    if not json_equal(record["inputs"], inputs.hashes) or not json_equal(record["counts"], inputs.counts):
        raise SystemExit("the inputs or counts differ from the protocol's")
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


def files(out: Path) -> dict:
    return {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "complete.json"}


def complete(out: Path, name: str, protocol: dict, protocol_sha: str, inputs: ab.Inputs, started: datetime,
             timings: dict, extra: dict) -> None:
    record = {"protocol_sha256": protocol_sha, "pinned": PROTOCOL_SHA256, "mode": name, **extra,
              "inputs": inputs.hashes, "counts": inputs.counts,
              "code": {"package_digest": protocol["code"]["digest"], "script_sha256": sha256_file(Path(__file__))},
              "git": detector_records.git_state(), "timings_s": timings,
              "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
              "files": files(out)}
    write_json(out / "complete.json", record)
    print(json.dumps({"mode": name, "timings_s": timings} | extra, indent=2, default=float))


def run_controls(out: Path, inputs: ab.Inputs, started: datetime, replicates: int, start: int, stop: int,
                 jobs: int) -> None:
    """One chunk of the label-free control simulations (no label is read); the chunk that completes the run writes
    the summary and complete.json."""
    digest = records.package_hashes()["digest"]
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete")
    marker = out / "package_digest.txt"
    if marker.exists() and marker.read_text().strip() != digest:
        raise SystemExit(f"{out} was started under other code")
    parts = out / "parts"
    done = sorted(parts.glob("rows_*.csv")) if parts.exists() else []
    for p in done:
        a, b = map(int, p.stem.split("_")[1:])
        if a < stop and start < b:
            raise SystemExit(f"chunk [{start}, {stop}) overlaps {p.name}")
    if not 0 <= start < stop <= replicates:
        raise SystemExit("need 0 <= start < stop <= replicates")
    clock = time.perf_counter()
    with no_statsbomb_reads():
        prepared = ab.prepare(inputs, primary_arms=dc.ARMS, full=False)
        truth, frame = dc.chunk(prepared, inputs.primary.table, replicates, start, stop, jobs, log=print)
    seconds = time.perf_counter() - clock
    parts.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n")
    frame.to_csv(parts / f"rows_{start}_{stop}.csv", index=False, float_format="%.9g")
    write_json(parts / f"timing_{start}_{stop}.json", {"seconds": round(seconds, 1), "jobs": jobs, "truth": truth,
                                                       "finished": datetime.now(UTC).isoformat(timespec="seconds")})
    print(f"chunk [{start}, {stop}): {seconds:.0f} s with {jobs} workers")
    if len(read_rows(parts)) == 2 * replicates:
        write_controls(out, inputs, started, replicates, digest, truth, None)


def read_rows(parts: Path) -> pd.DataFrame:
    """Every chunk's rows; "null" is a scenario name, not a missing value."""
    frames = [pd.read_csv(p, keep_default_na=False, na_values=["", "nan", "NaN"]) for p in sorted(parts.glob("rows_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_controls(out: Path, inputs: ab.Inputs, started: datetime, replicates: int, digest: str, truth: dict,
                   replay: dict | None) -> None:
    parts = out / "parts"
    rows = read_rows(parts)
    summary = dc.finish(truth, rows.sort_values(["scenario", "replicate"]).reset_index(drop=True))
    write_json(out / "controls.json", summary)
    chunks = [json.loads(p.read_text()) for p in sorted(parts.glob("timing_*.json"))]
    record = {"mode": f"controls_r{replicates}", "controls": summary, "replay_check": replay, "inputs": inputs.hashes,
              "counts": inputs.counts, "code": {"package_digest": digest, "script_sha256": sha256_file(Path(__file__))},
              "git": detector_records.git_state(),
              "timings_s": {"chunks": [{k: c[k] for k in ("seconds", "jobs")} for c in chunks],
                            "total": round(sum(c["seconds"] for c in chunks), 1)},
              "run": {"finished": datetime.now(UTC).isoformat(timespec="seconds"), "last_step_started": started.isoformat(timespec="seconds")},
              "files": files(out)}
    write_json(out / "complete.json", record)
    print(json.dumps({"gates": summary["gates"], "timings_s": record["timings_s"]}, indent=2, default=float))


def finish_controls(out: Path, inputs: ab.Inputs, started: datetime, replicates: int, replay: tuple[int, ...]) -> None:
    """Assemble chunks written under earlier code (the marker's digest): refused unless every chunk is present and the
    replicates `replay` of both scenarios, refitted with the current code, equal the stored rows exactly."""
    parts = out / "parts"
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete")
    rows = read_rows(parts)
    if len(rows) != 2 * replicates:
        raise SystemExit(f"{len(rows)} rows, not {2 * replicates}: run the missing chunks")
    digest = records.package_hashes()["digest"]
    with no_statsbomb_reads():
        prepared = ab.prepare(inputs, primary_arms=dc.ARMS, full=False)
        checks = []
        for i in replay:
            truth, fresh = dc.chunk(prepared, inputs.primary.table, replicates, i, i + 1)
            for name in dc.SCENARIOS:
                stored = rows[(rows.scenario == name) & (rows.replicate == i)].drop(columns="scenario").reset_index(drop=True)
                again = fresh[fresh.scenario == name].drop(columns="scenario").reset_index(drop=True)
                stored = stored[again.columns].astype(again.dtypes.to_dict())
                equal = bool(len(stored) == 1 and all(
                    (pd.isna(a) and pd.isna(b)) or (a == b if not isinstance(a, float) else abs(a - b) <= 1e-8 * max(1e-12, abs(b)))
                    for a, b in zip(stored.iloc[0], again.iloc[0], strict=True)))
                checks.append({"scenario": name, "replicate": i, "equal": equal})
    if not all(c["equal"] for c in checks):
        raise SystemExit(f"the replay differs from the stored rows: {checks}")
    marker = (out / "package_digest.txt").read_text().strip()
    write_controls(out, inputs, started, replicates, digest, truth,
                   {"chunks_written_under": marker, "replayed": checks,
                    "tolerance": "relative 1e-8 (the CSV keeps 9 significant digits)"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", action="store_true")
    mode.add_argument("--dryrun", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--controls-finish", action="store_true",
                      help="assemble chunks written under earlier code after a replay check")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--replicates", type=int, default=2000, help="replicates per scenario (--controls)")
    parser.add_argument("--start", type=int, default=0, help="first replicate of this chunk (--controls)")
    parser.add_argument("--stop", type=int, default=None, help="end of this chunk (--controls; default all)")
    parser.add_argument("--jobs", type=int, default=8, help="worker processes (--controls)")
    parser.add_argument("--check-only", action="store_true", help="--real: every gate, then stop before any label")
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL, help="the protocol --dryrun / --real check")
    parser.add_argument("--controls-dir", type=Path, default=CONTROLS, help="the controls --protocol embeds")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dryrun-dir", type=Path, default=OUT / "dryrun", help="the dry run --real checks")
    args = parser.parse_args()
    started, clock = datetime.now(UTC), time.perf_counter()
    if args.controls or args.controls_finish:
        out = args.out or OUT / f"controls_r{args.replicates}"
        if args.controls_finish:
            finish_controls(out, ab.load_inputs(), started, args.replicates, (0, args.replicates - 1))
        else:
            run_controls(out, ab.load_inputs(), started, args.replicates, args.start, args.stop or args.replicates, args.jobs)
        return
    if args.protocol:
        out = args.out or PROTOCOL
        if out.exists():
            raise SystemExit(f"{out} exists; the protocol is written once")
        inputs = ab.load_inputs()
        code = records.package_hashes()
        record = {"purpose": "xg-dino tickets 01 and 01b: the frozen A/B/C ablation protocol, written before any real fit",
                  "frozen": frozen_part(inputs, args.controls_dir, code["digest"]), "inputs": inputs.hashes,
                  "counts": inputs.counts, "code": code, "git": detector_records.git_state(),
                  "written": started.isoformat(timespec="seconds")}
        out.write_text(json.dumps(record, indent=2, allow_nan=False, default=float) + "\n")
        print(f"inputs {time.perf_counter() - clock:.0f} s; protocol sha256 {sha256_file(out)}")
        return

    name = "dryrun" if args.dryrun else "real"
    out = args.out or OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder")
    inputs = ab.load_inputs()
    t_inputs = time.perf_counter() - clock
    protocol, protocol_sha = checked_protocol(args.protocol_file, inputs, PROTOCOL_SHA256, need_pin=args.real)
    if args.real:
        check_dryrun(args.dryrun_dir, protocol_sha)
        if args.check_only:
            print(f"--real accepts: protocol {protocol_sha} pinned, code, inputs, controls and dry run match; no label read")
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


if __name__ == "__main__":
    main()
