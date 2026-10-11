"""Freeze the SoccerNet <-> StatsBomb pairing (xg-vendor ticket 02) and write `results/xgv_pairing/`.

Loads the index v1.2, the Labels-v2 files, the attempts table (only for Goal labels without a shot annotation) and
the verified StatsBomb data version, runs the pure `xgpoc.vendor.pairing.pair_sources`, enforces kill rule K0 (at least
85% of the index shots paired and 99% goal agreement) and publishes:

    match_map.csv          every SoccerNet match with its StatsBomb match or the reason it has none
    pairs.parquet          every index shot of a paired match: paired or not, residual, StatsBomb partner fields
    half_offsets.csv       per (game, half) half offset SoccerNet - StatsBomb; ticket 04 and ticket 12 read it
    recovered_goals.csv    every dropped SoccerNet Goal label of a paired match with the StatsBomb goal it took
    sb_unpaired.parquet    every unpaired StatsBomb shot of a paired match: population and reason
    secondary.parquet      the secondary population (the rows of sb_unpaired with population "secondary")
    unpaired_sn.parquet    the unpaired index shots with the nearest StatsBomb shot (ticket 04)
    cut_targets.parquet    where each secondary shot and recovered goal sits in its SoccerNet half (ticket 12)
    halves_to_cut.csv      the (game, half) SoccerNet halves that hold such targets, with counts and time range
    pairing_report.json    counts, window sensitivity, disagreement lists with reasons, K0
    complete.json          input hashes (index, attempts, Labels-v2, StatsBomb completion record), source hashes, the
                           code commit and the sha256 of every file above

If K0 fails, the report goes to `pairing_report_k0_failed.json` and no data is published. A finished folder is
verified instead of rebuilt; changed inputs or sources are an error (use another --out).

    ./run.sh python scripts/vendor_pair.py [--out DIR]
"""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow
from xgpoc.bundle import code_version
from xgpoc.paths import INDEX, LABELS_ROOT, SOCCERNET, STATSBOMB
from xgpoc.vendor import pairing, pairing_inputs, pairing_report
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
VERSION = "xgv_pairing_v1"
OUT = ROOT / "results/xgv_pairing"
ATTEMPTS = SOCCERNET / "attempts_all.parquet"
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/pairing.py", ROOT / "xgpoc/vendor/pairing_report.py", ROOT / "xgpoc/vendor/match_pairing.py",
                ROOT / "xgpoc/vendor/shot_pairing.py", ROOT / "xgpoc/vendor/pairing_inputs.py")
# The expected result on the real data (the prototype's): a different number is reported in the summary, not hidden.
PROTOTYPE_COUNTS = {"matches": 135, "pairs": 2741, "index_shots": 3039, "recovered_goals": 50}


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def tables_of(result: pairing.Pairing) -> dict[str, pd.DataFrame]:
    return {"match_map.csv": result.match_map, "pairs.parquet": result.pairs, "half_offsets.csv": result.half_offsets,
            "recovered_goals.csv": result.recovered_goals, "sb_unpaired.parquet": result.sb_unpaired,
            "secondary.parquet": result.secondary, "unpaired_sn.parquet": result.unpaired_sn,
            "cut_targets.parquet": result.cut_targets, "halves_to_cut.csv": pairing.halves_to_cut(result.cut_targets)}


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False, compression="zstd")


def identity_of(statsbomb: dict, labels_digest: str) -> dict:
    return {"version": VERSION,
            "inputs": {"index": {"path": str(INDEX), "sha256": file_sha256(INDEX)},
                       "attempts": {"path": str(ATTEMPTS), "sha256": file_sha256(ATTEMPTS)},
                       "labels_v2": {"root": str(LABELS_ROOT), "sha256": labels_digest},
                       "statsbomb": statsbomb},
            "sources": {str(p): file_sha256(p) for p in SOURCE_FILES}}


def verify_complete(out: Path, identity: dict) -> dict:
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


def publish(out: Path, result: pairing.Pairing, identity: dict) -> dict:
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name, frame in tables_of(result).items():
        write_table(frame, stage / name)
    (stage / "pairing_report.json").write_text(pairing_report.report_text(result.report))
    files = {p.name: file_sha256(p) for p in sorted(stage.iterdir())}
    if {str(p): file_sha256(p) for p in SOURCE_FILES} != identity["sources"]:
        raise ValueError("A source changed during the run")
    shot = result.report["shot_pairing"]
    summary = {"matches": result.report["match_pairing"]["paired"], "index_shots": shot["index_shots"], "pairs": shot["paired"],
               "paired_rate": shot["paired_rate"], "goal_agreement": shot["goal_agreement"],
               "recovered_goals": result.report["sb_side"]["recovery"]["recovered"],
               "secondary_shots": len(result.secondary), "halves_to_cut": result.report["sb_side"]["cut_targets"]["halves"]}
    summary["differs_from_prototype"] = {k: [summary[k], v] for k, v in PROTOTYPE_COUNTS.items() if summary[k] != v}
    runtime = {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    (stage / "complete.json").write_text(json_text({"identity": identity, "code": code_version(ROOT), "summary": summary,
                                                    "runtime": runtime, "files": files}))
    shutil.rmtree(out, ignore_errors=True)
    stage.rename(out)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    tables, statsbomb = pairing_inputs.read_statsbomb(STATSBOMB)
    labels, labels_digest = pairing_inputs.read_labels(LABELS_ROOT)
    identity = identity_of(statsbomb, labels_digest)
    if (args.out / "complete.json").exists():
        record = verify_complete(args.out, identity)
        print(f"Verified {args.out}; nothing rebuilt\n{json_text(record['summary'])}")
        return
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and holds no completion record; refusing to replace it. Remove it or use another --out")
    result = pairing.pair_sources(pd.read_parquet(INDEX), labels, pd.read_parquet(ATTEMPTS), tables)
    try:
        pairing_report.check_k0(result.report)
    except pairing_report.K0Failed as exc:
        failed = args.out.parent / (args.out.name + "_k0_failed.json")
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(pairing_report.report_text(result.report))
        sys.exit(f"STOPPED, nothing published. {exc}. Report: {failed}")
    summary = publish(args.out, result, identity)
    print(json_text(summary))


if __name__ == "__main__":
    main()
