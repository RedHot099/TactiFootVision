"""Run the timing-source test (xg-vendor ticket 04) and write `results/xgv_timing_source/`.

The rule is `research/xg_vendor_timing_rule.md`; the script refuses to run when its sha256 is not the one pinned in
`xgpoc.vendor.timing_source.RULE_SHA256`. Inputs, each checked before use:

    StatsBomb data version   tables timing and shots and the raw events of the 135 paired matches, against the
                             hashes in its completion record; the open-data commit must be `OPEN_DATA_SHA` (the note's)
    frozen pairing           results/xgv_pairing (ticket 02): its completion record must be `PAIRING_COMPLETE_SHA256`
                             (the run frozen with commit 177d65e), every file must match it, and it must name the same
                             StatsBomb completion record and the same Labels-v2 digest as found here
    Labels-v2                the annotation files (their digest is the pairing's)

Outputs:

    label_matches.parquet           every compared SoccerNet label of an aligned half: StatsBomb partner, residual, exclusion
    shot_residuals.parquet          the paired shots: frozen and leave-one-out residual
    agreement.csv                   agreement distribution per event type (all, by visibility, the rule's substitution set)
    substitution_sensitivity.csv    the substitution statistics under other windows and for "visible" labels only
    unpaired_reasons.csv            every unpaired SoccerNet index shot: nearest StatsBomb event of any type and a reason
                                    (xgpoc.vendor.unpaired_reasons)
    timing_verdict.json             the verdict record (schema xgv-timing-verdict-1, read by ticket 09)
    complete.json                   identity (rule, inputs, sources), code commit, summary and the sha256 of every file

A finished folder is verified instead of rebuilt; changed inputs or sources are an error (use another --out).

    ./run.sh python scripts/vendor_timing_source.py [--out DIR]
"""

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow
from xgpoc.bundle import code_version
from xgpoc.paths import LABELS_ROOT, STATSBOMB
from xgpoc.vendor import pairing_inputs
from xgpoc.vendor import statsbomb_fetch as fetching
from xgpoc.vendor import timing_source as ts
from xgpoc.vendor import unpaired_reasons as ur
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
VERSION = "xgv_timing_source_v1"
OUT = ROOT / "results/xgv_timing_source"
PAIRING = ROOT / "results/xgv_pairing"
PAIRING_FILES = ("half_offsets.csv", "match_map.csv", "pairs.parquet", "unpaired_sn.parquet")
STATSBOMB_TABLES = ("tables/timing.parquet", "tables/shots.parquet")
OPEN_DATA_SHA = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
PAIRING_COMPLETE_SHA256 = "8807ee42cba33062a07ca9136faab8b63427e2bd9a2e445ba902716fc14efc75"
SHOT_COLUMNS = ["match_id", "event_id", "period", "timestamp_ms", "team", "outcome", "event_json"]
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/timing_source.py", ROOT / "xgpoc/vendor/unpaired_reasons.py",
                ROOT / "xgpoc/vendor/timing_verdict.py",
                ROOT / "xgpoc/vendor/shot_pairing.py", ROOT / "xgpoc/vendor/match_pairing.py", ROOT / "xgpoc/vendor/statsbomb_tables.py",
                ROOT / "xgpoc/vendor/pairing_inputs.py")


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def verify_record(folder: Path, names: Iterable[str] | None = None) -> dict:
    """The completion record of `folder` after checking the named files (default: every listed file) against it."""
    record_path = Path(folder) / "complete.json"
    if not record_path.exists():
        raise ValueError(f"{folder} has no completion record")
    record = json.loads(record_path.read_text())
    for name in record["files"] if names is None else names:
        if name not in record["files"]:
            raise ValueError(f"The completion record of {folder} does not list {name}")
        if file_sha256(Path(folder) / name) != record["files"][name]:
            raise ValueError(f"{Path(folder) / name} differs from the hash in its completion record")
    return record


@dataclass(frozen=True)
class Inputs:
    labels: list
    half_offsets: pd.DataFrame
    match_map: pd.DataFrame
    pairs: pd.DataFrame
    unpaired_sn: pd.DataFrame
    timing: pd.DataFrame
    shots: pd.DataFrame
    events: pd.DataFrame
    provenance: dict


def load_events(statsbomb_root: Path, record: dict, match_ids: Iterable[int]) -> tuple[pd.DataFrame, dict]:
    """Every raw event of the given matches (`unpaired_reasons.event_frame`), each file checked against the record."""
    sha = record["identity"]["open_data"]["sha"]
    names = [f"raw/events/{m}.json.gz" for m in sorted(match_ids)]
    verify_record(statsbomb_root, names)
    frames = [ur.event_frame(m, json.loads(fetching.load_raw(fetching.events_path(statsbomb_root, m), sha, f"data/events/{m}.json")))
              for m in sorted(match_ids)]
    return pd.concat(frames, ignore_index=True), {name: record["files"][name] for name in names}


def load_inputs(statsbomb_root: Path, pairing_root: Path, labels_root: Path, open_data_sha: str, pairing_complete_sha256: str) -> Inputs:
    """Read and check every input (module docstring) against the pinned open-data commit and pairing record."""
    sb_record = verify_record(statsbomb_root, STATSBOMB_TABLES)
    if sb_record["identity"]["open_data"]["sha"] != open_data_sha:
        raise ValueError(f"The StatsBomb data version is open-data commit {sb_record['identity']['open_data']['sha']}, the rule note's is {open_data_sha}")
    if file_sha256(Path(pairing_root) / "complete.json") != pairing_complete_sha256:
        raise ValueError(f"{pairing_root} is not the frozen pairing (completion record sha256 {pairing_complete_sha256})")
    pairing_record = verify_record(pairing_root)
    sb_record_sha = file_sha256(Path(statsbomb_root) / "complete.json")
    pairing_inputs_record = pairing_record["identity"]["inputs"]
    if pairing_inputs_record["statsbomb"]["complete_json_sha256"] != sb_record_sha:
        raise ValueError("The frozen pairing was built from another StatsBomb data version")
    labels, labels_digest = pairing_inputs.read_labels(labels_root)
    if pairing_inputs_record["labels_v2"]["sha256"] != labels_digest:
        raise ValueError("The Labels-v2 files differ from the ones the frozen pairing read")
    match_map = pd.read_csv(Path(pairing_root) / "match_map.csv")
    paired_ids = match_map.sb_match_id[match_map.reason.eq("paired")].astype(int)
    events, event_hashes = load_events(statsbomb_root, sb_record, paired_ids)
    provenance = {"statsbomb": {"root": str(statsbomb_root), "complete_json_sha256": sb_record_sha,
                                "open_data": sb_record["identity"]["open_data"],
                                "files": {name: sb_record["files"][name] for name in STATSBOMB_TABLES} | event_hashes},
                  "pairing": {"root": str(pairing_root), "complete_json_sha256": file_sha256(Path(pairing_root) / "complete.json"),
                              "files": {name: pairing_record["files"][name] for name in PAIRING_FILES}},
                  "labels_v2": {"root": str(labels_root), "sha256": labels_digest}}
    return Inputs(labels=labels, half_offsets=pd.read_csv(Path(pairing_root) / "half_offsets.csv"), match_map=match_map,
                  pairs=pd.read_parquet(Path(pairing_root) / "pairs.parquet"),
                  unpaired_sn=pd.read_parquet(Path(pairing_root) / "unpaired_sn.parquet"),
                  timing=pd.read_parquet(Path(statsbomb_root) / "tables/timing.parquet"),
                  shots=pd.read_parquet(Path(statsbomb_root) / "tables/shots.parquet", columns=SHOT_COLUMNS),
                  events=events, provenance=provenance)


def identity_of(provenance: dict, rule_sha256: str) -> dict:
    return {"version": VERSION, "rule": {"note": str(ts.RULE_NOTE), "sha256": rule_sha256}, "inputs": provenance,
            "sources": {str(p): file_sha256(p) for p in SOURCE_FILES}}


def verify_complete(out: Path, identity: dict) -> dict:
    record = json.loads((out / "complete.json").read_text())
    if record["identity"] != identity:
        raise ValueError(f"Rule, inputs or sources changed since {out / 'complete.json'}; use a new --out")
    present = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - {"complete.json"}
    if unlisted := sorted(present - set(record["files"])):
        raise ValueError(f"{len(unlisted)} files are not in the completion record, e.g. {unlisted[0]}")
    return verify_record(out)


def publish(out: Path, result: ts.TimingSource, reasons: pd.DataFrame, identity: dict) -> dict:
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    result.label_matches.to_parquet(stage / "label_matches.parquet", index=False, compression="zstd")
    result.shot_residuals.to_parquet(stage / "shot_residuals.parquet", index=False, compression="zstd")
    result.agreement.to_csv(stage / "agreement.csv", index=False)
    result.substitution_sensitivity.to_csv(stage / "substitution_sensitivity.csv", index=False)
    reasons.to_csv(stage / "unpaired_reasons.csv", index=False)
    (stage / "timing_verdict.json").write_text(json_text(result.verdict))
    files = {p.name: file_sha256(p) for p in sorted(stage.iterdir())}
    if {str(p): file_sha256(p) for p in SOURCE_FILES} != identity["sources"]:
        raise ValueError("A source changed during the run")
    summary = result.summary | ur.reason_summary(reasons)
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

    rule_sha256 = ts.rule_note_sha256(ROOT)
    inputs = load_inputs(STATSBOMB, PAIRING, LABELS_ROOT, OPEN_DATA_SHA, PAIRING_COMPLETE_SHA256)
    identity = identity_of(inputs.provenance, rule_sha256)
    if (args.out / "complete.json").exists():
        record = verify_complete(args.out, identity)
        print(f"Verified {args.out}; nothing rebuilt\n{json_text(record['summary'])}")
        return
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and holds no completion record; refusing to replace it. Remove it or use another --out")
    result = ts.timing_source_test(inputs.labels, inputs.half_offsets, inputs.match_map, inputs.pairs, inputs.timing, inputs.shots,
                                   rule_sha256)
    summary = publish(args.out, result, ur.unpaired_reasons(inputs.unpaired_sn, inputs.events), identity)
    print(json_text(summary))
    print(json_text(result.verdict))


if __name__ == "__main__":
    main()
