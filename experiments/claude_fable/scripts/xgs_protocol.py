"""Write the frozen xg-stack protocol, research/xgs_stack_protocol.json (xg-stack ticket 01). Label-free.

    ./run.sh --no-sync python scripts/xgs_protocol.py
    ./run.sh --no-sync python scripts/xgs_protocol.py --refreeze --reason "..."    # after a code change only

Builds the population (`xgpoc.stack.inputs.build_population`, inside `no_statsbomb_reads()`; no label is read), checks
that every arm's design matrix has one column per frozen name and no banned name (`provenance.check_columns`), and writes the C5 design (`xgpoc.stack.protocol.design`),
the readings, the inputs' hashes, the population's label-free counts and the code hashes of `xgpoc.stack` and the
modules it uses (`xgpoc.stack.records.code_hashes`). Refuses an existing protocol unless `--refreeze`, which needs a
`--reason`, keeps the previous file's SHA-256 and the reason in `history`, and refuses any change to the models, the
read-out or the controls (a re-freeze is for code changes, never for thresholds). Prints the file's SHA-256, which the
ticket records and `scripts/xgs_run.py` pins.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack import provenance as sp
from xgpoc.stack import records

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SECTIONS = ("models", "read_out", "controls")


def protocol_record(population: si.Population, history: list[dict]) -> dict:
    for arm in sf.ARMS.values():
        sp.check_columns(arm, arm.columns, sf.Preprocessor.fit(arm, population.table)(population.table.iloc[:5]))
    return {
        "purpose": "xg-stack: the frozen design of council C5 (arms, columns, flag windows, models, read-out, controls and "
                   "their pass criteria), the inputs it runs on and the code that runs it, written before any arm is "
                   "fitted on real labels (xg-stack ticket 01)",
        "provenance": "Transcribed from .scratch/xg-stack/spec.md, Decisions -> C5 (council record in "
                      ".scratch/xg-detector/council.md). scripts/xgs_run.py refuses to run unless this file's SHA-256 is the "
                      "one it pins, the design equals xgpoc.stack.protocol.design(), the code digest and every input hash "
                      "are the ones below. This file is not edited; a code change writes a new one (history).",
        "design": pr.design(),
        "readings": pr.READINGS,
        "inputs": population.inputs,
        "counts": population.counts,
        "code": records.code_hashes(),
        "git": detector_records.git_state(),
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": history,
    }


def refreeze_history(path: Path, reason: str | None) -> list[dict]:
    """The history of a re-frozen protocol: the previous file's entries plus its own SHA-256 and the reason. Refused
    without a reason or a previous file, and when the models, the read-out or the controls would change."""
    if not reason or not path.exists():
        raise SystemExit("a re-freeze needs an existing protocol and --reason")
    previous = json.loads(path.read_text())
    if changed := [s for s in FROZEN_SECTIONS if previous["design"][s] != json.loads(json.dumps(pr.design()[s]))]:
        raise SystemExit(f"a re-freeze may not change {changed}")
    return [*previous["history"], {"sha256": sha256_file(path), "written": previous["written"], "git": previous["git"],
                                   "replaced_because": reason}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / pr.PROTOCOL)
    parser.add_argument("--refreeze", action="store_true", help="replace the protocol after a code change")
    parser.add_argument("--reason", default=None, help="why the code changed (required with --refreeze)")
    args = parser.parse_args()
    history = refreeze_history(args.out, args.reason) if args.refreeze else []
    if args.out.exists() and not args.refreeze:
        raise SystemExit(f"{args.out} exists; a re-freeze needs --refreeze and --reason")
    with no_statsbomb_reads():
        population = si.build_population()
    record = protocol_record(population, history)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(sha256_file(args.out))


if __name__ == "__main__":
    main()
