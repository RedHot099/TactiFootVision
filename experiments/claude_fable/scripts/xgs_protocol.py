"""Write the frozen xg-stack files of council C6 (xg-stack ticket 01). Label-free.

    ./run.sh --no-sync python scripts/xgs_protocol.py --plan      # research/xgs_stack_c6_plan.json, before the selection
    ./run.sh --no-sync python scripts/xgs_protocol.py --final     # research/xgs_stack_protocol_c6.json, after it
    ... --refreeze --reason "..."                                  # after a code change only

Both build the population (`xgpoc.stack.inputs.build_population`, inside `no_statsbomb_reads()`; no label is read)
and record the inputs' hashes, the label-free counts, the code hashes of `xgpoc.stack` and the modules it uses
(`xgpoc.stack.records.code_hashes`) and the git state.

--plan   every candidate's arms after the zero-variance rule (each design matrix checked against its column list and
         the banned names), the selection rule and seeds, (b)'s predicted recovery and the final controls
         (`xgpoc.stack.protocol.plan`).
--final  refuses unless the selection's completion record (results/xgs_stack/selection/complete.json) matches its
         files, was written under the plan of `--plan-sha256` and under the current code; then the chosen
         candidate's design (`xgpoc.stack.protocol.design`) with the selection's hashes and result.

An existing file is refused unless `--refreeze --reason`, which keeps the previous file's SHA-256 and the reason in
`history` and refuses any change to the frozen parts (`frozen_parts`: a re-freeze is for code changes, never for a rule
or a threshold). Prints the file's SHA-256, which the ticket records and `scripts/xgs_run.py` pins.
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
SELECTION = ROOT / "results/xgs_stack/selection"


def frozen_parts(record: dict, mode: str) -> dict:
    """The parts of a record a re-freeze may not change."""
    if mode == "plan":
        return {"selection": record["plan"]["selection"], "final_controls": record["plan"]["common"]["final_controls"],
                "read_out": record["plan"]["common"]["read_out"], "stop_rule": record["plan"]["common"]["stop_rule"]}
    design = record["design"]
    return {k: design[k] for k in ("chosen", "model", "final_controls", "read_out", "stop_rule")}


def refreeze_history(path: Path, reason: str | None, new: dict, mode: str) -> list[dict]:
    """The history of a re-frozen file: the previous entries plus its own SHA-256 and the reason. Refused without a
    reason or a previous file, and when a frozen part would change."""
    if not reason or not path.exists():
        raise SystemExit("a re-freeze needs an existing file and --reason")
    previous = json.loads(path.read_text())
    old, current = frozen_parts(previous, mode), json.loads(json.dumps(frozen_parts(new, mode)))
    if changed := [k for k in old if old[k] != current[k]]:
        raise SystemExit(f"a re-freeze may not change {changed}")
    return [*previous["history"], {"sha256": sha256_file(path), "written": previous["written"], "git": previous["git"],
                                   "replaced_because": reason}]


def checked_spec(name: str, population: si.Population) -> sf.Spec:
    spec = sf.spec(name, population.table)
    for arm in spec.arms.values():
        sp.check_columns(arm, arm.columns, sf.Preprocessor.fit(arm, population.table)(population.table.iloc[:5]))
    for columns in spec.gbm_columns.values():
        sp.check_names(columns)
    return spec


def selection_record(plan_sha256: str) -> dict:
    """The selection's completion record, refused unless it matches its files, the plan and the current code."""
    path = SELECTION / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run scripts/xgs_run.py --select first")
    record = json.loads(path.read_text())
    for name, digest in record["files"].items():
        if sha256_file(SELECTION / name) != digest:
            raise SystemExit(f"{SELECTION / name} does not match the selection's completion record")
    if record["plan_sha256"] != plan_sha256 or record["code"]["package_digest"] != records.code_hashes()["digest"]:
        raise SystemExit("the selection ran under another plan or other code")
    return record


def base(population: si.Population) -> dict:
    return {"inputs": population.inputs, "counts": population.counts, "code": records.code_hashes(),
            "git": detector_records.git_state(), "written": datetime.now(UTC).isoformat(timespec="seconds")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--plan-sha256", default=None, help="the frozen plan's SHA-256 (required with --final)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--refreeze", action="store_true", help="replace the file after a code change")
    parser.add_argument("--reason", default=None, help="why the code changed (required with --refreeze)")
    args = parser.parse_args()
    kind = "plan" if args.plan else "final"
    out = args.out or ROOT / (pr.PLAN if args.plan else pr.PROTOCOL)
    if out.exists() and not args.refreeze:
        raise SystemExit(f"{out} exists; a re-freeze needs --refreeze and --reason")
    with no_statsbomb_reads():
        population = si.build_population()
        if args.plan:
            specs = {name: checked_spec(name, population) for name in sf.SIMPLICITY}
            record = {"purpose": "xg-stack, council C6: the selection plan, written before any simulation of the "
                                 "selection (xg-stack ticket 01)",
                      "plan": pr.plan(specs)} | base(population)
        else:
            if args.plan_sha256 is None:
                raise SystemExit("--final needs --plan-sha256")
            selected = selection_record(args.plan_sha256)
            summary_path = SELECTION / "selection.json"
            summary = json.loads(summary_path.read_text())
            spec = checked_spec(summary["chosen"], population)
            record = {"purpose": "xg-stack, council C6: the frozen design of the chosen candidate, written after the "
                                 "selection on simulated outcomes and before any arm is fitted on real labels (xg-stack "
                                 "ticket 01)",
                      "plan": {"path": str(pr.PLAN), "sha256": args.plan_sha256},
                      "selection": {"path": "results/xgs_stack/selection", "complete_sha256": sha256_file(SELECTION / "complete.json"),
                                    "summary_sha256": selected["files"]["selection.json"], "chosen": summary["chosen"],
                                    "averaged_power_G": summary["averaged_power_G"],
                                    "observed_recovery_b": summary["observed_recovery_b"]},
                      "design": pr.design(spec)} | base(population)
    history = refreeze_history(out, args.reason, record, kind) if args.refreeze else []
    out.write_text(json.dumps(record | {"history": history}, indent=2, allow_nan=False) + "\n")
    print(sha256_file(out))


if __name__ == "__main__":
    main()
