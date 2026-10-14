"""The readings of the xg-stack test report (ticket 04; CPU, seconds): the test's, the gate's, the power's and the
StatsBomb look's numbers in one flat JSON, so the report's number check can address each of them (its JSON paths
split on dots, and the rule keys intervals and levels as "0.95").

    ./run.sh --no-sync python scripts/xgs_test_report.py
        # -> results/xgs_test_report/{readings.json, before_scoring.json, complete.json}

Inputs, each checked against its folder's completion record: results/xgs_test/result.json (the single test),
results/xgs_test_prep/{gate.json, power.json} (written before scoring) and results/xgs_test_statsbomb/agreement.json
(the one StatsBomb look, after the test; Data: StatsBomb.). readings.json holds the test's and the StatsBomb look's
numbers, before_scoring.json the gate's and the power's, which were fixed before any test label was read. It refuses
unless the test and the StatsBomb look name the same hash list and that hash list pins the gate and power files read
here (`check_inputs`). It runs without the StatsBomb guard on purpose: agreement.json is the StatsBomb look's
aggregate, read after the test. Nothing is computed beyond renaming and selecting; a rerun overwrites the outputs (only
the record's time stamp and git state change).
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.stack_heldout import rule as tr

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "results/xgs_test"
PREP = ROOT / "results/xgs_test_prep"
STATSBOMB = ROOT / "results/xgs_test_statsbomb"
OUT = ROOT / "results/xgs_test_report"
HASHLIST = ROOT / "research/xg_stack_test_hashes.json"


def undotted(name: str) -> str:
    """A level or margin in a key without its dot: 0.95 -> 95, 0.975 -> 975, 0.9 -> 90, 0.002 -> 002."""
    return re.sub(r"0\.(\d+)", lambda m: m.group(1).ljust(2, "0"), name)


def flat_delta(delta: dict) -> dict:
    """A rule.delta record with its intervals as lo/hi per level (95, 975, 99, 995, 90)."""
    out = {k: delta[k] for k in ("shots", "delta_ll", "ll_variant", "ll_base", "boot_sd", "draws")}
    for level, (lo, hi) in delta["intervals"].items():
        out |= {f"lo{undotted(level)}": lo, f"hi{undotted(level)}": hi}
    return out


def flat_power(power: dict) -> dict:
    """se scenario -> effect scenario -> the scenario's probabilities, levels and margins without dots."""
    out = {name: {"se": value} for name, value in power["standard_errors"].items()}
    for row in power["scenarios"]:
        out[row["se"]][row["effect"]] = {"delta": row["delta"]} | {
            undotted(k): v for k, v in row.items() if k.startswith("p_")}
    return out


def readings(result: dict, agreement: dict) -> dict:
    """The test's and the StatsBomb look's readings (module docstring)."""
    d, reading = result["descriptive"], result["primary"]["reading"]
    return {
        "level_in_force": result["level_in_force"], "population": result["population"],
        "primary": flat_delta(result["primary"]["delta"]) | {
            "label": reading["label"], "sentence": result["primary"]["sentence"],
            "equivalent_002": reading["equivalent"][tr.key(tr.MARGIN)],
            "equivalent_005": reading["equivalent"][tr.key(tr.ALSO_MARGIN)],
            "smallest_margin": reading["smallest_equivalence_margin"]},
        "descriptive": {name: flat_delta(d[name]) for name in ("GS_vs_R", "G_vs_g_combo", "R_vs_g_combo")}
        | {role: flat_delta(d["G_vs_R_by_role"][role]) for role in ("evaluation", "gate")},
        "scores": d["scores"],
        "statsbomb": {"shots": agreement["shots"], "spearman": agreement["spearman"]},
    }


def before_scoring(gate: dict, power: dict) -> dict:
    """The gate's and the power's readings, written before scoring (module docstring)."""
    return {
        "gate": {"false_success": {undotted(k): v for k, v in gate["false_success_rate"].items()},
                 "mc_se_95": gate["mc_se"]["0.95"], "false_worse_95": gate["false_worse_rate_95"],
                 "mean_null_delta": gate["mean_null_delta"], "replicates": gate["replicates"], "seed": gate["seed"],
                 "level_in_force": gate["level_in_force"]},
        "power": flat_power(power) | {"recovery": power["recovery"], "development": power["development"]},
    }


def check_inputs(test: Path, statsbomb: Path, before: list[Path], hashlist: Path) -> str:
    """The hash list both the test and the StatsBomb look ran under, refused unless they name the same one, the hash
    list on disk hashes to it, and it pins every file of `before` (the gate and the power) with its current sha256."""
    expected = json.loads((test / "complete.json").read_text())["hashlist_sha256"]
    if json.loads((statsbomb / "complete.json").read_text())["hashlist_sha256"] != expected:
        raise SystemExit("the StatsBomb look ran under another hash list than the test")
    if sha256_file(hashlist) != expected:
        raise SystemExit(f"{hashlist} is not the hash list the test ran under")
    pins = json.loads(hashlist.read_text())["files"]
    for path in before:
        if not any(str(path).endswith("/" + name) and digest == sha256_file(path) for name, digest in pins.items()):
            raise SystemExit(f"{path} is not the file the hash list pins")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    paths = {"result": TEST / "result.json", "gate": PREP / "gate.json", "power": PREP / "power.json",
             "agreement": STATSBOMB / "agreement.json"}
    inputs = {str(p.relative_to(ROOT)): records.verified_file(p) for p in paths.values()}
    hashlist_sha256 = check_inputs(TEST, STATSBOMB, [paths["gate"], paths["power"]], HASHLIST)
    loaded = {name: json.loads(p.read_text()) for name, p in paths.items()}
    out = readings(loaded["result"], loaded["agreement"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "readings.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    (args.out / "before_scoring.json").write_text(
        json.dumps(before_scoring(loaded["gate"], loaded["power"]), indent=2, ensure_ascii=False) + "\n")
    record = {"hashlist_sha256": hashlist_sha256, "inputs": inputs,
              "code": {"scripts/xgs_test_report.py": sha256_file(Path(__file__))}, "git": records.git_state(),
              "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(out["primary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
