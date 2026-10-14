"""Controls and the real run of xg-stack under the frozen protocol (xg-stack ticket 01; CPU, no GPU).

    ./run.sh --no-sync python scripts/xgs_run.py --controls    # controls 1-4 -> results/xgs_stack/controls
    ./run.sh --no-sync python scripts/xgs_run.py --real        # the single real run -> results/xgs_stack/real

Both modes open `research/xgs_stack_protocol.json` through `xgpoc.stack.protocol.load` with the SHA-256 pinned here,
and refuse to run unless its design equals `protocol.design()`, its code digest equals the current code's
(`xgpoc.stack.records.code_hashes`) and the population's input hashes and counts equal the protocol's. Everything runs
inside `no_statsbomb_reads()`; the outcomes are read inside the held-out label seal (`inputs.load_outcomes`).

--controls  control 1 (`provenance.control`), 2 (synthetic nulls), 3 (permutation of the real labels within matches,
            the full read-out) and 4 (positive control), then `complete.json` with each control's result, `all_passed`,
            the protocol hash, the input and code hashes and the output hashes. Refuses an existing output folder.
--real      refuses to start unless `controls/complete.json` matches its files, was written under this protocol and
            this code, and shows every control passed (`check_controls`); then the read-out on the real labels
            (`analysis.analyse`). Refuses an existing output folder, so the run is made once.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.eval.calibration import sigmoid
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.stack import analysis as sa
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack import provenance as sp
from xgpoc.stack import records
from xgpoc.stack import stats as ss

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = None          # ticket 01 records it after the freeze
OUT = ROOT / "results/xgs_stack"
CONTROLS = ("1_provenance", "2_synthetic_nulls", "3_permutation", "4_positive")


def opened_protocol(path: Path, expected: str | None) -> dict:
    """The protocol, refused unless its hash, design and code digest are the pinned and current ones."""
    if expected is None:
        raise SystemExit("no protocol hash is pinned yet (PROTOCOL_SHA256)")
    try:
        protocol = pr.load(path, expected)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if protocol["design"] != json.loads(json.dumps(pr.design())):
        raise SystemExit("the protocol's design differs from xgpoc.stack.protocol.design()")
    if protocol["code"]["digest"] != records.code_hashes()["digest"]:
        raise SystemExit("the code differs from the code the protocol pins; re-freeze after a code change")
    return protocol


def check_population(population: si.Population, protocol: dict) -> None:
    if population.inputs != protocol["inputs"] or population.counts != protocol["counts"]:
        raise SystemExit("the population's inputs or counts differ from the protocol's")


def check_controls(folder: Path, protocol_sha256: str, code_digest: str) -> dict:
    """The controls' completion record, refused unless every file matches it, it was written under this protocol and
    this code digest, and every control passed."""
    path = folder / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --controls first")
    record = json.loads(path.read_text())
    for name, digest in record["files"].items():
        if not (folder / name).exists() or sha256_file(folder / name) != digest:
            raise SystemExit(f"{folder / name} does not match the controls' completion record")
    if record["protocol_sha256"] != protocol_sha256 or record["code"]["package_digest"] != code_digest:
        raise SystemExit("the controls ran under another protocol or other code")
    if set(record["controls"]) != set(CONTROLS) or not record["all_passed"] or not all(c["passed"] for c in record["controls"].values()):
        raise SystemExit("not every control passed: the real run does not start")
    return record


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def read_out_files(out: Path, population: si.Population, result: dict) -> None:
    """The read-out's files: summary.json, probabilities.csv (no outcome) and categories.csv."""
    write_json(out / "summary.json", {k: result[k] for k in ("scores", "deltas", "readings", "gbm")})
    pd.DataFrame({"shot_id": population.table.index, "game": population.games, "fold": population.folds}
                 | {f"p_{k}": v for k, v in result["probabilities"].items()}).to_csv(out / "probabilities.csv", index=False,
                                                                                       float_format="%.9g")
    result["categories"].to_csv(out / "categories.csv", index=False, float_format="%.9g")


def run_controls(out: Path, population: si.Population, y: np.ndarray, protocol: dict) -> dict:
    table, folds = population.table, population.folds
    plan = pd.read_csv(a1.PLAN_DIR / "plan.csv", usecols=["game", "role"])
    index_shots = pd.read_parquet(INDEX, columns=["shot_id", "game"])
    annotations = {g: si.read_annotations(g, LABELS_ROOT) for g in sorted(table.game.unique())}
    frozen = {name: arm["columns"] for name, arm in protocol["design"]["arms"].items()}
    summary = {"1_provenance": sp.control(population, frozen, plan, index_shots, annotations)}
    print(f"control 1: {'passed' if summary['1_provenance']['passed'] else 'FAILED'}")

    designs = sa.all_designs(table, folds)
    weights = ss.bootstrap_weights(population.games)
    a, b = sf.reference_fit(table, y)
    z = table[sf.Z].to_numpy(float)
    p_r = sigmoid(a + b * z)
    nulls, summary["2_synthetic_nulls"] = sc.synthetic_nulls(designs, weights, p_r, log=print)
    nulls.to_csv(out / "synthetic_nulls.csv", index=False, float_format="%.9g")
    summary["2_synthetic_nulls"]["reference"] = {"a": a, "b": b}
    print(f"control 2: {json.dumps(summary['2_synthetic_nulls'])}")

    permuted, summary["3_permutation"] = sc.permutation(table, folds, y, weights, designs)
    (out / "permutation").mkdir()
    read_out_files(out / "permutation", population, permuted)
    print(f"control 3: {json.dumps(summary['3_permutation'])}")

    positive, summary["4_positive"] = sc.positive_control(designs, weights, table, a, b, log=print)
    positive.to_csv(out / "positive.csv", index=False, float_format="%.9g")
    print(f"control 4: {json.dumps(summary['4_positive'])}")
    write_json(out / "controls.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or OUT / ("controls" if args.controls else "real")
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    protocol = opened_protocol(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    package_digest = protocol["code"]["digest"]
    controls_record = check_controls(OUT / "controls", PROTOCOL_SHA256, package_digest) if args.real else None

    with no_statsbomb_reads():
        population = si.build_population()
        check_population(population, protocol)
        y = si.load_outcomes(population).aligned(population)
        out.mkdir(parents=True)
        if args.controls:
            summary = run_controls(out, population, y, protocol)
            result = {"controls": {k: {"passed": bool(v["passed"]), "criterion": v.get("criterion", "every check passes")}
                                   for k, v in summary.items()}}
            result["all_passed"] = all(c["passed"] for c in result["controls"].values())
        else:
            read = sa.analyse(population.table, population.folds, y, ss.bootstrap_weights(population.games))
            read_out_files(out, population, read)
            result = {"controls_record_sha256": sha256_file(OUT / "controls" / "complete.json"),
                      "controls_passed": controls_record["all_passed"], "readings": read["readings"]}
    complete = {"protocol_sha256": PROTOCOL_SHA256, "mode": "controls" if args.controls else "real", **result,
                "inputs": population.inputs, "counts": population.counts,
                "code": {"package_digest": package_digest, **records.code_hashes(Path(__file__))},
                "git": detector_records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "complete.json", complete)
    print(json.dumps({k: complete[k] for k in complete if k in ("controls", "all_passed", "readings")}, indent=2))


if __name__ == "__main__":
    main()
