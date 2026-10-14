"""The selection, the controls and the real run of xg-stack under the frozen files of council C6 (xg-stack ticket 01;
CPU, no GPU).

    ./run.sh --no-sync python scripts/xgs_run.py --select      # C6's selection -> results/xgs_stack/selection
    ./run.sh --no-sync python scripts/xgs_run.py --controls    # final controls 1-4 and reports -> results/xgs_stack/controls
    ./run.sh --no-sync python scripts/xgs_run.py --real        # the single real run -> results/xgs_stack/real

Every mode opens its frozen file through `xgpoc.stack.protocol.load` with the SHA-256 pinned here (`--select` the
plan, the others the protocol) and refuses to run unless the file's code digest equals the current code's
(`xgpoc.stack.records.code_hashes`), its content equals what the code writes (`protocol.plan` / `protocol.design`)
and the population's input hashes and counts equal the file's. Everything runs inside `no_statsbomb_reads()`; the
outcomes are read inside the held-out label seal (`inputs.load_outcomes`).

--select    the selection (`xgpoc.stack.selection.run`) on simulated outcomes; the real labels enter only through the
            unpenalised fit of R that fixes the simulation's truth.
--controls  refuses unless the protocol's selection record still matches; control 1 (`provenance.control`), 2
            (synthetic nulls), 3 (permutation of the real labels within matches, the full read-out) and 4 (the dist
            plant, the gate), then the reports: the dist + angle plant, the power curve with the oracle and the smallest
            detectable effect, the clustered nulls' bound coverage. `complete.json` holds each control's result and
            `all_passed`.
--real      refuses unless `controls/complete.json` matches its files, was written under this protocol and this code,
            and shows every control passed (`check_controls`); then the read-out on the real labels (`analysis.analyse`).
Every mode refuses an existing output folder, so each is made once.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

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
from xgpoc.stack import selection as sl
from xgpoc.stack import stats as ss

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = None              # the C6 plan, pinned after its freeze
PROTOCOL_SHA256 = None          # the C6 protocol, pinned after its freeze
OUT = ROOT / "results/xgs_stack"
CONTROLS = ("1_provenance", "2_synthetic_nulls", "3_permutation", "4_positive")


def opened(path: Path, expected: str | None) -> dict:
    """A frozen file, refused unless its hash is the pinned one and its code digest the current code's."""
    if expected is None:
        raise SystemExit(f"no SHA-256 is pinned for {path.name} yet")
    try:
        record = pr.load(path, expected)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if record["code"]["digest"] != records.code_hashes()["digest"]:
        raise SystemExit("the code differs from the code the file pins; re-freeze after a code change")
    return record


def same(a, b) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def check_population(population: si.Population, record: dict) -> None:
    if not same(population.inputs, record["inputs"]) or not same(population.counts, record["counts"]):
        raise SystemExit("the population's inputs or counts differ from the frozen file's")


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
    write_json(out / "summary.json", {k: result[k] for k in ("scores", "deltas", "readings", "chosen_c", "gbm")})
    pd.DataFrame({"shot_id": population.table.index, "game": population.games, "fold": population.folds}
                 | {f"p_{k}": v for k, v in result["probabilities"].items()}).to_csv(out / "probabilities.csv", index=False,
                                                                                       float_format="%.9g")
    result["categories"].to_csv(out / "categories.csv", index=False, float_format="%.9g")


def run_selection(out: Path, population: si.Population, y) -> dict:
    table, folds = population.table, population.folds
    a, b = sf.unpenalised_reference(table, y)
    frames, summary = sl.run(table, folds, ss.bootstrap_weights(population.games), a, b, log=print)
    pd.concat([f.assign(candidate=c, plant=k) for (c, k), f in frames.items()]).to_csv(
        out / "replicates.csv", index=False, float_format="%.9g")
    write_json(out / "selection.json", summary)
    return {"chosen": summary["chosen"], "averaged_power_G": summary["averaged_power_G"]}


def run_controls(out: Path, population: si.Population, y, protocol: dict) -> dict:
    table, folds = population.table, population.folds
    spec = sf.spec(protocol["design"]["chosen"], table)
    seeds, n = pr.FINAL_SEEDS, sc.N_REPLICATES
    plan = pd.read_csv(a1.PLAN_DIR / "plan.csv", usecols=["game", "role"])
    index_shots = pd.read_parquet(INDEX, columns=["shot_id", "game"])
    annotations = {g: si.read_annotations(g, LABELS_ROOT) for g in sorted(table.game.unique())}
    frozen = {name: arm["columns"] for name, arm in protocol["design"]["arms"].items()}
    summary = {"1_provenance": sp.control(population, spec, frozen, plan, index_shots, annotations)}
    print(f"control 1: {'passed' if summary['1_provenance']['passed'] else 'FAILED'}")

    designs = sf.all_designs(table, folds, spec)
    weights = ss.bootstrap_weights(population.games)
    a, b = sf.reference_fit(table, y, spec.model)
    z = table[sf.Z].to_numpy(float)
    frames = {}
    frames["synthetic_nulls"], summary["2_synthetic_nulls"] = sc.synthetic_nulls(
        designs, weights, spec.model, sigmoid(a + b * z), n, seeds["synthetic_nulls"], print)
    summary["2_synthetic_nulls"]["reference"] = {"a": a, "b": b}
    print(f"control 2: {json.dumps(summary['2_synthetic_nulls'])}")
    permuted, summary["3_permutation"] = sc.permutation(table, folds, y, weights, spec, seeds["permutation"], designs)
    (out / "permutation").mkdir()
    read_out_files(out / "permutation", population, permuted)
    print(f"control 3: {json.dumps(summary['3_permutation'])}")
    frames["positive_dist"], summary["4_positive"] = sc.positive_control(
        designs, weights, spec.model, table, a, b, "dist", n, seeds["positive_dist"], log=print)
    print(f"control 4: {json.dumps(summary['4_positive'])}")

    reports = {}
    frames["positive_dist_angle"], reports["dist_angle_plant"] = sc.positive_control(
        designs, weights, spec.model, table, a, b, "dist_angle", n, seeds["positive_dist_angle"], log=print)
    curve = [summary["4_positive"]]
    for target in sc.CURVE[1:]:
        frames[f"curve_{target}"], point = sc.positive_control(designs, weights, spec.model, table, a, b, "dist", n,
                                                               seeds[f"curve_{target}"], target, log=print)
        curve.append(point)
    reports["power_curve"] = {"points": curve, "smallest_detectable": sc.smallest_detectable(curve),
                              "power": sc.DETECTABLE_POWER}
    frames["clustered_nulls"], reports["clustered_nulls"] = sc.clustered_nulls(
        designs, weights, spec.model, table, a, b, n, seeds["clustered_nulls"], log=print)
    print(f"reports: {json.dumps(reports, default=float)}")
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.9g")
    write_json(out / "controls.json", summary | {"reports": reports, "angles_wrapped": population.counts["angles_wrapped"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    name = "selection" if args.select else "controls" if args.controls else "real"
    out = args.out or OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    if args.select:
        frozen_sha, frozen = PLAN_SHA256, opened(ROOT / pr.PLAN, PLAN_SHA256)
    else:
        frozen_sha, frozen = PROTOCOL_SHA256, opened(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
        selection = OUT / "selection/complete.json"
        if not selection.exists() or sha256_file(selection) != frozen["selection"]["complete_sha256"]:
            raise SystemExit("the selection's completion record is not the one the protocol pins")
    package_digest = frozen["code"]["digest"]
    controls_record = check_controls(OUT / "controls", PROTOCOL_SHA256, package_digest) if args.real else None

    with no_statsbomb_reads():
        population = si.build_population()
        check_population(population, frozen)
        if args.select:
            if not same(frozen["plan"], pr.plan({c: sf.spec(c, population.table) for c in sf.SIMPLICITY})):
                raise SystemExit("the plan differs from xgpoc.stack.protocol.plan()")
        elif not same(frozen["design"], pr.design(sf.spec(frozen["design"]["chosen"], population.table))):
            raise SystemExit("the protocol's design differs from xgpoc.stack.protocol.design()")
        y = si.load_outcomes(population).aligned(population)
        out.mkdir(parents=True)
        if args.select:
            result = run_selection(out, population, y)
        elif args.controls:
            summary = run_controls(out, population, y, frozen)
            result = {"controls": {k: {"passed": bool(v["passed"]), "criterion": v.get("criterion", "every check passes")}
                                   for k, v in summary.items()}}
            result["all_passed"] = all(c["passed"] for c in result["controls"].values())
        else:
            spec = sf.spec(frozen["design"]["chosen"], population.table)
            read = sa.analyse(population.table, population.folds, y, ss.bootstrap_weights(population.games), spec)
            read_out_files(out, population, read)
            result = {"controls_record_sha256": sha256_file(OUT / "controls" / "complete.json"),
                      "controls_passed": controls_record["all_passed"], "readings": read["readings"]}
    complete = {("plan_sha256" if args.select else "protocol_sha256"): frozen_sha, "mode": name, **result,
                "inputs": population.inputs, "counts": population.counts,
                "code": {"package_digest": package_digest, **records.code_hashes(Path(__file__))},
                "git": detector_records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "complete.json", complete)
    print(json.dumps({k: complete[k] for k in complete if k in ("chosen", "averaged_power_G", "controls", "all_passed", "readings")},
                     indent=2))


if __name__ == "__main__":
    main()
