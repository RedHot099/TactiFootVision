"""Write the frozen protocol of xg-temporal, `research/xgt_protocol.json` (ticket 03). Label-free; CPU only.

    ./run.sh --no-sync python scripts/xgt_protocol.py                       # after 01b and the label-free controls
    ./run.sh --no-sync python scripts/xgt_protocol.py --draft --source g_xgc --out outputs/xgt_protocol_draft.json
    ... --refreeze --reason "..."                                            # after a code change only

The file holds `frozen` (`xgpoc.temporal.protocol.frozen()`, every rule the code runs) and `data`, built here inside
`no_statsbomb_reads()` without any label:
    main                  the main population of g_xgv at (-1600, -600): input hashes (plan, folds, oof.csv with
                          its two columns, index with its three, the source's table and embedding array), counts, the
                          shot list's SHA-256, the arms' columns, the leak assertions; `source_record` adds the
                          SHA-256 of the source folder's complete.json (results/xgt_xgv_embeddings/complete.json)
    cells                 every window of the ticket 07 grid and the leakage-control rows, its population (window
                          population ∩ main) recorded the same way
    leak_common           the leak curve's common population (the shots of every leak-curve window), its shot list and
                          every window's population on it: the jump rule's population
    descriptive_sources   g_xgc and dinov2: their files and their main-window populations
    safe_margin           ticket 02's result.json (pinned SHA-256, checked against its completion record): STOP and the
                          owner's decision
    e3                    the categories' E3 source after the label-free coverage check of P_G (C9 Q5:
                          cache/xgd_full3/E3, else cache/xgd_full4_feet_t0/E3), its hashes and the category counts
    label_free_controls   the record of `scripts/xgt_run.py --controls --c10` (results/xgt/controls/label_free_c10),
                          refused unless it matches its files, is the protocol's controls (council C10's run on
                          `controls.RUN_SEEDS`, T's design `features.T_DESIGN`, source g_xgv, 200 replicates, every
                          simulation but at most the descriptive rich_nulls_-0.002, `controls.DROPPABLE`, recorded as
                          omitted), is not one of the earlier attempts, was written under the current package code, has
                          the same settings and population, and passed every label-free gate; the first attempt's and
                          the remedy's records (or any record without --c10) are refused
    attempts              the attempt trail, both required: the failed first run (results/xgt/controls/label_free) and
                          the failed C9 Q2 remedy (results/xgt/controls/label_free_remedy), each with its controls.json
                          at the pinned SHA-256 (`protocol.ATTEMPT_CONTROLS_SHA256`), its complete.json's SHA-256, its
                          gates and its positive miss; the first must be a miss by shrinkage (above -0.001, the
                          remedy's trigger), the remedy must be the run on `controls.REMEDY_SEEDS` with a positive miss
    code                  `xgpoc.temporal.records.package_hashes()`
Without results/xgt_xgv_embeddings (ticket 01b) or the label-free controls the protocol cannot be written; `--draft`
writes a draft instead (another --source allowed, missing parts recorded as pending, `draft: true`, never to the
protocol's path), which `scripts/xgt_run.py` refuses. An existing protocol is refused unless `--refreeze --reason`, which
keeps the previous file's SHA-256 and the reason in `history` and refuses any change of `frozen` and an unchanged
package digest (a re-freeze is for a code change, never for a rule). Prints the file's SHA-256; the ticket records it
and `scripts/xgt_run.py` pins it.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX
from xgpoc.temporal import categories as tc
from xgpoc.temporal import controls as tco
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import protocol as tpr
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import records as tr_records

ROOT = Path(__file__).resolve().parents[1]
PENDING = "pending"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", choices=sorted(ti.SOURCES), default=tpr.MAIN_SOURCE)
    parser.add_argument("--draft", action="store_true", help="a draft: another source, pending parts; never the protocol")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--controls", type=Path, default=ROOT / tpr.LABEL_FREE, help="the label-free controls' folder")
    parser.add_argument("--first-attempt", type=Path, default=ROOT / tpr.FIRST_ATTEMPT,
                        help="the failed first attempt's folder")
    parser.add_argument("--remedy-attempt", type=Path, default=ROOT / tpr.REMEDY_ATTEMPT,
                        help="the failed C9 Q2 remedy's folder")
    parser.add_argument("--refreeze", action="store_true")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args(argv)
    protocol_path = (ROOT / tpr.PROTOCOL).resolve()
    if args.out is None:
        if args.draft:
            parser.error("--draft needs its own --out")
        args.out = protocol_path
    if args.draft and Path(args.out).resolve() == protocol_path:
        parser.error(f"a draft is never written to {tpr.PROTOCOL}")
    if not args.draft and args.source != tpr.MAIN_SOURCE:
        parser.error(f"the protocol's main source is {tpr.MAIN_SOURCE}; another source needs --draft")
    if args.refreeze and not args.reason:
        parser.error("--refreeze needs --reason")
    return args


def source_record(files: ti.SourceFiles) -> dict:
    return files.record() | {"complete_json": {"path": ti.recorded_path(files.source.folder / ti.RECORD),
                                               "sha256": sha256_file(files.source.folder / ti.RECORD)}}


def main_record(source: str, plan) -> tuple[dict, ti.Population]:
    population, files = ti.build_population(ti.SOURCES[source], ti.Window(), plan)
    columns = tf.arm_columns(population.window, tf.K)
    provenance = tp.control(population, columns, plan)
    if not provenance["passed"]:
        raise SystemExit(f"provenance failed: {[c for c in provenance['checks'] if not c['passed']]}")
    return {"source": source, **population.record(), "source_files": source_record(files), "columns": columns,
            "provenance": provenance}, population


def cell_records(source: str, plan, main: ti.Population) -> list[dict]:
    """Every grid cell's population (window population ∩ main), recorded like the main one."""
    files = ti.verify_source(ti.SOURCES[source])
    out = []
    for cell in tpr.grid_cells():
        window = tpr.cell_window(cell)
        population, _ = ti.build_population(ti.SOURCES[source], window, plan, files=files)
        population = ti.restrict(population, main.table.index)
        provenance = tp.control(population, tf.arm_columns(window, tf.K), plan)
        if not provenance["passed"]:
            raise SystemExit(f"provenance failed in cell {cell['start']}/{cell['o_max']}")
        out.append({"start": cell["start"], "o_max": cell["o_max"], **population.record(),
                    "provenance_passed": provenance["passed"]})
    return out


def leak_common_record(source: str, plan, main: ti.Population) -> dict:
    """The leak curve's common population (`protocol.common_populations`): its size, shot list and every window's
    population on it, recorded like the cells."""
    files = ti.verify_source(ti.SOURCES[source])
    common = tpr.common_populations(tpr.leak_populations(source, plan, main, files))
    for o_max, population in common.items():
        window = tpr.leak_window(o_max)
        if not tp.control(population, tf.arm_columns(window, tf.K), plan)["passed"]:
            raise SystemExit(f"provenance failed in the common leak population at -1600/{o_max}")
    first = common[tpr.LEAK_CURVE[0]]
    return {"rule": tpr.leak_record()["common_population"], "shots": first.counts["shots"],
            "shot_list_sha256": ti.shot_list_sha256(first.table),
            "cells": {str(o_max): population.record() for o_max, population in common.items()}}


def descriptive_sources(plan) -> dict:
    out = {}
    for name in tpr.DESCRIPTIVE_SOURCES:
        population, files = ti.build_population(ti.SOURCES[name], ti.Window(), plan)
        out[name] = {**population.record(), "source_files": source_record(files)}
    return out


def safe_margin() -> dict:
    path = ROOT / tpr.SAFE_MARGIN
    digest = sha256_file(path)
    recorded = json.loads((path.parent / "complete.json").read_text())["outputs"].get(path.name)
    if digest != tpr.SAFE_MARGIN_RESULT_SHA256 or recorded != digest:
        raise SystemExit(f"{path} is not ticket 02's result (sha256 {digest})")
    result = json.loads(path.read_text())
    return {"path": str(tpr.SAFE_MARGIN), "sha256": digest, "verdict": result["verdict"], "o_max": result["o_max_ms"],
            "violations": result["violations"], "n": result["n"], "cp_upper_95": result["cp_upper_95"],
            "max_rate": result["max_rate"],
            "worst_case": {o: v["worst_case_struck"] for o, v in result["per_offset"].items()},
            "owner_decision": "2026-10-25: o_max stays -600 despite the STOP; the report (ticket 05) discloses the STOP "
                              "and the deviation from the stop rule; boundary sensitivity from the ticket 07 grid"}


def e3_record(population: ti.Population) -> dict:
    """The categories' E3 source chosen by the label-free coverage check of C9 Q5 (`categories.covering_source`)."""
    try:
        return tc.covering_source(population.table.index)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def label_free_controls(folder: Path, main: dict, draft: bool) -> dict:
    """The label-free controls' record (module docstring); pending in a draft when it is missing."""
    path = folder / "complete.json"
    if not path.exists():
        if draft:
            return {"status": PENDING, "path": ti.recorded_path(folder)}
        raise SystemExit(f"{path} does not exist: run scripts/xgt_run.py --controls first")
    record = verified(folder)
    if sha256_file(folder / "controls.json") in tpr.ATTEMPT_CONTROLS_SHA256.values():
        problems = ["an earlier failed attempt, not the C10 run"]       # nothing else of it is read
        if not draft:
            raise SystemExit(f"the label-free controls cannot enter the protocol: {problems}")
        return {"status": "refused", "path": ti.recorded_path(folder), "complete_sha256": sha256_file(path),
                "problems": problems}
    problems = []
    if not record.get("protocol_controls"):
        problems.append("not the protocol's controls (a trial run)")
    if not record.get("c10") or record.get("seeds") != tco.RUN_SEEDS or record.get("t_design") != tf.T_DESIGN:
        problems.append("not council C10's run on RUN_SEEDS (two earlier attempts failed)")
    if record["source"] != main["source"]:
        problems.append(f"source {record['source']}")
    if record["code"]["package"]["digest"] != tr_records.package_hashes()["digest"]:
        problems.append("other package code")
    if json.loads(json.dumps(record["settings"])) != json.loads(json.dumps(tco.settings())):
        problems.append("other settings")
    if {k: record[k] for k in ("inputs", "counts", "shot_list_sha256")} != json.loads(json.dumps(
            {k: main[k] for k in ("inputs", "counts", "shot_list_sha256")})):
        problems.append("another population")
    if not record["label_free_passed"]:
        problems.append("a label-free gate failed")
    omitted = record.get("omitted_simulations")
    if omitted is None or not set(omitted) <= set(tco.DROPPABLE) \
            or record.get("simulations") != [s for s in tco.SIMULATIONS if s not in omitted]:
        problems.append(f"simulations other than the protocol's (only {list(tco.DROPPABLE)} may be left out)")
    if problems and not draft:
        raise SystemExit(f"the label-free controls cannot enter the protocol: {problems}")
    return {"status": "recorded", "path": ti.recorded_path(folder), "complete_sha256": sha256_file(path),
            "gates": record["gates"], "label_free_passed": record["label_free_passed"],
            "simulations": record.get("simulations"), "omitted_simulations": omitted,
            "protocol_controls": record.get("protocol_controls"), "c10": record.get("c10"),
            "t_design": record.get("t_design"), "seeds": record.get("seeds"), "problems": problems}


def verified(folder: Path) -> dict:
    """A folder's complete.json, refused unless every file it names matches."""
    path = folder / "complete.json"
    record = json.loads(path.read_text())
    for name, digest in record["files"].items():
        if not (folder / name).exists() or sha256_file(folder / name) != digest:
            raise SystemExit(f"{folder / name} does not match {path}")
    return record


def attempt(name: str, folder: Path, draft: bool) -> dict:
    """One failed earlier attempt of the trail (module docstring): refused unless its files match its record, its
    controls.json is the pinned one and its positive gate missed (the first: by shrinkage, the remedy's trigger; the
    remedy: on REMEDY_SEEDS); pending in a draft when missing."""
    path, controls = folder / "complete.json", folder / "controls.json"
    if not path.exists():
        if draft:
            return {"status": PENDING, "path": ti.recorded_path(folder)}
        raise SystemExit(f"{path} does not exist: the protocol cites the {name} attempt of the controls")
    record = verified(folder)
    pinned = tpr.ATTEMPT_CONTROLS_SHA256[name]
    if sha256_file(controls) != pinned:
        raise SystemExit(f"{controls} is not the {name} attempt the code pins ({pinned[:8]}...)")
    summary = json.loads(controls.read_text())
    positive, rich = summary["gates"]["positive"], summary["gates"]["rich_nulls"]
    recovered = positive[f"mean_recovered_delta_{tco.MOTION_CONTRAST}"]
    if positive["passed"]:
        raise SystemExit(f"the {name} attempt's positive gate passed: it is no failed attempt")
    if name == "first" and not recovered > tco.REMEDY_TRIGGER:
        raise SystemExit("the first attempt's positive gate did not miss by shrinkage: the remedy has no trigger")
    if name == "remedy" and (not record.get("remedy") or summary["seeds"] != tco.REMEDY_SEEDS):
        raise SystemExit("the remedy attempt is not the C9 Q2 remedy's run on REMEDY_SEEDS")
    return {"status": "recorded", "path": ti.recorded_path(folder), "complete_sha256": sha256_file(path),
            "controls_sha256": sha256_file(controls), "protocol_controls": record.get("protocol_controls"),
            "label_free_passed": record["label_free_passed"],
            "gates": {g: bool(summary["gates"][g]["passed"]) for g in tco.LABEL_FREE_GATES},
            "positive": {"mean_recovered_T-S": recovered, "power_T-S": positive["power_T-S"],
                         "power_T-S_mc_se": positive["power_T-S_mc_se"], "oracle_power": positive["oracle_power"],
                         "grid_floor_share": positive["grid_floor_share"], "trigger": tco.REMEDY_TRIGGER},
            "rich_nulls": {"holm_rejection_T-S": rich["holm_rejection"]["T-S"], "mean_delta_T-S": rich["mean_delta"]["T-S"]},
            "seeds": summary["seeds"], "seconds": summary["run"]["seconds"],
            "consequence": {"first": "the pre-registered remedy of C9 Q2 (controls.remedy, model.t_variant), every "
                                     "label-free gate rerun once on controls.REMEDY_SEEDS (attempts.remedy)",
                            "remedy": "the second positive miss stopped the phase; owner decision 2026-10-28 "
                                      "(redesign); council C10: T = S + 8 motion columns, every label-free gate "
                                      "rerun once on controls.C10_SEEDS (label_free_controls)"}[name]}


def attempts(folders: dict[str, Path], draft: bool) -> dict:
    """The attempt trail (`protocol.ATTEMPTS` order): every earlier attempt is required."""
    if list(folders) != list(tpr.ATTEMPTS):
        raise SystemExit(f"the attempt trail needs {list(tpr.ATTEMPTS)} in that order")
    return {name: attempt(name, folder, draft) for name, folder in folders.items()}


def refreeze_history(path: Path, reason: str | None, record: dict) -> list[dict]:
    if not path.exists():
        raise SystemExit("a re-freeze needs an existing protocol")
    previous = json.loads(path.read_text())
    if previous["frozen"] != json.loads(json.dumps(record["frozen"], default=tr_records.json_default)):
        raise SystemExit("a re-freeze may not change the frozen rules")
    if previous["data"]["code"]["digest"] == record["data"]["code"]["digest"]:
        raise SystemExit("a re-freeze needs a code change: the package digest is the previous protocol's")
    return [*previous.get("history", []), {"sha256": sha256_file(path), "written": previous["written"],
                                           "git": previous["git"], "replaced_because": reason}]


def build(args: argparse.Namespace) -> dict:
    with no_statsbomb_reads():
        plan = a1.verify_plan(INDEX)
        try:
            main, population = main_record(args.source, plan)
        except FileNotFoundError as e:
            raise SystemExit(f"{e}: the {args.source} embeddings are not there yet (ticket 01b for g_xgv); a draft on "
                             "another source needs --draft --source g_xgc") from None
        data = {"main": main, "cells": cell_records(args.source, plan, population),
                "leak_common": leak_common_record(args.source, plan, population),
                "descriptive_sources": descriptive_sources(plan), "safe_margin": safe_margin(),
                "e3": e3_record(population),
                "label_free_controls": label_free_controls(args.controls, main, args.draft),
                "attempts": attempts({"first": args.first_attempt, "remedy": args.remedy_attempt}, args.draft),
                "code": tr_records.package_hashes()}
    if args.draft:
        xgv = ti.SOURCES[tpr.MAIN_SOURCE].folder / ti.RECORD
        data["pending"] = {"g_xgv_complete_json": None if xgv.exists() else ti.recorded_path(xgv),
                           "label_free_controls": data["label_free_controls"]["status"] == PENDING,
                           "attempts": [n for n, a in data["attempts"].items() if a["status"] == PENDING]}
    purpose = ("xg-temporal ticket 03: the frozen protocol, written before any label is read" if not args.draft else
               "xg-temporal ticket 03: a DRAFT of the protocol (not frozen; refused by scripts/xgt_run.py)")
    return {"purpose": purpose, "draft": bool(args.draft), "frozen": tpr.frozen(), "data": data,
            "written": datetime.now(UTC).isoformat(timespec="seconds"), "git": records.git_state()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.refreeze:
        raise SystemExit(f"{out} exists; a re-freeze needs --refreeze and --reason")
    record = build(args)
    history = refreeze_history(out, args.reason, record) if args.refreeze else []
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record | {"history": history}, indent=1, default=tr_records.json_default,
                              allow_nan=False) + "\n")
    print(sha256_file(out))


if __name__ == "__main__":
    main()
