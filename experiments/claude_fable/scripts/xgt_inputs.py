"""Label-free inputs of xg-temporal (ticket 01): the extraction list of ticket 01b, the populations of the coverage
table and the PCA explained variance. CPU only; no outcome is read (`xgpoc.temporal.inputs`).

    ./run.sh --no-sync python scripts/xgt_inputs.py [--out results/xgt_inputs] [--sources g_xgc dinov2]

Writes to --out (a new folder):
    extraction_shots.csv      shot_id, game of every non-penalty training shot with at least one decoded frame without
                              a cut at the 11 offsets -1600 ... +400 ms (the --shots-csv of ticket 01b)
    population_<source>.csv   shot_id, game, fold, frames, complete of the main window's population P (o_max -600)
    summary.json              the extraction counts and list hash, per source the populations of the coverage table's
                              windows against the spec's numbers, the population hash and the PCA explained variance
                              by 16, 32 and 64 components per outer training fold
    complete.json             input, code and output hashes, git state, timing; written last

The coverage rows with o_max -400 are built with the control flag set by the script (`provenance.check_window`): they
are counts of the spec's coverage table, never a design or a candidate. After ticket 01b, `--sources g_xgv --out
results/xgt_inputs_xgv` adds the g_xgv coverage (checked against the DINOv2 row) and its PCA variance.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.paths import INDEX
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import records as tr_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgt_inputs"
COVERAGE_WINDOWS = (-800, -600, -400)          # o_max of the spec's coverage table, start -1600
SPEC_TABLE = {                                  # spec, "Pokrycie na populacji rozwojowej": shots >= 1 frame, complete, matches
    "g_xgc": {"-800": [6278, 5981, 297], "-600": [6320, 5980, 297], "-400": [6339, 5978, 297]},
    "dinov2": {"-600": [7010, 6620, 328]},
    "g_xgv": {"-600": [7010, 6620, 328]},       # "oczekiwane jak DINOv2" (ticket 01b checks it)
}


def source_summary(name: str, plan, out: Path) -> dict:
    source = ti.SOURCES[name]
    files = ti.verify_source(source)
    windows = {}
    for o_max in COVERAGE_WINDOWS:
        window = ti.Window(ti.START, o_max, control=o_max > tp.LEAK_LIMIT)    # -400 is a count, never a candidate
        c = ti.build_population(source, window, plan, files=files)[0].counts
        windows[str(o_max)] = {"shots": c["shots"], "complete_shots": c["complete_shots"], "matches": c["matches"],
                               "offsets": len(window.offsets)}
        if (spec := SPEC_TABLE.get(name, {}).get(str(o_max))) is not None:
            windows[str(o_max)] |= {"spec": spec, "matches_spec": [c["shots"], c["complete_shots"], c["matches"]] == spec}
    main, _ = ti.build_population(source, ti.Window(), plan, files=files)
    path = out / f"population_{name}.csv"
    main.table.reset_index()[["shot_id", "game", "fold", "frames", "complete"]].to_csv(path, index=False)
    embeddings, position = ti.load_embeddings(files, main.refs)
    data = tf.DesignData(main.window, main.z, main.folds, position, embeddings)
    started = time.time()
    variance = tf.explained_variance(data)
    return {"source": files.record(), "coverage": windows, "main_population": main.counts,
            "main_population_csv_sha256": sha256_file(path), "embedding_dim": int(embeddings.shape[1]),
            "pca_explained_variance": variance, "pca_seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sources", nargs="+", default=["g_xgc", "dinov2"], choices=sorted(ti.SOURCES))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started, clock = datetime.now(UTC), time.time()
    plan = a1.verify_plan(INDEX)
    base = ti.base_shots(plan)
    mask = ti.verify_source(ti.SOURCES["dinov2"])
    shots, counts = ti.extraction_shots(base, ti.read_frames(mask, base.index, ti.GRID))
    args.out.mkdir(parents=True)
    shots.to_csv(args.out / "extraction_shots.csv", index=False)
    summary = {"label": "label-free inputs of xg-temporal ticket 01; no outcome read",
               "base": {"shots": len(base), "matches": int(base.game.nunique()), "rule": "non-penalty out-of-fold shots of "
                        "the plan's training role (oof.csv: shot_id and z_g_combo only)"},
               "extraction_01b": counts | {"rule": "base shots with >= 1 decoded frame without a cut at the 11 offsets "
                                           "-1600..+400 ms (frame mask of the dinov2_frozen rows of xgc_offset_scores)",
                                           "csv_sha256": sha256_file(args.out / "extraction_shots.csv")},
               "sources": {name: source_summary(name, plan, args.out) for name in args.sources}}
    summary["run"] = {"seconds": round(time.time() - clock, 1), "peak_rss_gib": round(tr_records.peak_rss_gib(), 2)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=tr_records.json_default) + "\n")
    complete = {"inputs": {"plan": a1.PLAN_SHA256, "folds": sha256_file(a1.FOLDS), "index": sha256_file(INDEX),
                           "oof": sha256_file(ti.OOF), "mask_source": mask.record()},
                "code": tr_records.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"),
                        "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1) + "\n")
    print(json.dumps({"extraction_01b": summary["extraction_01b"],
                      "sources": {n: {"coverage": s["coverage"], "pca": s["pca_explained_variance"]["mean"]}
                                  for n, s in summary["sources"].items()}, "run": summary["run"]}, indent=1))


if __name__ == "__main__":
    main()
