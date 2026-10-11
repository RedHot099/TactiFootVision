"""Dry run of the whole confirmatory analysis on labels permuted within matches (xg-vendor ticket 14).

    ./run.sh python scripts/vendor_dryrun.py [--pin primary_label_free_table] [--table-dir results/xgv_eval_primary]
        [--table-name eval_table_labels_absent] [--out results/xgv_dryrun_primary]

Every parameter comes from the pre-registration document's parameter block (`xgpoc.vendor.prereg`); while the margin is
a placeholder the dry run uses the block's `dry_run_margin`. Steps:
1. Open the label-free table (validated, confirmatory, labels absent) and check its sha256 against the pin `--pin`.
2. Write the same table with `table_kind` "dryrun" (nothing else differs; checked) into `<out>/table`, with links to
   its embedding files, so the analyses read it exactly as they will read the confirmatory table.
3. Build the labels of every row with the confirmatory rules (`labels.table_labels`: the frozen pair table, whose sha256
   must be the one the table records, and for secondary rows the pairing's population table `sb_unpaired.parquet`),
   permute them within matches (`xgpoc.vendor.dryrun.permute_labels`) and join them with the logged step
   (`evaltable.join_labels`). No unpermuted label is joined to a prediction or printed.
4. Write a draft pre-registration record of the current document (`prereg.draft_record`) and the StatsBomb shot rows of
   the table's events with their payload permuted within matches (`dryrun.permute_shots`).
5. Run `scripts/vendor_ladder.py`, `vendor_probes.py` and `vendor_leakage.py` exactly as the confirmatory run does
   (`confirm.analysis_commands`: every parameter from the draft record, the permuted shots, the timing verdict) into
   `<out>/ladder`, `probes`, `leakage`. A failed probe control gate is recorded, not fatal (the shells mark it).
6. Fill the table shells (`xgpoc.vendor.shells`) into `<out>/shells.md` and write `<out>/complete.json`.
Everything written is labelled DRY RUN; none of it is a result.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import code_version, sha256_file
from xgpoc.vendor import confirm, dryrun, labels, shells
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import prereg as pr
from xgpoc.vendor.probe_targets import SHOT_INPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
DOC = ROOT / "research/xg_vendor_preregistration.md"
PAIRS = ROOT / "results/xgv_pairing/pairs.parquet"
UNPAIRED = ROOT / "results/xgv_pairing/sb_unpaired.parquet"
SHOTS = Path("/home/kuba/projects/football_data/statsbomb_open_v1/tables/shots.parquet")
VERDICT = ROOT / "results/xgv_timing_source/timing_verdict.json"
LABEL_FREE, JOINED, LOG = "dryrun_labels_absent", "dryrun", "label_join_log.jsonl"
BANNER = "DRY RUN on labels and probe targets permuted within matches. These numbers are not results."


def dryrun_table(table: pd.DataFrame) -> pd.DataFrame:
    """The label-free table as a dry-run table; refuses anything but a change of `table_kind`."""
    out = table.assign(table_kind="dryrun")
    if not out.drop(columns="table_kind").equals(table.drop(columns="table_kind")):
        raise AssertionError("the dry-run table differs from the label-free table beyond table_kind")
    return out


def run(step: str, args: list[str], log: Path) -> int:
    """Run one analysis script with this interpreter; its output goes to `log`. Returns the exit status when it is 0 or
    the probe script's control-gate failure; anything else stops the dry run."""
    started = time.time()
    with log.open("w") as handle:
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / step), *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                              env=os.environ | {"PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"})
    if done.returncode not in confirm.ACCEPTED_EXITS[step]:
        sys.exit(f"{step} failed (exit {done.returncode}); see {log}")
    print(f"{step}: {time.time() - started:.0f} s (exit {done.returncode})", flush=True)
    return done.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--pin", default="primary_label_free_table", choices=pr.TABLE_PINS)
    parser.add_argument("--table-dir", type=Path, default=ROOT / "results/xgv_eval_primary")
    parser.add_argument("--table-name", default="eval_table_labels_absent")
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--unpaired", type=Path, default=UNPAIRED, help="the pairing's population table of the unpaired StatsBomb shots")
    parser.add_argument("--shots", type=Path, default=SHOTS)
    parser.add_argument("--timing-verdict", type=Path, default=VERDICT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgv_dryrun_primary")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; delete it to rerun")
    started = time.time()

    params = pr.read_parameters(args.doc.read_text())
    table, meta = et.load_table(args.table_dir, args.table_name)
    if meta["table_kind"] != "confirmatory" or meta["labels_state"] != "absent":
        sys.exit("the dry run starts from the confirmatory label-free table")
    if meta["table_sha256"] != params["pins"][args.pin]["sha256"]:
        sys.exit(f"the table's sha256 {meta['table_sha256']} is not the pin {args.pin} of the document")
    pairs_sha = sha256_file(args.pairs)
    if pairs_sha != meta["sources"].get("pairs.parquet"):
        sys.exit("the pair table is not the one the label-free table was built from")

    table_dir = out / "table"
    table_dir.mkdir(parents=True)
    for entry in meta["embedding_sets"].values():   # relative embedding files are read from the table directory
        if not Path(entry["file"]).is_absolute():
            (table_dir / entry["file"]).symlink_to(args.table_dir.resolve() / entry["file"])
    notes = (BANNER, f"label-free table {meta['table_sha256']} with table_kind dryrun; labels permuted within matches "
             f"(seed {params['permutation_seed']})")
    sources = {"label_free_table": meta["table_sha256"], "pairs.parquet": pairs_sha}
    et.write_table(dryrun_table(table), table_dir, LABEL_FREE, kind="dryrun", embedding_sets=meta["embedding_sets"], notes=notes, sources=sources)
    unpaired = pd.read_parquet(args.unpaired) if table.population.ne("primary").any() else None
    permuted = dryrun.permute_labels(labels.table_labels(table, pd.read_parquet(args.pairs), unpaired), table, params["permutation_seed"])
    joined_meta = et.join_labels(table_dir, LABEL_FREE, permuted, JOINED, out / LOG, now=datetime.now(UTC))

    prereg = pr.draft_record(args.doc, args.pin, datetime.now(UTC))
    prereg_path, shots_path = out / "prereg_draft.json", out / "shots_permuted.parquet"
    prereg_path.write_text(json.dumps(et.preregistration_record(prereg), indent=2, sort_keys=True) + "\n")
    shots = pd.read_parquet(args.shots, columns=SHOT_INPUT_COLUMNS)
    dryrun.permute_shots(shots, table, params["permutation_seed"]).to_parquet(shots_path, index=False)

    exits = {step: run(step, step_args, out / f"{step.removeprefix('vendor_').removesuffix('.py')}.log")
             for step, step_args in confirm.analysis_commands(table_dir, JOINED, prereg_path, shots_path, args.timing_verdict, out).items()}

    filled = shells.render(params, shells.Outputs(out / "ladder", out / "probes", out / "leakage"), prereg.margin)
    (out / "shells.md").write_text(f"# Szablony tabel wypełnione w dry run\n\n**{BANNER}**\n\nδ w tym przebiegu: {prereg.margin:g} "
                                   f"({'parametr dokumentu' if params['margin'] is not None else 'dry_run_margin; δ jest jeszcze do decyzji'}).\n\n"
                                   f"{filled}")
    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and not p.is_symlink())
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), "label": et.RUN_LABELS["dryrun"],
              "identity": {"analysis": "xg-vendor dry run (ticket 14)", "document": {"path": str(args.doc), "sha256": prereg.sha256},
                           "pin": args.pin, "label_free_table": {"directory": str(args.table_dir), "name": args.table_name,
                                                                 "sha256": meta["table_sha256"]},
                           "dryrun_table": {"label_free_sha256": joined_meta["label_join"]["label_free_sha256"],
                                            "joined_sha256": joined_meta["table_sha256"]},
                           "inputs": {"pairs": {"file": str(args.pairs), "sha256": pairs_sha}, "shots": {"file": str(args.shots), "sha256": sha256_file(args.shots)},
                                      "unpaired": None if unpaired is None else {"file": str(args.unpaired), "sha256": sha256_file(args.unpaired)},
                                      "timing_verdict": {"file": str(args.timing_verdict), "sha256": sha256_file(args.timing_verdict)}},
                           "parameters": {**{k: v for k, v in params.items() if k != "pins"}, "margin_used": prereg.margin},
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in pr.code_closure(ROOT)}, "code": code_version(ROOT)},
              "exits": exits,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "seconds": round(time.time() - started, 1)},
              "files": {name: sha256_file(out / name) for name in names}}
    (out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out} ({record['runtime']['seconds']} s); shells in {out / 'shells.md'}")


if __name__ == "__main__":
    main()
