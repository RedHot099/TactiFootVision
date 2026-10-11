"""The confirmatory run of the vendor comparison (xg-vendor ticket 15): verify every recorded hash, join the labels once,
run H1-H4 and fill the pre-registered table shells (`xgpoc.vendor.confirm`).

    ./run.sh python scripts/vendor_confirm.py --record results/xgv_prereg/preregistration_record.json \\
        --table-dir <directory of the combined label-free table> [--table-name eval_table_labels_absent] [--out results/xgv_confirmatory]

1. `confirm.verify` and `confirm.committed_unchanged`: the record, the committed hash list, every pin (combined label-free
   table, split plan, comparator predictions, timing verdict and rule, pairing, ...), every source file of the analysis
   code, the dry-run evidence, the document at the recorded commit and a clean working tree. Any mismatch stops here,
   before a label is read.
2. `confirm.verified_join`: the label-free table is copied into `<out>/table`, the labels are built with the frozen rules
   (`labels.table_labels` on the pinned pair table and the pinned pairing's `sb_unpaired.parquet`, label exclusions
   included) and joined once with the logged step at the current time, which must not precede the record.
3. `vendor_ladder.py`, `vendor_probes.py`, `vendor_leakage.py` with `confirm.analysis_commands` (every parameter from the
   record; the StatsBomb shots and the timing verdict are the pinned files). A failed probe control gate is recorded and
   marked in the shells, not retried.
4. `<out>/shells.md` (`shells.render`) and `<out>/complete.json`. The run happens once: an existing `<out>` refuses.
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
from xgpoc.vendor import confirm, labels, shells
from xgpoc.vendor import evaltable as et

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
DOC = ROOT / "research/xg_vendor_preregistration.md"
HASHES = ROOT / "research/xg_vendor_prereg_hashes.json"
UNPAIRED = ROOT / "results/xgv_pairing/sb_unpaired.parquet"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def run(step: str, args: list[str], log: Path) -> int:
    """One analysis script with this interpreter, output to `log`; stops unless the exit status is accepted."""
    with log.open("w") as handle:
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / step), *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                              env=os.environ | {"PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"})
    if done.returncode not in confirm.ACCEPTED_EXITS[step]:
        sys.exit(f"{step} failed (exit {done.returncode}); see {log}. A rerun needs the same inputs and a note in the deviation log.")
    return done.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", type=Path, required=True, help="the pre-registration record (vendor_prereg_hashes.py --record)")
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--table-name", default="eval_table_labels_absent")
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--hashes", type=Path, default=HASHES)
    parser.add_argument("--unpaired", type=Path, default=UNPAIRED)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgv_confirmatory")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        sys.exit(f"{out} exists; the confirmatory run happens once")
    started = time.time()
    inputs = {"root": ROOT, "document": args.doc.resolve(), "hashes_path": args.hashes.resolve(), "record_path": args.record,
              "table_dir": args.table_dir, "table_name": args.table_name}
    try:
        verified = confirm.verify(**inputs)
        confirm.committed_unchanged(ROOT, verified, inputs["document"], inputs["hashes_path"], git)
        pairs, unpaired = confirm.label_sources(ROOT, verified.hashes, args.unpaired)
    except confirm.ConfirmationRefused as refusal:
        sys.exit(f"REFUSED, no label joined: {refusal}")
    print(f"verified: record {verified.prereg.sha256[:12]} (commit {verified.prereg.commit[:12]}), "
          f"{len(verified.hashes['code'])} code files, {len(verified.hashes['pins'])} pins", flush=True)

    verified, joined = confirm.verified_join(inputs, lambda table, _: labels.table_labels(table, pairs, unpaired), out, datetime.now(UTC))
    def pinned(name: str) -> Path:
        path = Path(verified.hashes["pins"][name]["path"])
        return path if path.is_absolute() else ROOT / path

    commands = confirm.analysis_commands(out / "table", confirm.JOINED, args.record.resolve(), pinned("statsbomb_shots"), pinned("timing_verdict"), out)
    exits = {step: run(step, step_args, out / f"{step.removeprefix('vendor_').removesuffix('.py')}.log") for step, step_args in commands.items()}
    params = verified.hashes["parameters"]
    filled = shells.render(params, shells.Outputs(out / "ladder", out / "probes", out / "leakage"), verified.prereg.margin)
    (out / "shells.md").write_text(f"# Szablony tabel wypełnione w przebiegu konfirmacyjnym\n\nδ = {verified.prereg.margin:g}.\n\n{filled}")
    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and not p.is_symlink())
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), "label": et.RUN_LABELS["confirmatory"],
              "identity": {"analysis": "xg-vendor confirmatory run (ticket 15)", "record": {"path": str(args.record), "sha256": sha256_file(args.record)},
                           "prereg": et.preregistration_record(verified.prereg), "hash_list_sha256": sha256_file(inputs["hashes_path"]),
                           "label_free_table_sha256": verified.prereg.label_free_table_sha256, "label_join": joined["label_join"],
                           "joined_table_sha256": joined["table_sha256"], "unpaired_sha256": sha256_file(args.unpaired),
                           "sources": verified.hashes["code"], "code": code_version(ROOT)},
              "exits": exits,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "seconds": round(time.time() - started, 1)},
              "files": {name: sha256_file(out / name) for name in names}}
    (out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}; shells in {out / 'shells.md'}")


if __name__ == "__main__":
    main()
