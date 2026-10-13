"""The xg-compare analysis under the frozen protocol (xg-compare ticket 02; CPU, no frame read).

    ./run.sh --no-sync python scripts/xgc_analysis.py --dry-run     # labels permuted within matches -> results/xgc_compare/dryrun
    ./run.sh --no-sync python scripts/xgc_analysis.py               # the run, once -> results/xgc_compare/analysis

The protocol `research/xgc_compare_protocol.json` is opened through `xgpoc.compare.protocol.load` with the SHA-256
recorded in ticket 01, and its design must equal the code's (`protocol.design`). The shot table, its label file and
its coverage are checked against their completion record and against the hashes the protocol pins; the paper's
recalibration of the primary population (`results/xgv_confirmatory/ladder/coefficients.json`) is checked against
its completion record, and the run's observed main scale must reproduce it. The analysis is
`xgpoc.compare.pipeline.run`. The run refuses to start without a complete dry run, and both refuse an existing
output folder, so the scan's candidate file is frozen once. complete.json, written last, holds the protocol hash, the
input hashes, the candidate file's hash, the code hashes and the output hashes.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.compare import pipeline as pl
from xgpoc.compare import protocol as pr
from xgpoc.compare import records as xr
from xgpoc.compare import table as tb
from xgpoc.detector import records
from xgpoc.vendor import ladder as ld

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "0bc8a44b527efebe43178ed9c726022bd81893483f578a6f067ce588638a974f"   # ticket 01
TABLE = ROOT / "results/xgc_compare/table"
LADDER = ROOT / "results/xgv_confirmatory/ladder"
DRY_RUN_OUT = ROOT / "results/xgc_compare/dryrun"
RUN_OUT = ROOT / "results/xgc_compare/analysis"


def verified(path: Path) -> str:
    try:
        return records.verified_file(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="permute the labels within matches (seed pipeline.DRY_RUN_SEED)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (DRY_RUN_OUT if args.dry_run else RUN_OUT)
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder and the run is made once")
    if not args.dry_run and not (DRY_RUN_OUT / "complete.json").exists():
        raise SystemExit(f"run the dry run first ({DRY_RUN_OUT} has no complete.json)")
    started = datetime.now(UTC)

    try:
        protocol = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if protocol["design"] != json.loads(json.dumps(pr.design())):
        raise SystemExit("the protocol's design differs from xgpoc.compare.protocol.design()")
    inputs = {"protocol": {"path": str(pr.PROTOCOL), "sha256": PROTOCOL_SHA256}}
    for name, path in (("table", TABLE / tb.TABLE_FILE), ("labels", TABLE / tb.LABEL_FILE), ("coverage", TABLE / tb.COVERAGE_FILE),
                       ("paper_coefficients", LADDER / "coefficients.json")):
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": verified(path)}
    pinned = protocol["shot_table"]
    if (inputs["table"]["sha256"], inputs["labels"]["sha256"]) != (pinned["table_sha256"], pinned["labels_sha256"]):
        raise SystemExit("the shot table or its label file is not the one the protocol pins")
    table, meta = tb.read(TABLE)
    if meta["cuts"] != protocol["dictionary"]["cuts"]:
        raise SystemExit("the table's label-free cuts are not the protocol's")
    labels = tb.read_labels(TABLE, meta)
    coverage = pd.read_csv(TABLE / tb.COVERAGE_FILE)
    paper = json.loads((LADDER / "coefficients.json").read_text())["primary"]
    draws = ld.bootstrap(table.match.to_numpy(), pr.N_BOOT, pr.BOOT_SEED)

    context = {"protocol_sha256": PROTOCOL_SHA256, "table_sha256": inputs["table"]["sha256"], "labels_sha256": inputs["labels"]["sha256"]}
    summary = pl.run(table, labels, coverage, draws, out, dry_run=args.dry_run, paper_coefficients=paper, context=context,
                     log=print)
    complete = {"protocol_sha256": PROTOCOL_SHA256, "inputs": inputs, "dry_run": args.dry_run,
                "candidates_sha256": summary["candidates_sha256"], "notice": pl.NOTICE,
                "code": xr.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(out)}
    (out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("family_bh_rejections", "family_by_rejections", "scan_candidates", "scan_replicated")}))


if __name__ == "__main__":
    main()
