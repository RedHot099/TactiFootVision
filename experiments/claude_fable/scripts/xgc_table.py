"""Build the xg-compare shot table, its label file and the label-free coverage (xg-compare ticket 01; CPU, about a minute).

    ./run.sh python scripts/xgc_table.py [--out results/xgc_compare/table]

Inputs, each checked against its completion record or sidecar hash: the confirmatory evaluation table with labels
(`results/xgv_confirmatory/table/confirmatory`, validated against the pre-registration record, whose frame exclusion
it applies), the comparator's StatsBomb features (`results/xgv_comparator/features.parquet`), the offset-0 detector
descriptors (`cache/xgd_full3/E3/features.parquet`), the A1 test predictions (`results/xgd_test/predictions.csv`) and
the raw StatsBomb events of the table's matches (`xgpoc.paths.STATSBOMB`, hash-checked per file).

Writes shots.parquet (no label), labels.parquet (key, goal_sn, on_target_sn, sb_outcome), coverage.csv (per category
level: shots and matches, no goal count), shots.json (hashes, counts, cuts, sources) and complete.json, last.
The script copies the labels into the label file and uses them for nothing else. It computes no category-level outcome or gap.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.compare import records as xr
from xgpoc.compare import sequence as seq
from xgpoc.compare import table as tb
from xgpoc.detector import records
from xgpoc.paths import STATSBOMB
from xgpoc.vendor import evaltable as et

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = ROOT / "results/xgv_confirmatory/table"
PREREG = ROOT / "results/xgv_prereg/preregistration_record.json"
FEATURES = ROOT / "results/xgv_comparator/features.parquet"
E3 = ROOT / "cache/xgd_full3/E3/features.parquet"
A1 = ROOT / "results/xgd_test/predictions.csv"
PRIMARY_EXPECTED = (2721, 135)   # spec: 2,721 non-penalty shots with frames, 135 matches


def verified(path: Path) -> str:
    try:
        return records.verified_file(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgc_compare/table")
    parser.add_argument("--statsbomb", type=Path, default=STATSBOMB)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started = datetime.now(UTC)

    prereg = et.load_preregistration(PREREG)
    joined, joined_meta = et.load_table(CONFIRMATORY, "confirmatory", prereg)
    sources = {"confirmatory": {"path": str(CONFIRMATORY / "confirmatory.parquet"), "sha256": joined_meta["table_sha256"]},
               "preregistration_record": {"path": str(PREREG), "sha256": sha256_file(PREREG),
                                          "frame_exclusions": dict(prereg.frame_exclusions)},
               "features": {"path": str(FEATURES), "sha256": verified(FEATURES)},
               "e3": {"path": str(E3), "sha256": verified(E3), "offset_ms": 0},
               "a1": {"path": str(A1), "sha256": verified(A1)}}
    features = pd.read_parquet(FEATURES)
    e3 = pd.read_parquet(E3, columns=["shot_id", "offset_ms", *tb.E3_DESCRIPTORS])
    a1 = pd.read_csv(A1, usecols=["shot_id", "p_a1"])
    match_ids = joined.statsbomb_match_id.dropna().astype(int).unique()
    shots = seq.load_shots(args.statsbomb, match_ids)
    sources["statsbomb_events"] = {"root": str(args.statsbomb), "pin": json.loads((args.statsbomb / "pin.json").read_text()),
                                   "matches": len(match_ids), "shot_events": len(shots)}

    built = tb.build(joined, dict(prereg.frame_exclusions), features, e3, a1, seq.sequence_table(shots))
    table = built.table
    got = (int(table.in_primary.sum()), int(table.match[table.in_primary].nunique()))
    if got != PRIMARY_EXPECTED:
        raise SystemExit(f"primary population {got} differs from the spec's {PRIMARY_EXPECTED}")
    meta = tb.write(built, args.out, sources)
    complete = {"sidecar": meta, "code": xr.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in ("primary", "combined", "by_population", "matches_by_half", "a1_coverage", "cuts")}))


if __name__ == "__main__":
    main()
