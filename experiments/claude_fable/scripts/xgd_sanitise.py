"""Sanitise the D3 re-extraction for approach 1 (xg-detector ticket 03, council C1; CPU, a few seconds).

    ./run.sh python scripts/xgd_sanitise.py [--extract cache/xgd_full2/D3] [--out cache/xgd_sanitised/D3]

Reads the extraction's frames and features (each checked against its completion record), replays the extraction's
features from the frames (they must equal the cached ones exactly, else nothing is written), and writes one row per
shot and study offset (t0, -400 ms): the sanitised columns (`xgpoc.detector.sanitation`, rules frozen in ticket 03)
and the raster points of A2. Everything runs inside `no_statsbomb_reads()`; no label, StatsBomb or audit file is read.

Writes <out>/sanitised.parquet and <out>/complete.json (last; an existing record is refused).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import records
from xgpoc.detector import sanitation as sn
from xgpoc.detector.features import COLUMNS
from xgpoc.detector.guard import no_statsbomb_reads

ROOT = Path(__file__).resolve().parents[1]


def coverage(table: pd.DataFrame) -> dict:
    """Label-free shares per offset: the sanitised flags, what was masked, and the mean detection counts."""
    out = {}
    for offset, rows in table.groupby("offset_ms", sort=False):
        out[str(offset)] = {c: round(float(rows[c].mean()), 4) for c in [*sn.FLAGS, *sn.INDICATORS, *sn.COUNTS, "aligned"]}
        out[str(offset)]["def_cone"] = round(float(rows.ff_def_cone.notna().mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--extract", type=Path, default=ROOT / "cache/xgd_full2/D3")
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgd_sanitised/D3")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out / 'complete.json'} exists; use a new --out")

    with no_statsbomb_reads():
        try:
            inputs = {name: {"path": str(args.extract / name), "sha256": records.verified_file(args.extract / name)}
                      for name in ("frames.parquet", "features.parquet")}
        except ValueError as e:
            raise SystemExit(str(e)) from e
        extraction = (args.extract / "complete.json").read_text()
        config = ex.CONFIGS[json.loads(extraction)["config"]["name"]]
        frames = pd.read_parquet(args.extract / "frames.parquet")
        cached = pd.read_parquet(args.extract / "features.parquet").set_index(["shot_id", "offset_ms"]).sort_index()
        replayed = ex.replay_features(frames, config, ex.STUDY_OFFSETS).set_index(["shot_id", "offset_ms"]).sort_index()
        if not cached.index.equals(replayed.index) or (n := int(ex.feature_differences(cached, replayed, COLUMNS, 0.0).any(axis=1).sum())):
            raise SystemExit(f"the replayed features differ from the cache ({n if cached.index.equals(replayed.index) else 'rows'})")
        started = datetime.now(UTC)
        table = sn.sanitised_table(frames, config)
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.out / "sanitised.parquet", index=False)
        record = {"config": config.name, "rule": {"margin_m": sn.MARGIN, "goalkeeper_zone": "attacked penalty area + margin",
                                                  "offsets_ms": list(ex.STUDY_OFFSETS)},
                  "inputs": inputs | {"complete.json": {"path": str(args.extract / "complete.json"),
                                                        "sha256": sha256_file(args.extract / "complete.json")}},
                  "replay": {"rows": len(replayed), "rows_differing": 0},
                  "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "run": {"started": started.isoformat(timespec="seconds"),
                          "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                  "rows": len(table), "coverage": coverage(table), "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(ex.json_text(record))
    print(ex.json_text(record["coverage"]))


if __name__ == "__main__":
    main()
