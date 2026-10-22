"""Builds the feet_t0 E3 cache, cache/xgd_full4_feet_t0/E3, with the library rule `xgpoc.detector.location.FEET_T0`
(frozen in research/feet_location_rule.md): the retrained detector's cached frames (cache/xgd_full4/E3, unchanged,
linked) with the raw features recomputed under the feet rule (ball order 0, -400, -800 ms; shooter = the player whose
image box is nearest the ball's image centre; shot location = his feet). CPU, label-free.

    ./run.sh --no-sync python scripts/feet_e3.py

Refuses an existing folder. Writes features.parquet (`location.feature_rows`, one row per shot at t0 and -400 ms) and
complete.json, last. It replaces the 2026-10-22 monkey-patch build (cache/xgd_full5_feet_t0/E3, from the deleted
scripts/feet_patch.py); scripts/location_check.py checks the two are equal.
"""

import json
import os
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import location as loc
from xgpoc.detector import records

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache/xgd_full4/E3"
DST = ROOT / "cache/xgd_full4_feet_t0/E3"
RULE = loc.FEET_T0


def main() -> None:
    DST.mkdir(parents=True, exist_ok=False)
    os.symlink(SRC / "frames.parquet", DST / "frames.parquet")
    frames = pd.read_parquet(SRC / "frames.parquet")
    features = loc.feature_rows(frames, RULE)
    features.to_parquet(DST / "features.parquet", index=False)
    record = {"rule": {"name": RULE.name, "source": list(RULE.source.source), "situation": RULE.situation.__name__,
                       "e3_flags_at": list(RULE.e3_flags_at), "definition": "research/feet_location_rule.md"},
              "frames_from": {"path": str(SRC.relative_to(ROOT)), "sha256": records.verified_file(SRC / "frames.parquet")},
              "rows": len(features), "shots": int(features.shot_id.nunique()),
              "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
              "files": {n: sha256_file(DST / n) for n in ("frames.parquet", "features.parquet")}}
    (DST / "complete.json").write_text(json.dumps(record, indent=1) + "\n")
    print(len(features), "rows,", record["shots"], "shots;", record["files"])


if __name__ == "__main__":
    main()
