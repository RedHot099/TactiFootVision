"""Builds cache/xgd_full5_feet/E3: the retrained detector's cached frames (xgd_full4, unchanged) with features
recomputed under `feet_patch` (shot location = shooter's feet). Descriptive variant, CPU."""

import json
import os
import sys
from pathlib import Path

import feet_patch
import pandas as pd

T0 = "--t0" in sys.argv
feet_patch.apply(t0_ball=T0)
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.detector import extract3 as x3  # noqa: E402
from xgpoc.detector.features import missing_features, shot_features  # noqa: E402
from xgpoc.stack import inputs as si  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache/xgd_full4/E3"
DST = ROOT / ("cache/xgd_full5_feet_t0/E3" if T0 else "cache/xgd_full5_feet/E3")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=False)
    os.symlink(SRC / "frames.parquet", DST / "frames.parquet")
    frames = pd.read_parquet(SRC / "frames.parquet")
    rows = []
    for shot_id, offset, filled, _ in si.replay(frames):
        raw = (
            shot_features(filled, x3.COMMON["homography_min_confidence"])
            if filled is not None
            else missing_features()
        )
        rows.append({"shot_id": shot_id, "offset_ms": offset} | raw)
    pd.DataFrame(rows).to_parquet(DST / "features.parquet", index=False)
    (DST / "complete.json").write_text(
        json.dumps(
            {
                "variant": "feet_patch (shot location = shooter's feet, image-nearest player to the ball)",
                "t0_ball": T0,
                "frames_from": str(SRC),
                "files": {
                    n: sha256_file(DST / n)
                    for n in ("frames.parquet", "features.parquet")
                },
            },
            indent=1,
        )
    )
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
