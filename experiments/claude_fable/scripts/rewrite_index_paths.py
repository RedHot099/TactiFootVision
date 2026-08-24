"""Rewrite path prefixes in index parquets after moving data. Usage: old_prefix new_prefix files..."""

import sys
from pathlib import Path

import pandas as pd

old, new = sys.argv[1], sys.argv[2]
for f in sys.argv[3:]:
    df = pd.read_parquet(f)
    cols = [c for c in df.columns if c.startswith("frame_path") or c in ("clip_path", "gt_positions")]
    for c in cols:
        df[c] = df[c].astype(str).str.replace(old, new, regex=False)
    df.to_parquet(f, index=False)
    ok = df["frame_path"].map(lambda p: Path(p).exists()).mean() if "frame_path" in df else 1
    print(f, "rewritten", cols, f"frames exist: {ok:.1%}")
