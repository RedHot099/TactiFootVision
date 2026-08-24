"""Merge per-source sample indexes into one contract-compliant parquet."""

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/kuba/projects/tactifoot_vision/data/xg_shots_sample/claude_fable")
parts = [pd.read_parquet(p) for p in sorted(root.glob("*/index.parquet"))]
df = pd.concat(parts, ignore_index=True)
df.to_parquet(root / "index.parquet", index=False)
print(f"{len(df)} shots, {int(df.goal.sum())} goals, sources: {df.source.value_counts().to_dict()}")
print("->", root / "index.parquet")
