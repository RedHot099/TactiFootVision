"""Build a random-label smoke-test index from local SoccerNet tracking frames."""

import sys
from pathlib import Path

from xgpoc.data.index import make_dummy_index

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/kuba/projects/tactifoot_vision/data/soccernet/tracking/extracted")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "cache/dummy/index.parquet")
df = make_dummy_index(root, out, n=int(sys.argv[3]) if len(sys.argv) > 3 else 240)
print(df["split"].value_counts().to_dict(), "goal rate", df["goal"].mean().round(3), "->", out)
