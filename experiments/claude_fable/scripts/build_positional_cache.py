"""Usage: ./run.sh python scripts/build_positional_cache.py <index.parquet> <cache_dir> [--limit N]"""

import argparse
from pathlib import Path

from xgpoc.data.index import load_index
from xgpoc.features.cache import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("index")
ap.add_argument("cache_dir")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
df = load_index(Path(a.index))
if a.limit:
    df = df.head(a.limit)
build_cache(df, Path(a.cache_dir))
print("cached", len(df), "shots to", a.cache_dir)
