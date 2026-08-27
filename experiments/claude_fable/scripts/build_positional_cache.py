"""Usage: ./run.sh python scripts/build_positional_cache.py <index.parquet> <cache_dir> [--limit N]"""

import argparse
from pathlib import Path

from xgpoc.data.index import load_index
from xgpoc.features.cache import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("index")
ap.add_argument("cache_dir")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ball-fallback", action="store_true", help="retry ball on -400/-800 ms frames")
a = ap.parse_args()
df = load_index(Path(a.index))
if a.limit:
    df = df.head(a.limit)
cols = ("frame_path__400", "frame_path__800")
# itertuples mangles +/- in column names; resolve real names dynamically
if a.ball_fallback:
    real = [c for c in df.columns if c in ("frame_path_-400", "frame_path_-800")]
    df = df.rename(columns={c: c.replace("-", "m") for c in real})
    cols = tuple(c.replace("-", "m") for c in real)
else:
    cols = ()
build_cache(df, Path(a.cache_dir), fallback_cols=cols)
print("cached", len(df), "shots to", a.cache_dir)
