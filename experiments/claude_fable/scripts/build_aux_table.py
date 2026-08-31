"""Build aux.parquet (P0/P1.1 features) from the fallback positional cache."""

import sys
from pathlib import Path


from xgpoc.data.index import load_index
from xgpoc.features.aux import build_aux_table
from xgpoc.features.cache import load_cache

index = load_index(Path(sys.argv[1] if len(sys.argv) > 1 else "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"))
cache_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "cache/full_v2_fallback/positional")
_, tokens, boxes = load_cache(cache_dir)
aux = build_aux_table(index, tokens, boxes)
out = cache_dir / "aux.parquet"
aux.to_parquet(out, index=False)
print(len(aux), "rows ->", out)
print(aux.drop(columns="shot_id").mean().round(3).to_string())
