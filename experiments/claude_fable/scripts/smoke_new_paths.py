"""Smoke test for the pre-merge additions on a tiny subsample (minutes, not hours).

    ./run.sh python scripts/smoke_new_paths.py

Covers: overlay masking end to end, flip TTA in C, the unfrozen-backbone PoC G,
and that validation predictions land next to the test ones.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_poc as rp  # noqa: E402

from xgpoc.data.index import load_index  # noqa: E402
from xgpoc.models.common import TrainConfig  # noqa: E402

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
OUT = Path("results/smoke_new")

df = load_index(Path(INDEX))
small = pd.concat([g.head(80) for _, g in df.groupby("split")]).reset_index(drop=True)
tr, va, te = (small[small.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
print(f"smoke on {len(tr)}/{len(va)}/{len(te)}")

cfg = TrainConfig(epochs=2, max_minutes=2, batch_size=8, seed=0)
rp.run_c(tr, va, te, Path("cache/smoke"), OUT, cfg, mask="overlay", tta=True, tag="_overlay_tta")
rp.run_g(tr, va, te, Path("cache/smoke"), OUT, cfg, backbone="dinov2_b", n_blocks=2, tta=True, tag="_tta")

for f in sorted(OUT.glob("*_predictions.csv")):
    print(f.name, len(pd.read_csv(f)))
