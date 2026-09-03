"""Cache frozen backbone tokens for every shot in the index (all splits), so that the
D head and the ensemble can compare backbones in seconds.

    ./run.sh python scripts/extract_features.py --backbone dinov3_l --mask overlay

Output: cache/full_v1/features_<backbone>[_<mask>][_flip].npz - the same naming run_poc
uses, so `run_poc.py --poc d --backbone dinov3_l --mask overlay` picks it up. Gated
weights (DINOv3) need HF_TOKEN in the environment; see .env.
"""

import argparse
import os
import time
from pathlib import Path

from xgpoc.data.index import load_index
from xgpoc.models.d_foundation import BACKBONES, extract_patch_features
from xgpoc.paths import INDEX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--mask", default="none", choices=["none", "overlay"])
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("cache/full_v1"))
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    if "dinov3" in a.backbone and not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN not set - DINOv3 weights are gated", flush=True)
    df = load_index(Path(a.index))
    name = f"features_{a.backbone}" + ("" if a.mask == "none" else f"_{a.mask}") + ("_flip" if a.flip else "")
    out = a.cache / f"{name}.npz"
    t0 = time.time()
    feats = extract_patch_features(df, out, a.backbone, batch_size=a.batch_size, mask=a.mask, flip=a.flip)
    shape = next(iter(feats.values())).shape
    print(f"{out}: {len(feats)} shots, tokens {shape}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
