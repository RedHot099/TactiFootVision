"""Learning curve: train on nested 25 / 50 / 75 / 100 % subsets of the training *games*,
validate on the official valid split, evaluate on the test split.

    ./run.sh python scripts/run_learning_curve.py --stage dl_mask --seeds 3
    ./run.sh python scripts/run_learning_curve.py --stage g_l4 --seeds 1

This is the decision gate of the roadmap: a curve still climbing steeply at 100 % says
that more labelled games beat any architecture change; a flat one says the opposite.
Subsets are nested (the 25 % games are inside the 50 % ones) and drawn once with a fixed
seed, so the only thing that varies along the curve is the amount of data.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_poc as rp
from run_v12 import CACHE, STAGES, done
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig
from xgpoc.paths import INDEX

OUT = Path("results/v12_lc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="dl_mask")
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    df = rp.add_targets(load_index(Path(a.index)))
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    games = np.array(sorted(tr.game.unique()))
    perm = np.random.default_rng(0).permutation(games)
    runner, name, cfg_over, kw = STAGES[a.stage]
    for frac in a.fracs:
        keep = set(perm[: int(round(frac * len(perm)))])
        sub = tr[tr.game.isin(keep)].reset_index(drop=True)
        for s in range(a.seeds):
            out = a.out / a.stage / f"frac{int(round(frac * 100)):03d}" / f"seed{s}"
            if done(out, name):
                continue
            print(f"=== {out} ({time.strftime('%H:%M')}) {len(keep)} games / {len(sub)} shots / {int(sub.goal.sum())} goals ===", flush=True)
            t0 = time.time()
            rp.RUNNERS[runner](sub, va, te, CACHE, out, TrainConfig(seed=s, **cfg_over), **kw)
            print(f"--- done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    for f in sorted((a.out / a.stage).glob("frac*/seed*/*_metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"frac": f.parent.parent.name, "seed": f.parent.name, "auc": m["auc"], "log_loss": m["log_loss"],
                     "bss": m["brier_skill_score"], "best_epoch": m.get("best_epoch")})
    t = pd.DataFrame(rows)
    if len(t):
        agg = t.groupby("frac").agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), ll_mean=("log_loss", "mean"),
                                    bss_mean=("bss", "mean"), seeds=("seed", "size"))
        agg.to_csv(a.out / f"{a.stage}_learning_curve.csv")
        print(agg.round(4).to_string())
    print(f"LEARNING CURVE DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
