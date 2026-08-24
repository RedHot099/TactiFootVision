"""Sanity report for a shot index and (optionally) its positional cache.

    ./run.sh python scripts/check_index_quality.py <index.parquet> [<cache_dir>/positional]

Prints what step 1 of the handoff needs: split sizes, goal rate per split/source,
missing frames, and (with cache) homography / ball / team-detection rates.
"""

import sys
from pathlib import Path

import pandas as pd

from xgpoc.data.index import REQUIRED, load_index

idx_path = Path(sys.argv[1])
raw = pd.read_parquet(idx_path)
print("columns:", list(raw.columns))
missing_cols = [c for c in REQUIRED if c not in raw.columns]
if missing_cols:
    sys.exit(f"MISSING REQUIRED COLUMNS: {missing_cols}")
df = load_index(idx_path)
print(f"rows: {len(raw)}  with existing frame: {len(df)}  missing frames: {len(raw) - len(df)}")
print("\nper split:")
print(df.groupby("split").agg(shots=("shot_id", "size"), games=("game", "nunique"), goal_rate=("goal", "mean")).round(3))
print("\nper source:")
print(df.groupby("source").agg(shots=("shot_id", "size"), games=("game", "nunique"), goal_rate=("goal", "mean")).round(3))
if "visibility" in df:
    print("\nvisibility:", df["visibility"].value_counts().to_dict())
if "is_penalty" in df:
    print("penalties:", int(df["is_penalty"].sum()))
leak = set(df[df.split == "train"].game) & set(df[df.split == "test"].game)
print("\ngames shared between train and test:", len(leak), "(must be 0)")

if len(sys.argv) > 2:
    tab = pd.read_parquet(Path(sys.argv[2]) / "tabular.parquet").merge(df[["shot_id", "split", "source", "goal"]], on="shot_id")
    print(f"\npositional cache rows: {len(tab)} / {len(df)}")
    g = tab.groupby("source").agg(homography_ok=("homography_ok", "mean"), has_ball=("has_ball", "mean"),
                                  team_known=("attack_team_known", "mean"), gk_present=("gk_present", "mean"),
                                  n_players=("n_players", "mean"), n_keypoints=("n_keypoints", "mean"))
    print(g.round(3))
    ok = tab[(tab.homography_ok > 0) & (tab.has_ball > 0)]
    print(f"\nusable for A/B (homography AND ball): {len(ok)} ({len(ok) / max(len(tab), 1):.1%})")
    print("dist_to_goal on usable rows (m):", ok["dist_to_goal"].describe()[["mean", "25%", "50%", "75%"]].round(1).to_dict())
    print("goal rate by dist bucket (usable rows):")
    print(ok.groupby(pd.cut(ok["dist_to_goal"], [0, 8, 12, 18, 25, 40, 80]))["goal"].agg(["size", "mean"]).round(3))
