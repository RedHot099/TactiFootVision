"""Compare PoCs on the common test set with game-level bootstrap confidence intervals.

    ./run.sh python scripts/compare.py --index <index.parquet> --results results/<run>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.eval.metrics import evaluate


def bootstrap(df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    games = df["game"].unique()
    by_game = {g: d for g, d in df.groupby("game")}
    samples = {"brier": [], "log_loss": [], "auc": []}
    for _ in range(n_boot):
        pick = rng.choice(games, len(games), replace=True)
        s = pd.concat([by_game[g] for g in pick])
        m = evaluate(s["goal"].to_numpy(), s["p"].to_numpy())
        for k in samples:
            samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)[["shot_id", "game"]]
    rows = []
    for f in sorted(Path(a.results).glob("*_test_predictions.csv")):
        p = pd.read_csv(f).merge(idx, on="shot_id")
        m = evaluate(p["goal"].to_numpy(), p["p"].to_numpy())
        ci = bootstrap(p)
        rows.append({"model": f.name.replace("_test_predictions.csv", ""), "n": m["n"], "games": p["game"].nunique(),
                     "brier": m["brier"], "brier_ci": ci["brier"], "bss": m["brier_skill_score"],
                     "log_loss": m["log_loss"], "log_loss_ci": ci["log_loss"], "auc": m["auc"], "auc_ci": ci["auc"], "ece": m["ece"]})
    out = pd.DataFrame(rows).sort_values("log_loss")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(Path(a.results) / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
