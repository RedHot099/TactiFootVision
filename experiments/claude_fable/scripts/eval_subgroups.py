"""Subgroup metrics and reliability diagrams for the final models.

    ./run.sh python scripts/eval_subgroups.py --preds results/final/*.csv --out results/subgroups

A single test-set number hides where a model actually works. This splits the test set
by distance, league, season, annotation visibility, penalty vs open play, camera
framing (grass ratio) and - following the leakage audit - by whether the home team of
the test game was ever seen in training, which is the venue-memorisation question.

Every subgroup number is reported with the count and the observed goal rate, because
a Brier skill score computed against a subgroup's own base rate is not comparable
across subgroups with very different base rates.
"""

import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xgpoc.eval.metrics import evaluate  # noqa: E402

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
TAB = "cache/full_v2_fallback/positional/tabular.parquet"
DIST_BINS = [0, 6, 11, 16, 25, 200]
DIST_LABELS = ["0-6 m", "6-11 m", "11-16 m", "16-25 m", "25+ m"]


def context(index: str, tab_path: str) -> pd.DataFrame:
    df = pd.read_parquet(index).drop_duplicates("shot_id")
    df["league"] = df.game.str.split("/").str[0]
    df["season"] = df.game.str.split("/").str[1]
    df["home"] = df.game.str.extract(r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ ")[0]
    seen = set(df[df.split == "train"].home.dropna())
    df["venue"] = np.where(df.home.isin(seen), "home team seen in train", "unseen home team")
    df["shot_type"] = np.where(df.is_penalty, "penalty", "open play")
    df["framing"] = pd.qcut(df.grass_ratio, 4, labels=["grass q1", "q2", "q3", "q4"])
    tab = pd.read_parquet(tab_path).drop_duplicates("shot_id")
    if {"dist_to_goal", "homography_ok", "has_ball"} <= set(tab.columns):
        d = tab[["shot_id", "dist_to_goal", "homography_ok", "has_ball"]].copy()
        # a distance from a failed homography or a missing ball is a guess, not a measurement
        d["distance"] = pd.cut(d.dist_to_goal, DIST_BINS, labels=DIST_LABELS).astype(object)
        d.loc[(d.homography_ok < 0.5) | (d.has_ball < 0.5), "distance"] = "geometry unavailable"
        df = df.merge(d[["shot_id", "distance"]], on="shot_id", how="left")
    return df


def subgroup_table(pred: pd.DataFrame, ctx: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = pred.merge(ctx, on="shot_id")
    rows = []
    for col in cols:
        for value, g in d.groupby(col, observed=True):
            if len(g) < 30:
                continue
            m = evaluate(g.goal_x.to_numpy() if "goal_x" in g else g.goal.to_numpy(), g.p.to_numpy())
            rows.append({"dimension": col, "group": str(value), "n": m["n"], "goal_rate": m["goal_rate"],
                         "mean_xg": float(g.p.mean()), "auc": m["auc"], "log_loss": m["log_loss"],
                         "brier": m["brier"], "bss": m["brier_skill_score"], "ece": m["ece"]})
    return pd.DataFrame(rows)


def reliability(preds: dict[str, pd.DataFrame], out: Path, n_bins: int = 8) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    edges = np.quantile(np.concatenate([p.p for p in preds.values()]), np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    for name, p in preds.items():
        idx = np.clip(np.digitize(p.p, edges) - 1, 0, len(edges) - 2)
        xs, ys, ns = [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(p.p[m].mean())
            ys.append(p.goal[m].mean())
            ns.append(m.sum())
        axes[0].plot(xs, ys, "o-", label=name, alpha=0.85)
        axes[1].hist(p.p, bins=30, histtype="step", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="idealna kalibracja")
    lim = max(0.6, max(max(p.p.max() for p in preds.values()), 0.6))
    axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="przewidywane xG (średnia w kubełku)",
                ylabel="zaobserwowana częstość goli", title="Krzywa niezawodności (kubełki kwantylowe)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="przewidywane xG", ylabel="liczba strzałów", title="Rozkład predykcji", yscale="log")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="prediction CSVs (globs allowed)")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--tab", default=TAB)
    ap.add_argument("--out", type=Path, default=Path("results/subgroups"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for pat in a.preds for f in sorted(glob.glob(pat))]
    names = a.names or [f.stem.replace("_test_predictions", "") for f in files]
    preds = {n: pd.read_csv(f).drop_duplicates("shot_id") for n, f in zip(names, files)}
    ctx = context(a.index, a.tab)

    cols = ["distance", "league", "season", "visibility", "shot_type", "venue", "framing"]
    tables = []
    for name, p in preds.items():
        t = subgroup_table(p[["shot_id", "goal", "p"]], ctx.drop(columns=["goal"]), cols)
        tables.append(t.assign(model=name))
    out = pd.concat(tables)[["model", "dimension", "group", "n", "goal_rate", "mean_xg", "auc",
                             "log_loss", "brier", "bss", "ece"]]
    out.to_csv(a.out / "subgroups.csv", index=False)
    reliability(preds, a.out / "reliability.png")
    pd.set_option("display.width", 220)
    for dim in cols:
        s = out[out.dimension == dim]
        if len(s):
            print(f"\n--- {dim} ---")
            print(s.drop(columns="dimension").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
