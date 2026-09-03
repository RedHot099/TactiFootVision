"""Paired, game-clustered bootstrap for differences between models.

    ./run.sh python scripts/significance.py --preds a.csv b.csv --names A B --out results/significance

Overlapping confidence intervals do not mean "no difference": the models are scored on
the *same* shots, so the interesting quantity is the paired difference, whose variance
is far smaller than that of either model alone. Games are resampled as whole clusters
because shots from one match share camera, teams and conditions, and both models are
re-evaluated on the identical resample, which preserves the pairing.

Reported per pair: median delta, 95 % interval and P(delta favours the first model).
No p-value dressing - with 100 test games the honest statement is the interval.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.eval.metrics import evaluate

from xgpoc.paths import INDEX as _INDEX  # noqa: E402
INDEX = str(_INDEX)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, games: pd.DataFrame, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """a, b: shot_id/goal/p frames. Positive delta = a better (AUC up, losses down)."""
    if "goal" in games.columns:  # labels come from the index, never from the CSVs
        m = (a.drop(columns="goal", errors="ignore").merge(b.drop(columns="goal", errors="ignore"), on="shot_id", suffixes=("_a", "_b"))
             .merge(games, on="shot_id"))
    else:
        m = a.merge(b, on=["shot_id", "goal"], suffixes=("_a", "_b")).merge(games, on="shot_id")
    by_game = {g: d for g, d in m.groupby("game")}
    keys = np.array(list(by_game))
    rng = np.random.default_rng(seed)
    deltas = {"auc": [], "log_loss": [], "brier": []}
    for _ in range(n_boot):
        s = pd.concat([by_game[g] for g in rng.choice(keys, len(keys), replace=True)])
        y = s.goal.to_numpy()
        if y.min() == y.max():
            continue
        ma, mb = evaluate(y, s.p_a.to_numpy()), evaluate(y, s.p_b.to_numpy())
        deltas["auc"].append(ma["auc"] - mb["auc"])
        deltas["log_loss"].append(mb["log_loss"] - ma["log_loss"])
        deltas["brier"].append(mb["brier"] - ma["brier"])
    ya, yb = evaluate(m.goal.to_numpy(), m.p_a.to_numpy()), evaluate(m.goal.to_numpy(), m.p_b.to_numpy())
    out = {"n": len(m), "games": m.game.nunique(),
           "auc_a": ya["auc"], "auc_b": yb["auc"],
           "log_loss_a": ya["log_loss"], "log_loss_b": yb["log_loss"]}
    for k, v in deltas.items():
        v = np.array(v)
        out[f"d_{k}"] = float(np.median(v))
        out[f"d_{k}_lo"] = float(np.percentile(v, 2.5))
        out[f"d_{k}_hi"] = float(np.percentile(v, 97.5))
        out[f"p_{k}_favours_a"] = float((v > 0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("results/significance"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pairs", nargs="*", default=None, help="'A:B' pairs; default = all pairs")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in a.preds]
    names = a.names or [f.stem.replace("_test_predictions", "") for f in files]
    preds = {n: pd.read_csv(f).drop_duplicates("shot_id")[["shot_id", "goal", "p"]]
             for n, f in zip(names, files)}
    games = pd.read_parquet(a.index)[["shot_id", "game", "goal"]].drop_duplicates("shot_id")
    games["goal"] = games.goal.astype(int)

    pairs = ([tuple(p.split(":")) for p in a.pairs] if a.pairs
             else list(itertools.combinations(names, 2)))
    rows = []
    for x, y in pairs:
        r = paired_bootstrap(preds[x], preds[y], games, a.n_boot)
        rows.append({"model_a": x, "model_b": y, **r})
        print(f"{x} vs {y}: dAUC {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}] "
              f"P(a>b)={r['p_auc_favours_a']:.3f} | dlogloss {r['d_log_loss']:+.4f} "
              f"[{r['d_log_loss_lo']:+.4f}, {r['d_log_loss_hi']:+.4f}] P={r['p_log_loss_favours_a']:.3f}",
              flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "paired_tests.csv", index=False)
    pd.set_option("display.width", 240)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
