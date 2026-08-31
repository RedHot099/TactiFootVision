"""Aggregate the big run: per-seed metrics, cross-seed mean/std, seed-ensemble with
bootstrap CI per PoC.

    ./run.sh python scripts/aggregate_big.py --index ... --results results/big_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.eval.metrics import evaluate

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compare import bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)[["shot_id", "game"]]
    root = Path(a.results)
    preds: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(root.glob("**/*_test_predictions.csv")):
        name = f.name.replace("_test_predictions.csv", "")
        group = f.parent.name if not f.parent.name.startswith("seed") else f.parent.parent.name
        if group not in (root.name, ""):
            name += "@" + group
        preds.setdefault(name, []).append(pd.read_csv(f).drop_duplicates("shot_id"))
    rows = []
    for name, dfs in preds.items():
        per_seed = [evaluate(d["goal"].to_numpy(), d["p"].to_numpy()) for d in dfs]
        base = dfs[0][["shot_id", "goal"]].copy()
        ens = base.copy()
        ens["p"] = np.mean([d.set_index("shot_id").loc[base["shot_id"], "p"].to_numpy() for d in dfs], axis=0)
        em = evaluate(ens["goal"].to_numpy(), ens["p"].to_numpy())
        ci = bootstrap(ens.merge(idx, on="shot_id"))
        rows.append({
            "model": name, "seeds": len(dfs), "n": em["n"],
            "auc_mean": np.mean([m["auc"] for m in per_seed]), "auc_std": np.std([m["auc"] for m in per_seed]),
            "bss_mean": np.mean([m["brier_skill_score"] for m in per_seed]),
            "ens_brier": em["brier"], "ens_bss": em["brier_skill_score"], "ens_log_loss": em["log_loss"],
            "ens_auc": em["auc"], "ens_auc_ci": ci["auc"], "ens_log_loss_ci": ci["log_loss"], "ens_ece": em["ece"],
        })
    out = pd.DataFrame(rows).sort_values("ens_log_loss")
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    out.to_csv(root / "aggregate.csv", index=False)
    hist = {}
    for f in sorted(root.glob("**/*_metrics.json")):
        m = json.loads(f.read_text())
        if "history" in m:
            hist[f.parent.name + "/" + m["model"]] = {"epochs": len(m["history"]), "minutes": round(m["history"][-1]["minutes"], 1),
                                                      "best_valid_ll": round(min(h["valid_log_loss"] for h in m["history"]), 4)}
    print(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
