"""Re-score saved test predictions against the labels of a given index version.

    ./run.sh python scripts/rescore_predictions.py --preds results/ensemble/*.csv [--index ...]

Prediction CSVs carry the labels they were trained against; after a label revision
(v1.1 -> v1.2) the only fair "before" number is the old prediction scored on the new
labels. Labels are always taken from the index, never from the CSV.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import bootstrap  # noqa: E402

from xgpoc.eval.metrics import evaluate  # noqa: E402
from xgpoc.paths import INDEX  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    idx = pd.read_parquet(a.index).drop_duplicates("shot_id")[["shot_id", "game", "goal", "is_penalty"]]
    idx["goal"] = idx.goal.astype(int)
    rows = []
    for f in [Path(x) for pat in a.preds for x in sorted(glob.glob(pat))]:
        p = pd.read_csv(f).drop_duplicates("shot_id")[["shot_id", "p"]].merge(idx, on="shot_id")
        m = evaluate(p.goal.to_numpy(), p.p.to_numpy())
        ci = bootstrap(p)
        pen = p[p.is_penalty.astype(bool)]
        rows.append({"file": f.stem.replace("_test_predictions", ""), "n": m["n"], "auc": m["auc"], "auc_lo": ci["auc"][0],
                     "auc_hi": ci["auc"][1], "log_loss": m["log_loss"], "bss": m["brier_skill_score"], "ece": m["ece"],
                     "mean_xg": float(p.p.mean()), "goal_rate": m["goal_rate"], "pen_xg": float(pen.p.mean()) if len(pen) else float("nan"),
                     "pen_rate": float(pen.goal.mean()) if len(pen) else float("nan")})
    t = pd.DataFrame(rows).sort_values("log_loss")
    pd.set_option("display.width", 250)
    print(f"labels: {a.index}")
    print(t.round(4).to_string(index=False))
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
