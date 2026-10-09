"""Test-split log-loss, Brier and AUC of the head-free control next to the ticket-04 G-combo.

    ./run.sh python scripts/compare_head_free.py

Both are the ten-model deployment (`report_complexity.deploy`). The interval is the paired,
game-clustered bootstrap of `significance.py` (same 2,000 draws, `report_complexity.game_draws`);
differences are control - G-combo, so a positive log-loss/Brier difference means the control is worse.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from report_complexity import configurations, deploy, game_draws, interval, scores
from verify_xgc_g import load_runs
from xgpoc.eval.calibration import penalty_rate
from xgpoc.paths import INDEX

METRICS = ("auc", "log_loss", "brier")


def ensemble(root: Path, labels: pd.Series, is_pen: pd.Series, pooled_rate: float, p_pen: float) -> pd.Series:
    records = {p.parent: json.loads(p.read_text()) for p in sorted(root.glob("fold*/seed*/complete.json"))}
    runs = load_runs(records, next(iter(records.values()))["identity"]["name"])
    keys = configurations(runs)["g10"][0][1]
    if len(keys) != 10:
        raise SystemExit(f"{root}: {len(keys)} fold models, expected 10")
    return deploy(runs, keys, labels, is_pen, pooled_rate, p_pen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g-combo", type=Path, default=Path("results/xgc_g_combo"))
    ap.add_argument("--control", type=Path, default=Path("results/xgc_g_combo_no_head"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None, help="CSV (default: <control>/comparison.csv)")
    a = ap.parse_args()

    idx = pd.read_parquet(INDEX).drop_duplicates("shot_id")
    idx["goal"], idx["is_penalty"] = idx.goal.astype(bool), idx.is_penalty.astype(bool)
    labels, is_pen = idx.set_index("shot_id").goal, idx.set_index("shot_id").is_penalty
    pooled_rate = float(idx[idx.split != "test"].goal.mean())
    pen_train = idx[(idx.split == "train") & idx.is_penalty]
    p_pen = penalty_rate(int(pen_train.goal.sum()), len(pen_train))

    ref = ensemble(a.g_combo, labels, is_pen, pooled_rate, p_pen)
    ctl = ensemble(a.control, labels, is_pen, pooled_rate, p_pen)
    ids = ref.index.intersection(ctl.index)
    if not (len(ids) == len(ref) == len(ctl)):
        raise SystemExit("the two ensembles do not cover the same test shots")
    test = idx.set_index("shot_id").loc[ids]
    y = test.goal.to_numpy().astype(float)
    pa, pb = ref.loc[ids].to_numpy(), ctl.loc[ids].to_numpy()

    point_a, point_b = scores(y, pa), scores(y, pb)
    diffs = np.array([scores(y[r], pb[r]) - scores(y[r], pa[r]) for r in game_draws(test.game.to_numpy(), a.n_boot)])
    lo, hi = np.array([interval(diffs[:, i]) for i in range(3)]).T
    out = pd.DataFrame({"metric": METRICS, "g_combo": point_a, "no_head": point_b, "diff_no_head_minus_g_combo": point_b - point_a,
                        "ci_lo": lo, "ci_hi": hi, "n_shots": len(ids), "n_games": test.game.nunique(), "n_boot": a.n_boot})
    path = a.out or a.control / "comparison.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
