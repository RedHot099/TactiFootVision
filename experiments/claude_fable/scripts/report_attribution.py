"""Does the point-1 result survive other goal attribution rules? (xg-credibility 03)

    ./run.sh python scripts/report_attribution.py

Recomputes the ensemble-minus-G-combo comparison of scripts/report_complexity.py on the
archived test predictions (results/v16_complexity/predictions) with the evaluation labels
changed by the rules and bounds of xgpoc.eval.attribution. Predictions and models are
untouched: only the test labels change, so the effect of the rules on training is NOT
measured (a limitation, stated in the report).

Writes attribution_sensitivity.csv (one row per rule or bound) and attribution_recount.csv
(rebound chains and chain goals per split) into the report folder. Before anything else it
checks that the nearest rule reproduces the `goal` column of the index, and that its row
reproduces headline_ens_vs_g10.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from report_complexity import (
    TAB,
    analyse,
    calibration_diff,
    calibration_row,
    context,
    diff_row,
)
from xgpoc.eval.attribution import (
    RULES,
    attribute,
    bound_labels,
    recount,
    relabel,
    source_events,
)
from xgpoc.paths import INDEX

PREDICTIONS = Path("results/v16_complexity/predictions")
FIGURES = Path("results_reports/ensemble_vs_g_2026-10")
# (name, model whose log-loss is bounded, reference model or None, bound); with a reference the
# bound is on the log-loss difference model - reference
BOUNDS = [("worst for ens", "ens", None, "worst"), ("best for ens", "ens", None, "best"),
          ("worst for g10", "g10", None, "worst"), ("best for g10", "g10", None, "best"),
          ("worst for ens vs g10", "ens", "g10", "worst"), ("best for ens vs g10", "ens", "g10", "best")]


def sensitivity_row(name: str, kind: str, ctx: pd.DataFrame, preds: dict[str, pd.Series], near: np.ndarray,
                    goal: np.ndarray, n_boot: int) -> dict:
    """One table row: ens - g10 differences (positive = ens better) and the calibration slopes."""
    c = ctx.assign(goal=goal)
    res = analyse(preds, c, [], n_boot)
    d = diff_row(res, ["ens"], ["g10"])
    g10, ens = calibration_row(res, ["g10"]), calibration_row(res, ["ens"])
    slope_diff = calibration_diff(res, "ens", "g10").set_index("quantity").loc["slope"]
    row = {"rule": name, "kind": kind, "n_shots": len(c), "goals": int(goal.sum()), "goals_moved": int((goal & ~near).sum())}
    for m in ("auc", "log_loss", "brier"):
        row |= {f"d_{m}": d[f"d_{m}"], f"d_{m}_lo": d[f"d_{m}_lo"], f"d_{m}_hi": d[f"d_{m}_hi"]}
    row["p_log_loss_ens_better"] = d["p_log_loss_a_better"]
    for label, cal in (("g10", g10), ("ens", ens)):
        row |= {f"slope_{label}": cal["slope"], f"slope_{label}_lo": cal["slope_lo"], f"slope_{label}_hi": cal["slope_hi"]}
    row |= {"slope_diff": slope_diff["diff"], "slope_diff_lo": slope_diff["lo"], "slope_diff_hi": slope_diff["hi"]}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS)
    ap.add_argument("--figures", type=Path, default=FIGURES)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    idx = pd.read_parquet(a.index).drop_duplicates("shot_id").reset_index(drop=True)
    idx["goal"], idx["is_penalty"] = idx.goal.astype(bool), idx.is_penalty.astype(bool)
    attr = attribute(source_events(idx.game))
    nearest, _ = relabel(idx, attr, "nearest")
    mismatch = int((nearest != idx.goal.to_numpy()).sum())
    print(f"nearest rule vs index goal column: {mismatch} mismatches of {len(idx)} shots")
    assert mismatch == 0, "the nearest rule does not reproduce the index labels"
    counts = recount(idx, attr)
    counts.to_csv(a.figures / "attribution_recount.csv", index=False)
    print(counts.to_string(index=False))

    ctx = context(idx, TAB).sort_values("shot_id").reset_index(drop=True)
    preds = {m: pd.read_csv(a.predictions / f"{m}.csv").set_index("shot_id").p for m in ("g10", "ens")}
    assert set(ctx.shot_id) == set(preds["ens"].index) == set(preds["g10"].index)
    p = {m: s.loc[ctx.shot_id].to_numpy() for m, s in preds.items()}
    near = ctx.goal.to_numpy()  # the nearest rule, checked against the full index above

    rows = []
    for rule in RULES:
        goal, keep = relabel(ctx, attr, rule)
        rows.append(sensitivity_row(rule, "rule", ctx[keep].reset_index(drop=True), preds, near[keep], goal[keep], a.n_boot))
    for name, model, ref, bound in BOUNDS:
        goal = bound_labels(ctx, attr, p[model], bound, None if ref is None else p[ref])
        rows.append(sensitivity_row(name, "bound", ctx, preds, near, goal, a.n_boot))
    table = pd.DataFrame(rows)

    headline = pd.read_csv(a.figures / "headline_ens_vs_g10.csv").iloc[0]
    for m in ("auc", "log_loss", "brier"):
        for end in ("", "_lo", "_hi"):
            assert abs(table.loc[0, f"d_{m}{end}"] - headline[f"d_{m}{end}"]) < 1e-9, (m, end)
    table.to_csv(a.figures / "attribution_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
