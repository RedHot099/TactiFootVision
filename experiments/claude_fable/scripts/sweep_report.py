"""Readings of the Polish sweep report (results_reports/sweep_2026_10/REPORT.md), flattened from files that already
exist. CPU, seconds. No model fit, no label read: the label-bearing columns of results/err_decomp/shots.parquet are
not loaded, and the goal counts come from the sweep's own scores.csv.

    ./run.sh --no-sync python scripts/sweep_report.py
        # -> results/sweep_report/{location_error.csv, summary.json, complete.json}
        #    results_reports/sweep_2026_10/delta_forest.png

- location_error.csv: shot-location error against StatsBomb (metres, on the 2,721 paired primary shots of the 135
  evaluation matches) of three E3 caches at t0: the retrained detector as frozen (dd2c, cache/xgd_full4), the feet
  variant (cache/xgd_full5_feet) and the feet_t0 variant (cache/xgd_full5_feet_t0); overall, by StatsBomb body part,
  by StatsBomb distance band, and (dd2c only) by the frame the ball of the t0 row came from. Data: StatsBomb.
- summary.json: goalkeeper detection and depth error (from results/err_decomp/shots.parquet), how often the retrained
  detector saw a shot (seen.csv with the goal counts of scores.csv), and the share of the vendor gap the dd2c methods
  close (results/sweep_2026_10/statsbomb/deltas.csv; Data: StatsBomb), and a verbatim copy of the decision-log row
  of the feet follow-up (research/quality_v13_decisions.tsv, phase err-decomp): its feet vendor-gap shares were
  printed by scripts/feet_vendor.py and saved nowhere else, and the log grows, so the report cites this copy.
- delta_forest.png: ΔLL against G-combo with its 95% interval for the retrained-detector methods, dd2c sweep
  (results/sweep_2026_10) beside the feet sweep (results/sweep_feet_2026_10), on all shots and on unseen shots.
"""

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from report_complexity import INK, INK2, SURFACE, _axes_style  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.detector.conventions import YARD, statsbomb_to_metres  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sweep_report"
FIGURE = ROOT / "results_reports/sweep_2026_10/delta_forest.png"
SHOTS = ROOT / "results/err_decomp/shots.parquet"
SB_TABLE = ROOT / "results/xgc_compare/table/shots.parquet"
SWEEP = ROOT / "results/sweep_2026_10"
FEET_SWEEP = ROOT / "results/sweep_feet_2026_10"
DECISIONS = ROOT / "research/quality_v13_decisions.tsv"
CACHES = {
    "dd2c": ROOT / "cache/xgd_full4/E3/features.parquet",
    "feet": ROOT / "cache/xgd_full5_feet/E3/features.parquet",
    "feet_t0": ROOT / "cache/xgd_full5_feet_t0/E3/features.parquet",
}
BANDS = ([0, 6, 11, 16.5, 25, np.inf], ["<6m", "6-11m", "11-16.5m", "16.5-25m", ">=25m"])
METHODS = (("stack_G_new", "stack G"), ("stack_GS_new", "stack GS"), ("stack_Q_new", "stack Q"),
           ("stack_ALL_new", "stack ALL"), ("pos_DA_new", "pos_DA"), ("pos_ALL_new", "pos_ALL"))
COLOUR = {"dd2c": "#2a78d6", "feet": "#eb6834"}   # categorical slots of the dataviz reference palette (validated)


def location_rows(paired: pd.Index, sb: pd.DataFrame) -> pd.DataFrame:
    sbm = pd.DataFrame(statsbomb_to_metres(sb[["sb_x", "sb_y"]].to_numpy(float)), index=sb.index, columns=["x", "y"])
    err, source = {}, None
    for name, path in CACHES.items():
        e = pd.read_parquet(path)
        e = e[e.offset_ms == 0].set_index("shot_id").reindex(paired)
        err[name] = np.hypot(e.shot_x - sbm.x, e.shot_y - sbm.y)
        if name == "dd2c":
            source = e.ball_offset_ms.where(e.shot_from_ball == 1).map({0: "t0", -400: "-400ms", -800: "-800ms"})
            source = source.fillna("no ball")
    groups = {
        "all": pd.Series("all", index=paired),
        "body": sb.sb_body,
        "band": pd.cut(sb.sb_dist * YARD, BANDS[0], right=False, labels=BANDS[1]).astype(str),
        "dd2c_ball_frame": source,
    }
    rows = []
    for group, key in groups.items():
        for level, idx in key.groupby(key).groups.items():
            row = {"group": group, "level": level, "shots": len(idx)}
            for name, e in err.items():
                if group == "dd2c_ball_frame" and name != "dd2c":
                    continue
                v = e.loc[idx].dropna()
                row |= {f"{name}_n": len(v), f"{name}_median_m": v.median(), f"{name}_ge5_share": (v >= 5).mean()}
            rows.append(row)
    return pd.DataFrame(rows)


def summary(shots: pd.DataFrame) -> dict:
    paired = shots[shots.sb_paired]
    scores = pd.read_csv(SWEEP / "scores.csv")
    g = scores[scores.method == "g_combo"].set_index("subset")
    seen = pd.read_csv(SWEEP / "seen.csv")
    goals, unseen_goals = int(g.goals["all"]), int(g.goals["unseen"])
    shots_all, unseen_shots = int(g.shots["all"]), int(g.shots["unseen"])
    seen_shots = int(seen.seen.sum())
    assert seen_shots == shots_all - unseen_shots
    seen_goals = goals - unseen_goals
    seen_other = seen_shots - seen_goals
    deltas = pd.read_csv(SWEEP / "statsbomb/deltas.csv").set_index("method").delta_ll
    with DECISIONS.open(newline="") as f:
        row = [r for r in csv.DictReader(f, delimiter="\t") if r["phase"] == "err-decomp"][-1]
    return {
        "decision_log_err_decomp": {"ts": row["ts"], "result": row["result"]},
        "goalkeeper": {
            "detected_share_all": float(shots.gk_detected.mean()),
            "detected_share_paired": float(paired.gk_detected.mean()),
            "depth_error_median_m": float(paired.gk_depth_err_m.median()),
        },
        "seen": {
            "shots": shots_all, "seen_shots": seen_shots, "goals": goals, "seen_goals": seen_goals,
            "other": shots_all - goals, "seen_other": seen_other,
            "share_goals": seen_goals / goals, "share_other": seen_other / (shots_all - goals),
        },
        "vendor_gap_share_dd2c": {m: float(deltas[m] / deltas["sb_vendor"]) for m in ("stack_Q_new", "stack_ALL_new")},
    }


def forest(out: Path) -> None:
    sweeps = {"dd2c": pd.read_csv(SWEEP / "deltas.csv"), "feet": pd.read_csv(FEET_SWEEP / "deltas.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, facecolor=SURFACE)
    for ax, (subset, title) in zip(axes, (("all", "wszystkie strzały"), ("unseen", "strzały niewidziane przez detektor")), strict=True):
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
        for i, (method, _) in enumerate(METHODS[::-1]):
            for k, (name, d) in enumerate(sweeps.items()):
                r = d[(d.subset == subset) & (d.method == method)].iloc[0]
                y = i + (0.17 if k == 0 else -0.17)
                ax.hlines(y, r.lo95, r.hi95, color=COLOUR[name], lw=2, zorder=2)
                ax.scatter(r.delta_ll, y, s=40, color=COLOUR[name], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_yticks(range(len(METHODS)), [label for _, label in METHODS[::-1]], fontsize=9, color=INK)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.3f}".replace("-", "−") if v else "0"))
        ax.tick_params(axis="x", labelsize=8.5, colors=INK2)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("ΔLL wobec G-combo (ujemne = lepiej)", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=COLOUR[k], marker="o", lw=2, markersize=6) for k in COLOUR]
    fig.legend(handles, ["detektor dd2c, miejsce strzału z piłki (18.10)", "wariant stóp feet_t0, flagi piłki z dd2c (22.10)"],
               frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, labelcolor=INK2)
    fig.suptitle("172 mecze odłożone (zużyte), odczyt opisowy: 95% przedziały z 5 000 losowań meczów", fontsize=10,
                 x=0.01, ha="left", color=INK)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shots = pd.read_parquet(SHOTS, columns=["sb_paired", "gk_detected", "gk_depth_err_m"])   # no label column
    sb = pd.read_parquet(SB_TABLE, columns=["soccernet_shot_id", "population", "sb_x", "sb_y", "sb_dist", "sb_body"])
    sb = sb[sb.population == "primary"].set_index("soccernet_shot_id")
    paired = shots.index[shots.sb_paired]
    sb = sb.reindex(paired)
    location_rows(paired, sb).to_csv(OUT / "location_error.csv", index=False, float_format="%.6g")
    (OUT / "summary.json").write_text(json.dumps(summary(shots), indent=1) + "\n")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    forest(FIGURE)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts/sweep_report.py"],
                                capture_output=True, text=True).stdout.strip())
    inputs = [SHOTS, SB_TABLE, *CACHES.values(), SWEEP / "scores.csv", SWEEP / "seen.csv", SWEEP / "deltas.csv",
              SWEEP / "statsbomb/deltas.csv", FEET_SWEEP / "deltas.csv", DECISIONS]
    record = {
        "data": "Data: StatsBomb.",
        "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
        "code": {"scripts/sweep_report.py": sha256_file(Path(__file__))},
        "git": {"sha": git, "script_dirty": dirty},
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {n: sha256_file(OUT / n) for n in ("location_error.csv", "summary.json")},
        "figures": {str(FIGURE.relative_to(ROOT)): sha256_file(FIGURE)},
    }
    (OUT / "complete.json").write_text(json.dumps(record, indent=1) + "\n")
    print(pd.read_csv(OUT / "location_error.csv").to_string(index=False))
    print((OUT / "summary.json").read_text())


if __name__ == "__main__":
    main()
