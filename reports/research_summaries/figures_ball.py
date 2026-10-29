"""Charts for the ball summary.

Run: uv run python reports/research_summaries/figures_ball.py
Numbers come from the ball-detection reports on research/ball-detection (paths next to each table);
the final test row is read from the ticket 09 record, the accuracy curve and its numbers from ticket 17.
"""

import json

import matplotlib

matplotlib.use("Agg")
import ball_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures_xg import GRID, INK, MUTED  # same style as the xG summary
from sources import OUT

FIG = OUT / "ball/figures"
DETECTOR = "#2a78d6"
TRACKING = "#1baf7a"
NEUTRAL = "#9a9994"

# results_reports/ball_detection/01/results.md, games 2+3 of GSR valid, dd2c at 1280, conf 0.15
SIZE_BINS = ["< 8 px", "8–12", "12–16", "16–24", "≥ 24 px"]
SIZE_BALLS = [365, 1853, 1579, 1544, 160]
SIZE_RECALL = [0.071, 0.385, 0.518, 0.747, 0.800]  # centre rule
MISSES = [
    ("no candidate at all", 42.6),
    ("candidate below threshold", 33.1),
    ("found, loose box", 24.3),
]

# results_reports/ball_detection/REPORT.md, the registered primary comparison of each ticket:
# paired gain in the share of frames with the ball placed within 5 m, 95% clip bootstrap
REGISTERED = [
    ("Native 1920 px input", "games 2+3", -0.21, -2.87, 2.10, DETECTOR),
    (
        "Fine-tuning, hard balls oversampled ×2",
        "games 2+3",
        0.26,
        -0.80,
        1.35,
        DETECTOR,
    ),
    ("2×2 tiles + 45 px size filter", "game 5", 5.29, 3.15, 8.02, DETECTOR),
    ("Temporal tracking", "games 2+3", 17.32, 11.86, 22.82, TRACKING),
    ("Temporal tracking", "game 5", 9.15, 4.24, 14.58, TRACKING),
]


# research/detector-data: results_reports/detector_data_2026-10/REPORT.md, section 4, stage 2b: fine-tuning on
# SoccerSynth alone from the 2a checkpoint; AP50 at 1280 on the three Ekstraklasa dev clips
SYNTH_STAGES = ["start", "after epoch 1", "after epoch 5"]
SYNTH_AP50 = {
    "player": [0.878, 0.765, 0.554],
    "goalkeeper": [0.864, 0.317, 0.005],
    "referee": [0.778, 0.207, 0.001],
    "ball": [0.161, 0.004, 0.0003],
}


plt.rcParams.update(
    {
        "font.family": ["Noto Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


def errors_chart() -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.6, 1.9), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    x = np.arange(len(SIZE_BINS))
    a.bar(x, SIZE_RECALL, color=DETECTOR, width=0.6)
    for i, r in enumerate(SIZE_RECALL):
        a.text(i, r + 0.03, f"{r:.2f}", ha="center", fontsize=8, color=INK)
    labels = [f"{b}\nn={n:,}" for b, n in zip(SIZE_BINS, SIZE_BALLS, strict=True)]
    a.set_xticks(x, labels, fontsize=7.5)
    a.set_ylim(0, 1)
    a.set_ylabel("Recall")
    a.set_title("Detector recall by ball size", fontsize=9, loc="left")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    y = np.arange(len(MISSES))
    b.barh(
        y, [m[1] for m in MISSES], color=["#184f95", "#3987e5", "#9ec5f4"], height=0.55
    )
    for i, (_, share) in enumerate(MISSES):
        b.text(share + 1.5, i, f"{share:.0f}%", va="center", fontsize=8, color=INK)
    b.set_yticks(y, [m[0] for m in MISSES], fontsize=8)
    b.invert_yaxis()
    b.set_xlim(0, 55)
    b.set_xlabel("Share of missed balls", fontsize=8)
    b.set_title("Why a ball is missed (3,515 misses)", fontsize=9, loc="left")
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "errors.svg")
    plt.close(fig)


def registered_chart(test: tuple[float, float, float]) -> None:
    rows = [
        *REGISTERED,
        ("Tracking + size filter (final)", "GSR test", *test, TRACKING),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    for i, (_name, _split, m, lo, hi, color) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=color, lw=2)
        ax.scatter(m, i, color=color, s=40, zorder=3, edgecolor="white", lw=0.8)
        ax.text(
            hi + 0.6,
            i,
            f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]",
            va="center",
            fontsize=7.8,
            color=INK,
        )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [f"{r[0]}  ·  {r[1]}" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-5, 36)
    ax.set_xlabel(
        "Paired gain over the baseline detector, percentage points", fontsize=8.5
    )
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D(
            [], [], color=DETECTOR, marker="o", lw=2, label="single-frame detector"
        ),
        plt.Line2D([], [], color=TRACKING, marker="o", lw=2, label="temporal tracking"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "registered.svg")
    plt.close(fig)


def breakdown_chart(status: list[dict]) -> None:
    """Well-placed share by kind of tracked frame (GSR test, ticket 09 record)."""
    rows = {r["status"]: r for r in status}
    BREAKDOWN = [
        (
            f"{kind}\n({100 * rows[kind]['share']:.1f}% of frames)",
            100 * rows[kind]["top1_well"],
            100 * rows[kind]["track_well"],
        )
        for kind in ("measured", "interpolated")
    ]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    x = np.arange(len(BREAKDOWN))
    for k, (label, color) in enumerate(
        (("detector top-1", NEUTRAL), ("tracked", TRACKING))
    ):
        vals = [row[1 + k] for row in BREAKDOWN]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=label)
        for i, v in enumerate(vals):
            ax.text(
                x[i] + (k - 0.5) * 0.36,
                v + 2,
                f"{v:.0f}%",
                ha="center",
                fontsize=7.8,
                color=INK,
            )
    ax.set_xticks(x, [row[0] for row in BREAKDOWN], fontsize=7.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Ball within 5 m (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "breakdown.svg")
    plt.close(fig)


def accuracy_numbers() -> dict:
    """Games 2+3 thresholds and percentiles of ticket 17 (read.json), value and 95% clip-bootstrap CI."""
    groups = json.loads((ball_data.BALL / "17/read.json").read_text())["splits"]
    g = groups["valid"]["groups"]["games_2_3"]
    keys = ("le1_all", "le2_all", "le5_all", "p50_m", "p90_m", "no_position")
    return {
        "reference_centre": {k: g["reference"]["centre"][k] for k in keys},
        **{
            name: {k: g["predictors"][name]["summary"][k] for k in keys}
            for name in ("top1", "phase1")
        },
    }


def accuracy_chart(acc: dict) -> None:
    """Share of all denominator frames (games 2+3) with the ball within d metres (ticket 17 per-frame tables).

    A frame without a position (no ball or no homography) never counts as within d, as in the report.
    """
    root = ball_data.BALL / "17"
    curves = [
        (
            "GSR reference",
            pd.read_parquet(root / "truth_valid.parquet")
            .query("game in ['2', '3']")
            .ref_centre_m,
            NEUTRAL,
            "--",
        ),
        (
            "detector top-1",
            pd.read_parquet(root / "placement_games_2_3_top1.parquet").error_m,
            DETECTOR,
            "-",
        ),
        (
            "final pipeline",
            pd.read_parquet(root / "placement_games_2_3_phase1.parquet").error_m,
            TRACKING,
            "-",
        ),
    ]
    d = np.logspace(np.log10(0.1), np.log10(30), 300)
    fig, ax = plt.subplots(figsize=(4.1, 2.15))
    for label, err, color, ls in curves:
        e = err.to_numpy(dtype=float)
        share = [(e <= x).sum() / len(e) * 100 for x in d]  # NaN counts as not within
        ax.plot(d, share, color=color, lw=1.8, ls=ls, label=label)
    ref_p50 = acc["reference_centre"]["p50_m"]["value"]
    ax.axvline(ref_p50, color=MUTED, lw=0.8, ls=":")
    ax.text(
        ref_p50 * 1.07,
        97,
        f"reference median\n{ref_p50:.2f} m",
        ha="left",
        va="top",
        fontsize=7,
        color=INK,
    )
    ax.axvline(5, color=INK, lw=0.8)
    ax.text(5.3, 40, "5 m", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(
        [0.1, 0.25, 0.5, 1, 2, 5, 10, 25],
        ["0.1", "0.25", "0.5", "1", "2", "5", "10", "25"],
    )
    ax.set_xlim(0.1, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Distance on the pitch d (m)", fontsize=8.5)
    ax.set_ylabel("Frames within d (%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy.svg")
    plt.close(fig)
    for label, err, *_ in curves:  # the curve must match the report's thresholds
        e = err.to_numpy(dtype=float)
        print(label, {x: round(100 * (e <= x).sum() / len(e), 1) for x in (1, 2, 5)})


def synthetic_chart() -> None:
    """AP50 per class while fine-tuning on SoccerSynth alone (web-only figure)."""
    colors = {
        "player": TRACKING,
        "goalkeeper": "#184f95",
        "referee": "#eb6834",
        "ball": DETECTOR,
    }
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    x = np.arange(len(SYNTH_STAGES))
    for name, values in SYNTH_AP50.items():
        ax.plot(x, values, color=colors[name], lw=1.8, marker="o", ms=4, label=name)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax.set_xticks(x, SYNTH_STAGES, fontsize=8)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("AP50 at 1280 px")
    ax.set_title(
        "Fine-tuning on SoccerSynth alone · 3 Ekstraklasa clips (indicative)",
        fontsize=8.5,
        loc="left",
    )
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "synthetic.svg")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    reads = ball_data.phase_numbers()
    test = reads["test"]["groups"]["test"]
    numbers = {
        "test_track": test["track"]["well_placed"],
        "test_top1": test["top1"]["well_placed"],
        "test_track_median_m": test["track"]["median_m"],
    }
    diff = test["track_vs_top1"]["well_placed"]
    gain = (
        100 * diff["difference"],
        100 * diff["ci"][0],
        100 * diff["ci"][1],
    )  # shares -> points
    numbers["test_gain_pp"] = gain
    numbers["status"] = test["status"]
    errors_chart()
    registered_chart(gain)
    breakdown_chart(test["status"])
    synthetic_chart()
    numbers["accuracy_games_2_3"] = accuracy_numbers()
    accuracy_chart(numbers["accuracy_games_2_3"])
    (OUT / "ball/numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
