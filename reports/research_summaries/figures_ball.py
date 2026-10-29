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


def _pct(df: pd.DataFrame, col: str, mask: pd.Series) -> float:
    """Well-placed frames of `col` within `mask`, as a share of all scored frames, in %."""
    return 100 * df.loc[mask, col].sum() / len(df)


def bridge_numbers() -> dict:
    """Additive decomposition of the gain from the old detector to the final pipeline, per split.

    Per frame of the scored denominator (every 5th frame with one ball on the pitch): well placed = homography and error
    <= 5 m. Old detector from ball-detection 25, retrained-detector top-1 and pipeline from ticket 09, pipeline status
    from ticket 09's tracks. No per-frame "top-1 + size filter" row exists for test, so the filter is not a separate
    step: the pipeline - top-1 difference is split by the pipeline's status (interpolated vs measured or no position).
    The steps sum exactly to the end bar; start, end and the retraining step reproduce ball-detection 25.
    """
    key = ["clip", "frame"]
    names = {"test": "test", "valid": "full"}  # ticket 09 file names
    read25 = json.loads((ball_data.BALL / "25/read.json").read_text())
    out = {}
    for split, name in names.items():

        def well(path: str) -> pd.DataFrame:
            d = pd.read_parquet(ball_data.BALL / path)
            return d[key].assign(well=(d.homography & (d.error_m <= 5)).to_numpy())

        df = (
            well(f"25/placement_{split}_base.parquet")
            .rename(columns={"well": "base"})
            .merge(
                well(f"09/read_{split}/placement_{name}_top1.parquet").rename(
                    columns={"well": "top1"}
                ),
                on=key,
            )
            .merge(
                well(f"09/read_{split}/placement_{name}_track.parquet").rename(
                    columns={"well": "pipeline"}
                ),
                on=key,
            )
            .merge(
                pd.read_parquet(
                    ball_data.BALL / f"09/track_{split}/tracks.parquet",
                    columns=[*key, "status"],
                ),
                on=key,
                how="left",
            )
        )
        n = len(df)
        interp = df.status.eq("interpolated")
        everything = pd.Series(True, index=df.index)
        row = {
            "frames": n,
            "start": _pct(df, "base", everything),
            "retraining": _pct(df, "top1", everything) - _pct(df, "base", everything),
            "reselection": _pct(df, "pipeline", ~interp) - _pct(df, "top1", ~interp),
            "gap_filling": _pct(df, "pipeline", interp) - _pct(df, "top1", interp),
            "end": _pct(df, "pipeline", everything),
        }
        steps = (
            row["start"] + row["retraining"] + row["reselection"] + row["gap_filling"]
        )
        rows = read25[split]["rows"]
        assert abs(steps - row["end"]) < 1e-9
        assert abs(row["start"] - 100 * rows["base"]["well_placed"]["value"]) < 1e-9
        assert abs(row["end"] - 100 * rows["pipeline"]["well_placed"]["value"]) < 1e-9
        assert (
            abs(
                row["retraining"]
                - 100 * read25[split]["base_to_dd2c"]["well_placed"]["difference"]
            )
            < 1e-9
        )
        out[split] = row
    return out


def bridge_tips(numbers: dict) -> dict:
    """Hover text per bar. Totals carry their existing CIs; the two tracking parts have none."""

    def ci(d: dict, value: str = "difference") -> str:
        lo, hi = (100 * c for c in d["ci"])
        return f"{100 * d[value]:+.1f} pp [{lo:+.1f}, {hi:+.1f}]"

    tips = {}
    for split, b in numbers["bridge"].items():
        base = numbers["baseline"][split]
        lo, hi = (100 * c for c in base["base"]["well_placed"]["ci"])
        plo, phi = (100 * c for c in base["pipeline"]["well_placed"]["ci"])
        tips |= {
            f"bridge-{split}-0": f"Old detector: {b['start']:.1f}% [{lo:.1f}, {hi:.1f}]",
            f"bridge-{split}-1": f"Retraining the detector: {ci(base['base_to_dd2c'])}",
            f"bridge-{split}-2": f"Filter + tracking on measured frames and frames without a position: {b['reselection']:+.1f} pp",
            f"bridge-{split}-3": f"Filter + tracking on interpolated frames: {b['gap_filling']:+.1f} pp",
            f"bridge-{split}-4": f"Final pipeline: {b['end']:.1f}% [{plo:.1f}, {phi:.1f}]; "
            f"{ci(base['base_to_pipeline'])} over the old detector",
        }
    return tips


def bridge_chart(bridge: dict, tips: dict) -> None:
    """Waterfall from the old detector to the final pipeline: GSR test (left) and the whole of GSR valid (right)."""
    labels = [
        "old\ndetector",
        "retraining",
        "re-selection",
        "gap\nfilling",
        "final\npipeline",
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.6, 2.25),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
    )
    for ax, (split, title) in zip(
        axes, (("test", "GSR test"), ("valid", "GSR validation")), strict=True
    ):
        b = bridge[split]
        steps = [b["retraining"], b["reselection"], b["gap_filling"]]
        bottoms = [
            0,
            b["start"],
            b["start"] + steps[0],
            b["start"] + steps[0] + steps[1],
            0,
        ]
        heights = [b["start"], *steps, b["end"]]
        colors = [NEUTRAL, DETECTOR, DETECTOR, DETECTOR, INK]
        x = np.arange(5)
        for k in range(5):
            bar = ax.bar(
                x[k], heights[k], bottom=bottoms[k], width=0.62, color=colors[k]
            )[0]
            bar.set_gid(f"bridge-{split}-{k}")
            top = bottoms[k] + heights[k]
            text = f"{heights[k]:.1f}%" if k in (0, 4) else f"+{heights[k]:.1f}"
            ax.text(
                x[k], top + 2, text, ha="center", va="bottom", fontsize=7.8, color=INK
            )
            if k < 4:
                ax.plot([x[k] + 0.31, x[k + 1] - 0.31], [top, top], color=MUTED, lw=0.6)
        ax.set_xticks(x, labels, fontsize=7.4)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Ball within 5 m (%)", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.9, wspace=0.08)
    path = FIG / "bridge.svg"
    fig.savefig(path)
    plt.close(fig)
    svg = path.read_text()
    for gid, tip in tips.items():  # hover text for the inline HTML copy
        svg = svg.replace(f'<g id="{gid}">', f'<g id="{gid}"><title>{tip}</title>', 1)
    path.write_text(svg)


def baseline_numbers() -> dict:
    """Old detector vs retrained detector vs final pipeline on GSR test and the whole of GSR valid
    (ball-detection 25, read.json): shares within 5 m, > 5 m among placed, paired gains with 95% CIs."""
    read = json.loads((ball_data.BALL / "25/read.json").read_text())
    return {
        split: {
            "base": read[split]["rows"]["base"],
            "dd2c": read[split]["rows"]["dd2c_top1"],
            "pipeline": read[split]["rows"]["pipeline"],
            "base_to_dd2c": read[split]["base_to_dd2c"]["well_placed"],
            "base_to_pipeline": read[split]["base_to_pipeline"]["well_placed"],
        }
        for split in ("test", "valid")
    }


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
    errors_chart()
    synthetic_chart()
    numbers["accuracy_games_2_3"] = accuracy_numbers()
    numbers["baseline"] = baseline_numbers()
    numbers["bridge"] = bridge_numbers()
    bridge_chart(numbers["bridge"], bridge_tips(numbers))
    accuracy_chart(numbers["accuracy_games_2_3"])
    (OUT / "ball/numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
