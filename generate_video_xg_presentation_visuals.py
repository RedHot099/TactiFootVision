import json
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MATCH_DIR = Path(
    "/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/"
    "3775567_Chelsea_FCW_vs_Manchester_United"
)
FP_RUN_DIR = MATCH_DIR / "experiments" / "video_xg_fp_reduction_10fps_20260526"
WINNERS_RUN_DIR = (
    MATCH_DIR / "experiments" / "video_only_xg_end_to_end_winners_10fps_20260526"
)
BASELINE_1FPS_RUN_DIR = (
    MATCH_DIR / "experiments" / "video_only_xg_end_to_end_yolo11m_1fps_20260526_131429"
)
OUTPUT_DIR = Path("docs/video_xg/visualizations")

DPI = 180
FIG_BG = "#f8fafc"
TEXT = "#172033"
MUTED = "#64748b"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#16a34a"
ORANGE = "#f97316"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#94a3b8"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_theme()

    generated = [
        make_pipeline_architecture(),
        make_metric_waterfall(),
        make_candidate_funnel(),
        make_ball_trajectory_overlay(),
        make_shot_detection_tradeoff(),
        make_xg_scatter_small_multiples(),
        make_xg_total_error_chart(),
        make_failure_mode_taxonomy(),
        make_video_overlay_frame_example(),
    ]
    write_manifest(generated)
    print("Generated presentation visuals:")
    for path in generated:
        print(f"- {path}")
    print(f"- {OUTPUT_DIR / 'README.md'}")


def _set_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_BG,
            "axes.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
        }
    )


def _save(fig: plt.Figure, stem: str, *, svg: bool = False) -> Path:
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    if svg:
        fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _short_label(value: str) -> str:
    labels = {
        "baseline_contact_kinematic": "contact\nkinematic",
        "learned_temporal": "learned\ntemporal",
        "hard_negative_calibrated": "hard-negative\ncalibrated",
        "high_recall_cascade": "high-recall\ncascade",
        "windowed_temporal": "windowed\ntemporal",
        "rule_sweep": "rule\nsweep",
        "dense_local_refinement": "dense local\nrefinement",
        "video_geometry": "geometry",
        "video_freeze_context": "freeze\ncontext",
        "video_kinematic_context": "kinematic\ncontext",
        "quality_aware_ensemble": "quality-aware\nensemble",
        "coefficient_fit": "coefficient\nfit",
        "video_geometry_isotonic_platt": "geometry\nisotonic",
        "video_kinematic_context_isotonic_platt": "kinematic\nisotonic",
        "video_freeze_context_isotonic_platt": "freeze\nisotonic",
    }
    return labels.get(value, value.replace("_", "\n"))


def make_pipeline_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ("Input video\npart1 + part2", 0.04, 0.58, BLUE),
        ("Detections\nYOLO11m / chunks", 0.20, 0.58, TEAL),
        ("Tracking\nplayers + ball stream", 0.36, 0.58, TEAL),
        ("Ball trajectory\nKalman + interpolation", 0.52, 0.58, GREEN),
        ("Shot spotting\ncandidate + filter", 0.68, 0.58, ORANGE),
        ("xG models\ngeometry / context / fit", 0.84, 0.58, PURPLE),
    ]
    for label, x, y, color in boxes:
        _draw_box(ax, x, y, 0.12, 0.16, label, color)

    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][1] + 0.12
        x2 = boxes[idx + 1][1]
        _arrow(ax, x1, 0.66, x2, 0.66, color="#334155")

    _draw_box(
        ax,
        0.68,
        0.24,
        0.28,
        0.15,
        "StatsBomb reference\nloaded only after predictions",
        "#475569",
    )
    _arrow(ax, 0.90, 0.58, 0.82, 0.39, color="#475569", dashed=True)

    ax.text(
        0.50,
        0.88,
        "Video-only xG pipeline: runtime vs post-inference evaluation",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.49,
        "runtime: only video",
        ha="center",
        va="center",
        fontsize=13,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        0.82,
        0.17,
        "evaluation / calibration boundary",
        ha="center",
        va="center",
        fontsize=12,
        color=MUTED,
    )
    return _save(fig, "01_pipeline_architecture", svg=True)


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    dashed: bool = False,
) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        linestyle="--" if dashed else "-",
        color=color,
    )
    ax.add_patch(arrow)


def make_metric_waterfall() -> Path:
    baseline_metrics = json.loads(
        (BASELINE_1FPS_RUN_DIR / "10_metrics.json").read_text()
    )
    current_metrics = json.loads((FP_RUN_DIR / "00_baseline_summary.json").read_text())
    shot_ablation = pd.read_csv(FP_RUN_DIR / "13_shot_fp_ablation.csv")
    final_row = shot_ablation[
        shot_ablation["variant"] == "hard_negative_calibrated"
    ].iloc[0]

    stages = ["1 FPS\nbaseline", "10 FPS\nhigh recall", "FP-reduced\nfinal"]
    hit2 = [
        baseline_metrics["hit@2.0s"],
        current_metrics["hit@2s"],
        final_row["hit@2s"],
    ]
    hit1 = [
        baseline_metrics["hit@1.0s"],
        current_metrics["hit@1s"],
        final_row["hit@1s"],
    ]
    false_positives = [np.nan, 61, final_row["false_positives"]]
    fallback = [61, current_metrics["missing_center_fallback_count"], 0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "Pipeline quality improved before xG became meaningful",
        fontsize=21,
        fontweight="bold",
    )
    _bar_metric(axes[0, 0], stages, hit2, "hit@2s", GREEN, percent=True)
    _bar_metric(axes[0, 1], stages, hit1, "hit@1s", BLUE, percent=True)
    _bar_metric(
        axes[1, 0],
        stages,
        false_positives,
        "false positives",
        RED,
        lower_is_better=True,
    )
    _bar_metric(
        axes[1, 1],
        stages,
        fallback,
        "missing-center fallback features",
        ORANGE,
        lower_is_better=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "02_pipeline_quality_waterfall")


def _bar_metric(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    title: str,
    color: str,
    *,
    percent: bool = False,
    lower_is_better: bool = False,
) -> None:
    x = np.arange(len(labels))
    colors = [GRAY if np.isnan(value) else color for value in values]
    bars = ax.bar(
        x,
        [0 if np.isnan(value) else value for value in values],
        color=colors,
        alpha=0.88,
    )
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if lower_is_better:
        ax.text(
            0.98,
            0.92,
            "lower is better",
            ha="right",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
        )
    for bar, value in zip(bars, values, strict=True):
        if np.isnan(value):
            label = "n/a"
        elif percent:
            label = f"{value:.1%}"
        else:
            label = f"{value:.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.02, ax.get_ylim()[1] * 0.02),
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def make_candidate_funnel() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.88,
        "Shot candidate funnel after FP reduction",
        ha="center",
        fontsize=22,
        fontweight="bold",
    )
    funnel = [
        ("High-recall candidates", 80, 0.16, BLUE),
        ("Matched GT candidates", 19, 0.42, GREEN),
        ("False positives removed", 61, 0.42, RED),
        ("Final selected shots", 19, 0.72, PURPLE),
    ]
    _draw_box(
        ax, 0.08, 0.54, 0.20, 0.18, f"{funnel[0][0]}\n{funnel[0][1]}", funnel[0][3]
    )
    _draw_box(
        ax, 0.40, 0.66, 0.20, 0.16, f"{funnel[1][0]}\n{funnel[1][1]}", funnel[1][3]
    )
    _draw_box(
        ax, 0.40, 0.36, 0.20, 0.16, f"{funnel[2][0]}\n{funnel[2][1]}", funnel[2][3]
    )
    _draw_box(
        ax, 0.72, 0.54, 0.20, 0.18, f"{funnel[3][0]}\n{funnel[3][1]}", funnel[3][3]
    )
    _arrow(ax, 0.28, 0.63, 0.40, 0.74, color=GREEN)
    _arrow(ax, 0.28, 0.63, 0.40, 0.44, color=RED)
    _arrow(ax, 0.60, 0.74, 0.72, 0.63, color=GREEN)
    _arrow(ax, 0.60, 0.44, 0.72, 0.57, color=RED, dashed=True)
    ax.text(
        0.50,
        0.23,
        "Recall is preserved; FP are filtered before xG aggregation.",
        ha="center",
        fontsize=15,
    )
    return _save(fig, "03_candidate_funnel", svg=True)


def make_ball_trajectory_overlay() -> Path:
    selection = _select_shot_for_overlay()
    frame = _read_video_frame(selection["part_index"], selection["part_frame_index"])
    trajectory = pd.read_parquet(WINNERS_RUN_DIR / "05_ball_trajectory.parquet")
    window = trajectory[
        trajectory["global_seconds"].between(
            selection["predicted_seconds"] - 2.0, selection["predicted_seconds"] + 2.0
        )
        & trajectory["image_x"].notna()
        & trajectory["image_y"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(
        f"Ball trajectory around shot: {selection['shot_id']} "
        f"(video xG={selection['predicted_xg']:.3f}, SB xG={selection['reference_xg']:.3f})",
        fontsize=18,
        color=TEXT,
    )

    source_colors = {
        "observed": GREEN,
        "kalman_rts_interpolated": ORANGE,
        "optical_flow_template": PURPLE,
        "missing": GRAY,
    }
    if not window.empty:
        ax.plot(
            window["image_x"],
            window["image_y"],
            color="#111827",
            linewidth=2.0,
            alpha=0.65,
        )
        for source, data in window.groupby("source"):
            ax.scatter(
                data["image_x"],
                data["image_y"],
                s=52,
                c=source_colors.get(source, BLUE),
                label=source,
                edgecolors="#ffffff",
                linewidths=0.8,
                alpha=0.95,
            )
        closest_idx = (
            (window["global_seconds"] - selection["predicted_seconds"]).abs().idxmin()
        )
        closest = window.loc[closest_idx]
        ax.scatter(
            [closest["image_x"]],
            [closest["image_y"]],
            s=210,
            c=RED,
            marker="x",
            linewidths=3.0,
            label="selected shot frame",
        )
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.88)
    return _save(fig, "04_ball_trajectory_overlay")


def _select_shot_for_overlay() -> dict[str, float | int | str]:
    eval_df = pd.read_csv(FP_RUN_DIR / "xg" / "coefficient_fit" / "per_shot_eval.csv")
    selected = pd.read_parquet(FP_RUN_DIR / "14_selected_refined_shots.parquet")
    trajectory = pd.read_parquet(WINNERS_RUN_DIR / "05_ball_trajectory.parquet")
    candidates = eval_df.sort_values("predicted_xg", ascending=False)
    for row in candidates.to_dict("records"):
        shot_rows = selected[selected["shot_id"] == row["shot_id"]]
        if shot_rows.empty:
            continue
        shot = shot_rows.iloc[0]
        points = trajectory[
            trajectory["global_seconds"].between(
                row["predicted_seconds"] - 2.0, row["predicted_seconds"] + 2.0
            )
            & trajectory["image_x"].notna()
            & trajectory["image_y"].notna()
        ]
        if len(points) >= 8:
            return {
                **row,
                "part_index": int(shot["part_index"]),
                "part_frame_index": int(shot["part_frame_index"]),
                "feature_source": "ball trajectory overlay",
            }
    row = candidates.iloc[0].to_dict()
    shot = selected[selected["shot_id"] == row["shot_id"]].iloc[0]
    return {
        **row,
        "part_index": int(shot["part_index"]),
        "part_frame_index": int(shot["part_frame_index"]),
        "feature_source": "ball trajectory overlay",
    }


def _read_video_frame(part_index: int, frame_index: int) -> np.ndarray:
    timeline = json.loads((WINNERS_RUN_DIR / "00_video_timeline.json").read_text())
    video_path = Path(timeline[part_index]["path"])
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_shot_detection_tradeoff() -> Path:
    df = pd.read_csv(FP_RUN_DIR / "13_shot_fp_ablation.csv").copy()
    df["label"] = df["variant"].map(_short_label)

    fig, ax = plt.subplots(figsize=(16, 9))
    scatter = ax.scatter(
        df["hit@2s"],
        df["precision@2s"],
        s=120 + df["false_positives"] * 9,
        c=df["false_positives"],
        cmap="Reds",
        edgecolor="#111827",
        linewidth=0.9,
        alpha=0.88,
    )
    offsets = {
        "hard_negative_calibrated": (12, 12),
        "learned_temporal": (12, 38),
        "baseline_contact_kinematic": (12, 10),
        "high_recall_cascade": (12, 62),
        "windowed_temporal": (12, 30),
        "rule_sweep": (12, -16),
        "dense_local_refinement": (12, -56),
    }
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            (row["hit@2s"], row["precision@2s"]),
            xytext=offsets.get(row["variant"], (8, 6)),
            textcoords="offset points",
            fontsize=10,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": FIG_BG,
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
    ax.axhline(0.40, color=MUTED, linestyle="--", linewidth=1.2, label="precision gate")
    ax.axvline(0.78, color=GREEN, linestyle="--", linewidth=1.2, label="recall floor")
    ax.set_title("Shot detection trade-off: recall vs precision", fontsize=20)
    ax.set_xlabel("hit@2s")
    ax.set_ylabel("precision@2s")
    ax.set_xlim(-0.02, 0.90)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="lower right")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("false positives")
    fig.tight_layout()
    return _save(fig, "05_shot_detection_tradeoff")


def make_xg_scatter_small_multiples() -> Path:
    methods = [
        ("none", "video_geometry"),
        ("none", "video_freeze_context"),
        ("none", "video_kinematic_context"),
        ("quality_aware_ensemble", "quality_aware_ensemble"),
        ("coefficient_fit", "coefficient_fit"),
        ("isotonic_platt", "video_geometry_isotonic_platt"),
    ]
    frames = []
    for variant, method in methods:
        data = pd.read_csv(FP_RUN_DIR / "xg" / variant / "per_shot_eval.csv")
        frames.append(data[data["method"] == method].copy())
    all_df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    max_value = float(
        max(all_df["reference_xg"].max(), all_df["predicted_xg"].max()) * 1.08
    )
    for ax, (_, method) in zip(axes.flat, methods, strict=True):
        data = all_df[all_df["method"] == method]
        ax.scatter(
            data["reference_xg"],
            data["predicted_xg"],
            c=np.where(data["is_goal"] > 0, GREEN, BLUE),
            s=58,
            alpha=0.86,
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        ax.plot(
            [0, max_value], [0, max_value], color=MUTED, linestyle="--", linewidth=1.0
        )
        mae = data["abs_xg_error"].mean()
        ax.set_title(
            f"{_short_label(method).replace(chr(10), ' ')}\nMAE={mae:.3f}", fontsize=12
        )
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "StatsBomb xG vs video-derived xG by method", fontsize=20, fontweight="bold"
    )
    fig.supxlabel("StatsBomb xG")
    fig.supylabel("Video xG")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.93))
    return _save(fig, "06_xg_statsbomb_vs_video_scatter")


def make_xg_total_error_chart() -> Path:
    df = pd.read_csv(FP_RUN_DIR / "05_xg_ablation.csv").copy()
    df["display"] = df["method"].map(_short_label)
    df = df.sort_values("total_xg_error")

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = [RED if value < 0 else ORANGE for value in df["total_xg_error"]]
    bars = ax.barh(df["display"], df["total_xg_error"], color=colors, alpha=0.86)
    ax.axvline(0, color="#111827", linewidth=1.2)
    ax.set_title("Total xG error by method after FP reduction", fontsize=20)
    ax.set_xlabel("predicted total xG - StatsBomb total xG")
    ax.set_ylabel("")
    for bar, value in zip(bars, df["total_xg_error"], strict=True):
        offset = 0.035 if value >= 0 else -0.035
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha=ha,
        )
    ax.text(
        0.98,
        0.95,
        "reference StatsBomb total xG = 2.532",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
        ha="right",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": FIG_BG,
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
    )
    fig.tight_layout()
    return _save(fig, "07_xg_total_error_by_method")


def make_failure_mode_taxonomy() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "Current failure modes to audit before publication",
        fontsize=21,
        fontweight="bold",
    )
    cards = [
        (
            "Occluded or tiny ball",
            "Cause: ball detector loses the object during contact.\nImpact: noisy speed/contact features.\nNext: local 30 FPS refinement.",
            BLUE,
        ),
        (
            "Rebound / shot clusters",
            "Cause: several events in a short window.\nImpact: event-level matching can understate quality.\nNext: cluster-level metrics.",
            ORANGE,
        ),
        (
            "Weak homography",
            "Cause: broadcast view lacks stable pitch geometry.\nImpact: distance and angle are degraded.\nNext: line/box calibration backend.",
            PURPLE,
        ),
        (
            "Passes similar to shots",
            "Cause: high ball speed and direction after contact.\nImpact: false positives before hard-negative filter.\nNext: richer temporal model.",
            RED,
        ),
    ]
    for ax, (title, body, color) in zip(axes.flat, cards, strict=True):
        ax.axis("off")
        patch = FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.82,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(patch)
        ax.text(
            0.10,
            0.74,
            title,
            fontsize=17,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
        )
        ax.text(
            0.10,
            0.54,
            "\n".join(textwrap.wrap(body, width=42)),
            fontsize=12.5,
            va="top",
            color=TEXT,
            transform=ax.transAxes,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "08_failure_mode_taxonomy")


def make_video_overlay_frame_example() -> Path:
    selection = _select_shot_for_overlay()
    frame = _read_video_frame(selection["part_index"], selection["part_frame_index"])
    frame = _crop_wide(frame)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(
        "Example research overlay for video xG clips", fontsize=19, fontweight="bold"
    )

    delta = selection["predicted_xg"] - selection["reference_xg"]
    panel = FancyBboxPatch(
        (0.62, 0.10),
        0.34,
        0.42,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#0f172a",
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.88,
    )
    ax.add_patch(panel)
    lines = [
        ("Method", "coefficient_fit"),
        ("Video xG", f"{selection['predicted_xg']:.3f}"),
        ("StatsBomb xG", f"{selection['reference_xg']:.3f}"),
        ("Delta", f"{delta:+.3f}"),
        ("Time error", f"{selection['time_error_seconds']:+.2f}s"),
    ]
    ax.text(
        0.65,
        0.47,
        selection["shot_id"],
        transform=ax.transAxes,
        color="#ffffff",
        fontsize=15,
        fontweight="bold",
    )
    y = 0.40
    for key, value in lines:
        ax.text(
            0.65,
            y,
            key,
            transform=ax.transAxes,
            color="#cbd5e1",
            fontsize=12,
            ha="left",
        )
        ax.text(
            0.93,
            y,
            value,
            transform=ax.transAxes,
            color="#ffffff",
            fontsize=12,
            ha="right",
        )
        y -= 0.06
    ax.text(
        0.65,
        0.12,
        "Overlay should support audit, not replace metrics.",
        transform=ax.transAxes,
        color="#e2e8f0",
        fontsize=10.5,
    )
    return _save(fig, "09_video_overlay_frame_example")


def _crop_wide(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_ratio = 16 / 9
    current_ratio = width / height
    if current_ratio >= target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return frame[:, left : left + new_width]
    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    return frame[top : top + new_height, :]


def write_manifest(paths: list[Path]) -> None:
    descriptions = {
        "01_pipeline_architecture.png": "Slajd 2/4: architektura pipeline'u i granica runtime vs evaluation.",
        "02_pipeline_quality_waterfall.png": "Slajdy 4-7: jak zmienialy sie kluczowe metryki po poprawkach.",
        "03_candidate_funnel.png": "Slajd 7: redukcja kandydatow i false positive.",
        "04_ball_trajectory_overlay.png": "Slajd 5: przyklad trajektorii pilki wokol strzalu.",
        "05_shot_detection_tradeoff.png": "Slajd 7: trade-off recall/precision/false positives dla rankerow.",
        "06_xg_statsbomb_vs_video_scatter.png": "Slajdy 8-10: porownanie xG per strzal dla metod.",
        "07_xg_total_error_by_method.png": "Slajd 10: blad sumy xG wzgledem StatsBomb.",
        "08_failure_mode_taxonomy.png": "Slajd 13: obecne failure modes.",
        "09_video_overlay_frame_example.png": "Slajdy 11-12: przyklad panelu metryk na materiale wideo.",
    }
    lines = [
        "# Video-Only xG Presentation Visuals",
        "",
        "Generated with `uv run python generate_video_xg_presentation_visuals.py`.",
        "",
        "| File | Suggested use |",
        "|---|---|",
    ]
    for path in paths:
        lines.append(f"| `{path.name}` | {descriptions.get(path.name, '')} |")
    lines.append("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
