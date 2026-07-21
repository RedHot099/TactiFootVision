from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})

RESULTS_DIR = Path("results/detection_tracking")
DATA_DIR_CANDIDATES = [
    RESULTS_DIR / "raw" / "sam2_full_experiment_snmot118",
    RESULTS_DIR / "raw" / "sam2_seg_geom_5seq",
    RESULTS_DIR / "raw" / "soccernet_tracking_2023_tiny_seg",
    RESULTS_DIR / "raw" / "soccernet_tracking_2023_detection_tracking",
]

TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 11
DPI = 300

TRACKER_LABELS = {
    "bytetrack": "ByteTrack",
    "botsort": "BoT-SORT",
    "botsort_reid": "BoT-SORT",
    "sam2": "SAM2",
}
DETECTOR_LABELS = {
    "rfdetr_base": "Base",
    "rfdetr_seg": "Seg",
}

MODEL_ORDER = [
    "Base + ByteTrack",
    "Seg + ByteTrack",
    "Base + BoT-SORT",
    "Seg + BoT-SORT",
    "Base + SAM2",
    "Seg + SAM2",
]

MODEL_COLORS = {
    "Base + ByteTrack": "#1f77b4",
    "Seg + ByteTrack": "#aec7e8",
    "Base + BoT-SORT": "#ff7f0e",
    "Seg + BoT-SORT": "#ffbb78",
    "Base + SAM2": "#2ca02c",
    "Seg + SAM2": "#98df8a",
}


def find_data_dir() -> Path:
    for candidate in DATA_DIR_CANDIDATES:
        if (candidate / "summary.csv").exists() and (candidate / "metrics_per_class.csv").exists():
            return candidate
    raise FileNotFoundError("No detection tracking dataset found in results/detection_tracking/raw.")


def add_model_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tracker_label"] = df["tracker"].map(TRACKER_LABELS).fillna(df["tracker"])
    df["detector_label"] = df["detector"].map(DETECTOR_LABELS).fillna(df["detector"])
    df["model"] = df["detector_label"] + " + " + df["tracker_label"]
    return df


data_dir = find_data_dir()
out_dir = RESULTS_DIR / "plots" / f"presentation_{data_dir.name}"
out_dir.mkdir(parents=True, exist_ok=True)

summary_df = pd.read_csv(data_dir / "summary.csv")
metrics_per_class_df = pd.read_csv(data_dir / "metrics_per_class.csv")
per_sequence_stats_df = pd.read_csv(data_dir / "per_sequence_stats.csv")

summary_df = add_model_label(summary_df)
metrics_per_class_df = add_model_label(metrics_per_class_df)
per_sequence_stats_df = add_model_label(per_sequence_stats_df)

available_models = [model for model in MODEL_ORDER if model in summary_df["model"].unique()]
model_palette = {model: MODEL_COLORS[model] for model in available_models}


def format_metric_labels(metric: str) -> str:
    return {
        "weighted_HOTA": "HOTA",
        "weighted_IDF1": "IDF1",
        "weighted_MOTA": "MOTA",
        "HOTA": "HOTA",
        "IDF1": "IDF1",
        "MOTA": "MOTA",
    }.get(metric, metric)


def apply_horizontal_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#d0d0d0", alpha=0.45, linewidth=0.8)
    ax.grid(False, axis="x")


# Plot 1: Overall weighted metrics grouped by metric (x-axis)
metrics_cols = ["weighted_HOTA", "weighted_IDF1", "weighted_MOTA"]
overall_df = summary_df[["model"] + metrics_cols].melt(
    id_vars="model",
    value_vars=metrics_cols,
    var_name="metric",
    value_name="score",
)
overall_df["metric"] = overall_df["metric"].map(format_metric_labels)
overall_df["score"] = overall_df["score"] * 100

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=overall_df,
    x="metric",
    y="score",
    hue="model",
    hue_order=available_models,
    palette=model_palette,
    ax=ax,
)
ax.set_title("Overall Tracking Quality - Weighted Metrics", fontsize=TITLE_SIZE)
ax.set_xlabel("Metric", fontsize=LABEL_SIZE)
ax.set_ylabel("Score (%)", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
apply_horizontal_grid(ax)
ax.legend(title="Model", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig(out_dir / "plot_overall_weighted_metrics.png", dpi=DPI)
plt.close(fig)


# Plot 2: Weighted HOTA + FPS (secondary axis)
quality_df = summary_df[["model", "weighted_HOTA", "fps"]].copy()
quality_df["weighted_HOTA"] = quality_df["weighted_HOTA"] * 100
quality_df["model"] = pd.Categorical(quality_df["model"], categories=available_models, ordered=True)
quality_df = quality_df.sort_values("model")

fig, ax1 = plt.subplots(figsize=(12, 6))
bar_colors = [model_palette[model] for model in quality_df["model"]]
ax1.bar(quality_df["model"], quality_df["weighted_HOTA"], color=bar_colors, alpha=0.85)
ax1.set_title("Tracking Quality vs Inference Speed - HOTA + FPS", fontsize=TITLE_SIZE)
ax1.set_xlabel("Model", fontsize=LABEL_SIZE)
ax1.set_ylabel("Weighted HOTA (%)", fontsize=LABEL_SIZE)
ax1.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE)
ax1.tick_params(axis="y", labelsize=TICK_SIZE)
apply_horizontal_grid(ax1)

ax2 = ax1.twinx()
ax2.plot(quality_df["model"], quality_df["fps"], color="#9a9a9a", marker="o", linewidth=2)
ax2.set_ylabel("Inference Speed (FPS)", fontsize=LABEL_SIZE)
ax2.tick_params(axis="y", labelsize=TICK_SIZE, length=0)
ax2.grid(False)

legend_handles = [Patch(color=model_palette[m], label=m) for m in available_models]
legend_handles.append(Line2D([0], [0], color="#9a9a9a", marker="o", label="FPS"))
fig.legend(handles=legend_handles, title="Model", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(out_dir / "plot_quality_vs_speed.png", dpi=DPI)
plt.close(fig)


# Plot 3: Player tracking quality grouped by metric (x-axis)
player_df = metrics_per_class_df[metrics_per_class_df["class"] == "player"].copy()
player_metrics = ["HOTA", "IDF1", "MOTA"]
player_plot_df = player_df[["model"] + player_metrics].melt(
    id_vars="model",
    value_vars=player_metrics,
    var_name="metric",
    value_name="score",
)
player_plot_df["metric"] = player_plot_df["metric"].map(format_metric_labels)
player_plot_df["score"] = player_plot_df["score"] * 100

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=player_plot_df,
    x="metric",
    y="score",
    hue="model",
    hue_order=available_models,
    palette=model_palette,
    ax=ax,
)
ax.set_title("Player Tracking Quality - HOTA/IDF1/MOTA", fontsize=TITLE_SIZE)
ax.set_xlabel("Metric", fontsize=LABEL_SIZE)
ax.set_ylabel("Score (%)", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
apply_horizontal_grid(ax)
ax.legend(title="Model", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig(out_dir / "plot_player_metrics.png", dpi=DPI)
plt.close(fig)


# Plot 4: Player count vs GT (reference line)
player_stats = per_sequence_stats_df[per_sequence_stats_df["class"] == "player"].copy()
player_stats = player_stats.groupby("model", as_index=False).agg(
    pred_tracks=("pred_tracks", "mean"),
    gt_tracks=("gt_tracks", "mean"),
)
player_stats["model"] = pd.Categorical(player_stats["model"], categories=available_models, ordered=True)
player_stats = player_stats.sort_values("model")
gt_mean = player_stats["gt_tracks"].mean()

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = [model_palette[model] for model in player_stats["model"]]
ax.bar(player_stats["model"], player_stats["pred_tracks"], color=bar_colors, alpha=0.85)
ax.axhline(gt_mean, color="black", linestyle="--", linewidth=1.5, label=f"GT mean ({gt_mean:.0f})")
ax.set_title("Tracked Players per Sequence vs GT", fontsize=TITLE_SIZE)
ax.set_xlabel("Model", fontsize=LABEL_SIZE)
ax.set_ylabel("Unique Players per Sequence (mean)", fontsize=LABEL_SIZE)
ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
apply_horizontal_grid(ax)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(out_dir / "plot_player_count_vs_gt.png", dpi=DPI)
plt.close(fig)


# Plot 5: Identity stability (total ID switches)
id_switch_df = metrics_per_class_df.groupby("model", as_index=False)["ID-switch"].sum()
id_switch_df["model"] = pd.Categorical(id_switch_df["model"], categories=available_models, ordered=True)
id_switch_df = id_switch_df.sort_values("model")

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = [model_palette[model] for model in id_switch_df["model"]]
ax.bar(id_switch_df["model"], id_switch_df["ID-switch"], color=bar_colors, alpha=0.85)
ax.set_title("Identity Stability - Total ID Switches", fontsize=TITLE_SIZE)
ax.set_xlabel("Model", fontsize=LABEL_SIZE)
ax.set_ylabel("ID Switches (lower is better)", fontsize=LABEL_SIZE)
ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
apply_horizontal_grid(ax)
fig.tight_layout()
fig.savefig(out_dir / "plot_id_switches.png", dpi=DPI)
plt.close(fig)

print(f"Presentation plots generated in {out_dir}")
