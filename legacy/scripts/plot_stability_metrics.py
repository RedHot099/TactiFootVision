#!/usr/bin/env python3
"""Generate stability metrics visualization plots.

Style and colors matched to generate_presentation_plots.py for visual consistency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Same settings as generate_presentation_plots.py
sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})

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


def apply_horizontal_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#d0d0d0", alpha=0.45, linewidth=0.8)
    ax.grid(False, axis="x")


def plot_stability_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison bar charts for stability metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to main trackers only
    main_trackers = ["bytetrack", "botsort_reid", "sam2"]
    df_main = df[df["tracker_type"].isin(main_trackers)].copy()
    
    if df_main.empty:
        print("No main trackers found for plotting")
        return
    
    # Create model labels (same as generate_presentation_plots.py)
    df_main["tracker_label"] = df_main["tracker_type"].map(TRACKER_LABELS).fillna(df_main["tracker_type"])
    df_main["detector_label"] = df_main["detector"].map(DETECTOR_LABELS).fillna(df_main["detector"])
    df_main["model"] = df_main["detector_label"] + " + " + df_main["tracker_label"]
    
    # Aggregate across classes
    agg = df_main.groupby("model", as_index=False).agg({
        "isr_mean": "mean",
        "tci": "mean",
        "mss_mean": "mean",
    })
    
    # Get available models in correct order
    available_models = [model for model in MODEL_ORDER if model in agg["model"].unique()]
    model_palette = {model: MODEL_COLORS[model] for model in available_models}
    
    # Melt to long format: x=metric, hue=model
    melted = agg.melt(
        id_vars=["model"],
        value_vars=["isr_mean", "tci", "mss_mean"],
        var_name="metric",
        value_name="score",
    )
    
    # Pretty metric names
    metric_map = {
        "isr_mean": "ISR",
        "tci": "TCI",
        "mss_mean": "MSS",
    }
    melted["metric"] = melted["metric"].map(metric_map)
    
    # Convert to percentage (0-100 scale like other plots)
    melted["score"] = melted["score"] * 100
    
    # Main plot: metrics on X-axis, models as hue (colors)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="metric",
        y="score",
        hue="model",
        hue_order=available_models,
        palette=model_palette,
        ax=ax,
    )
    ax.set_title("Trajectory Stability Metrics - ISR / TCI / MSS", fontsize=TITLE_SIZE)
    ax.set_xlabel("Metric", fontsize=LABEL_SIZE)
    ax.set_ylabel("Score (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    apply_horizontal_grid(ax)
    ax.legend(title="Model", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(output_dir / "stability_metrics_overview.png", dpi=DPI)
    plt.close(fig)
    
    print(f"Generated plot: {output_dir / 'stability_metrics_overview.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate stability metrics plots")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to stability_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to parent of metrics CSV)",
    )
    args = parser.parse_args()
    
    if not args.metrics_csv.is_file():
        print(f"Error: Metrics file not found: {args.metrics_csv}")
        sys.exit(1)
    
    df = pd.read_csv(args.metrics_csv)
    output_dir = args.output_dir or (args.metrics_csv.parent / "plots" / "stability")
    
    plot_stability_comparison(df, output_dir)


if __name__ == "__main__":
    main()
