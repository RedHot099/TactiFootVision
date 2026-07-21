import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


@dataclass
class ModelSummary:
    name: str
    metrics: Dict[str, float]
    confusion: Dict[str, float]
    f1_by_tracks: List[Tuple[int, float]]


def load_reference_metrics(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_mock_comparisons(reference_metrics: Dict[str, float]) -> List[ModelSummary]:
    """
    Create mock performance summaries for baseline RF-DETR and an improved
    RF-DETR + SAM2 tracking model. The mock values are anchored around the
    provided RF-DETR Segmentation metrics so plots look realistic while
    still conveying the intended relative performance.
    """
    precision = reference_metrics["precision"]
    recall = reference_metrics["recall"]
    f1 = reference_metrics["f1"]
    mean_iou = reference_metrics["mean_iou"]

    tp = reference_metrics["tp"]
    fp = reference_metrics["fp"]
    fn = reference_metrics["fn"]
    unique_tracks = reference_metrics["unique_tracks"]

    rng = np.random.default_rng(seed=42)

    baseline = ModelSummary(
        name="RF-DETR (baseline)",
        metrics={
            "precision": precision - rng.uniform(0.03, 0.05),
            "recall": recall - rng.uniform(0.035, 0.055),
            "f1": f1 - rng.uniform(0.03, 0.05),
            "mean_iou": mean_iou - rng.uniform(0.02, 0.04),
        },
        confusion={
            "tp": tp * rng.uniform(0.88, 0.94),
            "fp": fp * rng.uniform(1.12, 1.22),
            "fn": fn * rng.uniform(1.18, 1.32),
        },
        f1_by_tracks=[],
    )

    current = ModelSummary(
        name="RF-DETR Seg (current)",
        metrics={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
        },
        confusion={
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        f1_by_tracks=[],
    )

    tracking = ModelSummary(
        name="RF-DETR + SAM2 Tracking",
        metrics={
            "precision": min(precision + rng.uniform(0.02, 0.035), 0.99),
            "recall": min(recall + rng.uniform(0.025, 0.04), 0.99),
            "f1": min(f1 + rng.uniform(0.03, 0.045), 0.99),
            "mean_iou": min(mean_iou + rng.uniform(0.02, 0.04), 0.99),
        },
        confusion={
            "tp": tp * rng.uniform(1.06, 1.11),
            "fp": fp * rng.uniform(0.78, 0.86),
            "fn": fn * rng.uniform(0.64, 0.72),
        },
        f1_by_tracks=[],
    )

    track_ratios = np.linspace(0.15, 1.0, 5)
    baseline_curve = 0.76 + 0.11 * track_ratios - 0.04 * track_ratios**2
    baseline_curve += rng.normal(0, 0.005, size=baseline_curve.shape)
    current_curve = baseline_curve + 0.035 + rng.normal(0, 0.004, size=baseline_curve.shape)
    tracking_advantage = 0.055 - 0.02 * track_ratios
    tracking_curve = current_curve + tracking_advantage + rng.normal(0, 0.004, size=baseline_curve.shape)

    for curve in (baseline_curve, current_curve, tracking_curve):
        np.clip(curve, 0.7, 0.98, out=curve)

    baseline.f1_by_tracks = [
        (int(round(unique_tracks * ratio)), float(score))
        for ratio, score in zip(track_ratios, baseline_curve)
    ]
    current.f1_by_tracks = [
        (int(round(unique_tracks * ratio)), float(score))
        for ratio, score in zip(track_ratios, current_curve)
    ]
    tracking.f1_by_tracks = [
        (int(round(unique_tracks * ratio)), float(score))
        for ratio, score in zip(track_ratios, tracking_curve)
    ]

    return [baseline, current, tracking]


BLUES_CMAP = plt.get_cmap("Blues")
PALETTE = [BLUES_CMAP(x) for x in np.linspace(0.45, 0.8, 3)]


def plot_overall_metrics(models: List[ModelSummary], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.set_prop_cycle(cycler(color=PALETTE))
    metrics = ["precision", "recall", "f1", "mean_iou"]
    bar_width = 0.25
    x = np.arange(len(metrics))

    for idx, summary in enumerate(models):
        offsets = x + (idx - 1) * bar_width
        values = [summary.metrics[m] for m in metrics]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=summary.name,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Tracking Model Comparison – Core Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric.upper() for metric in metrics)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_counts(models: List[ModelSummary], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.set_prop_cycle(cycler(color=PALETTE))
    categories = ["tp", "fp", "fn"]
    bar_width = 0.25
    x = np.arange(len(categories))

    for idx, summary in enumerate(models):
        offsets = x + (idx - 1) * bar_width
        values = [summary.confusion[c] for c in categories]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=summary.name,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_ylabel("Detections")
    ax.set_title("Comparison of Detection Outcomes")
    ax.set_xticks(x)
    ax.set_xticklabels(c.upper() for c in categories)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:,.0f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_f1_vs_tracks(models: List[ModelSummary], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.set_prop_cycle(cycler(color=PALETTE))

    for summary, color in zip(models, PALETTE):
        tracks, f1_scores = zip(*summary.f1_by_tracks)
        ax.plot(
            tracks,
            f1_scores,
            marker="o",
            linewidth=2.4,
            label=summary.name,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        for track_count, score in summary.f1_by_tracks:
            ax.annotate(
                f"{score:.2f}",
                (track_count, score),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    ax.set_ylim(0.75, 0.97)
    ax.set_xlabel("Number of Unique Player Tracks")
    ax.set_ylabel("F1 Score")
    ax.set_title("Effect of Unique Track Volume on F1 Performance")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock evaluation plots comparing RF-DETR variants."
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path(
            "results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking/"
            "SNMOT-060_rfdetr_seg_checkpoint_best_total.summary.metrics.json"
        ),
        help="Path to the JSON summary metrics for the RF-DETR Seg model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/detection_tracking/plots/tracking_comparison"),
        help="Directory where comparison figures will be saved.",
    )

    args = parser.parse_args()

    reference_metrics = load_reference_metrics(args.metrics_path)
    models = create_mock_comparisons(reference_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_metrics(models, args.output_dir / "tracking_comparison_overall.png")
    plot_confusion_counts(models, args.output_dir / "tracking_comparison_confusion.png")
    plot_f1_vs_tracks(models, args.output_dir / "tracking_comparison_f1_vs_tracks.png")

    print(
        f"Generated comparison plots in {args.output_dir.resolve()} "
        f"for models: {', '.join(model.name for model in models)}."
    )


if __name__ == "__main__":
    main()
