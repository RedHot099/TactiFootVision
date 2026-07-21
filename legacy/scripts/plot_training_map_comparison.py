from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT_DIR = Path("results/project/plots/first5")
OUT_PATH = OUT_DIR / "training_map_comparison.png"


def _yolo_train_metrics(results_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    m50 = "metrics/mAP50(B)"
    m5095 = "metrics/mAP50-95(B)"
    precision = "metrics/precision(B)"
    recall = "metrics/recall(B)"

    best_idx = df[m5095].idxmax()
    best = df.loc[best_idx]
    return {
        "best_epoch": int(best["epoch"]),
        "map50": float(best[m50]),
        "map50_95": float(best[m5095]),
        "precision": float(best[precision]),
        "recall": float(best[recall]),
    }


def _rfdetr_metrics(results_json: Path) -> dict[str, float]:
    obj = json.loads(results_json.read_text(encoding="utf-8"))
    valid = obj.get("class_map", {}).get("valid")
    if not isinstance(valid, list):
        raise ValueError("Unexpected RF-DETR results format (missing class_map.valid)")
    all_row = next((r for r in valid if r.get("class") == "all"), None)
    if not all_row:
        raise ValueError("Missing class=all in RF-DETR class_map.valid")
    return {
        "map50": float(all_row["map@50"]),
        "map50_95": float(all_row["map@50:95"]),
        "precision": float(all_row["precision"]),
        "recall": float(all_row["recall"]),
    }


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    y_max = ax.get_ylim()[1]
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(height + 0.015, y_max - 0.01),
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main() -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep", 2)

    metrics = {
        "yolov8m": _yolo_train_metrics(
            Path("results/project/raw/yolov8m/training/detect_yolov8m2/results.csv")
        ),
        "yolo11m": _yolo_train_metrics(
            Path("results/project/raw/yolo11m/training/detect_yolo11m/results.csv")
        ),
        "yolo12m": _yolo_train_metrics(
            Path("results/project/raw/yolo12m/training/detect_yolo12m/results.csv")
        ),
        "rfdetr_base": _rfdetr_metrics(Path("results/project/raw/rfdetr_base/training/results.json")),
    }

    order = ["yolov8m", "yolo11m", "yolo12m", "rfdetr_base"]
    df = pd.DataFrame.from_dict(metrics, orient="index").loc[order]

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_5095 = ax.bar(
        x - width / 2,
        df["map50_95"].to_numpy(dtype=float),
        width,
        label="mAP@50:95 (val)",
        color=palette[0],
    )
    bars_50 = ax.bar(
        x + width / 2,
        df["map50"].to_numpy(dtype=float),
        width,
        label="mAP@50 (val)",
        color=palette[1],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.set_title("Jakość detekcji na walidacji (trening)")
    ax.set_ylabel("mAP")
    ax.legend(loc="lower right")

    _annotate_bars(ax, bars_5095)
    _annotate_bars(ax, bars_50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
