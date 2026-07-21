from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT_DIR = Path("results/project/plots/first5")
OUT_PATH = OUT_DIR / "map_vs_fps.png"


def _yolo_best_map(results_csv: Path) -> dict[str, float]:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    m50 = "metrics/mAP50(B)"
    m5095 = "metrics/mAP50-95(B)"
    best = df.loc[df[m5095].idxmax()]
    return {"map50": float(best[m50]), "map50_95": float(best[m5095])}


def _rfdetr_map(results_json: Path) -> dict[str, float]:
    obj = json.loads(results_json.read_text(encoding="utf-8"))
    valid = obj.get("class_map", {}).get("valid")
    if not isinstance(valid, list):
        raise ValueError("Unexpected RF-DETR results format (missing class_map.valid)")
    all_row = next((r for r in valid if r.get("class") == "all"), None)
    if not all_row:
        raise ValueError("Missing class=all in RF-DETR class_map.valid")
    return {"map50": float(all_row["map@50"]), "map50_95": float(all_row["map@50:95"])}


def _fps_summary(inference_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(inference_csv)
    return (
        df.groupby("model", as_index=False)
        .agg(
            fps_mean=("detection_fps", "mean"),
        )
        .copy()
    )


def main() -> None:
    models = ["yolov8m", "yolo11m", "yolo12m", "rfdetr_base"]

    map_rows = []
    map_rows.append(
        {
            "model": "yolov8m",
            **_yolo_best_map(
                Path("results/project/raw/yolov8m/training/detect_yolov8m2/results.csv")
            ),
        }
    )
    map_rows.append(
        {
            "model": "yolo11m",
            **_yolo_best_map(
                Path("results/project/raw/yolo11m/training/detect_yolo11m/results.csv")
            ),
        }
    )
    map_rows.append(
        {
            "model": "yolo12m",
            **_yolo_best_map(
                Path("results/project/raw/yolo12m/training/detect_yolo12m/results.csv")
            ),
        }
    )
    map_rows.append(
        {
            "model": "rfdetr_base",
            **_rfdetr_map(Path("results/project/raw/rfdetr_base/training/results.json")),
        }
    )
    map_df = pd.DataFrame(map_rows)

    fps_df = _fps_summary(Path("results/project/numeric/inference_timings_all_models.csv"))

    df = map_df.merge(fps_df, on="model", how="inner")
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df = df.sort_values("model")

    long_map = df.melt(
        id_vars=["model", "fps_mean"],
        value_vars=["map50_95", "map50"],
        var_name="metric",
        value_name="value",
    )
    long_map["metric"] = long_map["metric"].map(
        {"map50_95": "mAP@50:95 (val)", "map50": "mAP@50 (val)"}
    )

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep", 3)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=long_map,
        x="model",
        y="value",
        hue="metric",
        palette=[palette[0], palette[1]],
        ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP (validation)")
    ax.set_xlabel("")
    ax.set_title("mAP vs FPS")

    # Annotate bar values.
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=3, fontsize=9)

    # Secondary axis for FPS.
    ax2 = ax.twinx()
    x = np.arange(len(df))
    ax2.plot(
        x,
        df["fps_mean"].to_numpy(),
        color=palette[2],
        marker="^",
        linewidth=2,
        label="FPS (mean)",
    )
    # Keep FPS values on the plot, but remove the right-side axis styling (ticks/spine).
    ax2.set_ylabel("FPS (detector-only)")
    ax2.set_yticks([])
    ax2.spines["right"].set_visible(False)

    # Make room for FPS labels.
    fps_max = float(np.nanmax(df["fps_mean"].to_numpy()))
    ax2.set_ylim(0, max(50.0, fps_max * 1.15))
    for i, row in df.reset_index(drop=True).iterrows():
        ax2.text(
            i,
            float(row["fps_mean"]) + max(2.0, 0.02 * fps_max),
            f"{row['fps_mean']:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=9,
            color=palette[2],
        )

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True)
    ax.get_legend().set_title("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
