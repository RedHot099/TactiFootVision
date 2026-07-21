from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


IN_PATH = Path("results/project/numeric/inference_timings_all_models.csv")
OUT_DIR = Path("results/project/plots/first5")
OUT_PATH = OUT_DIR / "inference_fps_boxplot.png"


def main() -> None:
    if not IN_PATH.is_file():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    order = ["yolov8m", "yolo11m", "yolo12m", "rfdetr_base"]
    df = df[df["model"].isin(order)].copy()

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(
        data=df,
        x="model",
        y="detection_fps",
        order=order,
        ax=ax,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2ca02c",
            "markeredgecolor": "#2ca02c",
            "markersize": 8,
        },
    )

    ax.set_title("Szybkość inferencji (FPS) — rozkład po meczach")
    ax.set_ylabel("FPS (tylko detektor)")
    ax.set_xlabel("")

    # Annotate summary stats above each box.
    y_max = float(np.nanmax(df["detection_fps"].to_numpy()))
    y_pad = max(2.0, 0.03 * y_max)
    for i, model in enumerate(order):
        fps = df.loc[df["model"] == model, "detection_fps"].astype(float).to_numpy()
        if fps.size == 0:
            continue
        mean = float(np.mean(fps))
        median = float(np.median(fps))
        q1 = float(np.quantile(fps, 0.25))
        q3 = float(np.quantile(fps, 0.75))
        top = max(float(np.max(fps)), q3) + y_pad
        ax.text(
            i,
            top,
            f"mean {mean:.1f}\nmed {median:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
