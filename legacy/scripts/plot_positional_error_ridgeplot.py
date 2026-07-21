from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

SAMPLES_PATH = Path("results/project/numeric/statsbomb360_eval/positional_error_samples.parquet")
SUMMARY_PATH = Path("results/project/numeric/statsbomb360_eval/positional_error_summary.csv")

OUT_PNG = Path("results/project/plots/statsbomb360_eval/positional_error_ridgeplot.png")
OUT_SVG = Path("results/project/plots/statsbomb360_eval/positional_error_ridgeplot.svg")

X_MAX = 100.0
MAX_SAMPLES_PER_MODEL = 120_000
RANDOM_SEED = 0

MODEL_ORDER = ["rfdetr_base", "yolov8m", "yolo11m", "yolo12m"]
MODEL_LABELS = {
    "rfdetr_base": "RF-DETR",
    "yolov8m": "YOLOv8",
    "yolo11m": "YOLOv11",
    "yolo12m": "YOLOv12",
}
MODEL_COLORS = {
    "rfdetr_base": "#5cb85c",  # green
    "yolov8m": "#b39ddb",  # purple
    "yolo11m": "#f6b26b",  # orange
    "yolo12m": "#fff176",  # yellow
}


def _load_samples() -> pd.DataFrame:
    if not SAMPLES_PATH.is_file():
        raise FileNotFoundError(f"Missing samples parquet: {SAMPLES_PATH}")
    df = pd.read_parquet(SAMPLES_PATH, columns=["model", "distance"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance"])
    df = df[(df["distance"] >= 0.0) & (df["distance"] <= X_MAX)].copy()
    return df


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    df = df.set_index("model")
    return df


def main() -> None:
    df = _load_samples()
    summary = _load_summary()

    rng = np.random.default_rng(RANDOM_SEED)
    xs = np.linspace(0.0, X_MAX, 800)

    fig = plt.figure(figsize=(10.6, 6.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(0.0, X_MAX)
    ax.set_xlabel("Positional error (Euclidean distance, pitch units)")
    ax.set_ylabel("")

    ridge_height = 0.9
    ridge_gap = 1.0

    y_positions = {}
    for i, model in enumerate(MODEL_ORDER):
        y0 = (len(MODEL_ORDER) - 1 - i) * ridge_gap
        y_positions[model] = y0

        d = df.loc[df["model"] == model, "distance"].to_numpy(dtype=np.float64)
        if d.size == 0:
            continue
        if d.size > MAX_SAMPLES_PER_MODEL:
            d = rng.choice(d, size=MAX_SAMPLES_PER_MODEL, replace=False)

        kde = gaussian_kde(d, bw_method="scott")
        dens = kde(xs)
        dens = np.clip(dens, 0.0, np.inf)
        if np.max(dens) > 0:
            dens = dens / np.max(dens)
        dens = dens * ridge_height

        color = MODEL_COLORS.get(model, "#999999")
        ax.fill_between(xs, y0, y0 + dens, color=color, alpha=0.7, linewidth=0)
        ax.plot(xs, y0 + dens, color="black", linewidth=0.8)

        # y-axis label at the ridge baseline (like in the reference plot)
        ax.text(
            0.0,
            y0,
            MODEL_LABELS.get(model, model),
            ha="right",
            va="center",
            fontsize=10,
            color="black",
            clip_on=False,
        )

        # mean/median markers
        if model in summary.index:
            med = float(summary.loc[model, "median"])
            mean = float(summary.loc[model, "mean"])
            ax.vlines(med, y0, y0 + ridge_height, color="black", linewidth=1.4, alpha=0.9)
            ax.vlines(mean, y0, y0 + ridge_height, color="black", linewidth=0.8, alpha=0.6)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(0.0, X_MAX + 0.001, 25.0))
    ax.grid(axis="x", linestyle="-", alpha=0.15)
    ax.set_ylim(-0.2, (len(MODEL_ORDER) - 1) * ridge_gap + ridge_height + 0.3)

    # Remove plot frame/spines (no frame like the reference)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", bottom=True, top=False)

    # Table as part of the plot (inset)
    table_rows = []
    for model in MODEL_ORDER:
        if model not in summary.index:
            continue
        row = summary.loc[model]
        table_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"{float(row['median']):.2f}",
                f"{float(row['mean']):.2f}",
                f"{float(row['sd']):.2f}",
            ]
        )

    col_labels = ["Model", "Median", "Mean", "SD"]
    ax_inset = inset_axes(
        ax,
        width="44%",
        height="62%",
        loc="upper right",
        borderpad=0.6,
    )
    ax_inset.axis("off")
    table = ax_inset.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="upper right",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.08, 1.7)

    # Color rows to match ridges
    for r, model in enumerate(MODEL_ORDER, start=1):  # +1 for header row
        if model not in summary.index:
            continue
        color = MODEL_COLORS.get(model, "#dddddd")
        for c in range(len(col_labels)):
            table[(r, c)].set_facecolor(color)
            table[(r, c)].set_alpha(0.75)

    for c in range(len(col_labels)):
        table[(0, c)].set_facecolor("#f0f0f0")
        table[(0, c)].set_alpha(1.0)

    # Remove table borders/gridlines
    for cell in table.get_celld().values():
        cell.set_linewidth(0.0)
        cell.set_edgecolor((0, 0, 0, 0))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")


if __name__ == "__main__":
    main()
