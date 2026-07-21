#!/usr/bin/env python3
"""Plot per-step comparisons (color space, crop ratios/methods, UMAP) from metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

STAGE_RATIO_SWEEP = "ratio_sweep"
STAGE_OPENCV_COMPARISON = "opencv_comparison"
STAGE_COLOR_UMAP = "umap_color"


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df


def best_per_group(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        df.sort_values(by + ["accuracy"], ascending=[True] * len(by) + [False])
        .groupby(by, as_index=False)
        .first()
    )


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out: Path, hue: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 4))
    if hue:
        pivot = df.pivot(index=x, columns=hue, values=y)
        pivot.plot(kind="bar", ax=plt.gca())
    else:
        plt.bar(df[x], df[y])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pipeline step comparisons from a metrics CSV.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/team_classification/numeric/team_classification_metrics_aug.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/team_classification/plots/plots_aug"))
    args = parser.parse_args()

    df = load_metrics(args.metrics_csv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Color spaces: best per color across all configs.
    color_best = best_per_group(df, ["color_space"])
    plot_bar(color_best, "color_space", "accuracy", "Best accuracy per color space", out_dir / "colors.png")

    # 2. Crop ratios (center crops only, ratio_sweep stage).
    ratio_df = df[df["experiment_stage"] == STAGE_RATIO_SWEEP]
    ratio_best = best_per_group(ratio_df, ["crop_center_ratio", "color_space"])
    plot_bar(
        ratio_best,
        "crop_center_ratio",
        "accuracy",
        "Accuracy vs center crop ratio (best per color)",
        out_dir / "ratios.png",
        hue="color_space",
    )

    # 3. Crop method comparison (best per method).
    crop_best = best_per_group(df, ["crop_method"])
    plot_bar(crop_best, "crop_method", "accuracy", "Best accuracy per crop method", out_dir / "crop_methods.png")

    # 4. UMAP vs no-UMAP (use color_UMAP stage for reduced, ratio_sweep/ocv for non-UMAP best).
    umap_rows = df[df["experiment_stage"] == STAGE_COLOR_UMAP]
    umap_best = best_per_group(umap_rows, ["crop_method", "color_space"])
    if not umap_best.empty:
        plot_bar(
            umap_best,
            "crop_method",
            "accuracy",
            "UMAP (n=3) accuracy per crop method/color",
            out_dir / "umap_methods.png",
            hue="color_space",
        )

    if "embedding_backend" in df.columns:
        embed_best = best_per_group(df, ["embedding_backend"])
        plot_bar(
            embed_best,
            "embedding_backend",
            "accuracy",
            "Best accuracy per embedding backend",
            out_dir / "embedding_backends.png",
        )


if __name__ == "__main__":
    main()
