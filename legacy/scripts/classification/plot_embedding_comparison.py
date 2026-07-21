#!/usr/bin/env python3
"""Compare embedding backends (e.g., SigLIP vs ResNet) from multiple metrics CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(files: Dict[str, Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name, path in files.items():
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["source_file"] = name
        if "embedding_backend" not in df.columns:
            df["embedding_backend"] = name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No metrics CSVs found for embedding comparison.")
    return pd.concat(frames, ignore_index=True)


def best_per_backend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["embedding_backend", "accuracy"], ascending=[True, False])
        .groupby("embedding_backend", as_index=False)
        .first()
    )


def best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            ["embedding_backend", "crop_method", "accuracy"], ascending=[True, True, False]
        )
        .groupby(["embedding_backend", "crop_method"], as_index=False)
        .first()
    )


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out: Path, hue: str | None = None) -> None:
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
    parser = argparse.ArgumentParser(description="Compare embedding backends from metrics CSVs.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=Path,
        default=[
            Path("results/team_classification/numeric/team_classification_metrics_full_sam2.csv"),
            Path("results/team_classification/numeric/team_classification_metrics_full_sam2_resnet.csv"),
        ],
        help="List of metrics CSVs, ideally one per embedding backend.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/team_classification/plots/plots_full"))
    args = parser.parse_args()

    mapping = {path.stem: path for path in args.metrics}
    df = load_metrics(mapping)

    overall = best_per_backend(df)
    per_method = best_per_method(df)

    plot_bar(
        overall,
        x="embedding_backend",
        y="accuracy",
        title="Best accuracy per embedding backend",
        out=args.output_dir / "embedding_overall.png",
    )
    plot_bar(
        per_method,
        x="embedding_backend",
        y="accuracy",
        title="Best accuracy per backend and crop method",
        out=args.output_dir / "embedding_by_crop.png",
        hue="crop_method",
    )


if __name__ == "__main__":
    main()
