from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results/team_classification")
NUMERIC_DIR = RESULTS_DIR / "numeric"
OUT_DIR = RESULTS_DIR / "plots" / "plots_classification_improved"


def _configure_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 120,
        }
    )


def _infer_clustering_method(path: Path) -> str | None:
    name = path.name.lower()
    if "kmeans" in name:
        return "kmeans"
    if "dbscan" in name:
        return "dbscan"
    if "cmeans" in name:
        return "cmeans"
    return None


def _load_metrics() -> pd.DataFrame:
    paths = [NUMERIC_DIR / "soccernet_tracking_team_classification_metrics_structured.csv"]
    if not paths[0].is_file():
        raise FileNotFoundError(
            "Missing metrics file: results/team_classification/numeric/"
            "soccernet_tracking_team_classification_metrics_structured.csv"
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name

        if "cluster_method" in df.columns and "clustering_method" not in df.columns:
            df["clustering_method"] = df["cluster_method"].astype(str).str.lower()
        else:
            clustering_method = _infer_clustering_method(path)
            if clustering_method is not None:
                df["clustering_method"] = clustering_method

        if "embedding_backend" not in df.columns:
            df["embedding_backend"] = "clip"
        else:
            df["embedding_backend"] = df["embedding_backend"].astype(str).str.lower()

        df["crop_method"] = df["crop_method"].astype(str).str.lower()
        df["color_space"] = df["color_space"].astype(str).str.lower()
        df["umap_applied"] = df["umap_applied"].astype(bool)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Normalize display labels.
    data["crop_method_display"] = (
        data["crop_method"]
        .replace(
            {
                "center": "Center crop",
                "opencv_mask": "OpenCV",
                "sam2_mask": "SAM2",
            }
        )
        .astype("category")
    )
    data["color_space_display"] = (
        data["color_space"].replace({"rgb": "RGB", "h": "Hue"}).astype("category")
    )
    data["embedding_backend_display"] = (
        data["embedding_backend"]
        .replace(
            {
                "clip": "CLIP",
                "siglip": "SigLIP",
                "resnet": "ResNet",
                "resnet16": "ResNet",
            }
        )
        .astype("category")
    )
    if "clustering_method" in data.columns:
        data["clustering_method_display"] = (
            data["clustering_method"]
            .replace({"kmeans": "KMeans", "dbscan": "DBSCAN", "cmeans": "CMeans"})
            .astype("category")
        )

    return data


def _save(fig: mpl.figure.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_color_space_comparison(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = sns.color_palette("mako", n_colors=2)

    order = ["RGB", "Hue"]
    sns.violinplot(
        data=data,
        x="color_space_display",
        hue="color_space_display",
        y="accuracy",
        order=order,
        inner=None,
        cut=0,
        linewidth=1,
        palette=palette,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=data,
        x="color_space_display",
        y="accuracy",
        order=order,
        color="black",
        size=2.5,
        alpha=0.35,
        jitter=0.25,
        ax=ax,
    )

    ax.set_title("Impact of Color Space on Classification Accuracy")
    ax.set_xlabel("Color Space")
    ax.set_ylabel("Accuracy")

    for i, label in enumerate(order):
        subset = data.loc[data["color_space_display"] == label, "accuracy"].dropna().to_numpy()
        if subset.size == 0:
            continue
        mean = float(np.mean(subset))
        median = float(np.median(subset))
        ax.hlines(
            mean,
            i - 0.35,
            i + 0.35,
            colors="black",
            linestyles="--",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.text(
            i,
            min(1.0, float(np.quantile(subset, 0.98)) + 0.015),
            f"median={median:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    sns.despine(ax=ax)
    _save(fig, "plot_color_space_comparison.png")


def plot_crop_method_comparison(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    subset = data.copy()
    subset = subset[(subset["umap_applied"] == False)].copy()  # noqa: E712

    def _center_label(r: float) -> str:
        if pd.isna(r):
            return "?"
        return f"{float(r):.1f}"

    subset["preprocessing_label"] = np.where(
        subset["crop_method"].eq("center"),
        subset["crop_center_ratio"].apply(_center_label),
        subset["crop_method_display"].astype(str),
    )
    subset["preprocessing_label"] = subset["preprocessing_label"].replace({"OpenCV": "GC"}).astype(str)

    center_ratios = (
        subset.loc[subset["crop_method"].eq("center"), "crop_center_ratio"]
        .dropna()
        .astype(float)
        .sort_values()
        .unique()
        .tolist()
    )
    order = [f"{r:.1f}" for r in center_ratios]
    for extra in ["GC", "SAM2"]:
        if extra in subset["preprocessing_label"].unique():
            order.append(extra)

    # Encode color by performance (bar height), not by x-axis order.
    center_palette = sns.color_palette("viridis", n_colors=max(2, len(center_ratios)))
    palette: dict[str, tuple[float, float, float]] = {}
    if "?" in subset["preprocessing_label"].unique():
        palette["?"] = (0.65, 0.65, 0.65)
    if "GC" in order:
        palette["GC"] = sns.color_palette("mako", n_colors=6)[4]
    if "SAM2" in order:
        palette["SAM2"] = sns.color_palette("mako", n_colors=6)[2]

    means = (
        subset.groupby("preprocessing_label", observed=True)["accuracy"]
        .mean()
        .reindex(order)
        .astype(float)
    )
    valid_means = means.dropna()
    if not valid_means.empty:
        vmin = float(valid_means.min())
        vmax = float(valid_means.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        cmap = plt.get_cmap("viridis")
        for label, value in valid_means.items():
            if label in {"GC", "SAM2", "?"}:
                continue
            t = (float(value) - vmin) / (vmax - vmin)
            palette[str(label)] = cmap(t)[:3]
    sns.barplot(
        data=subset,
        x="preprocessing_label",
        hue="preprocessing_label",
        y="accuracy",
        order=order,
        errorbar=("ci", 95),
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=1.1,
        legend=False,
    )

    baseline_idx = order.index("0.2") if "0.2" in order else None
    for j, patch in enumerate(ax.patches[: len(order)]):
        if baseline_idx is not None and j == baseline_idx:
            patch.set_alpha(1.0)
            patch.set_linewidth(1.8)
        else:
            patch.set_alpha(0.92)

    ax.set_title("Effect of Preprocessing on Classification Accuracy")
    ax.set_xlabel("Preprocessing method")
    ax.set_ylabel("Mean Accuracy")

    for j, value in enumerate(means.to_numpy()):
        if pd.isna(value):
            continue
        ax.text(j, float(value) + 0.01, f"{float(value):.2f}", ha="center", va="bottom", fontsize=11)

    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax)
    _save(fig, "plot_crop_method_comparison.png")


def plot_embedding_comparison(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    order = ["CLIP", "SigLIP", "ResNet"]
    palette = sns.color_palette("viridis", n_colors=len(order))

    subset = data[data["embedding_backend_display"].isin(order)].copy()
    sns.boxplot(
        data=subset,
        x="embedding_backend_display",
        y="accuracy",
        order=order,
        palette=palette,
        ax=ax,
        showfliers=False,
        width=0.55,
        linewidth=1.2,
    )
    sns.stripplot(
        data=subset,
        x="embedding_backend_display",
        y="accuracy",
        order=order,
        color="black",
        size=2.2,
        alpha=0.25,
        jitter=0.25,
        ax=ax,
    )

    ax.set_title("Performance of Feature Extraction Models")
    ax.set_xlabel("Embedding backbone")
    ax.set_ylabel("Accuracy")

    sns.despine(ax=ax)
    _save(fig, "plot_embedding_comparison.png")


def plot_dim_reduction_impact(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    subset = data.copy()
    subset = subset[(subset["color_space_display"] == "RGB") & (subset["experiment_stage"] == "umap_color")].copy()

    # Keep a single comparable preprocessing setting (most common center ratio for center-crop).
    center = subset[subset["crop_method"].eq("center")].copy()
    if not center.empty:
        mode_ratio = float(center["crop_center_ratio"].mode().iloc[0])
        subset = subset[(subset["crop_method"].eq("center")) & (subset["crop_center_ratio"].astype(float).eq(mode_ratio))].copy()

    baseline = subset[subset["umap_applied"] == False]["accuracy"].dropna()  # noqa: E712
    if baseline.empty and not center.empty:
        baseline = data[
            (data["color_space_display"] == "RGB")
            & (data["crop_method"].eq("center"))
            & (data["crop_center_ratio"].astype(float).eq(mode_ratio))
            & (data["umap_applied"] == False)  # noqa: E712
        ]["accuracy"].dropna()
    umap = subset[(subset["umap_applied"] == True) & (subset["umap_components"].astype(float) < subset["original_feature_dim"].astype(float))].copy()  # noqa: E712

    if umap.empty:
        raise ValueError("No UMAP-applied rows found for RGB in experiment_stage='umap_color'.")

    stats = (
        umap.groupby("umap_components", observed=True)["accuracy"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("umap_components")
    )
    x = stats["umap_components"].astype(float).to_numpy()
    y = stats["mean"].astype(float).to_numpy()
    y_std = stats["std"].astype(float).fillna(0.0).to_numpy()

    line_color = sns.color_palette("mako", n_colors=6)[2]
    best_color = sns.color_palette("mako", n_colors=6)[4]

    ax.fill_between(x, y - y_std, y + y_std, color=line_color, alpha=0.18, linewidth=0, label="_nolegend_")
    ax.plot(x, y, color=line_color, linewidth=2.2, marker="o", markersize=5, label="UMAP mean")

    if baseline.size:
        base_mean = float(baseline.mean())
        ax.axhline(base_mean, color="black", linestyle="--", linewidth=1.4, alpha=0.9, label="No Reduction (mean)")

    # Highlight best point.
    best_idx = int(np.nanargmax(y))
    best_x = float(x[best_idx])
    best_y = float(y[best_idx])
    ax.scatter([best_x], [best_y], s=110, color=best_color, edgecolor="black", linewidth=1.2, zorder=5, label="Best UMAP setting")
    ax.text(
        best_x,
        best_y + 0.012,
        f"{best_y:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    ax.set_title("Impact of UMAP Dimensionality Reduction (RGB)")
    ax.set_xlabel("UMAP Components")
    ax.set_ylabel("Classification Accuracy")
    ax.legend(loc="lower right", frameon=True)

    sns.despine(ax=ax)
    _save(fig, "plot_dim_reduction_impact.png")


def plot_clustering_comparison(data: pd.DataFrame) -> None:
    if "clustering_method_display" not in data.columns:
        raise ValueError("No clustering_method data found. Expected metrics files containing kmeans/dbscan/cmeans.")

    subset = data.dropna(subset=["clustering_method_display"]).copy()
    order = (
        subset.groupby("clustering_method_display", observed=True)["accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Light-to-dark blue palette based on performance rank.
    palette = sns.color_palette("Blues", n_colors=max(3, len(order)))[-len(order) :]

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    sns.violinplot(
        data=subset,
        x="clustering_method_display",
        hue="clustering_method_display",
        y="accuracy",
        order=order,
        palette=palette,
        ax=ax,
        inner="quartile",
        cut=0,
        dodge=False,
        linewidth=1.1,
        legend=False,
    )

    ax.set_title("Comparison of Clustering Algorithms")
    ax.set_xlabel("Clustering algorithm")
    ax.set_ylabel("Accuracy")

    sns.despine(ax=ax)
    _save(fig, "plot_clustering_comparison.png")


def main() -> None:
    _configure_style()
    data = _load_metrics()

    plot_color_space_comparison(data)
    plot_crop_method_comparison(data)
    plot_embedding_comparison(data)
    plot_dim_reduction_impact(data)
    plot_clustering_comparison(data)

    print(f"Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
