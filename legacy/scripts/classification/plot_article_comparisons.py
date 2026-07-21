#!/usr/bin/env python3
"""Czytelne, jednowątkowe wykresy porównawcze dla eksperymentu klasyfikacji drużyn.

Każdy wykres to jedno porównanie na zagregowanym zbiorze (bez podziału na sekwencje):
- ratio_bar_<method>.png        : accuracy vs. crop ratio, bez UMAP, osobno dla każdej metody crop (H i RGB).
- umap_elbow_per_color.png      : krzywe accuracy vs. #komponentów UMAP dla najlepszego cropu w danej barwie.
- umap_gain_bar.png             : baseline (no UMAP) vs. najlepszy UMAP dla wybranego cropu (per barwa).
- cluster_bar.png               : najlepsza konfiguracja każdej metody klastrowania (accuracy zagregowane).
- top_configs_lollipop.png      : top 6 konfiguracji pipeline (accuracy ważone), lollipop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

DEFAULT_METRICS = Path("experiments/team_classification_quick/metrics.csv")
DEFAULT_CLUSTER_FILES: Dict[str, Path] = {
    "kmeans_full": Path("results/team_classification/numeric/team_classification_metrics_kmeans.csv"),
    "dbscan": Path("results/team_classification/numeric/team_classification_metrics_dbscan.csv"),
}


def configure_style() -> None:
    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["umap_components"] = df["umap_components"].astype(int)
    df["crop_center_ratio"] = df["crop_center_ratio"].astype(float)
    df["crop_method"] = df["crop_method"].astype(str)
    df["color_space"] = df["color_space"].astype(str)
    if "umap_applied" in df.columns:
        df["umap_applied"] = df["umap_applied"].astype(bool)
    else:
        df["umap_applied"] = True
    return df


def compute_weighted_accuracy(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(list(group_cols))
        .agg(
            tp=("true_positive", "sum"),
            fp=("false_positive", "sum"),
            tn=("true_negative", "sum"),
            fn=("false_negative", "sum"),
        )
        .reset_index()
    )
    grouped["total"] = grouped[["tp", "fp", "tn", "fn"]].sum(axis=1)
    grouped = grouped.loc[grouped["total"] > 0].copy()
    grouped["accuracy"] = (grouped["tp"] + grouped["tn"]) / grouped["total"]
    return grouped


def overall_accuracy(df: pd.DataFrame) -> float:
    totals = df[["true_positive", "false_positive", "true_negative", "false_negative"]].sum()
    denom = totals.sum()
    if denom <= 0:
        return float("nan")
    return float((totals["true_positive"] + totals["true_negative"]) / denom)


def select_best_baseline_per_color(df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
    baseline = df.loc[~df["umap_applied"]]
    if baseline.empty:
        return {}
    acc = compute_weighted_accuracy(baseline, ["color_space", "crop_method", "crop_center_ratio"])
    acc = acc.sort_values(["color_space", "accuracy"], ascending=[True, False])
    best: Dict[str, Tuple[str, float]] = {}
    for _, row in acc.iterrows():
        color = str(row["color_space"])
        if color in best:
            continue
        best[color] = (str(row["crop_method"]), float(row["crop_center_ratio"]))
    return best


def best_config(df: pd.DataFrame) -> Optional[pd.Series]:
    grouped = compute_weighted_accuracy(
        df, ["color_space", "crop_method", "crop_center_ratio", "umap_applied", "umap_components"]
    )
    if grouped.empty:
        return None
    return grouped.sort_values("accuracy", ascending=False).iloc[0]


def plot_ratio_bars(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    subset = df.loc[~df["umap_applied"]].copy()
    if subset.empty:
        return outputs
    agg = compute_weighted_accuracy(subset, ["crop_method", "color_space", "crop_center_ratio"])
    for method, group in agg.groupby("crop_method"):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        sns.barplot(
            data=group.sort_values("crop_center_ratio"),
            x="crop_center_ratio",
            y="accuracy",
            hue="color_space",
            palette="Set2",
            ax=ax,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Crop center ratio")
        ax.set_ylabel("Accuracy (ważone)")
        ax.set_title(f"Accuracy vs ratio (bez UMAP) | {method}")
        ax.legend(title="Przestrzeń barw")
        ax.yaxis.set_major_locator(MaxNLocator(6))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
        fig.tight_layout()
        path = output_dir / f"ratio_bar_{method}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_umap_elbow(df: pd.DataFrame, output_path: Path) -> None:
    best_per_color = select_best_baseline_per_color(df)
    if not best_per_color:
        return
    fig, axes = plt.subplots(1, len(best_per_color), figsize=(6.4 * len(best_per_color), 4.0), sharey=True)
    axes = [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, (color_space, (method, ratio)) in zip(axes, best_per_color.items()):
        subset = df.loc[
            (df["color_space"] == color_space)
            & (df["crop_method"] == method)
            & (np.isclose(df["crop_center_ratio"], ratio))
        ]
        if subset.empty:
            ax.set_visible(False)
            continue
        grouped = compute_weighted_accuracy(subset, ["umap_applied", "umap_components"])
        grouped["dim_value"] = np.where(grouped["umap_applied"], grouped["umap_components"], 0)
        grouped = grouped.sort_values("dim_value")
        sns.lineplot(
            data=grouped,
            x="dim_value",
            y="accuracy",
            marker="o",
            linewidth=2.2,
            color="#4C72B0",
            ax=ax,
        )
        ax.set_xticks(grouped["dim_value"])
        ax.set_xticklabels(["0 (brak UMAP)" if v == 0 else str(int(v)) for v in grouped["dim_value"]])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Liczba komponentów")
        ax.set_ylabel("Accuracy" if ax is axes[0] else "")
        ax.set_title(f"{color_space.upper()} | {method} | r={ratio:.2f}")
    fig.suptitle("UMAP sweep dla najlepszego cropu (accuracy ważone)", y=1.05)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_umap_gain_bar(df: pd.DataFrame, output_path: Path) -> None:
    best_per_color = select_best_baseline_per_color(df)
    records: List[Dict[str, object]] = []
    for color_space, (method, ratio) in best_per_color.items():
        base = df.loc[
            (df["color_space"] == color_space)
            & (df["crop_method"] == method)
            & (np.isclose(df["crop_center_ratio"], ratio))
            & (~df["umap_applied"])
        ]
        umap = df.loc[
            (df["color_space"] == color_space)
            & (df["crop_method"] == method)
            & (np.isclose(df["crop_center_ratio"], ratio))
            & (df["umap_applied"])
        ]
        if base.empty or umap.empty:
            continue
        base_acc = overall_accuracy(base)
        best_umap = best_config(umap)
        if best_umap is None:
            continue
        best_dim = int(best_umap["umap_components"])
        best_umap_acc = overall_accuracy(umap.loc[umap["umap_components"] == best_dim])
        records.append(
            {
                "color_space": color_space.upper(),
                "method": method,
                "ratio": ratio,
                "baseline": base_acc,
                "umap_acc": best_umap_acc,
                "umap_dim": best_dim,
            }
        )
    if not records:
        return
    plot_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    positions = np.arange(len(plot_df))
    width = 0.35
    ax.bar(positions - width / 2, plot_df["baseline"], width, label="brak UMAP", color="#4C72B0")
    ax.bar(positions + width / 2, plot_df["umap_acc"], width, label="najlepszy UMAP", color="#DD8452")
    labels = [
        f"{row.color_space} | {row.method} | r={row.ratio:.2f} | UMAP={row.umap_dim}"
        for row in plot_df.itertuples()
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy (ważone)")
    ax.set_title("UMAP vs baseline dla najlepszego cropu")
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(6))
    for idx, row in enumerate(plot_df.itertuples()):
        ax.text(idx - width / 2, row.baseline + 0.01, f"{row.baseline:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, row.umap_acc + 0.01, f"{row.umap_acc:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_bar(datasets: Dict[str, pd.DataFrame], output_path: Path) -> None:
    records: List[Dict[str, object]] = []
    for label, df in datasets.items():
        cfg = best_config(df)
        if cfg is None:
            continue
        mask = (
            (df["color_space"] == cfg["color_space"])
            & (df["crop_method"] == cfg["crop_method"])
            & (np.isclose(df["crop_center_ratio"], cfg["crop_center_ratio"]))
            & (
                (df["umap_applied"] & (df["umap_components"] == cfg["umap_components"]))
                | (~df["umap_applied"] & (cfg["umap_applied"] is False))
            )
        )
        acc_val = overall_accuracy(df.loc[mask])
        umap_label = "no UMAP" if not cfg["umap_applied"] else f"UMAP={int(cfg['umap_components'])}"
        records.append(
            {
                "cluster_method": label,
                "accuracy": acc_val,
                "label": f"{cfg['color_space'].upper()} | {cfg['crop_method']} | r={cfg['crop_center_ratio']:.2f} | {umap_label}",
            }
        )
    if not records:
        return
    plot_df = pd.DataFrame(records).sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(
        data=plot_df,
        x="cluster_method",
        y="accuracy",
        hue="cluster_method",
        dodge=False,
        palette="Set2",
        legend=False,
        ax=ax,
    )
    for idx, row in plot_df.iterrows():
        ax.text(idx, row["accuracy"] + 0.01, f"{row['accuracy']:.3f}\n{row['label']}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Metoda klastrowania (najlepsza konfiguracja)")
    ax.set_ylabel("Accuracy (ważone)")
    ax.set_title("Porównanie metod klastrowania (zagregowane)")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_configs(df: pd.DataFrame, output_path: Path, top_k: int = 6) -> None:
    grouped = compute_weighted_accuracy(
        df, ["color_space", "crop_method", "crop_center_ratio", "umap_applied", "umap_components"]
    )
    if grouped.empty:
        return
    def _label(row: pd.Series) -> str:
        umap_label = "no UMAP" if not row["umap_applied"] else f"UMAP={int(row['umap_components'])}"
        return f"{row['color_space'].upper()} | {row['crop_method']} | r={row['crop_center_ratio']:.2f} | {umap_label}"

    grouped["label"] = grouped.apply(_label, axis=1)
    top = grouped.sort_values("accuracy", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.6 * len(top))))
    ax.hlines(top["label"], xmin=0, xmax=top["accuracy"], color="#4C72B0", linewidth=2.2)
    ax.plot(top["accuracy"], top["label"], "o", color="#4C72B0")
    for acc, label in zip(top["accuracy"], top["label"]):
        ax.text(acc + 0.01, label, f"{acc:.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Accuracy (ważone)")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(top)} konfiguracji pipeline")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_label_path(raw: str) -> Tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("Expected format label:path for --cluster-metrics.")
    label, path_str = raw.split(":", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Cluster metrics label cannot be empty.")
    return label.strip(), Path(path_str.strip())


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Czytelne wykresy porównawcze (artykuł).")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="CSV metryk bazowych (kmeans).")
    parser.add_argument(
        "--cluster-metrics",
        type=parse_label_path,
        action="append",
        default=[],
        metavar="label:path",
        help="Dodatkowe CSV z innymi metodami klastrów (np. dbscan:results/...csv).",
    )
    parser.add_argument(
        "--no-auto-discovery",
        action="store_true",
        help="Wyłącz automatyczne dołączanie domyślnych plików z klastrów.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/team_classification/plots/article_clean")
    )
    args = parser.parse_args(argv)

    configure_style()
    df = load_metrics(args.metrics.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratio_paths = plot_ratio_bars(df, args.output_dir)
    elbow_path = args.output_dir / "umap_elbow_per_color.png"
    plot_umap_elbow(df, elbow_path)
    delta_path = args.output_dir / "umap_gain_bar.png"
    plot_umap_gain_bar(df, delta_path)
    top_path = args.output_dir / "top_configs_lollipop.png"
    plot_top_configs(df, top_path)

    clustering: Dict[str, pd.DataFrame] = {"kmeans": df}
    if not args.no_auto_discovery:
        for label, path in DEFAULT_CLUSTER_FILES.items():
            if path.exists():
                clustering[label] = load_metrics(path)
    for label, path in args.cluster_metrics:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Brak pliku metryk dla {label}: {resolved}")
        clustering[label] = load_metrics(resolved)
    cluster_path = args.output_dir / "cluster_bar.png"
    plot_cluster_bar(clustering, cluster_path)

    print("Zapisane wykresy:")
    for path in [
        *ratio_paths,
        elbow_path,
        delta_path,
        cluster_path,
        top_path,
    ]:
        if path.exists():
            print(f" - {path}")


if __name__ == "__main__":
    main()
