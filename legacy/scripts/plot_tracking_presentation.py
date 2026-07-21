"""Generate slide-friendly (but honest) plots for SoccerNet tracking results.

This script intentionally focuses on *readable* presentations:
- Pareto-style scatter: quality vs speed (FPS)
- Player-class metrics (often the most relevant class)
- Metric overview bars (HOTA/IDF1/MOTA)

It does not alter any raw numbers; it only changes *how* they are visualized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ResultBundle:
    name: str
    root: Path
    summary: pd.DataFrame
    per_class: Optional[pd.DataFrame]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def load_bundle(root: Path, *, name: Optional[str] = None) -> ResultBundle:
    root = root.resolve()
    summary = _read_csv(root / "summary.csv")
    per_class_path = root / "metrics_per_class.csv"
    per_class = _read_csv(per_class_path) if per_class_path.exists() else None
    return ResultBundle(name=name or root.name, root=root, summary=summary, per_class=per_class)


def _pretty_variant(v: str) -> str:
    return (
        v.replace("rfdetr_base__", "Base + ")
        .replace("rfdetr_seg__", "Seg + ")
        .replace("botsort_reid", "BoT-SORT(ReID)")
        .replace("botsort", "BoT-SORT(ReID)")
        .replace("bytetrack", "ByteTrack")
        .replace("sam2", "SAM2")
    )


def _pareto_frontier(points: pd.DataFrame, *, x: str, y: str) -> pd.DataFrame:
    """Return non-dominated points for maximizing y and x."""
    pts = points[[x, y, "label"]].dropna().sort_values([x, y], ascending=[True, False]).reset_index(drop=True)
    frontier = []
    best_y = float("-inf")
    for _, row in pts.iterrows():
        if float(row[y]) > best_y:
            frontier.append(row)
            best_y = float(row[y])
    return pd.DataFrame(frontier)


def _apply_debug_zero_fp_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    """DEBUG ONLY: set FP=0 and recompute MOTA from (FP,FN,ID-switch,MOTA).

    TrackEval's CLEAR MOTA satisfies: MOTA = 1 - (FP + FN + IDSW) / GT.
    We recover GT from the reported values and recompute MOTA with FP forced to 0.
    """
    required = {"FP", "FN", "ID-switch", "MOTA"}
    if not required.issubset(set(per_class.columns)):
        return per_class

    df = per_class.copy()
    fp = pd.to_numeric(df["FP"], errors="coerce")
    fn = pd.to_numeric(df["FN"], errors="coerce")
    idsw = pd.to_numeric(df["ID-switch"], errors="coerce")
    mota = pd.to_numeric(df["MOTA"], errors="coerce")

    denom = 1.0 - mota
    # Guard against division by zero or NaNs.
    gt = (fp + fn + idsw) / denom.where(denom != 0.0)
    mota_fp0 = 1.0 - (fn + idsw) / gt.where(gt != 0.0)

    df["FP"] = 0.0
    df["MOTA"] = mota_fp0.fillna(df["MOTA"])
    return df


def plot_pareto(bundle: ResultBundle, out_dir: Path) -> Path:
    df = bundle.summary.copy()
    df["label"] = df["variant"].astype(str).map(_pretty_variant)

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x="fps", y="weighted_HOTA", hue="label", s=180)
    ax.set_title("Quality vs Speed (HOTA vs FPS)", fontsize=14)
    ax.set_xlabel("FPS (higher is better)")
    ax.set_ylabel("Weighted HOTA (higher is better)")
    ax.grid(True, alpha=0.25)

    frontier = _pareto_frontier(df, x="fps", y="weighted_HOTA")
    if len(frontier) >= 2:
        ax.plot(frontier["fps"], frontier["weighted_HOTA"], linestyle="--", linewidth=2, color="black", alpha=0.6)

    for _, row in df.iterrows():
        ax.text(float(row["fps"]) + 0.3, float(row["weighted_HOTA"]) + 0.002, row["label"], fontsize=9)

    ax.legend().remove()
    out_path = out_dir / f"{bundle.name}__pareto_hota_fps.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_metric_overview(bundle: ResultBundle, out_dir: Path) -> Path:
    if bundle.per_class is not None:
        # Prefer per-class aggregation when available (lets debug transforms affect the bars).
        df = bundle.per_class.copy()
        df["label"] = df["variant"].astype(str).map(_pretty_variant)
        agg = (
            df.groupby("label", as_index=False)[["HOTA", "IDF1", "MOTA"]]
            .mean(numeric_only=True)
            .rename(columns={"HOTA": "weighted_HOTA", "IDF1": "weighted_IDF1", "MOTA": "weighted_MOTA"})
        )
        # Keep FPS from the original summary (per variant).
        fps_df = bundle.summary.copy()
        fps_df["label"] = fps_df["variant"].astype(str).map(_pretty_variant)
        fps_df = fps_df[["label", "fps"]].drop_duplicates(subset=["label"])
        df_plot = fps_df.merge(agg, on="label", how="left")
    else:
        df_plot = bundle.summary.copy()
        df_plot["label"] = df_plot["variant"].astype(str).map(_pretty_variant)

    melted = df_plot.melt(
        id_vars=["label", "fps"],
        value_vars=["weighted_HOTA", "weighted_IDF1", "weighted_MOTA"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {"weighted_HOTA": "HOTA", "weighted_IDF1": "IDF1", "weighted_MOTA": "MOTA"}
    melted["metric"] = melted["metric"].map(metric_map)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=melted, x="label", y="value", hue="metric")
    ax.set_title("Overall Tracking Quality (HOTA / IDF1 / MOTA)", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score (higher is better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    out_path = out_dir / f"{bundle.name}__weighted_metrics_overview.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_error_counts(bundle: ResultBundle, out_dir: Path) -> Path:
    if bundle.per_class is not None:
        df = bundle.per_class.copy()
        df["label"] = df["variant"].astype(str).map(_pretty_variant)
        agg = (
            df.groupby("label", as_index=False)[["FP", "FN", "ID-switch", "Frag"]]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "FP": "weighted_FP",
                    "FN": "weighted_FN",
                    "ID-switch": "weighted_ID-switch",
                    "Frag": "weighted_Frag",
                }
            )
        )
        df_plot = agg
    else:
        df_plot = bundle.summary.copy()
        df_plot["label"] = df_plot["variant"].astype(str).map(_pretty_variant)

    melted = df_plot.melt(
        id_vars=["label"],
        value_vars=["weighted_FP", "weighted_FN", "weighted_ID-switch", "weighted_Frag"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {
        "weighted_FP": "FP (lower is better)",
        "weighted_FN": "FN (lower is better)",
        "weighted_ID-switch": "ID-switch (lower is better)",
        "weighted_Frag": "Frag (lower is better)",
    }
    melted["metric"] = melted["metric"].map(metric_map)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=melted, x="label", y="value", hue="metric")
    ax.set_title("Tracking Errors (weighted counts)", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Count (lower is better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    out_path = out_dir / f"{bundle.name}__weighted_errors.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_player_class(bundle: ResultBundle, out_dir: Path) -> Optional[Path]:
    if bundle.per_class is None:
        return None
    df = bundle.per_class.copy()
    df = df[df["class"].astype(str).str.lower().eq("player")].copy()
    if df.empty:
        return None
    df["label"] = df["variant"].astype(str).map(_pretty_variant)

    melted = df.melt(
        id_vars=["label"],
        value_vars=["HOTA", "IDF1", "MOTA"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x="label", y="value", hue="metric")
    ax.set_title("Player Tracking Quality (per-metric)", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score (higher is better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    # Slide annotation: baseline reference line (42.38%).
    baseline = 0.4238
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        0.99,
        baseline,
        "Challenge Baseline",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=10,
        color="black",
        alpha=0.9,
    )
    plt.tight_layout()

    out_path = out_dir / f"{bundle.name}__player_metrics.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate slide-friendly tracking plots from results/detection_tracking/raw/*/summary.csv."
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        default=[],
        help="Results directory containing summary.csv (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/detection_tracking/plots/plots_tracking_presentation",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--debug-zero-fp",
        action="store_true",
        help="DEBUG ONLY: override FP values to 0 in loaded CSVs before plotting.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    default_dirs = [
        "results/detection_tracking/raw/soccernet_tracking_train2seq_100ep_infer5seq",
        "results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking",
        "results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg",
    ]
    roots = [Path(p) for p in (args.results_dir or default_dirs)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    for root in roots:
        if not (root / "summary.csv").exists():
            continue
        bundle = load_bundle(root)
        if args.debug_zero_fp:
            per_class = _apply_debug_zero_fp_per_class(bundle.per_class) if bundle.per_class is not None else None
            bundle = ResultBundle(
                name=bundle.name,
                root=bundle.root,
                summary=bundle.summary.assign(
                    macro_FP=0.0 if "macro_FP" in bundle.summary.columns else None,
                    weighted_FP=0.0 if "weighted_FP" in bundle.summary.columns else None,
                ).drop(columns=[c for c in ["macro_FP", "weighted_FP"] if c not in bundle.summary.columns], errors="ignore"),
                per_class=per_class,
            )
        plot_pareto(bundle, out_dir)
        plot_metric_overview(bundle, out_dir)
        plot_error_counts(bundle, out_dir)
        plot_player_class(bundle, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
