#!/usr/bin/env python3
"""
Recreates the 'paper_color_space_estimation.png' plot with a modified Y-axis.
This script is a partial copy of 'scripts/classification/plot_team_classification_results.py'
containing only the necessary functions to reproduce the plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, List

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# --- Copied Functions from the original script ---

def configure_style() -> None:
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        }
    )

def _normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["umap_components"] = df["umap_components"].astype(int)
    if "umap_applied" in df.columns:
        df["umap_applied"] = df["umap_applied"].astype(bool)
    else:
        df["umap_applied"] = True
    if "crop_center_ratio" not in df.columns:
        df["crop_center_ratio"] = 1.0
    if "crop_method" not in df.columns:
        df["crop_method"] = "center"
    if "embedding_backend" not in df.columns:
        df["embedding_backend"] = "unknown"
    if "color_space" in df.columns:
        df["color_space"] = df["color_space"].astype(str).str.lower()
    return df

def load_metrics_many(paths: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: List[pd.DataFrame] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df = _normalize_metrics(df)
        df["source_file"] = path.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sequences = sorted(str(seq) for seq in merged.get("sequence", pd.Series(dtype=str)).dropna().unique())
    mapping = {seq: f"match_{idx}" for idx, seq in enumerate(sequences, start=1)}
    if "sequence" in merged.columns:
        merged["match_label"] = merged["sequence"].astype(str).map(mapping).fillna("match_unknown")
    else:
        merged["match_label"] = "match_unknown"
    return merged, mapping

def _best_rows(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    sorted_df = df.sort_values(list(group_cols) + ["accuracy"], ascending=[True] * len(group_cols) + [False])
    return sorted_df.groupby(list(group_cols), as_index=False).first()

def _paired_best(
    df: pd.DataFrame,
    *,
    pair_col: str,
    a: str,
    b: str,
    id_cols: Sequence[str] = ("sequence",),
    restrict_cols: Sequence[str] = ("crop_method",),
) -> pd.DataFrame:
    extra_cols = []
    if "match_label" in df.columns and "sequence" in id_cols:
        extra_cols.append("match_label")
    cols = list(id_cols) + list(restrict_cols) + extra_cols + [pair_col, "accuracy"]
    sub = df.loc[df[pair_col].isin([a, b]), cols].copy()
    if sub.empty:
        return sub
    group_cols = list(id_cols) + list(restrict_cols)
    if extra_cols:
        group_cols = group_cols + extra_cols
    best = _best_rows(sub, group_cols + [pair_col])
    pivot = best.pivot_table(
        index=group_cols,
        columns=pair_col,
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    if a not in pivot.columns or b not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=[a, b]).copy()
    pivot[f"{a}_acc"] = pivot[a].astype(float)
    pivot[f"{b}_acc"] = pivot[b].astype(float)
    pivot["delta"] = pivot[f"{b}_acc"] - pivot[f"{a}_acc"]
    return pivot.drop(columns=[a, b])

def _bootstrap_ci(
    values: np.ndarray, *, statistic=np.mean, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = statistic(values[idx], axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    point = float(statistic(values))
    return point, lo, hi

def _extract_match_number(value: object) -> int:
    text = str(value)
    if text.startswith("match_"):
        suffix = text.split("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 10**9
    return 10**9
    
def apply_footer(fig: plt.Figure, *parts: str, bottom: float = 0.05) -> None:
    # A simplified version of the original footer logic
    clean = [p.strip() for p in parts if p and str(p).strip()]
    if not clean:
        fig.tight_layout()
        return
    fig.text(0.5, 0.01, " | ".join(clean), ha="center", va="bottom", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0, float(bottom), 1, 1))

# --- Modified Plotting Function ---

def plot_estimation_paired(
    paired: pd.DataFrame,
    *,
    a_label: str,
    b_label: str,
    title: str,
    output_path: Path,
    max_pairs: int = 35,
    footer: str = "",
) -> None:
    if paired.empty:
        return
    paired = paired.copy()
    if "match_label" in paired.columns:
        paired["_match_order"] = paired["match_label"].map(_extract_match_number).astype(int)
    else:
        paired["_match_order"] = 10**9
    crop_order = {"center": 0, "opencv_mask": 1, "sam2_mask": 2}
    if "crop_method" in paired.columns:
        paired["_crop_order"] = paired["crop_method"].map(lambda v: crop_order.get(str(v), 99)).astype(int)
    else:
        paired["_crop_order"] = 99
    if "color_space" in paired.columns:
        paired["_color_order"] = paired["color_space"].map(lambda v: 0 if str(v).lower() == "rgb" else 1).astype(int)
    else:
        paired["_color_order"] = 0
    backend_order = {"siglip": 0, "resnet": 1}
    if "embedding_backend" in paired.columns:
        paired["_backend_order"] = paired["embedding_backend"].map(lambda v: backend_order.get(str(v), 99)).astype(int)
    else:
        paired["_backend_order"] = 99

    paired = paired.sort_values(
        ["_match_order", "_crop_order", "_color_order", "_backend_order", "delta"],
        ascending=[True, True, True, True, False],
    )
    if len(paired) > max_pairs:
        paired = paired.iloc[:max_pairs].copy()

    fig = plt.figure(figsize=(10.8, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    a_col = f"{a_label}_acc"
    b_col = f"{b_label}_acc"
    y = np.arange(len(paired))
    ax_left.hlines(y, paired[a_col], paired[b_col], color="0.75", linewidth=1.0, zorder=1)
    ax_left.scatter(paired[a_col], y, s=22, color="#4C72B0", alpha=0.85, label=a_label, zorder=2)
    ax_left.scatter(paired[b_col], y, s=22, color="#C44E52", alpha=0.85, label=b_label, zorder=2)
    ax_left.set_xlim(0, 1)
    
    # --- MODIFICATION ---
    # Change Y-axis labeling as per user request
    ax_left.set_yticks([])
    ax_left.set_ylabel("Games", fontsize=11)
    # --- END MODIFICATION ---

    ax_left.set_xlabel("Accuracy (paired)")
    ax_left.set_title("Paired per sequence (dumbbell)")
    ax_left.legend(frameon=False, loc="lower right")

    deltas = paired["delta"].to_numpy()
    point, lo, hi = _bootstrap_ci(deltas, statistic=np.mean, n_boot=4000, seed=0)
    sns.violinplot(y=deltas, ax=ax_right, orient="v", inner=None, color="0.85", linewidth=0.0)
    sns.stripplot(y=deltas, ax=ax_right, orient="v", color="0.2", alpha=0.4, size=3, jitter=0.15)
    ax_right.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax_right.errorbar(
        x=0,
        y=point,
        yerr=[[point - lo], [hi - point]],
        fmt="o",
        color="#000000",
        capsize=4,
        markersize=5,
        zorder=5,
    )
    ax_right.set_xticks([])
    ax_right.set_ylabel(f"Δ accuracy ({b_label} − {a_label})")
    ax_right.set_title("Effect size (mean ± 95% CI)")
    max_abs = float(np.nanmax(np.abs(np.asarray([*deltas, point, lo, hi], dtype=float)))) if deltas.size else 0.0
    max_abs = max(0.05, max_abs * 1.25)
    ax_right.set_ylim(-max_abs, max_abs)

    fig.suptitle(title, y=1.02)
    apply_footer(fig, footer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

# --- Main execution block ---

def main():
    parser = argparse.ArgumentParser(description="Reproduce a specific plot from team classification results.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        nargs="+",
        help="One or more metrics files (CSV or Parquet).",
        required=True
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--paired-max-points", type=int, default=35)
    args = parser.parse_args()

    configure_style()
    metrics, _ = load_metrics_many(list(args.metrics_csv))
    
    # Replicate the logic from the original script to find the best data for the plot
    try:
        best_combo = None
        best_score = -1.0
        for crop_method in sorted(metrics["crop_method"].dropna().unique()):
            for backend in sorted(metrics["embedding_backend"].dropna().unique()):
                subset = metrics[(metrics["crop_method"] == crop_method) & (metrics["embedding_backend"] == backend)]
                paired = _paired_best(
                    subset,
                    pair_col="color_space",
                    a="h",
                    b="rgb",
                    restrict_cols=("crop_method", "embedding_backend"),
                )
                if paired.empty:
                    continue
                score = float(np.mean(np.abs(paired["delta"].to_numpy())))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_combo = (crop_method, backend, paired)
        
        if best_combo is None:
            raise ValueError("No valid RGB/H pairs found for any crop_method+embedding_backend combo.")
        
        crop_method, backend, color_paired = best_combo
        
        plot_estimation_paired(
            color_paired,
            a_label="h",
            b_label="rgb",
            title="Impact of Color Space on Classification Accuracy", # MODIFIED TITLE HERE
            output_path=args.output_path,
            max_pairs=int(args.paired_max_points),
        )
        print(f"Successfully generated and saved plot to {args.output_path}")

    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    main()
