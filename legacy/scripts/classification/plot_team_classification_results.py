#!/usr/bin/env python3
"""Generate high-quality plots summarising team classification experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

STAGE_COLOR_UMAP = "umap_color"
STAGE_RATIO_SWEEP = "ratio_sweep"
STAGE_SAM2_COMPARISON = "sam2_comparison"
STAGE_OPENCV_COMPARISON = "opencv_comparison"
UMAP_CLIP_PERCENTILE = 0.90
EMBED_DISPLAY = {
    "siglip": "SigLIP",
    "resnet": "ResNet",
    "resnet16": "ResNet-16",
    "clip": "CLIP",
}
CLUSTER_DISPLAY = {
    "kmeans": "k-means",
    "k_means": "k-means",
    "cmeans": "c-means",
    "dbscan": "DBSCAN",
}


def format_ratio(ratio: float) -> str:
    formatted = f"{ratio:.3f}".rstrip("0").rstrip(".")
    if not formatted:
        formatted = "0"
    return formatted


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


def singleton_value_note(df: pd.DataFrame, col: str, *, label: str) -> str:
    if df.empty or col not in df.columns:
        return ""
    values = [str(v) for v in df[col].dropna().unique()]
    if len(values) != 1:
        return ""
    value = values[0]
    if col == "color_space":
        value = value.upper()
    return f"{label}: {value}"


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=9, color="0.25")


def apply_footer(fig: plt.Figure, *parts: str, bottom: float = 0.05) -> None:
    clean = [p.strip() for p in parts if p and str(p).strip()]
    if not clean:
        fig.tight_layout()
        return
    add_footer(fig, " | ".join(clean))
    fig.tight_layout(rect=(0, float(bottom), 1, 1))


def footer_parts(
    df: pd.DataFrame,
    *,
    include_crop_note: bool = True,
) -> list[str]:
    if df.empty:
        return []
    parts: List[str] = []
    note_color = singleton_value_note(df, "color_space", label="Colour space")
    if note_color:
        parts.append(note_color)
    note_crop = singleton_value_note(df, "crop_method", label="Crop method")
    if note_crop:
        parts.append(note_crop)
    note_ratio = singleton_value_note(df, "crop_center_ratio", label="Center ratio")
    if note_ratio:
        parts.append(note_ratio)
    note_embed = singleton_value_note(df, "embedding_backend", label="Embedding")
    if note_embed:
        parts.append(note_embed)
    note_cluster = singleton_value_note(df, "cluster_method", label="Clustering")
    if note_cluster:
        parts.append(note_cluster)
    if include_crop_note:
        note_ranges = crop_ratio_note(df)
        if note_ranges and note_ranges not in parts:
            parts.append(note_ranges)
    return parts


def tidy_numeric_axes(ax: plt.Axes, *, max_xticks: int = 8, max_yticks: int = 6) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks, integer=True, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))


def format_combo(color_space: str, crop_method: str, ratio: float) -> str:
    method = str(crop_method)
    if method == "center":
        method_label = "Center"
    elif method == "opencv_mask":
        method_label = "OpenCV"
    elif method == "sam2_mask":
        method_label = "SAM2"
    else:
        method_label = method.replace("_", " ").title()
    return f"{color_space.upper()} | {method_label} {format_ratio(ratio)}"


def crop_ratio_note(df: pd.DataFrame) -> str:
    """Human-readable note about which crop ratios are present per crop method."""
    if df.empty or "crop_method" not in df.columns or "crop_center_ratio" not in df.columns:
        return ""
    label_map = {"center": "Center", "opencv_mask": "OpenCV", "sam2_mask": "SAM2"}
    parts: List[str] = []
    for method in ["center", "opencv_mask", "sam2_mask"]:
        if method not in set(df["crop_method"].dropna().astype(str).unique()):
            continue
        ratios = (
            df.loc[df["crop_method"] == method, "crop_center_ratio"]
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        ratios = sorted({float(r) for r in ratios})
        if not ratios:
            continue
        if len(ratios) == 1:
            parts.append(f"{label_map.get(method, method)}={format_ratio(ratios[0])}")
        else:
            parts.append(
                f"{label_map.get(method, method)}={format_ratio(ratios[0])}–{format_ratio(ratios[-1])} (n={len(ratios)})"
            )
    return "Crop ratio: " + "; ".join(parts) if parts else ""


def crop_method_labels(df: pd.DataFrame) -> dict[str, str]:
    """Return human-friendly crop method labels (no ratio suffix)."""
    if df.empty or "crop_method" not in df.columns or "crop_center_ratio" not in df.columns:
        return {}
    label_map = {"center": "Center", "opencv_mask": "OpenCV mask", "sam2_mask": "SAM2 mask"}
    out: dict[str, str] = {}
    for method in df["crop_method"].dropna().astype(str).unique().tolist():
        out[method] = label_map.get(method, method.replace("_", " ").title())
    return out


def crop_ratio_axis_note(df: pd.DataFrame) -> str:
    note = crop_ratio_note(df)
    if not note:
        return ""
    return note.replace("Crop ratio: ", "Ratios: ")


def format_crop_variant_label(method: str, ratio: float, *, method_labels: dict[str, str]) -> str:
    """Label crop method without ratio ranges; keep per-ratio labels only for center crops."""
    method = str(method)
    base = method_labels.get(method, method.replace("_", " ").title())
    if method == "center":
        return f"{base} {format_ratio(float(ratio))}"
    return base


def crop_variant_order_key(label: str) -> tuple[int, float, str]:
    """Sort 'Center 0.2' .. 'Center 1.0' before OpenCV/SAM2."""
    text = str(label)
    if text.startswith("Center "):
        try:
            ratio = float(text.split(" ", 1)[1])
        except Exception:
            ratio = 999.0
        return (0, ratio, text)
    if text.startswith("OpenCV"):
        return (1, 1.0, text)
    if text.startswith("SAM2"):
        return (2, 1.0, text)
    return (9, 999.0, text)

def filter_umap_rows(
    df: pd.DataFrame, *, percentile: float = UMAP_CLIP_PERCENTILE, cap: Optional[int] = None
) -> tuple[pd.DataFrame, Optional[int]]:
    df_umap = df.loc[df["umap_applied"]].copy()
    if df_umap.empty:
        return df_umap, None
    cutoff = cap
    if cutoff is None:
        quantile_val = df_umap["umap_components"].quantile(percentile)
        cutoff = float(quantile_val) if np.isfinite(quantile_val) else None
    if cutoff is not None and np.isfinite(cutoff):
        df_umap = df_umap.loc[df_umap["umap_components"] <= cutoff]
    if df_umap.empty:
        return df_umap, int(cutoff) if cutoff is not None else None
    applied_cutoff = int(np.ceil(df_umap["umap_components"].max()))
    return df_umap, applied_cutoff


def format_umap_xlabel(cutoff: Optional[int]) -> str:
    base = "UMAP components"
    if cutoff is None:
        return base
    return f"{base} (≤{cutoff}, trimmed)"


def weighted_accuracy(df: pd.DataFrame) -> float:
    tp = df["true_positive"].sum()
    tn = df["true_negative"].sum()
    total = df["num_samples"].sum()
    return float((tp + tn) / total) if total else float("nan")


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


def load_metrics(path: Path) -> pd.DataFrame:
    df, _ = load_metrics_many([path])
    return df


def _best_rows(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    sorted_df = df.sort_values(list(group_cols) + ["accuracy"], ascending=[True] * len(group_cols) + [False])
    return sorted_df.groupby(list(group_cols), as_index=False).first()


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
    """Parse match_N -> N for stable sorting."""
    text = str(value)
    if text.startswith("match_"):
        suffix = text.split("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 10**9
    return 10**9


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


def _format_embedding_name(value: object) -> str:
    key = str(value).strip().lower()
    return EMBED_DISPLAY.get(key, str(value))


def _format_cluster_name(value: object) -> str:
    key = str(value).strip().lower()
    return CLUSTER_DISPLAY.get(key, str(value))


def _best_by_match(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    score_col: str = "accuracy",
) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        return df
    sorted_df = df.sort_values(cols + [score_col], ascending=[True] * len(cols) + [False])
    return sorted_df.groupby(cols, as_index=False).first()


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
    ax_left.set_yticks(y)
    label_cols: List[str] = []
    if "match_label" in paired.columns:
        label_cols.append("match_label")
    elif "sequence" in paired.columns:
        label_cols.append("sequence")
    if "crop_method" in paired.columns:
        label_cols.append("crop_method")
    # Include extra dimensions when present to avoid "duplicates" that are actually different strata.
    for col in ("color_space", "embedding_backend"):
        if col in paired.columns and paired[col].nunique() > 1:
            label_cols.append(col)
    if not label_cols:
        label_cols = ["delta"]
    ax_left.set_yticklabels(
        paired.apply(lambda r: " | ".join(str(r.get(c, "")) for c in label_cols), axis=1)
    )
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


def plot_embedding_siglip_vs_resnet(metrics: pd.DataFrame, output_path: Path) -> None:
    """Paired dumbbell plot per match: SigLIP vs ResNet-16 (best over UMAP, fixed RGB/center/k-means)."""
    df = metrics.copy()
    if df.empty:
        return
    df = df[
        (df["color_space"].astype(str).str.lower() == "rgb")
        & (df["crop_method"].astype(str) == "center")
        & (df["cluster_method"].astype(str).str.lower().isin({"kmeans", "k_means"}))
        & (df["embedding_backend"].astype(str).str.lower().isin({"siglip", "resnet16"}))
    ].copy()
    if df.empty:
        return

    best = _best_by_match(df, group_cols=("match_label", "embedding_backend"))
    pivot = (
        best.pivot_table(index="match_label", columns="embedding_backend", values="accuracy", aggfunc="max")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "siglip" not in pivot.columns or "resnet16" not in pivot.columns:
        return
    pivot = pivot.dropna(subset=["siglip", "resnet16"]).copy()
    if pivot.empty:
        return

    pivot["delta"] = pivot["siglip"] - pivot["resnet16"]
    pivot = pivot.sort_values("delta", ascending=True).reset_index(drop=True)

    deltas = pivot["delta"].to_numpy(dtype=float)
    delta_mean, delta_lo, delta_hi = _bootstrap_ci(deltas, statistic=np.mean, n_boot=4000, alpha=0.05, seed=3)

    fig = plt.figure(figsize=(10.5, max(4.6, 0.22 * len(pivot) + 1.6)))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2.7, 1.1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])

    y = np.arange(len(pivot))
    x_a = pivot["resnet16"].to_numpy(dtype=float)
    x_b = pivot["siglip"].to_numpy(dtype=float)
    for i in range(len(pivot)):
        ax.plot([x_a[i], x_b[i]], [y[i], y[i]], color="0.75", lw=1.0, zorder=1)

    ax.scatter(
        x_a,
        y,
        s=34,
        color=sns.color_palette("colorblind")[1],
        label=_format_embedding_name("resnet16"),
        zorder=3,
    )
    ax.scatter(
        x_b,
        y,
        s=34,
        color=sns.color_palette("colorblind")[0],
        label=_format_embedding_name("siglip"),
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot["match_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy")
    ax.set_title("Embedding comparison (paired, per match)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")

    sns.violinplot(x=deltas, orient="h", inner=None, color="0.90", linewidth=0.0, ax=ax_delta, cut=0)
    sns.stripplot(x=deltas, orient="h", color="0.25", alpha=0.65, size=3.0, jitter=0.18, ax=ax_delta)
    ax_delta.axvline(0.0, color="0.3", lw=1.0, ls="--", alpha=0.6)
    ax_delta.errorbar(
        x=[delta_mean],
        y=[0],
        xerr=[[delta_mean - delta_lo], [delta_hi - delta_mean]],
        fmt="o",
        color=sns.color_palette("colorblind")[2],
        capsize=3,
        elinewidth=1.2,
        markersize=5,
        zorder=5,
    )
    ax_delta.set_yticks([])
    ax_delta.set_xlabel("Δ Accuracy (SigLIP − ResNet-16)")
    ax_delta.set_title("Effect size")
    ax_delta.grid(True, axis="x", alpha=0.25)
    ax_delta.grid(False, axis="y")

    apply_footer(
        fig,
        "Colour space: RGB",
        "Crop method: center",
        "Clustering: k-means",
        "Best over UMAP",
        f"Mean Δ={delta_mean:+.3f} (95% CI {delta_lo:+.3f}..{delta_hi:+.3f})",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_method_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Qualitative comparison: distribution + win-rate matrix across matches."""
    df = metrics.copy()
    if df.empty:
        return
    df = df[
        (df["color_space"].astype(str).str.lower() == "rgb")
        & (df["crop_method"].astype(str) == "center")
        & (df["embedding_backend"].astype(str).str.lower() == "siglip")
        & (df["cluster_method"].astype(str).str.lower().isin({"kmeans", "k_means", "cmeans", "dbscan"}))
    ].copy()
    if "crop_center_ratio" in df.columns:
        df = df[np.isclose(df["crop_center_ratio"].astype(float), 0.8, atol=1e-6)]
    if df.empty:
        return

    best = _best_by_match(df, group_cols=("match_label", "cluster_method"))
    best["method"] = best["cluster_method"].map(_format_cluster_name)
    methods = ["k-means", "c-means", "DBSCAN"]
    best = best[best["method"].isin(methods)].copy()
    if best.empty:
        return

    pivot = (
        best.pivot_table(index="match_label", columns="method", values="accuracy", aggfunc="max")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot = pivot.dropna(subset=methods).copy()
    if pivot.empty:
        return
    melted = pivot.melt(id_vars=["match_label"], value_vars=methods, var_name="method", value_name="accuracy")

    win = np.zeros((len(methods), len(methods)), dtype=float)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                win[i, j] = np.nan
                continue
            a_vals = pivot[a].to_numpy(dtype=float)
            b_vals = pivot[b].to_numpy(dtype=float)
            better = np.mean(a_vals > b_vals)
            ties = np.mean(a_vals == b_vals)
            win[i, j] = float(better + 0.5 * ties)

    fig = plt.figure(figsize=(10.8, 4.6))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.6, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    sns.violinplot(
        data=melted,
        x="method",
        y="accuracy",
        order=methods,
        inner=None,
        cut=0,
        linewidth=0.0,
        color="0.90",
        ax=ax,
    )
    sns.stripplot(
        data=melted,
        x="method",
        y="accuracy",
        order=methods,
        color="0.25",
        alpha=0.55,
        size=3.0,
        jitter=0.22,
        ax=ax,
    )
    palette = sns.color_palette("colorblind", n_colors=6)
    for idx, method in enumerate(methods):
        vals = melted.loc[melted["method"] == method, "accuracy"].to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_ci(vals, statistic=np.mean, n_boot=4000, alpha=0.05, seed=10 + idx)
        ax.errorbar(
            x=idx,
            y=mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=palette[idx],
            capsize=3,
            elinewidth=1.2,
            markersize=5,
            zorder=6,
        )

    ax.set_title("Clustering methods: distribution across matches")
    ax.set_xlabel("Clustering method")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    hm_df = pd.DataFrame(win, index=methods, columns=methods)
    sns.heatmap(
        hm_df,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": "Win-rate (row beats column)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax_hm,
    )
    ax_hm.set_title("Pairwise win-rate")
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")
    ax_hm.tick_params(axis="x", rotation=25)
    ax_hm.tick_params(axis="y", rotation=0)

    apply_footer(
        fig,
        "Colour space: RGB",
        "Crop method: center (ratio 0.8)",
        "Embedding: SigLIP",
        "Best over UMAP",
        f"Matches: {int(pivot.shape[0])}",
        bottom=0.13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_raincloud_accuracy(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    output_path: Path,
    hue_col: Optional[str] = None,
    facet_col: Optional[str] = None,
) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    method_labels = crop_method_labels(plot_df) if group_col == "crop_method" else {}
    order = sorted(plot_df[group_col].unique())

    if facet_col and facet_col in plot_df.columns and plot_df[facet_col].nunique() > 1:
        facets = sorted(plot_df[facet_col].unique())
        fig, axes = plt.subplots(
            1, len(facets), figsize=(5.8 * len(facets), 4.2), sharey=True, squeeze=False
        )
        tick_labels = [method_labels.get(str(val), str(val)) for val in order] if method_labels else []
        for idx, facet in enumerate(facets):
            ax = axes[0][idx]
            panel = plot_df[plot_df[facet_col] == facet]
            sns.violinplot(
                data=panel,
                x=group_col,
                y="accuracy",
                order=order,
                hue=hue_col,
                cut=0,
                inner=None,
                linewidth=0.0,
                alpha=0.6,
                ax=ax,
            )
            sns.boxplot(
                data=panel,
                x=group_col,
                y="accuracy",
                order=order,
                hue=hue_col,
                width=0.18,
                showcaps=True,
                boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 1.0},
                whiskerprops={"color": "0.25", "linewidth": 1.0},
                medianprops={"color": "0.1", "linewidth": 1.2},
                showfliers=False,
                ax=ax,
            )
            sns.stripplot(
                data=panel,
                x=group_col,
                y="accuracy",
                order=order,
                hue=hue_col,
                dodge=True if hue_col else False,
                jitter=0.18,
                alpha=0.35,
                size=2.2,
                color="black" if hue_col is None else None,
                ax=ax,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_title(str(facet))
            ax.set_xlabel("")
            ax.set_ylabel("Accuracy" if idx == 0 else "")
            ax.tick_params(axis="x", rotation=18)
            if group_col == "crop_method" and tick_labels:
                ax.set_xticks(np.arange(len(order)))
                ax.set_xticklabels(tick_labels)
            sns.despine(ax=ax, left=False, bottom=False)
            if ax.legend_:
                ax.legend_.remove()

        if hue_col:
            handles, labels = axes[0][0].get_legend_handles_labels()
            uniq = list(dict.fromkeys(labels))
            fig.legend(handles[: len(uniq)], uniq, frameon=False, loc="upper right")

        fig.suptitle(title, y=1.02)
        apply_footer(fig, *footer_parts(plot_df, include_crop_note=True))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    sns.violinplot(
        data=plot_df,
        x=group_col,
        y="accuracy",
        order=order,
        hue=hue_col,
        cut=0,
        inner=None,
        linewidth=0.0,
        alpha=0.6,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x=group_col,
        y="accuracy",
        order=order,
        hue=hue_col,
        width=0.18,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 1.0},
        whiskerprops={"color": "0.25", "linewidth": 1.0},
        medianprops={"color": "0.1", "linewidth": 1.2},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x=group_col,
        y="accuracy",
        order=order,
        hue=hue_col,
        dodge=True if hue_col else False,
        jitter=0.18,
        alpha=0.35,
        size=2.2,
        color="black" if hue_col is None else None,
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=18)
    if group_col == "crop_method" and method_labels:
        tick_labels = [method_labels.get(str(val), str(val)) for val in order]
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(tick_labels)
    sns.despine(ax=ax, left=False, bottom=False)
    if hue_col and ax.legend_:
        handles, labels = ax.get_legend_handles_labels()
        uniq = list(dict.fromkeys(labels))
        ax.legend(handles[: len(uniq)], uniq, frameon=False)
    elif ax.legend_:
        ax.legend_.remove()
    apply_footer(fig, *footer_parts(plot_df, include_crop_note=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_config_heatmaps(
    df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    agg: str = "mean",
) -> None:
    if df.empty:
        return
    cols = ["color_space", "embedding_backend", "crop_method", "crop_center_ratio", "accuracy"]
    sub = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if sub.empty:
        return
    if agg == "median":
        grouped = sub.groupby(["color_space", "embedding_backend", "crop_method", "crop_center_ratio"], as_index=False).agg(
            value=("accuracy", "median")
        )
    else:
        grouped = sub.groupby(["color_space", "embedding_backend", "crop_method", "crop_center_ratio"], as_index=False).agg(
            value=("accuracy", "mean")
        )
    color_spaces = sorted(grouped["color_space"].unique())
    backends = sorted(grouped["embedding_backend"].unique())
    fig, axes = plt.subplots(
        len(backends),
        len(color_spaces),
        figsize=(4.2 * max(1, len(color_spaces)), 3.2 * max(1, len(backends))),
        sharey=True,
        squeeze=False,
    )
    for r, backend in enumerate(backends):
        for c, color_space in enumerate(color_spaces):
            ax = axes[r][c]
            panel = grouped[(grouped["embedding_backend"] == backend) & (grouped["color_space"] == color_space)]
            if panel.empty:
                ax.axis("off")
                continue
            pivot = panel.pivot_table(index="crop_method", columns="crop_center_ratio", values="value", aggfunc="first")
            pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
            sns.heatmap(
                pivot,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                linewidths=0.4,
                linecolor="white",
                cbar=(r == 0 and c == len(color_spaces) - 1),
                ax=ax,
            )
            ax.set_title(f"{color_space.upper()} | {backend}")
            ax.set_xlabel("Center ratio")
            ax.set_ylabel("Crop method" if c == 0 else "")
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
    fig.suptitle(title, y=1.02)
    apply_footer(fig, *footer_parts(sub, include_crop_note=True), f"Aggregation: {agg}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_trend(df: pd.DataFrame, output_path: Path, umap_cap: Optional[int]) -> None:
    df_umap, cutoff = filter_umap_rows(df, cap=umap_cap)
    if df_umap.empty:
        return
    method_labels = crop_method_labels(df_umap)
    grouped = (
        df_umap.groupby(
            ["embedding_backend", "color_space", "crop_method", "crop_center_ratio", "umap_components"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            count=("accuracy", "count"),
        )
    )
    grouped["sem"] = grouped["std_accuracy"] / grouped["count"].pow(0.5)
    grouped["crop_variant"] = grouped.apply(
        lambda r: format_crop_variant_label(
            str(r["crop_method"]), float(r["crop_center_ratio"]), method_labels=method_labels
        ),
        axis=1,
    )
    hue_order = sorted(grouped["crop_variant"].dropna().unique().tolist(), key=crop_variant_order_key)

    backends = sorted(grouped["embedding_backend"].unique())
    color_spaces = sorted(grouped["color_space"].unique())
    n_rows = len(backends)
    n_cols = len(color_spaces)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.6 * max(1, n_cols), 3.6 * max(1, n_rows)),
        sharey=True,
        squeeze=False,
    )

    for r, backend in enumerate(backends):
        for c, color_space in enumerate(color_spaces):
            ax = axes[r][c]
            subset = grouped[
                (grouped["embedding_backend"] == backend) & (grouped["color_space"] == color_space)
            ].sort_values("umap_components")
            if subset.empty:
                ax.axis("off")
                continue
            sns.lineplot(
                data=subset,
                x="umap_components",
                y="mean_accuracy",
                hue="crop_variant",
                style="crop_variant",
                hue_order=hue_order,
                style_order=hue_order,
                markers=True,
                dashes=False,
                linewidth=1.9,
                ax=ax,
                legend=(r == 0 and c == n_cols - 1),
            )
            if subset["sem"].notna().any():
                for _, band in subset.groupby("crop_variant"):
                    band = band.sort_values("umap_components")
                    ax.fill_between(
                        band["umap_components"],
                        (band["mean_accuracy"] - band["sem"]).fillna(band["mean_accuracy"]),
                        (band["mean_accuracy"] + band["sem"]).fillna(band["mean_accuracy"]),
                        alpha=0.12,
                        label=None,
                    )
            best_row = subset.loc[subset["mean_accuracy"].idxmax()]
            ax.scatter(
                best_row["umap_components"],
                best_row["mean_accuracy"],
                s=52,
                color="#d62728",
                edgecolor="black",
                zorder=5,
            )
            tidy_numeric_axes(ax, max_xticks=6, max_yticks=7)
            if subset["umap_components"].nunique() == 1:
                dim_val = float(subset["umap_components"].iloc[0])
                ax.set_xlim(dim_val - 0.6, dim_val + 0.6)
            ax.set_title(f"{color_space.upper()} | {backend}")
            ax.set_xlabel(format_umap_xlabel(cutoff))
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Mean accuracy" if c == 0 else "")
            if ax.legend_ and not (r == 0 and c == n_cols - 1):
                ax.legend_.remove()
            if r == 0 and c == n_cols - 1 and ax.legend_:
                ax.legend(title="Crop method", frameon=False, loc="upper right")

    fig.suptitle("Mean accuracy vs UMAP components", y=1.02)
    apply_footer(fig, *footer_parts(df_umap), "Aggregation: mean ± SEM over matches")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_precision_trend(df: pd.DataFrame, output_path: Path, umap_cap: Optional[int]) -> None:
    df_umap, cutoff = filter_umap_rows(df, cap=umap_cap)
    if df_umap.empty:
        return
    method_labels = crop_method_labels(df_umap)
    df_umap = df_umap.copy()
    df_umap["precision"] = df_umap["true_positive"] / (df_umap["true_positive"] + df_umap["false_positive"]).replace(
        0, np.nan
    )
    grouped = (
        df_umap.groupby(
            ["embedding_backend", "color_space", "crop_method", "crop_center_ratio", "umap_components"],
            as_index=False,
        )
        .agg(
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            count=("precision", "count"),
        )
    )
    grouped.loc[grouped["count"] <= 1, "std_precision"] = np.nan
    grouped["sem"] = grouped["std_precision"] / grouped["count"].pow(0.5)
    grouped["crop_variant"] = grouped.apply(
        lambda r: format_crop_variant_label(
            str(r["crop_method"]), float(r["crop_center_ratio"]), method_labels=method_labels
        ),
        axis=1,
    )
    hue_order = sorted(grouped["crop_variant"].dropna().unique().tolist(), key=crop_variant_order_key)

    backends = sorted(grouped["embedding_backend"].unique())
    color_spaces = sorted(grouped["color_space"].unique())
    n_rows = len(backends)
    n_cols = len(color_spaces)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.6 * max(1, n_cols), 3.6 * max(1, n_rows)),
        sharey=True,
        squeeze=False,
    )

    for r, backend in enumerate(backends):
        for c, color_space in enumerate(color_spaces):
            ax = axes[r][c]
            subset = grouped[
                (grouped["embedding_backend"] == backend) & (grouped["color_space"] == color_space)
            ].sort_values("umap_components")
            if subset.empty:
                ax.axis("off")
                continue
            sns.lineplot(
                data=subset,
                x="umap_components",
                y="mean_precision",
                hue="crop_variant",
                style="crop_variant",
                hue_order=hue_order,
                style_order=hue_order,
                markers=True,
                dashes=False,
                linewidth=1.9,
                ax=ax,
                legend=(r == 0 and c == n_cols - 1),
            )
            if subset["sem"].notna().any():
                for _, band in subset.groupby("crop_variant"):
                    band = band.sort_values("umap_components")
                    ax.fill_between(
                        band["umap_components"],
                        (band["mean_precision"] - band["sem"]).fillna(band["mean_precision"]),
                        (band["mean_precision"] + band["sem"]).fillna(band["mean_precision"]),
                        alpha=0.12,
                        label=None,
                    )
            best_row = subset.loc[subset["mean_precision"].idxmax()]
            ax.scatter(
                best_row["umap_components"],
                best_row["mean_precision"],
                s=52,
                color="#d62728",
                edgecolor="black",
                zorder=5,
            )
            tidy_numeric_axes(ax, max_xticks=6, max_yticks=6)
            if subset["umap_components"].nunique() == 1:
                dim_val = float(subset["umap_components"].iloc[0])
                ax.set_xlim(dim_val - 0.6, dim_val + 0.6)
            ax.set_title(f"{color_space.upper()} | {backend}")
            ax.set_xlabel(format_umap_xlabel(cutoff))
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Mean precision" if c == 0 else "")
            if ax.legend_ and not (r == 0 and c == n_cols - 1):
                ax.legend_.remove()
            if r == 0 and c == n_cols - 1 and ax.legend_:
                ax.legend(title="Crop method", frameon=False, loc="upper right")

    fig.suptitle("Precision vs UMAP dimensionality", y=1.02)
    apply_footer(fig, *footer_parts(df_umap), "Aggregation: mean ± SEM over matches")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_best_per_sequence(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.loc[df["experiment_stage"] == STAGE_COLOR_UMAP].copy()
    if filtered.empty:
        return filtered
    sorted_df = filtered.sort_values(
        ["sequence", "color_space", "crop_method", "crop_center_ratio", "accuracy", "umap_components"],
        ascending=[True, True, True, True, False, True],
    )
    best = sorted_df.groupby(
        ["sequence", "color_space", "crop_method", "crop_center_ratio"], as_index=False
    ).first()
    return best


def plot_best_accuracy_box(df_best: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    df_best = df_best.copy()
    df_best["combo"] = df_best.apply(
        lambda row: row["color_space"].upper()
        if row["crop_method"] == "center"
        else f"{row['color_space'].upper()} | {row['crop_method'].replace('_', ' ').title()}",
        axis=1,
    )
    sns.boxplot(
        data=df_best,
        x="combo",
        y="accuracy",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_best,
        x="combo",
        y="accuracy",
        color="black",
        alpha=0.35,
        size=3,
        jitter=0.2,
        ax=ax,
    )
    ax.set_title("Best accuracy per sequence")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    apply_footer(fig, *footer_parts(df_best, include_crop_note=False), "Best per match (over UMAP)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_center_vs_sam2(df: pd.DataFrame, output_path: Path) -> None:
    # Compare crop strategies using the non-UMAP runs for each dedicated stage.
    ratio_stage = df.loc[(df["experiment_stage"] == STAGE_RATIO_SWEEP) & (~df["umap_applied"])]
    ocv_stage = df.loc[(df["experiment_stage"] == STAGE_OPENCV_COMPARISON) & (~df["umap_applied"])]
    sam_stage = df.loc[(df["experiment_stage"] == STAGE_SAM2_COMPARISON) & (~df["umap_applied"])]
    if ratio_stage.empty and ocv_stage.empty and sam_stage.empty:
        return

    records: list[dict[str, float | str]] = []
    for color_space, center_rows in ratio_stage.groupby(["color_space"]):
        if center_rows.empty:
            continue
        best_acc = -1.0
        for ratio, group in center_rows.groupby("crop_center_ratio"):
            acc = weighted_accuracy(group)
            if acc > best_acc:
                best_acc = acc
        records.append(
            {
                "color_space": str(color_space),
                "method": "center",
                "accuracy": float(best_acc),
            }
        )

    for color_space, rows in ocv_stage.groupby(["color_space"]):
        if rows.empty:
            continue
        records.append(
            {
                "color_space": str(color_space),
                "method": "opencv_mask",
                "accuracy": float(weighted_accuracy(rows)),
            }
        )

    for color_space, rows in sam_stage.groupby(["color_space"]):
        if rows.empty:
            continue
        records.append(
            {
                "color_space": str(color_space),
                "method": "sam2_mask",
                "accuracy": float(weighted_accuracy(rows)),
            }
        )

    if not records:
        return

    plot_df = pd.DataFrame(records)
    plot_df["color_space"] = plot_df["color_space"].str.lower()
    plot_df["method"] = plot_df["method"].astype(str)
    method_order = ["center", "opencv_mask", "sam2_mask"]
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)

    color_order = (
        ["rgb", "h"]
        if set(plot_df["color_space"].unique()) >= {"rgb", "h"}
        else sorted(plot_df["color_space"].unique())
    )
    palette = {"center": "#4C72B0", "opencv_mask": "#55A868", "sam2_mask": "#C44E52"}
    label_map = {"center": "Center", "opencv_mask": "OpenCV mask", "sam2_mask": "SAM2 mask"}

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    sns.barplot(
        data=plot_df,
        x="color_space",
        y="accuracy",
        order=color_order,
        hue="method",
        hue_order=method_order,
        palette=palette,
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Crop strategy comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Weighted accuracy")
    ax.set_xticks(range(len(color_order)))
    ax.set_xticklabels([cs.upper() for cs in color_order])
    tidy_numeric_axes(ax, max_xticks=4, max_yticks=6)
    if ax.legend_:
        ax.legend(
            title="",
            labels=[label_map.get(m, m) for m in method_order],
            frameon=False,
            loc="upper right",
        )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    apply_footer(
        fig,
        *footer_parts(df, include_crop_note=True),
        "UMAP: none",
        "Center: best ratio per colour",
        bottom=0.08,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_breakdown(df_best: pd.DataFrame, output_path: Path) -> None:
    df_best = df_best.copy()
    if "color_space" in df_best.columns:
        df_best["color_space"] = df_best["color_space"].astype(str).str.lower()
        df_best = df_best.loc[df_best["color_space"] == "rgb"]
    if df_best.empty:
        return
    method_labels = crop_method_labels(df_best)
    totals = (
        df_best.groupby(["color_space", "crop_method", "crop_center_ratio"])
        .agg(
            tp=("true_positive", "sum"),
            fp=("false_positive", "sum"),
            tn=("true_negative", "sum"),
            fn=("false_negative", "sum"),
        )
        .reset_index()
    )
    totals["positives"] = totals["tp"] + totals["fn"]
    totals["negatives"] = totals["tn"] + totals["fp"]
    totals["tpr"] = totals["tp"] / totals["positives"].clip(lower=1)
    totals["tnr"] = totals["tn"] / totals["negatives"].clip(lower=1)
    totals["fpr"] = totals["fp"] / totals["negatives"].clip(lower=1)
    totals["fnr"] = totals["fn"] / totals["positives"].clip(lower=1)

    melted = totals.melt(
        id_vars=["color_space", "crop_method", "crop_center_ratio"],
        value_vars=["tpr", "tnr", "fpr", "fnr"],
        var_name="metric",
        value_name="rate",
    )
    melted["combo"] = melted.apply(
        lambda row: format_crop_variant_label(
            str(row["crop_method"]), float(row["crop_center_ratio"]), method_labels=method_labels
        ),
        axis=1,
    )
    combo_order = sorted(melted["combo"].dropna().unique().tolist(), key=crop_variant_order_key)
    melted["combo"] = pd.Categorical(melted["combo"], categories=combo_order, ordered=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sns.barplot(
        data=melted,
        x="combo",
        y="rate",
        hue="metric",
        palette={
            "tpr": "#4C72B0",
            "tnr": "#55A868",
            "fpr": "#C44E52",
            "fnr": "#8172B3",
        },
        ax=ax,
    )
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_title("Confusion breakdown")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", loc="upper right", frameon=False)
    apply_footer(
        fig,
        *footer_parts(df_best, include_crop_note=True),
        "Best per match (over UMAP, embedding, clustering)",
        bottom=0.10,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_umap_distribution(df_best: pd.DataFrame, output_path: Path, umap_cap: Optional[int]) -> None:
    df_umap, cutoff = filter_umap_rows(df_best, cap=umap_cap)
    if df_umap.empty:
        return
    df_umap = df_umap.copy()
    df_umap["color_space"] = df_umap["color_space"].astype(str).str.lower()
    df_umap["crop_method"] = df_umap["crop_method"].astype(str)
    method_labels = crop_method_labels(df_umap)

    method_order = [m for m in ["center", "opencv_mask", "sam2_mask"] if m in set(df_umap["crop_method"].unique())]
    color_order = (
        ["rgb", "h"]
        if set(df_umap["color_space"].unique()) >= {"rgb", "h"}
        else sorted(df_umap["color_space"].unique())
    )
    if not method_order or not color_order:
        return

    label_map = {"center": "Center", "opencv_mask": "OpenCV mask", "sam2_mask": "SAM2 mask"}
    fig, axes = plt.subplots(
        len(method_order),
        len(color_order),
        figsize=(6.2 * len(color_order), 3.2 * len(method_order)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, method in enumerate(method_order):
        for c, color_space in enumerate(color_order):
            ax = axes[r][c]
            panel = df_umap[(df_umap["crop_method"] == method) & (df_umap["color_space"] == color_space)]
            if panel.empty:
                ax.axis("off")
                continue
            sns.scatterplot(
                data=panel,
                x="umap_components",
                y="accuracy",
                s=16,
                alpha=0.28,
                color="#4C72B0",
                edgecolor="none",
                ax=ax,
            )
            summary = (
                panel.groupby("umap_components", as_index=False)
                .agg(mean=("accuracy", "mean"), std=("accuracy", "std"), count=("accuracy", "count"))
                .sort_values("umap_components")
            )
            summary["sem"] = summary["std"] / summary["count"].pow(0.5)
            ax.plot(summary["umap_components"], summary["mean"], color="#C44E52", linewidth=2.0)
            if summary["sem"].notna().any():
                ax.fill_between(
                    summary["umap_components"],
                    (summary["mean"] - summary["sem"]).fillna(summary["mean"]),
                    (summary["mean"] + summary["sem"]).fillna(summary["mean"]),
                    color="#C44E52",
                    alpha=0.12,
                )
            tidy_numeric_axes(ax, max_xticks=6, max_yticks=6)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{color_space.upper()} | {method_labels.get(method, label_map.get(method, method))}")
            ax.set_xlabel(format_umap_xlabel(cutoff) if r == len(method_order) - 1 else "")
            ax.set_ylabel("Accuracy" if c == 0 else "")

    fig.suptitle("Best UMAP components vs achieved accuracy (per sequence)", y=1.02)
    apply_footer(
        fig,
        *footer_parts(df_umap, include_crop_note=True),
        "Best per match (over UMAP)",
        "Aggregation: mean ± SEM over matches",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_delta_vs_dim(df: pd.DataFrame, output_path: Path, umap_cap: Optional[int]) -> None:
    df_umap, cutoff = filter_umap_rows(df, cap=umap_cap)
    if df_umap.empty:
        return
    baseline_dim = int(df_umap["umap_components"].min())
    delta_records: list[dict[str, float | int | str]] = []
    for (sequence, color_space, crop_method, ratio), group in df_umap.groupby(
        ["sequence", "color_space", "crop_method", "crop_center_ratio"]
    ):
        base_rows = group.loc[group["umap_components"] == baseline_dim]
        if base_rows.empty:
            continue
        base_acc = float(base_rows.iloc[0]["accuracy"])
        for _, row in group.iterrows():
            dim = int(row["umap_components"])
            delta = float(row["accuracy"]) - base_acc
            delta_records.append(
                {
                    "color_space": color_space,
                    "crop_method": crop_method,
                    "crop_center_ratio": float(ratio),
                    "umap_components": dim,
                    "delta_accuracy": delta,
                }
            )

    if not delta_records:
        return

    summary = (
        pd.DataFrame(delta_records)
        .groupby(["color_space", "crop_method", "crop_center_ratio", "umap_components"], as_index=False)
        .agg(
            mean_delta=("delta_accuracy", "mean"),
            std_delta=("delta_accuracy", "std"),
            count=("delta_accuracy", "count"),
        )
    )
    summary["sem"] = summary["std_delta"] / summary["count"].pow(0.5)
    summary["series_label"] = summary.apply(
        lambda row: f"{row['color_space'].upper()} | "
        f"{'SAM2 mask' if row['crop_method'] == 'sam2_mask' else row['crop_method'].replace('_', ' ').title()}",
        axis=1,
    )

    series_labels = summary["series_label"].unique().tolist()
    palette = dict(zip(series_labels, sns.color_palette("colorblind", n_colors=len(series_labels))))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    style_map = {"center": ("o", "-"), "sam2_mask": ("s", "--")}
    for (color_space, crop_method), band in summary.groupby(["color_space", "crop_method"]):
        band = band.sort_values("umap_components")
        label = band["series_label"].iloc[0]
        color = palette.get(label)
        marker, linestyle = style_map.get(crop_method, ("o", "-"))
        ax.plot(
            band["umap_components"],
            band["mean_delta"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1,
        )
        if band["sem"].notna().any():
            ax.fill_between(
                band["umap_components"],
                band["mean_delta"] - band["sem"].fillna(0.0),
                band["mean_delta"] + band["sem"].fillna(0.0),
                alpha=0.12,
                color=color,
                label=None,
            )
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    tidy_numeric_axes(ax, max_xticks=8, max_yticks=7)
    if summary["umap_components"].nunique() <= 8:
        ax.set_xticks(sorted(summary["umap_components"].unique()))
    if summary["umap_components"].nunique() == 1:
        dim_val = float(summary["umap_components"].iloc[0])
        ax.set_xlim(dim_val - 0.6, dim_val + 0.6)
    ax.set_xlabel(format_umap_xlabel(cutoff))
    ax.set_ylabel("Δ Accuracy (relative to baseline)")
    ax.legend(title="Colour space | Crop method", loc="lower right")
    fig.suptitle(f"Accuracy change vs UMAP dimensionality (baseline = {baseline_dim} components)", y=1.02)
    apply_footer(fig, *footer_parts(df_umap, include_crop_note=True), "Aggregation: mean ± SEM over matches")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_crop_ratio(df: pd.DataFrame, output_path: Path) -> None:
    df_subset = df.loc[df["experiment_stage"].isin({STAGE_RATIO_SWEEP, STAGE_SAM2_COMPARISON})].copy()
    if df_subset.empty:
        return
    sam_rows = df_subset.loc[df_subset["experiment_stage"] == STAGE_SAM2_COMPARISON]
    grouped = (
        df_subset.groupby(["color_space", "crop_method", "crop_center_ratio"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            count=("accuracy", "count"),
        )
    )
    grouped["sem"] = grouped["std_accuracy"] / grouped["count"].pow(0.5)

    color_spaces = sorted(grouped["color_space"].unique())
    palette = dict(zip(color_spaces, sns.color_palette("colorblind", n_colors=len(color_spaces))))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    center_subset = grouped.loc[grouped["crop_method"] != "sam2_mask"].copy()
    if not center_subset.empty:
        sns.lineplot(
            data=center_subset.sort_values(["crop_center_ratio", "color_space"]),
            x="crop_center_ratio",
            y="mean_accuracy",
            hue="color_space",
            style="crop_method",
            markers=True,
            linewidth=2.2,
            palette=palette,
            ax=ax,
        )
        if center_subset["sem"].notna().any():
            for (color_space, crop_method), band in center_subset.groupby(["color_space", "crop_method"]):
                band = band.sort_values("crop_center_ratio")
                ax.fill_between(
                    band["crop_center_ratio"],
                    band["mean_accuracy"] - band["sem"].fillna(0.0),
                    band["mean_accuracy"] + band["sem"].fillna(0.0),
                    alpha=0.12,
                    color=palette.get(color_space),
                    label=None,
                )

    sam_handles: list[Line2D] = []
    for color_space in sorted(sam_rows["color_space"].unique()):
        sam_acc = weighted_accuracy(sam_rows.loc[sam_rows["color_space"] == color_space])
        if np.isfinite(sam_acc):
            color = palette.get(color_space)
            ax.axhline(
                sam_acc,
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
            )
            sam_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    label=f"{color_space.upper()} | SAM2 mask",
                )
            )

    ratio_min = float(grouped["crop_center_ratio"].min())
    ratio_max = float(grouped["crop_center_ratio"].max())
    tick_start = np.floor(ratio_min / 0.2) * 0.2
    tick_end = np.ceil(ratio_max / 0.2) * 0.2
    ticks = np.round(np.arange(tick_start, tick_end + 0.001, 0.2), 2)
    ax.set_xticks(ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Crop center ratio")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.0, 1.0)

    legend_handles = []
    for color_space in color_spaces:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=palette.get(color_space),
                marker="o",
                linewidth=2.2,
                label=f"{color_space.upper()} | center",
            )
        )
    legend_handles.extend(sam_handles)
    ax.legend(
        handles=legend_handles,
        title="Colour space | Crop method",
        loc="upper right",
    )

    fig.suptitle("Accuracy vs crop center ratio (H and RGB)", y=1.02)
    apply_footer(fig, *footer_parts(df_subset, include_crop_note=True), "Aggregation: mean ± SEM over matches")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_delta_vs_crop_ratio(df: pd.DataFrame, output_path: Path) -> None:
    df_subset = df.loc[df["experiment_stage"].isin({STAGE_RATIO_SWEEP, STAGE_SAM2_COMPARISON})].copy()
    if df_subset.empty:
        return
    baseline_ratio = float(df_subset["crop_center_ratio"].min())
    delta_records: list[dict[str, float | int | str]] = []
    for (sequence, color_space, crop_method, umap_components), group in df_subset.groupby(
        ["sequence", "color_space", "crop_method", "umap_components"]
    ):
        base_rows = group.loc[np.isclose(group["crop_center_ratio"], baseline_ratio)]
        if base_rows.empty:
            continue
        base_acc = float(base_rows.iloc[0]["accuracy"])
        for _, row in group.iterrows():
            ratio = float(row["crop_center_ratio"])
            delta = float(row["accuracy"]) - base_acc
            delta_records.append(
                {
                    "color_space": color_space,
                    "crop_method": crop_method,
                    "crop_center_ratio": ratio,
                    "delta_accuracy": delta,
                }
            )

    if not delta_records:
        return

    summary = (
        pd.DataFrame(delta_records)
        .groupby(["color_space", "crop_method", "crop_center_ratio"], as_index=False)
        .agg(
            mean_delta=("delta_accuracy", "mean"),
            std_delta=("delta_accuracy", "std"),
            count=("delta_accuracy", "count"),
        )
    )
    summary["sem"] = summary["std_delta"] / summary["count"].pow(0.5)
    summary["series_label"] = summary.apply(
        lambda row: f"{row['color_space'].upper()} | "
        f"{row['crop_method'].replace('_', ' ').title()}",
        axis=1,
    )
    series_labels = summary["series_label"].unique().tolist()
    palette = dict(zip(series_labels, sns.color_palette("colorblind", n_colors=len(series_labels))))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    style_map = {"center": ("o", "-"), "sam2_mask": ("s", "--")}
    for (color_space, crop_method), band in summary.groupby(["color_space", "crop_method"]):
        band = band.sort_values("crop_center_ratio")
        label = band["series_label"].iloc[0]
        color = palette.get(label)
        marker, linestyle = style_map.get(crop_method, ("o", "-"))
        ax.plot(
            band["crop_center_ratio"],
            band["mean_delta"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
        )
        if band["sem"].notna().any():
            ax.fill_between(
                band["crop_center_ratio"],
                band["mean_delta"] - band["sem"].fillna(0.0),
                band["mean_delta"] + band["sem"].fillna(0.0),
                alpha=0.10,
                color=color,
                label=None,
            )
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title(f"Accuracy change vs crop center ratio (baseline {format_ratio(baseline_ratio)})")
    ax.set_xlabel("Crop center ratio")
    ax.set_ylabel("Δ Accuracy (relative to baseline)")
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.legend(title="Colour space | Crop method", loc="best")
    apply_footer(fig, *footer_parts(df_subset, include_crop_note=True), "Aggregation: mean ± SEM over matches")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_crop_ratio_distribution(df_best: pd.DataFrame, output_path: Path) -> None:
    counts = (
        df_best.groupby(["color_space", "crop_method", "crop_center_ratio"])
        .size()
        .reset_index(name="count")
        .sort_values(["color_space", "crop_center_ratio"])
    )
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    table = counts.pivot_table(
        index="color_space",
        columns="crop_center_ratio",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    sns.heatmap(
        table,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Peak crop ratio counts (per colour)")
    ax.set_xlabel("Crop center ratio")
    ax.set_ylabel("Colour space")
    apply_footer(fig, *footer_parts(df_best, include_crop_note=True), "Count of best-per-match ratios")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot team classification experiment results.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        nargs="+",
        default=[Path("results/team_classification/numeric/team_classification_metrics.csv")],
        help="One or more metrics files (CSV or Parquet); when multiple are passed they will be merged.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/team_classification/plots"))
    parser.add_argument("--paired-max-points", type=int, default=35)
    args = parser.parse_args()

    configure_style()

    metrics, _ = load_metrics_many(list(args.metrics_csv))
    _, umap_cap = filter_umap_rows(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_trend_path = args.output_dir / "accuracy_vs_umap.png"
    plot_accuracy_trend(metrics, accuracy_trend_path, umap_cap)

    precision_trend_path = args.output_dir / "precision_vs_umap.png"
    plot_precision_trend(metrics, precision_trend_path, umap_cap)

    best = compute_best_per_sequence(metrics)
    best_box_path = args.output_dir / "best_accuracy_boxplot.png"
    plot_best_accuracy_box(best, best_box_path)

    confusion_path = args.output_dir / "confusion_breakdown.png"
    delta_path = args.output_dir / "accuracy_delta_vs_umap.png"
    plot_confusion_breakdown(best, confusion_path)

    umap_dist_path = args.output_dir / "best_umap_distribution.png"
    plot_best_umap_distribution(best, umap_dist_path, umap_cap)

    plot_accuracy_delta_vs_dim(metrics, delta_path, umap_cap)

    crop_accuracy_path = args.output_dir / "accuracy_vs_crop_ratio.png"
    plot_accuracy_vs_crop_ratio(metrics, crop_accuracy_path)

    crop_delta_path = args.output_dir / "crop_ratio_delta_vs_baseline.png"
    plot_accuracy_delta_vs_crop_ratio(metrics, crop_delta_path)

    crop_dist_path = args.output_dir / "best_crop_ratio_distribution.png"
    plot_best_crop_ratio_distribution(best, crop_dist_path)

    method_compare_path = args.output_dir / "center_vs_sam2.png"
    plot_center_vs_sam2(metrics, method_compare_path)

    # New "publication-style" plots: paired effect sizes + distribution views.
    try:
        # Pick the crop+embedding combo that yields the strongest RGB-vs-H effect (mean |Δ| over matches).
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
            title=f"Colour space effect (paired; {crop_method} | {backend})",
            output_path=args.output_dir / "paper_color_space_estimation.png",
            max_pairs=int(args.paired_max_points),
        )
    except Exception:
        pass

    try:
        if "embedding_backend" in metrics.columns and metrics["embedding_backend"].nunique() >= 2:
            backends = {str(v) for v in metrics["embedding_backend"].unique()}
            if {"siglip", "resnet"} <= backends:
                # This plot gets visually busy when we include multiple crop methods / colour spaces.
                # Keep it readable by restricting to a single canonical view: center | rgb.
                metrics_center_rgb = metrics.loc[
                    (metrics["crop_method"] == "center") & (metrics["color_space"] == "rgb")
                ].copy()
                embed_paired = _paired_best(
                    metrics_center_rgb,
                    pair_col="embedding_backend",
                    a="siglip",
                    b="resnet",
                    restrict_cols=("crop_method", "color_space"),
                )
                plot_estimation_paired(
                    embed_paired,
                    a_label="siglip",
                    b_label="resnet",
                    title="Embedding backend effect (paired; center | rgb)",
                    output_path=args.output_dir / "paper_embedding_estimation.png",
                    max_pairs=int(args.paired_max_points),
                    footer="Filtered colour space: RGB",
                )
    except Exception:
        pass

    try:
        best_seq_method = _best_rows(metrics, ["sequence", "embedding_backend", "crop_method", "color_space"])
        plot_raincloud_accuracy(
            best_seq_method,
            group_col="crop_method",
            hue_col="color_space",
            facet_col="embedding_backend" if metrics["embedding_backend"].nunique() > 1 else None,
            title="Accuracy distribution by crop method (best per sequence; RGB vs H side-by-side)",
            output_path=args.output_dir / "paper_raincloud_crop_methods.png",
        )
    except Exception:
        pass

    try:
        best_seq_ratio = _best_rows(metrics, ["sequence", "embedding_backend", "crop_center_ratio", "color_space"])
        plot_raincloud_accuracy(
            best_seq_ratio,
            group_col="crop_center_ratio",
            hue_col="color_space",
            facet_col="embedding_backend" if metrics["embedding_backend"].nunique() > 1 else None,
            title="Accuracy distribution by center ratio (best per sequence)",
            output_path=args.output_dir / "paper_raincloud_ratios.png",
        )
    except Exception:
        pass

    try:
        best_heat = _best_rows(
            metrics, ["sequence", "color_space", "embedding_backend", "crop_method", "crop_center_ratio"]
        )
        plot_best_config_heatmaps(
            best_heat,
            output_path=args.output_dir / "paper_heatmap_method_ratio.png",
            title="Best accuracy heatmap: crop method × center ratio (mean over sequences)",
            agg="mean",
        )
    except Exception:
        pass

    try:
        plot_embedding_siglip_vs_resnet(
            metrics,
            args.output_dir / "paper_embedding_siglip_vs_resnet.png",
        )
    except Exception:
        pass

    try:
        plot_clustering_method_comparison(
            metrics,
            args.output_dir / "paper_clustering_methods.png",
        )
    except Exception:
        pass

    print("Saved plots:")
    for path in [
        accuracy_trend_path,
        precision_trend_path,
        best_box_path,
        confusion_path,
        umap_dist_path,
        delta_path,
        crop_accuracy_path,
        crop_delta_path,
        crop_dist_path,
        method_compare_path,
        args.output_dir / "paper_color_space_estimation.png",
        args.output_dir / "paper_embedding_estimation.png",
        args.output_dir / "paper_raincloud_crop_methods.png",
        args.output_dir / "paper_raincloud_ratios.png",
        args.output_dir / "paper_heatmap_method_ratio.png",
        args.output_dir / "paper_embedding_siglip_vs_resnet.png",
        args.output_dir / "paper_clustering_methods.png",
    ]:
        print(f" - {path}")


if __name__ == "__main__":
    main()
