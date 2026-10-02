"""Matplotlib figures for notebooks: images, dataset samples, training curves, pitch maps, metrics.

matplotlib is imported inside the functions. Every helper returns a
``matplotlib.figure.Figure`` that pyplot does not manage, so a notebook shows
it exactly once (as the cell's value) and scripts can ``fig.savefig(...)``.
"""

import math
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.data.dataset import Dataset, Sample
from tactifoot_vision.evaluation.metrics import DetectionMetrics, KeypointMetrics
from tactifoot_vision.pipeline.result import NO_TEAM, PipelineResult
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.viz.annotate import CLASS_COLORS, draw_annotations
from tactifoot_vision.viz.radar import (
    DEFAULT_COLOR,
    LINE_COLOR,
    PITCH_COLOR,
    TEAM_COLORS,
    ColorLike,
    as_color,
    pitch_markings,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tactifoot_vision.augment.base import Transform
    from tactifoot_vision.models.base import TrainResult

# Chart chrome: quiet ink and hairlines on a near-white surface.
_SURFACE = "#FCFCFB"
_INK = "#0B0B0B"
_INK_2 = "#52514E"
_MUTED = "#898781"
_GRID = "#E1E0D9"
_AXIS = "#C3C2B7"
# Categorical series colours in fixed order (validated for colour-blind separation).
_SERIES = (
    "#2A78D6",
    "#EB6834",
    "#1BAF7A",
    "#EDA100",
    "#E87BA4",
    "#008300",
    "#4A3AA7",
    "#E34948",
)
_ALL_PLAYERS_HEAT = "#FFB000"
_STD_LENGTH, _STD_WIDTH = 105.0, 68.0


def _theme() -> AbstractContextManager:
    import matplotlib as mpl

    return mpl.rc_context({
        "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE, "savefig.facecolor": _SURFACE,
        "axes.edgecolor": _AXIS, "axes.linewidth": 0.8, "axes.labelcolor": _INK_2,
        "axes.labelsize": 9, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlecolor": _INK, "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": _GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": _AXIS, "ytick.color": _AXIS, "xtick.labelcolor": _MUTED,
        "ytick.labelcolor": _MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": _INK_2,
        "text.color": _INK_2, "font.family": "sans-serif",
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.titlesize": 12.5, "figure.titleweight": "semibold",
    })  # fmt: skip


def _new_figure(width: float, height: float) -> "Figure":
    from matplotlib.figure import Figure

    # Not managed by pyplot: no GUI window opens in scripts, and notebooks show the
    # figure once, as the cell's value.
    _enable_notebook_display()
    return Figure(figsize=(width, height), layout="constrained")


def _enable_notebook_display() -> None:
    """Make IPython render ``Figure`` objects (the inline backend's formatters).

    Other libraries can drop the formatters by switching backends (Ultralytics
    training does), so they are re-registered whenever they are missing.
    """
    if "IPython" not in sys.modules:
        return
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None or not hasattr(shell, "display_formatter"):
        return
    from IPython.core.pylabtools import select_figure_formats
    from matplotlib.figure import Figure

    formatters = shell.display_formatter.formatters.values()
    if any(Figure in formatter.type_printers for formatter in formatters):
        return
    try:
        from matplotlib_inline.backend_inline import InlineBackend
    except ImportError:  # not a Jupyter kernel
        return
    config = InlineBackend.instance(parent=shell)
    select_figure_formats(shell, config.figure_formats, **config.print_figure_kwargs)


def _titles(ax: "Axes", title: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional quieter line underneath."""
    if not subtitle:
        ax.set_title(title)
        return
    ax.set_title(subtitle, fontsize=9, fontweight="normal", color=_INK_2)
    ax.annotate(
        title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 23),
        textcoords="offset points", fontsize=11.5, fontweight="semibold", color=_INK,
        va="bottom", ha="left", annotation_clip=False,
    )  # fmt: skip


def _to_rgb(image: np.ndarray, bgr: bool) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3 and bgr:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4 and bgr:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def _rgb(color: ColorLike) -> tuple[float, float, float]:
    c = as_color(color)
    return c.r / 255, c.g / 255, c.b / 255


# ---------------------------------------------------------------------- images
def show(
    images: np.ndarray | Sequence[np.ndarray],
    titles: str | Sequence[str | None] | None = None,
    cols: int = 3,
    width: float = 12.0,
    bgr: bool = True,
) -> "Figure":
    """Show one image or a grid of images.

    Args:
        images: an ``H x W (x C)`` array or a list of them.
        titles: one title per image (or a single string for a single image).
        cols: images per row.
        width: figure width in inches (the height follows the images).
        bgr: images are OpenCV BGR (converted to RGB for display).
    """
    single = isinstance(images, np.ndarray) and (
        images.ndim == 2 or (images.ndim == 3 and images.shape[2] in (1, 3, 4))
    )
    images = [images] if single else list(images)  # type: ignore[list-item]
    if not images:
        raise ValueError("No images to show")
    if titles is None:
        titles = [None] * len(images)
    elif isinstance(titles, str):
        titles = [titles]
    if len(titles) != len(images):
        raise ValueError(f"Got {len(titles)} titles for {len(images)} images")
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    aspect = max(image.shape[0] / image.shape[1] for image in images)
    title_space = 0.3 if any(titles) else 0.0
    with _theme():
        fig = _new_figure(width, rows * (width / cols * aspect + title_space))
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, image, title in zip(axes.flat, images, titles, strict=False):
            cmap = "gray" if image.ndim == 2 or image.shape[2] == 1 else None
            ax.imshow(_to_rgb(image, bgr), cmap=cmap)
            if title:
                ax.set_title(
                    title, fontsize=9, fontweight="normal", color=_INK_2, loc="center"
                )
    return fig


def _pick(count: int, n: int, seed: int) -> np.ndarray:
    return np.sort(
        np.random.default_rng(seed).choice(count, size=min(n, count), replace=False)
    )


def _split_samples(dataset: Dataset, split: str) -> list[Sample]:
    samples = dataset[split]
    if not samples:
        raise ValueError(f"Split {split!r} is empty; available: {dataset.split_names}")
    return samples


def show_samples(
    dataset: Dataset, split: str = "train", n: int = 6, seed: int = 0, cols: int = 3
) -> "Figure":
    """Random images of a split with their ground-truth boxes, classes and keypoints."""
    samples = _split_samples(dataset, split)
    images, titles = [], []
    for i in _pick(len(samples), n, seed):
        sample = samples[i]
        images.append(
            draw_annotations(
                sample.read_image(), sample.annotations, dataset.class_names
            )
        )
        titles.append(
            f"{_short(sample.image_path.stem)} · {len(sample.annotations)} objects"
        )
    return _with_class_legend(show(images, titles, cols=cols), dataset.class_names)


def show_augmentations(
    dataset: Dataset,
    transform: "Transform",
    n: int = 4,
    split: str = "train",
    seed: int = 0,
) -> "Figure":
    """Original vs augmented image pairs (one row each), labels drawn on both.

    ``transform`` is called as ``transform(image, annotations, rng)`` with an
    ``np.random.Generator`` seeded from ``seed``.
    """
    samples = _split_samples(dataset, split)
    rng = np.random.default_rng(seed)
    images, titles = [], []
    for i in _pick(len(samples), n, seed):
        image, annotations = dataset.read(split, int(i))
        original = draw_annotations(image, annotations, dataset.class_names)
        new_image, new_annotations = transform(image, annotations, rng)
        images += [
            original,
            draw_annotations(new_image, new_annotations, dataset.class_names),
        ]
        titles += [f"{_short(samples[i].image_path.stem)} · original", "augmented"]
    return _with_class_legend(show(images, titles, cols=2), dataset.class_names)


def _with_class_legend(fig: "Figure", class_names: Sequence[str]) -> "Figure":
    """Box colour -> class name key under the image grid (skipped for a single class)."""
    if len(class_names) < 2:
        return fig
    from matplotlib.patches import Patch

    handles = [
        Patch(color=CLASS_COLORS[i % len(CLASS_COLORS)], label=name)
        for i, name in enumerate(class_names)
    ]
    with _theme():
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return fig


def _short(text: str, limit: int = 28) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -------------------------------------------------------------------- training
_TRAIN_PREFIXES = ("train/", "train_")
_VALID_PREFIXES = ("val/", "val_", "valid/", "valid_", "test/", "test_")
_METRIC_LABELS = {
    "map5095": "mAP50-95",
    "map50": "mAP50",
    "map75": "mAP75",
    "precision": "precision",
    "recall": "recall",
}
_METRIC_KINDS = {"B": "box", "P": "pose", "M": "mask"}
# Metric hues avoid the train/validation blue and orange of the loss panels.
_METRIC_COLORS = {
    "mAP50-95": "#4A3AA7",
    "mAP50": "#1BAF7A",
    "mAP75": "#EDA100",
    "precision": "#E87BA4",
    "recall": "#008300",
}


def _split_prefix(lower: str) -> tuple[str, str]:
    for prefix in _VALID_PREFIXES:
        if lower.startswith(prefix):
            return "valid", lower[len(prefix) :]
    for prefix in _TRAIN_PREFIXES:
        if lower.startswith(prefix):
            return "train", lower[len(prefix) :]
    return "train", lower


def _numeric(series: pd.Series) -> pd.Series | None:
    values = pd.to_numeric(series, errors="coerce")
    return values if values.notna().any() else None


def _loss_series(history: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """``{panel title: {"train"|"valid": values}}``; auxiliary decoder losses are skipped."""
    groups: dict[str, dict[str, pd.Series]] = {}
    for column in history.columns:
        lower = str(column).strip().lower()
        if "loss" not in lower or lower.startswith("ema"):  # RF-DETR EMA duplicates
            continue
        if (values := _numeric(history[column])) is None:
            continue
        split, name = _split_prefix(lower)
        if re.search(
            r"_\d+$|_enc$|unscaled", name
        ):  # RF-DETR per-layer / unscaled copies
            continue
        core = name.replace("loss", "").strip("_/ ") or "total"
        groups.setdefault(f"{core} loss", {})[split] = values
    return dict(sorted(groups.items(), key=lambda item: item[0] != "total loss"))


def _metric_series(history: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """``{kind: {metric label: values}}`` from Ultralytics or RF-DETR history columns."""
    groups: dict[str, dict[str, pd.Series]] = {}
    for column in history.columns:
        name = str(column).strip()
        lower = name.lower()
        if "loss" in lower or lower.startswith("ema"):
            continue
        if "coco_eval_bbox" in lower:  # RF-DETR: list of the 12 COCO stats per epoch
            stats = history[column].map(
                lambda v: list(v)[:3]
                if isinstance(v, list | tuple | np.ndarray)
                else [np.nan] * 3
            )
            table = pd.DataFrame(stats.tolist(), index=history.index, dtype=float)
            for i, label in enumerate(("mAP50-95", "mAP50", "mAP75")):
                groups.setdefault("box", {})[label] = table[i]
            continue
        kind = ""
        if lower.startswith("metrics/"):
            name = name[len("metrics/") :]
            if match := re.fullmatch(r"(.*)\((\w)\)", name):
                name, kind = (
                    match.group(1),
                    _METRIC_KINDS.get(match.group(2), match.group(2)),
                )
        elif not re.search(r"map|precision|recall", lower):
            continue
        else:
            name = _split_prefix(lower)[1]
        if (values := _numeric(history[column])) is None:
            continue
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        groups.setdefault(kind, {})[_METRIC_LABELS.get(key, name)] = values
    return groups


def plot_training(train_result: "TrainResult | pd.DataFrame") -> "Figure":
    """Loss curves (train vs validation, one panel per loss term) and validation metrics.

    Understands Ultralytics ``results.csv`` columns (``train/box_loss``,
    ``metrics/mAP50(B)``, ...) and RF-DETR logs (``train_loss``, ``test_loss``,
    ``test_coco_eval_bbox``, ...). The best epoch of the headline metric is marked.
    """
    history = getattr(train_result, "history", train_result)
    history = history.reset_index(drop=True)
    columns = {str(c).strip().lower(): c for c in history.columns}
    epochs = (
        pd.to_numeric(history[columns["epoch"]], errors="coerce").to_numpy()
        if "epoch" in columns
        else np.arange(1, len(history) + 1)
    )
    losses = _loss_series(history)
    metrics = _metric_series(history)
    panels: list[tuple[str, dict[str, pd.Series]]] = [
        *list(losses.items())[:6],
        *[
            (f"{kind} metrics" if kind else "validation metrics", series)
            for kind, series in metrics.items()
        ],
    ]
    name = getattr(train_result, "model_type", None)
    with _theme():
        if not panels:
            fig = _new_figure(6, 3)
            ax = fig.add_subplot()
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No loss or metric columns in the history",
                ha="center",
                va="center",
            )
            return fig
        from matplotlib.ticker import MaxNLocator

        cols = min(3, len(panels))
        rows = math.ceil(len(panels) / cols)
        fig = _new_figure(4.4 * cols, 3.1 * rows + 0.5)
        axes = list(fig.subplots(rows, cols, squeeze=False).flat)
        marker = "o" if len(history) < 3 else None
        for ax, (title, series) in zip(axes, panels, strict=False):
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            if len(epochs) <= 12:  # few epochs: tick each one (no fractional epochs)
                ax.set_xticks(epochs)
                ax.set_xlim(np.nanmin(epochs) - 0.5, np.nanmax(epochs) + 0.5)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if title.endswith("loss"):
                for split, color in (("train", _SERIES[0]), ("valid", _SERIES[1])):
                    if split in series:
                        ax.plot(epochs, series[split], color=color, marker=marker)
            else:
                _plot_metric_panel(ax, epochs, series, marker)
        for ax in axes[len(panels) :]:
            ax.set_visible(False)
        if losses:
            from matplotlib.lines import Line2D

            handles = [
                Line2D([], [], color=_SERIES[0]),
                Line2D([], [], color=_SERIES[1]),
            ]
            fig.legend(
                handles, ["train", "validation"], loc="outside upper right", ncols=2
            )
        title = " · ".join(
            filter(
                None,
                [
                    "Training history",
                    name,
                    f"{len(history)} epoch{'s' * (len(history) != 1)}",
                ],
            )
        )
        fig.suptitle(title, x=0.01, ha="left", color=_INK)
    return fig


def _plot_metric_panel(
    ax: "Axes", epochs: np.ndarray, series: dict[str, pd.Series], marker: str | None
) -> None:
    order = list(_METRIC_COLORS)
    labels = sorted(series, key=lambda m: order.index(m) if m in order else len(order))
    for label in labels:
        color = _METRIC_COLORS.get(label, _MUTED)
        ax.plot(epochs, series[label], color=color, label=label, marker=marker)
    headline = labels[0]
    values = series[headline].to_numpy(dtype=float)
    if np.isfinite(values).any():
        best = int(np.nanargmax(values))
        color = _METRIC_COLORS.get(headline, _MUTED)
        ax.plot(epochs[best], values[best], "o", ms=8, color=color, mec=_SURFACE, mew=2)
        left = epochs[best] > np.nanmean(epochs)
        ax.annotate(
            f"best {headline} {values[best]:.3f}\nepoch {epochs[best]:g}", (epochs[best], values[best]),
            xytext=(-8 if left else 8, -10), textcoords="offset points", fontsize=8,
            ha="right" if left else "left", va="top", color=_INK_2,
        )  # fmt: skip
    low = min(0.0, float(np.nanmin([np.nanmin(v) for v in series.values()])))
    high = max(1.0, float(np.nanmax([np.nanmax(v) for v in series.values()])))
    ax.set_ylim(low, high * 1.02)
    ax.legend(loc="best", fontsize=8)


# ----------------------------------------------------------------------- pitch
def _pitch_aspect(pitch: SoccerPitch) -> float:
    """Axes aspect so one metre is as long across as along the pitch."""
    return (_STD_WIDTH / pitch.width) / (_STD_LENGTH / pitch.length)


def _pitch_limits(ax: "Axes", pitch: SoccerPitch) -> None:
    mx, my = 3.5 * pitch.length / _STD_LENGTH, 3.5 * pitch.width / _STD_WIDTH
    ax.set_xlim(-mx, pitch.length + mx)
    ax.set_ylim(
        pitch.width + my, -my
    )  # y grows downwards, like the radar and the camera


def draw_pitch(
    ax: "Axes | None" = None,
    pitch: SoccerPitch | None = None,
    pitch_color: ColorLike = PITCH_COLOR,
    line_color: ColorLike = LINE_COLOR,
    size: float = 9.0,
) -> "Figure":
    """Draw a pitch with all markings (lines, circle, penalty arcs, spots).

    Coordinates are pitch units with ``y`` pointing down, as on the radar, so
    you can plot positions on top::

        fig = tv.viz.draw_pitch()
        fig.axes[0].scatter(df.pitch_x, df.pitch_y)

    Draws into ``ax`` when given, else into a new figure ``size`` inches wide.
    """
    return _pitch_axes(ax, pitch or SoccerPitch(), pitch_color, line_color, size).figure


def _pitch_axes(
    ax: "Axes | None",
    pitch: SoccerPitch,
    pitch_color: ColorLike = PITCH_COLOR,
    line_color: ColorLike = LINE_COLOR,
    size: float = 9.0,
) -> "Axes":
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    with _theme():
        if ax is None:
            ax = _new_figure(size, size * _STD_WIDTH / _STD_LENGTH + 0.6).add_subplot()
        mx, my = 3.5 * pitch.length / _STD_LENGTH, 3.5 * pitch.width / _STD_WIDTH
        ax.add_patch(
            Rectangle(
                (-mx, -my), pitch.length + 2 * mx, pitch.width + 2 * my,
                facecolor=_rgb(pitch_color), edgecolor="none", zorder=0,
            )
        )  # fmt: skip
        lines, spots = pitch_markings(pitch)
        line_rgb = _rgb(line_color)
        ax.add_collection(
            LineCollection(
                lines, colors=[line_rgb], linewidths=1.3, alpha=0.75, zorder=2,
                capstyle="round", joinstyle="round",
            )
        )  # fmt: skip
        ax.scatter(
            spots[:, 0], spots[:, 1], s=9, color=line_rgb, alpha=0.75, zorder=2, lw=0
        )
        ax.set_aspect(_pitch_aspect(pitch))
        _pitch_limits(ax, pitch)
        ax.axis("off")
    return ax


def _positions(result: PipelineResult) -> pd.DataFrame:
    """People (no referees) with finite pitch coordinates, one row per object per frame."""
    table = result.to_dataframe()
    table = table[(table["object"] == "person") & (table["class_name"] != "referee")]
    return table[np.isfinite(table[["pitch_x", "pitch_y"]].to_numpy(float)).all(axis=1)]


def _team_color(team_id: int, team_colors: Sequence[ColorLike]) -> ColorLike:
    return team_colors[team_id] if 0 <= team_id < len(team_colors) else DEFAULT_COLOR


def plot_heatmap(
    result: PipelineResult,
    team: int | None = None,
    ax: "Axes | None" = None,
    cell_m: float = 1.5,
    smoothing: float = 1.5,
    team_colors: Sequence[ColorLike] = TEAM_COLORS,
) -> "Figure":
    """Where players spent time, on a drawn pitch (referees excluded).

    Args:
        team: only this team id (drawn in its team colour); ``None`` for everybody.
        ax: axes to draw into; default a new figure.
        cell_m: histogram cell size in metres.
        smoothing: Gaussian blur in cells.
        team_colors: as in :class:`FrameAnnotator`, to match the video.
    """
    from matplotlib.colors import LinearSegmentedColormap

    pitch = result.pitch
    table = _positions(result)
    if team is not None:
        table = table[table["team_id"] == team]
    xy = table[["pitch_x", "pitch_y"]].to_numpy(float)
    hue = _team_color(team, team_colors) if team is not None else _ALL_PLAYERS_HEAT
    ax = _pitch_axes(ax, pitch)
    with _theme():
        who = "all players" if team is None else f"team {team}"
        _titles(
            ax,
            f"Presence heatmap · {who}",
            f"{len(xy):,} positions in {len(result)} frames",
        )
        if len(xy) == 0:
            _empty_pitch_note(ax, pitch, "No pitch positions")
            return ax.figure
        nx = max(2, round(_STD_LENGTH / cell_m))
        ny = max(2, round(_STD_WIDTH / cell_m))
        hist, _, _ = np.histogram2d(
            xy[:, 1],
            xy[:, 0],
            bins=(ny, nx),
            range=[[0, pitch.width], [0, pitch.length]],
        )
        hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), smoothing)
        hist /= hist.max() or 1.0
        r, g, b = _rgb(hue)
        light = tuple(c + (1 - c) * 0.55 for c in (r, g, b))
        cmap = LinearSegmentedColormap.from_list(
            "presence", [(r, g, b, 0.0), (r, g, b, 0.7), (*light, 0.95)]
        )
        image = ax.imshow(
            hist, extent=(0, pitch.length, pitch.width, 0), cmap=cmap, vmin=0, vmax=1,
            interpolation="bilinear", aspect=_pitch_aspect(pitch), zorder=1,
        )  # fmt: skip
        _pitch_limits(ax, pitch)
        bar = ax.figure.colorbar(
            image, ax=ax, orientation="horizontal", shrink=0.3, aspect=25, pad=0.02,
            ticks=[0, 1],
        )  # fmt: skip
        bar.ax.set_xticklabels(["rare", "frequent"])
        bar.outline.set_visible(False)
        bar.ax.set_facecolor(PITCH_COLOR)
    return ax.figure


def plot_tracks(
    result: PipelineResult,
    track_ids: Sequence[int] | None = None,
    max_tracks: int = 10,
    ax: "Axes | None" = None,
    team_colors: Sequence[ColorLike] = TEAM_COLORS,
) -> "Figure":
    """Pitch trajectories of tracks, coloured by team and tagged with their id.

    Without ``track_ids`` the ``max_tracks`` tracks with the most positions are
    shown. Hollow dots mark where a track starts, filled dots where it ends.
    ``team_colors`` works as in :class:`FrameAnnotator`.
    """
    from matplotlib.lines import Line2D

    pitch = result.pitch
    table = _positions(result)
    table = table[table["track_id"] >= 0]
    counts = (
        table.groupby("track_id").size().sort_values(ascending=False, kind="stable")
    )
    ids = list(counts.index[:max_tracks]) if track_ids is None else list(track_ids)
    tracks = dict(tuple(table.sort_values("frame").groupby("track_id")))
    ax = _pitch_axes(ax, pitch)
    with _theme():
        if track_ids is not None:
            subtitle = f"{len(ids)} selected tracks"
        elif len(ids) < len(counts):
            subtitle = f"{len(ids)} longest of {len(counts)} tracks"
        else:
            subtitle = f"{len(ids)} tracks"
        _titles(ax, "Player tracks", subtitle)
        teams_seen: set[int] = set()
        for track_id in ids:
            track = tracks.get(track_id)
            if track is None:
                continue
            values, votes = np.unique(
                track["team_id"].to_numpy(int), return_counts=True
            )
            team = int(values[votes.argmax()])  # majority vote, NO_TEAM included
            teams_seen.add(team)
            color = _rgb(_team_color(team, team_colors))
            x, y = track["pitch_x"].to_numpy(float), track["pitch_y"].to_numpy(float)
            ax.plot(x, y, color=color, lw=1.8, alpha=0.9, zorder=3)
            start = {"s": 22, "facecolor": PITCH_COLOR, "edgecolor": color, "lw": 1.4}
            end = {"s": 48, "color": color, "edgecolor": PITCH_COLOR, "lw": 1.5}
            ax.scatter(x[:1], y[:1], zorder=4, **start)
            ax.scatter(x[-1:], y[-1:], zorder=5, **end)
            ax.annotate(
                f"#{track_id}", (x[-1], y[-1]), xytext=(5, 4), textcoords="offset points",
                fontsize=8, color="white", zorder=6,
            )  # fmt: skip
        if not ids:
            _empty_pitch_note(ax, pitch, "No tracks with pitch positions")
        if teams_seen:
            labels = {
                t: ("no team" if t == NO_TEAM else f"team {t}")
                for t in sorted(teams_seen)
            }
            handles = [
                Line2D([], [], color=_rgb(_team_color(t, team_colors)), marker="o", ms=6,
                       lw=1.8, mec=PITCH_COLOR) for t in labels
            ]  # fmt: skip
            ax.legend(
                handles, labels.values(), loc="upper center", bbox_to_anchor=(0.5, 0.0),
                ncols=len(handles),
            )  # fmt: skip
    return ax.figure


def _empty_pitch_note(ax: "Axes", pitch: SoccerPitch, text: str) -> None:
    ax.text(
        pitch.length / 2, pitch.width / 2, text, ha="center", va="center", color="white",
        fontsize=11, zorder=6, bbox={"boxstyle": "round,pad=0.5", "fc": PITCH_COLOR, "ec": "none"},
    )  # fmt: skip


# --------------------------------------------------------------------- metrics
_DETECTION_METRICS = {"map50_95": "mAP50-95", "map50": "mAP50", "map75": "mAP75"}


def plot_metrics(
    metrics: DetectionMetrics | KeypointMetrics | Mapping[str, Any],
    metric: str = "map50_95",
) -> "Figure":
    """Per-class AP bars, per-keypoint error bars, or a model comparison.

    Args:
        metrics: a :class:`DetectionMetrics` (per-class AP), a
            :class:`KeypointMetrics` (mean error per keypoint), or a dict
            ``{model name: metrics}`` of one of those types for side-by-side bars
            (e.g. ``{"YOLO": yolo_metrics, "RF-DETR": rfdetr_metrics}``).
        metric: detection metric to plot: ``map50_95``, ``map50`` or ``map75``.
    """
    named = dict(metrics) if isinstance(metrics, Mapping) else {"": metrics}
    if not named:
        raise ValueError("No metrics to plot")
    kinds = {type(m) for m in named.values()}
    if kinds == {DetectionMetrics}:
        if metric not in _DETECTION_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; use one of {list(_DETECTION_METRICS)}"
            )
        return _plot_detection_metrics(named, metric)
    if kinds == {KeypointMetrics}:
        return _plot_keypoint_metrics(named)
    raise TypeError(
        "plot_metrics expects DetectionMetrics or KeypointMetrics (or a dict of one type), "
        f"got {sorted(k.__name__ for k in kinds)}"
    )


def _plot_detection_metrics(
    named: dict[str, DetectionMetrics], metric: str
) -> "Figure":
    classes: list[str] = []
    for m in named.values():
        classes += [str(c) for c in m.per_class.index if str(c) not in classes]
    first = next(iter(named.values()))
    counts = (
        first.per_class["instances"]
        if "instances" in first.per_class.columns
        else pd.Series()
    )
    rows = ["all classes"] + [
        f"{c}  n={int(counts[c]):,}" if c in counts.index else c for c in classes
    ]
    n = len(named)
    bar = min(0.3, 0.7 / n)
    label = _DETECTION_METRICS[metric]
    with _theme():
        fig = _new_figure(7.5, 1.3 + len(rows) * (0.25 + 0.25 * n))
        ax = fig.add_subplot()
        y = np.arange(len(rows))
        for j, (name, m) in enumerate(named.items()):
            per_class = m.per_class.copy()
            per_class.index = per_class.index.map(str)
            values = [getattr(m, metric)] + [
                float(per_class.at[c, metric]) if c in per_class.index else np.nan
                for c in classes
            ]
            offset = (j - (n - 1) / 2) * bar
            ax.barh(
                y + offset,
                values,
                height=bar * 0.85,
                color=_SERIES[j % len(_SERIES)],
                label=name,
            )
            for yy, value in zip(y + offset, values, strict=True):
                if np.isfinite(value):
                    ax.text(
                        value + 0.012,
                        yy,
                        f"{value:.2f}",
                        va="center",
                        fontsize=8,
                        color=_INK_2,
                    )
        ax.axhline(0.5, color=_AXIS, lw=0.8)  # overall score above, classes below
        ax.set_yticks(y, labels=rows)
        ax.tick_params(axis="y", length=0, labelcolor=_INK_2, labelsize=9)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, 1.08)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel(label)
        if n == 1:
            parts = [
                f"{v} {getattr(first, k):.3f}" for k, v in _DETECTION_METRICS.items()
            ]
            subtitle = " · ".join(parts) + f" · {first.num_images} images · n = objects"
            _titles(ax, f"Per-class {label}", subtitle)
        else:
            _titles(ax, f"Per-class {label} by model", "n = ground-truth objects")
            fig.legend(loc="outside upper right", ncols=n)
    return fig


def _plot_keypoint_metrics(named: dict[str, KeypointMetrics]) -> "Figure":
    index = sorted({int(i) for m in named.values() for i in m.per_keypoint.index})
    n = len(named)
    width = min(0.4, 0.8 / n)
    with _theme():
        fig = _new_figure(max(7.0, 0.22 * len(index) * max(1.0, 0.75 * n) + 1.5), 3.8)
        ax = fig.add_subplot()
        x = np.arange(len(index))
        for j, (name, m) in enumerate(named.items()):
            table = m.per_keypoint.copy()
            table.index = table.index.map(int)
            errors = table["mean_error_px"].reindex(index)
            if "count" in table.columns:
                errors = errors.where(table["count"].reindex(index).fillna(0) > 0)
            offset = (j - (n - 1) / 2) * width
            legend = f"{name} · mean {m.mean_error_px:.1f} px" if name else None
            ax.bar(
                x + offset,
                errors,
                width=width * 0.85,
                color=_SERIES[j % len(_SERIES)],
                label=legend,
            )
        ax.set_xticks(x, labels=[str(i) for i in index])
        ax.tick_params(axis="x", length=0, labelsize=7.5)
        ax.set_xlim(-0.7, len(index) - 0.3)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("keypoint")
        ax.set_ylabel("mean error (px)")
        if n == 1:
            m = next(iter(named.values()))
            ax.axhline(m.mean_error_px, color=_INK_2, lw=1, zorder=3)
            ax.annotate(
                f"mean\n{m.mean_error_px:.1f} px", (1, m.mean_error_px),
                xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8,
            )  # fmt: skip
            _titles(
                ax,
                "Keypoint error per landmark",
                f"PCK@{m.pck_threshold:g} {m.pck:.1%} · found in {m.detection_rate:.0%} "
                f"of {m.num_images} images",
            )
        else:
            _titles(ax, "Keypoint error per landmark by model")
            fig.legend(loc="outside upper right", ncols=n)
    return fig


def plot_distance_histogram(
    distances: Sequence[float] | np.ndarray | pd.Series,
    bins: int = 50,
    ax: "Axes | None" = None,
    title: str = "Distance to the nearest detection",
    xlabel: str = "distance (pitch units)",
) -> "Figure":
    """Histogram of distances (e.g. ``compare_with_statsbomb(...)["euclidean_distance"]``) with the median marked.

    Non-finite values (unmatched objects) are ignored.
    """
    values = pd.to_numeric(pd.Series(np.ravel(distances)), errors="coerce").to_numpy(
        float
    )
    values = values[np.isfinite(values)]
    with _theme():
        if ax is None:
            ax = _new_figure(7.5, 3.8).add_subplot()
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("objects")
        if len(values) == 0:
            _titles(ax, title, "no finite distances")
            ax.set_yticks([])
            ax.set_xticks([])
            return ax.figure
        ax.hist(values, bins=bins, color=_SERIES[0], edgecolor=_SURFACE, linewidth=0.8)
        median = float(np.median(values))
        ax.axvline(median, color=_INK_2, lw=1)
        ax.annotate(
            f"median {median:.2f}", (median, 1), xycoords=("data", "axes fraction"),
            xytext=(4, -2), textcoords="offset points", va="top", fontsize=8.5, color=_INK_2,
        )  # fmt: skip
        ax.set_xlim(left=0)
        _titles(
            ax,
            title,
            f"{len(values):,} objects · median {median:.2f} · "
            f"90th percentile {np.percentile(values, 90):.2f}",
        )
    return ax.figure
