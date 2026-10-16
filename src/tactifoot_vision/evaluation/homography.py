"""Pitch position error in metres of a pipeline result's homographies against a labelled clip."""

import logging

import numpy as np
import pandas as pd

from tactifoot_vision.data.labelled_clip import LabelledClip, Truth
from tactifoot_vision.evaluation.metrics import HomographyMetrics
from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.homography import project_points
from tactifoot_vision.pitch.pitch import SoccerPitch

logger = logging.getLogger(__name__)

_GRID = (22, 15)  # points along the length and across: 5 m and 4.9 m apart
_THRESHOLDS = (1, 2, 5)  # metres
_COLUMNS = [
    "median_error_m", "mean_error_m", "p90_error_m",
    *(f"under_{t}m" for t in _THRESHOLDS), "beyond_horizon", "jitter_m", "frames",
]  # fmt: skip


def evaluate_homography(
    result: PipelineResult, clip: LabelledClip
) -> HomographyMetrics:
    """Pitch position error in metres of ``result``'s homographies against ``clip``'s calibration.

    Every processed frame the clip has a calibration for is scored: the
    pitch's landmarks and a fixed grid of pitch points that the true
    homography shows inside the frame are mapped to their pixels, and the
    run's homography maps those pixels back to the pitch. A frame's error is
    the mean distance between the two, in metres: pitch units are scaled to a
    105 x 68 m pitch, so a ``SoccerPitch(120, 80)`` run is scored like any
    other. A frame without a homography counts against ``availability``, not
    as an error. A frame whose homography puts a point it shows beyond its
    horizon has an infinite error: ``beyond_horizon`` counts it, the
    percentiles rank it last and the mean leaves it out. ``jitter_m`` is the
    mean change of each point's error vector between consecutive processed
    frames, so a constant offset has none.

    Args:
        result: a run on the clip's video, with its frame indices.
        clip: the truth; it must carry calibration, frame -> pitch
            homographies in the units of ``result.pitch``.
    """
    if Truth.CALIBRATION not in clip.truth:
        raise ValueError(
            f"Labelled clip {clip.name} has no calibration; "
            "evaluate_homography needs a clip with calibration"
        )
    frame_size = (clip.width, clip.height)
    if tuple(result.frame_size) != frame_size:
        raise ValueError(
            f"The result's frames are {result.frame_size[0]} x "
            f"{result.frame_size[1]} px, labelled clip {clip.name}'s "
            f"{clip.width} x {clip.height}; run the pipeline on the clip's video"
        )
    pitch = result.pitch
    point_sets = {"landmarks": pitch.vertices.astype(float), "grid": _grid(pitch)}
    # Per point set, the error vectors of each processed frame, consecutive
    # like the frames (None without a homography or without truth).
    errors: dict[str, list[np.ndarray | None]] = {name: [] for name in point_sets}
    num_frames = available = 0
    for frame in result.frames:
        try:
            truth = clip.frame(frame.index).homography
        except IndexError:
            raise ValueError(
                f"Processed frame {frame.index} is not in labelled clip {clip.name}; "
                "run the pipeline on the clip's video, within its frames"
            ) from None
        scored = truth is not None and frame.homography is not None
        num_frames += truth is not None
        available += scored
        for name, points in point_sets.items():
            errors[name].append(
                _errors(points, truth, frame.homography, frame_size, pitch)
                if scored
                else None
            )
    if num_frames == 0:
        raise ValueError(
            f"No processed frame of {clip.name} has a calibration; "
            "run the pipeline on its calibrated frames"
        )

    per_points = pd.DataFrame(
        [_summary(errors[name]) for name in point_sets],
        index=pd.Index(list(point_sets), name="points"),
        columns=_COLUMNS,
    ).astype({"frames": int})
    grid = per_points.loc["grid"]
    metrics = HomographyMetrics(
        **{column: float(grid[column]) for column in _COLUMNS if column != "frames"},
        availability=available / num_frames,
        per_points=per_points,
        truth=Truth.CALIBRATION,
        num_frames=num_frames,
    )
    logger.info(
        "Homography on %s (%d frames, %s truth): median error %.2f m, "
        "p90 %.2f m, available %.0f%%",
        clip.name,
        num_frames,
        metrics.truth,
        metrics.median_error_m,
        metrics.p90_error_m,
        100 * metrics.availability,
    )
    return metrics


def _grid(pitch: SoccerPitch) -> np.ndarray:
    """The fixed grid of pitch points, edges included, in pitch units."""
    x, y = np.meshgrid(
        np.linspace(0, pitch.length, _GRID[0]), np.linspace(0, pitch.width, _GRID[1])
    )
    return np.column_stack([x.ravel(), y.ravel()])


def _errors(
    points: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    frame_size: tuple[int, int],
    pitch: SoccerPitch,
) -> np.ndarray:
    """``(N, 2)`` predicted minus true pitch position of each point, in metres.

    NaN for a point the frame does not show, infinite for one whose pixel the
    predicted homography puts beyond its horizon.
    """
    width, height = frame_size
    centre = (width / 2, height / 2)
    (front,), _ = project_points(np.array([centre]), truth, centre)
    pixels, in_front = project_points(points, np.linalg.inv(truth), tuple(front))
    shown = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height)
    )
    projected, valid = project_points(pixels[shown], predicted, centre)
    errors = np.full(points.shape, np.nan)
    errors[shown] = np.where(
        valid[:, None], (projected - points[shown]) * metres_per_unit(pitch), np.inf
    )
    return errors


def _summary(errors: list[np.ndarray | None]) -> list[float]:
    """The ``_COLUMNS`` of one point set from its per-frame error vectors."""
    # A frame is scored when it has a homography and shows at least one point.
    scored = [e for e in errors if e is not None and not np.isnan(e).all()]
    frame_errors = np.array([np.nanmean(np.linalg.norm(e, axis=1)) for e in scored])
    if len(frame_errors) == 0:
        return [np.nan] * (len(_COLUMNS) - 1) + [0]
    changes = [
        _change(previous, current)
        for previous, current in zip(errors, errors[1:], strict=False)
        if previous is not None and current is not None
    ]
    changes = [c for c in changes if not np.isnan(c)]
    finite = frame_errors[np.isfinite(frame_errors)]
    return [
        _percentile(frame_errors, 50),
        float(finite.mean()) if len(finite) else np.nan,
        _percentile(frame_errors, 90),
        *(float((frame_errors < t).mean()) for t in _THRESHOLDS),
        float(np.isinf(frame_errors).mean()),
        float(np.mean(changes)) if changes else np.nan,
        len(frame_errors),
    ]


def _change(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean distance between two frames' error vectors at the points with a finite error in both."""
    both = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    if not both.any():
        return np.nan
    return float(np.linalg.norm(current[both] - previous[both], axis=1).mean())


def _percentile(errors: np.ndarray, q: float) -> float:
    """``np.percentile`` (linear), infinite where it interpolates towards an infinite error.

    numpy gives NaN next to an infinite value, even where the percentile
    lands exactly on a finite one.
    """
    errors = np.sort(errors)
    position = q / 100 * (len(errors) - 1)
    low, high = errors[int(np.floor(position))], errors[int(np.ceil(position))]
    if low == high or position.is_integer():
        return float(low)
    return float(low + (high - low) * (position - np.floor(position)))
