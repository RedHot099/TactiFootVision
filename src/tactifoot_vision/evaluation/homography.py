"""Pitch position error in metres of a pipeline result's homographies against labelled clips."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tactifoot_vision.data.labelled_clip import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation.metrics import HomographyMetrics
from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import PERSON_ANCHOR, PipelineResult
from tactifoot_vision.pitch.homography import project_points
from tactifoot_vision.pitch.pitch import SoccerPitch

logger = logging.getLogger(__name__)

_GRID = (22, 15)  # points along the length and across: 5 m and 4.9 m apart
_THRESHOLDS = (1, 2, 5)  # metres
_COLUMNS = [
    "median_error_m", "mean_error_m", "p90_error_m",
    *(f"under_{t}m" for t in _THRESHOLDS), "beyond_horizon", "jitter_m", "frames",
]  # fmt: skip
# The point sets each truth scores, the top-level numbers on the last; a
# clip is scored against the first of these truths it carries.
_POINT_SETS = {
    Truth.CALIBRATION: ("landmarks", "grid"),
    Truth.PITCH_POSITIONS: ("people",),
}
# The top-level numbers, in HomographyMetrics.as_dict's order.
_NUMBERS = [
    "median_error_m", "mean_error_m", "p90_error_m", "under_1m", "under_2m",
    "under_5m", "beyond_horizon", "availability", "jitter_m",
]  # fmt: skip
_INTERVAL = (2.5, 97.5)  # percentiles of the bootstrap draws: a 95% interval

# One point set of one frame: the points' keys (which point or which
# identity, None when the people have no identities), their true pitch
# positions and the pixels that show them.
_Points = tuple[np.ndarray | None, np.ndarray, np.ndarray]


@dataclass
class _Scored:
    """One clip's scores: per point set, the error of each scored frame and each step's change."""

    frame_errors: dict[str, np.ndarray]
    changes: dict[str, np.ndarray]
    num_frames: int  # processed frames with truth
    available: int  # of them, those with a homography


def evaluate_homography(
    result: PipelineResult, clip: LabelledClip, margin_m: float = 3.0
) -> HomographyMetrics:
    """Pitch position error in metres of ``result``'s homographies against ``clip``'s truth.

    The truth is the clip's calibration where it has one, otherwise the pitch
    positions of its labelled people; ``truth`` in the metrics names it. With
    calibration, every processed frame the clip calibrates is scored: the
    pitch's landmarks and a fixed grid of pitch points that the true
    homography shows inside the frame are mapped to their pixels, and the
    run's homography maps those pixels back to the pitch. With pitch
    positions, every processed frame with a person whose pitch position is
    known and on the pitch (within ``margin_m`` of its lines, so the people
    stand where the calibration's grid does) is scored: the run's homography
    maps each such person's box foot (bottom centre) to the pitch, to be
    compared with that position. The ball is left out: in the air, its box
    foot is not where it is on the pitch.

    A frame's error is the mean distance between the two, in metres: pitch
    units are scaled to a 105 x 68 m pitch, so a ``SoccerPitch(120, 80)`` run
    is scored like any other. A frame without a homography counts against
    ``availability``, not as an error. A frame whose homography puts a point
    it shows beyond its horizon has an infinite error: ``beyond_horizon``
    counts it, the percentiles rank it last and the mean leaves it out.
    ``jitter_m`` is the mean change of each point's error vector between
    consecutive processed frames, so a constant offset has none; on pitch
    positions a point is one identity, its ``tracker_id``, which must be
    unique in a frame: a repeated or negative (untracked) id is not matched,
    and people without identities have no jitter.

    Args:
        result: a run on the clip's video, with its frame indices.
        clip: the truth; it must carry calibration (frame -> pitch
            homographies) or pitch positions (``detections.data["pitch_xy"]``,
            NaN where unknown), in the units of ``result.pitch``.
        margin_m: on pitch positions, how far beyond the pitch's lines, in
            metres, a person is still scored.
    """
    truth = _truth(clip)
    metrics = _metrics([_score(result, clip, truth, margin_m)], truth)
    _log(clip.name, metrics)
    return metrics


def evaluate_homography_clips(
    pairs: Iterable[tuple[PipelineResult, LabelledClip]],
    draws: int = 2000,
    seed: int = 0,
    margin_m: float = 3.0,
) -> HomographyMetrics:
    """The metrics of :func:`evaluate_homography` over several ``(result, clip)`` pairs, with intervals.

    The scored frames of every clip are pooled, so the numbers are those of
    one clip holding them all, jitter not measured from one clip to the next.
    ``intervals`` holds a 95% interval of each number from a bootstrap that
    resamples whole clips with replacement ``draws`` times, since frames of
    one sequence are not independent; the same ``seed`` gives the same
    intervals. With a single clip ``intervals`` is ``None``: resampling one
    clip always gives that clip, so its interval would have no width. Every
    clip must be scored against the same kind of truth; ``margin_m`` is
    :func:`evaluate_homography`'s.
    """
    pairs = list(pairs)
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws}")
    if not pairs:
        raise ValueError(
            "evaluate_homography_clips needs at least one (result, clip) pair"
        )
    truths = [_truth(clip) for _, clip in pairs]
    truth = truths[0]
    if any(t is not truth for t in truths):
        kinds = ", ".join(
            f"{c.name}: {t}" for (_, c), t in zip(pairs, truths, strict=True)
        )
        raise ValueError(
            f"The clips are scored against different truths ({kinds}); "
            "aggregate clips with the same truth"
        )
    scored = [_score(result, clip, truth, margin_m) for result, clip in pairs]
    intervals = _intervals(scored, truth, draws, seed) if len(scored) > 1 else None
    metrics = _metrics(scored, truth, intervals)
    _log(f"{len(pairs)} clips", metrics)
    return metrics


def _truth(clip: LabelledClip) -> Truth:
    """The truth ``clip`` is scored against: its calibration, else its pitch positions."""
    for truth in _POINT_SETS:
        if truth in clip.truth:
            return truth
    raise ValueError(
        f"Labelled clip {clip.name} has no calibration or pitch positions; "
        "evaluate_homography needs a clip with one of them"
    )


def _score(
    result: PipelineResult, clip: LabelledClip, truth: Truth, margin_m: float
) -> _Scored:
    """``result``'s per-frame errors against ``clip``'s ``truth``."""
    frame_size = (clip.width, clip.height)
    if tuple(result.frame_size) != frame_size:
        raise ValueError(
            f"The result's frames are {result.frame_size[0]} x "
            f"{result.frame_size[1]} px, labelled clip {clip.name}'s "
            f"{clip.width} x {clip.height}; run the pipeline on the clip's video"
        )
    pitch = result.pitch
    centre = (frame_size[0] / 2, frame_size[1] / 2)
    names = _POINT_SETS[truth]
    pitch_points = {"landmarks": pitch.vertices.astype(float), "grid": _grid(pitch)}
    # Per point set, the keys and error vectors of each processed frame,
    # consecutive like the frames (None without a homography or without truth).
    errors: dict[str, list[tuple[np.ndarray | None, np.ndarray] | None]] = {
        name: [] for name in names
    }
    num_frames = available = 0
    for frame in result.frames:
        try:
            labelled = clip.frame(frame.index)
        except IndexError:
            raise ValueError(
                f"Processed frame {frame.index} is not in labelled clip {clip.name}; "
                "run the pipeline on the clip's video, within its frames"
            ) from None
        if truth is Truth.CALIBRATION:
            point_sets = _calibration_points(labelled, pitch_points, frame_size, centre)
        else:
            point_sets = _position_points(labelled, pitch, margin_m)
        scored = point_sets is not None and frame.homography is not None
        num_frames += point_sets is not None
        available += scored
        for name in names:
            if scored:
                keys, points, pixels = point_sets[name]
                vectors = _errors(points, pixels, frame.homography, centre, pitch)
                errors[name].append((keys, vectors))
            else:
                errors[name].append(None)
    if num_frames == 0:
        kind = "a calibration" if truth is Truth.CALIBRATION else "pitch positions"
        raise ValueError(
            f"No processed frame of {clip.name} has {kind}; "
            "run the pipeline on its labelled frames"
        )
    frame_errors, changes = {}, {}
    for name in names:
        # A frame is scored when it has a homography and shows at least one point.
        frame_errors[name] = np.array(
            [
                np.linalg.norm(e, axis=1).mean()
                for _, e in filter(None, errors[name])
                if len(e)
            ]
        )
        steps = zip(errors[name], errors[name][1:], strict=False)
        steps = [_change(p, c) for p, c in steps if p is not None and c is not None]
        changes[name] = np.array([c for c in steps if not np.isnan(c)])
    return _Scored(frame_errors, changes, num_frames, available)


def _calibration_points(
    frame: LabelledFrame,
    pitch_points: dict[str, np.ndarray],
    frame_size: tuple[int, int],
    centre: tuple[float, float],
) -> dict[str, _Points] | None:
    """The ``pitch_points`` the true homography shows in the frame, keyed by index."""
    if frame.homography is None:
        return None
    width, height = frame_size
    (front,), _ = project_points(np.array([centre]), frame.homography, centre)
    inverse = np.linalg.inv(frame.homography)
    point_sets = {}
    for name, points in pitch_points.items():
        pixels, in_front = project_points(points, inverse, tuple(front))
        shown = (
            in_front
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= height)
        )
        point_sets[name] = (np.flatnonzero(shown), points[shown], pixels[shown])
    return point_sets


def _position_points(
    frame: LabelledFrame, pitch: SoccerPitch, margin_m: float
) -> dict[str, _Points] | None:
    """The people with a known pitch position within ``margin_m`` of the pitch and their box feet, keyed by identity."""
    people = frame.detections
    positions = people.data.get("pitch_xy")
    if positions is None:
        return None
    positions = np.asarray(positions, float).reshape(-1, 2)
    margin = margin_m / metres_per_unit(pitch)
    known = (
        (positions >= -margin) & (positions <= [pitch.length, pitch.width] + margin)
    ).all(axis=1)  # NaN, unknown, compares false
    if not known.any():
        return None
    feet = people.get_anchors_coordinates(PERSON_ANCHOR)[known]
    keys = None if people.tracker_id is None else people.tracker_id[known]
    return {"people": (keys, positions[known], feet)}


def _grid(pitch: SoccerPitch) -> np.ndarray:
    """The fixed grid of pitch points, edges included, in pitch units."""
    x, y = np.meshgrid(
        np.linspace(0, pitch.length, _GRID[0]), np.linspace(0, pitch.width, _GRID[1])
    )
    return np.column_stack([x.ravel(), y.ravel()])


def _errors(
    points: np.ndarray,
    pixels: np.ndarray,
    predicted: np.ndarray,
    centre: tuple[float, float],
    pitch: SoccerPitch,
) -> np.ndarray:
    """``(N, 2)`` predicted minus true pitch position of each point, in metres.

    Infinite for a point whose pixel the predicted homography puts beyond its horizon.
    """
    projected, valid = project_points(pixels, predicted, centre)
    return np.where(
        valid[:, None], (projected - points) * metres_per_unit(pitch), np.inf
    )


def _change(
    previous: tuple[np.ndarray | None, np.ndarray],
    current: tuple[np.ndarray | None, np.ndarray],
) -> float:
    """Mean distance between two frames' error vectors at the points with a finite error in both."""
    (previous_keys, previous), (current_keys, current) = previous, current
    if previous_keys is None or current_keys is None:
        return np.nan
    previous_keys, previous = _matchable(previous_keys, previous)
    current_keys, current = _matchable(current_keys, current)
    _, before, after = np.intersect1d(previous_keys, current_keys, return_indices=True)
    if len(before) == 0:
        return np.nan
    return float(np.linalg.norm(current[after] - previous[before], axis=1).mean())


def _matchable(keys: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The keys and errors of the points a frame can be matched on.

    Those with a finite error and a key that names one point: not negative
    (an untracked person) and not repeated in the frame.
    """
    values, counts = np.unique(keys, return_counts=True)
    keep = (
        np.isfinite(errors).all(axis=1)
        & (keys >= 0)
        & ~np.isin(keys, values[counts > 1])
    )
    return keys[keep], errors[keep]


def _metrics(
    scored: list[_Scored], truth: Truth, intervals: pd.DataFrame | None = None
) -> HomographyMetrics:
    """The metrics of these clips' scored frames pooled."""
    names = _POINT_SETS[truth]
    per_points = pd.DataFrame(
        [_summary(scored, name) for name in names],
        index=pd.Index(list(names), name="points"),
        columns=_COLUMNS,
    ).astype({"frames": int})
    headline = per_points.loc[names[-1]]
    return HomographyMetrics(
        **{
            column: float(headline[column]) for column in _COLUMNS if column != "frames"
        },
        availability=_availability(scored),
        per_points=per_points,
        truth=truth,
        num_frames=sum(s.num_frames for s in scored),
        intervals=intervals,
    )


def _availability(scored: list[_Scored]) -> float:
    """The share of these clips' frames with truth that have a homography."""
    return sum(s.available for s in scored) / sum(s.num_frames for s in scored)


def _summary(scored: list[_Scored], name: str) -> list[float]:
    """The ``_COLUMNS`` of one point set over these clips' scored frames."""
    frame_errors = np.concatenate([s.frame_errors[name] for s in scored])
    if len(frame_errors) == 0:
        return [np.nan] * (len(_COLUMNS) - 1) + [0]
    changes = np.concatenate([s.changes[name] for s in scored])
    finite = frame_errors[np.isfinite(frame_errors)]
    return [
        _percentile(frame_errors, 50),
        float(finite.mean()) if len(finite) else np.nan,
        _percentile(frame_errors, 90),
        *(float((frame_errors < t).mean()) for t in _THRESHOLDS),
        float(np.isinf(frame_errors).mean()),
        float(changes.mean()) if len(changes) else np.nan,
        len(frame_errors),
    ]


def _intervals(
    scored: list[_Scored], truth: Truth, draws: int, seed: int
) -> pd.DataFrame:
    """The ``low`` and ``high`` ends of each top-level number over clip bootstrap draws."""
    rng = np.random.default_rng(seed)
    name = _POINT_SETS[truth][-1]
    samples = []
    for _ in range(draws):
        sample = [scored[i] for i in rng.integers(len(scored), size=len(scored))]
        values = dict(zip(_COLUMNS, _summary(sample, name), strict=True))
        values["availability"] = _availability(sample)
        samples.append([values[number] for number in _NUMBERS])
    # A draw where a number is undefined (NaN: no jitter step, say) is left
    # out of that number's interval.
    samples = np.array(samples)
    return pd.DataFrame(
        [
            [_percentile(column[~np.isnan(column)], q) for q in _INTERVAL]
            for column in samples.T
        ],
        index=pd.Index(_NUMBERS),
        columns=["low", "high"],
    )


def _percentile(errors: np.ndarray, q: float) -> float:
    """``np.percentile`` (linear), infinite where it interpolates towards an infinite error.

    numpy gives NaN next to an infinite value, even where the percentile
    lands exactly on a finite one.
    """
    if len(errors) == 0:
        return np.nan
    errors = np.sort(errors)
    position = q / 100 * (len(errors) - 1)
    low, high = errors[int(np.floor(position))], errors[int(np.ceil(position))]
    if low == high or position.is_integer():
        return float(low)
    return float(low + (high - low) * (position - np.floor(position)))


def _log(label: str, metrics: HomographyMetrics) -> None:
    logger.info(
        "Homography on %s (%d frames, %s truth): median error %.2f m, "
        "p90 %.2f m, available %.0f%%",
        label,
        metrics.num_frames,
        metrics.truth,
        metrics.median_error_m,
        metrics.p90_error_m,
        100 * metrics.availability,
    )
