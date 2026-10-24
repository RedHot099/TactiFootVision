"""Velocities: how fast and where each tracked person moves on the pitch."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import UNTRACKED, FrameResult, PipelineResult

_EPS = 1e-6  # seconds: timestamps are index / fps
_MIN_SAMPLES = 3  # positions a velocity is fitted through
_MAX_SPEED = 12.0  # m/s: a sprint's top speed


def velocities(
    result: PipelineResult, window: float = 0.5, max_speed: float = _MAX_SPEED
) -> list[np.ndarray]:
    """Each person's pitch velocity in m/s, smoothed along their track.

    One ``(N, 2)`` array per frame of ``result.frames``, row for row with the
    frame's ``detections``: the velocity along the pitch length (``x``) and
    width (``y``), NaN for a person without one. Run :func:`join_tracks
    <tactifoot_vision.pipeline.join_tracks>` first so a track is one player's.

    A person's velocity is the slope of the straight line fitted (least
    squares) through the pitch positions of their ``tracker_id`` within
    ``window`` seconds centred on the frame, never across a different id.
    Times are the frames' ``timestamp`` and pitch units are scaled to a
    105 x 68 m pitch, so any frame stride gives the same velocities; a
    window shorter than two of the result's median time steps is widened to
    them, so a coarse stride still has the frames either side. A velocity
    needs three positions in its window (the track's ends at a coarse stride
    have none) and is dropped above ``max_speed`` m/s, the jump of a bad
    homography rather than a run (``np.inf`` keeps every one; 12 m/s is a
    sprint's top speed, not measured on this package's data). People without
    a tracker id or a pitch position have none. A tracker id twice in one
    frame is a ``ValueError``.
    """
    return _velocities(
        result, result.frames, _half(result, window, max_speed), max_speed
    )


def _velocities_near(
    result: PipelineResult,
    positions: Sequence[int],
    window: float,
    max_speed: float = _MAX_SPEED,
) -> dict[int, np.ndarray]:
    """:func:`velocities` of the frames at ``positions``, by position.

    Only the frames within the fitting window of them are read; ``result``
    is in video order, as :meth:`PipelineResult.window` needs.
    """
    half = _half(result, window, max_speed)
    if not positions:
        return {}
    times = [frame.timestamp for frame in result.frames]
    first = bisect_left(times, times[min(positions)] - half - _EPS)
    last = bisect_right(times, times[max(positions)] + half + _EPS)
    near = _velocities(result, result.frames[first:last], half, max_speed)
    return {position: near[position - first] for position in positions}


def _half(result: PipelineResult, window: float, max_speed: float) -> float:
    """Half the fitting window in seconds, at least the median time step."""
    if not np.isfinite(window) or window <= 0:
        raise ValueError(f"window must be a finite number > 0, got {window}")
    if not max_speed > 0:
        raise ValueError(f"max_speed must be > 0, got {max_speed}")
    times = [frame.timestamp for frame in result.frames]
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return max(window / 2, step)


def _velocities(
    result: PipelineResult, frames: list[FrameResult], half: float, max_speed: float
) -> list[np.ndarray]:
    """:func:`velocities` of ``frames``, consecutive frames of ``result``."""
    per_frame = [np.full((len(f.detections), 2), np.nan) for f in frames]
    samples: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for position, frame in enumerate(frames):
        ids = frame.detections.tracker_id
        if ids is None:
            continue
        tracked = ids[ids != UNTRACKED]
        if len(np.unique(tracked)) < len(tracked):
            raise ValueError(
                f"Frame {frame.index} has a tracker id twice; velocities follow "
                "one person per id"
            )
        positioned = np.isfinite(frame.pitch_xy).all(axis=1)
        for row in np.flatnonzero(positioned & (ids != UNTRACKED)):
            samples[int(ids[row])].append((frame.timestamp, position, int(row)))
    scale = metres_per_unit(result.pitch)
    for track in samples.values():
        track.sort()
        times = np.array([time for time, _, _ in track])
        xy = np.array([frames[p].pitch_xy[row] for _, p, row in track]) * scale
        slopes = _slopes(times, xy, half)
        slopes[np.hypot(*slopes.T) > max_speed] = np.nan
        for (_, position, row), velocity in zip(track, slopes, strict=True):
            per_frame[position][row] = velocity
    return per_frame


def _slopes(times: np.ndarray, xy: np.ndarray, half: float) -> np.ndarray:
    """Least-squares slope of ``xy`` within ``half`` s of each time (NaN: < 3).

    ``times`` are sorted. The windows are gathered into a padded
    ``(samples, width)`` array and centred on their own mean time, which
    keeps the sums exact however long the video.
    """
    start = np.searchsorted(times, times - half - _EPS, "left")
    stop = np.searchsorted(times, times + half + _EPS, "right")
    index = start[:, None] + np.arange((stop - start).max())
    inside = index < stop[:, None]
    index = np.minimum(index, len(times) - 1)
    dt = np.where(inside, times[index] - times[:, None], 0.0)
    count = inside.sum(axis=1)
    dt = np.where(inside, dt - dt.sum(axis=1, keepdims=True) / count[:, None], 0.0)
    spread = (dt**2).sum(axis=1)
    # The centred dt sum to zero, so xy need not be centred.
    slope = (dt[..., None] * xy[index]).sum(axis=1)
    slope /= np.where(spread > 0, spread, 1)[:, None]
    slope[(count < _MIN_SAMPLES) | (spread < _EPS**2)] = np.nan
    return slope
