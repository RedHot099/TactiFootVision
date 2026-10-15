"""Velocities: how fast and where each tracked person moves on the pitch."""

from collections import defaultdict

import numpy as np

from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import UNTRACKED, PipelineResult

_EPS = 1e-6  # seconds: timestamps are index / fps


def velocities(result: PipelineResult, window: float = 0.5) -> list[np.ndarray]:
    """Each person's pitch velocity in metres per second, smoothed along their track.

    One ``(N, 2)`` array per frame of ``result.frames``, row for row with the
    frame's ``detections``: the velocity along the pitch length (``x``) and
    width (``y``), NaN for a person without one. Run :func:`join_tracks
    <tactifoot_vision.pipeline.join_tracks>` first so a track is one player's.

    A person's velocity is the slope of the straight line fitted (least
    squares) through the pitch positions of their ``tracker_id`` within
    ``window`` seconds centred on the frame: fewer at the track's ends, never
    across a different id. Times are the frames' ``timestamp``, so any frame
    stride gives the same velocities, and pitch units are scaled to a
    105 x 68 m pitch. People without a tracker id or a pitch position, and
    those whose window holds one frame only, have none. A tracker id twice
    in one frame is a ``ValueError``.
    """
    if not np.isfinite(window) or window <= 0:
        raise ValueError(f"window must be a finite number > 0, got {window}")
    per_frame = [np.full((len(f.detections), 2), np.nan) for f in result.frames]
    samples: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for position, frame in enumerate(result.frames):
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
        xy = np.array([result.frames[p].pitch_xy[row] for _, p, row in track]) * scale
        for (_, position, row), velocity in zip(
            track, _slopes(times, xy, window / 2), strict=True
        ):
            per_frame[position][row] = velocity
    return per_frame


def _slopes(times: np.ndarray, xy: np.ndarray, half: float) -> np.ndarray:
    """Slope of the least-squares line through ``xy`` within ``half`` s of each time.

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
    dt = np.where(
        inside, dt - dt.sum(axis=1, keepdims=True) / inside.sum(axis=1)[:, None], 0.0
    )
    spread = (dt**2).sum(axis=1)
    # The centred dt sum to zero, so xy need not be centred.
    slope = (dt[..., None] * xy[index]).sum(axis=1) / np.where(spread > 0, spread, 1)[
        :, None
    ]
    slope[spread < _EPS**2] = np.nan
    return slope
