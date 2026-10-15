"""The ``smooth_homographies`` post-process: smooth the homographies over time without lag.

:class:`~tactifoot_vision.pitch.HomographyEstimator` is causal: it sees only
past frames, so averaging its fits makes the drawn lines trail a pan. With
the whole video at hand, :func:`smooth_homographies` fits each frame's matrix
from the frames on both sides of it, which removes the jitter and not the
motion, as the line-refinement prototype did.
"""

import logging
import math

import numpy as np
from scipy.signal import savgol_filter

from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, reproject
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.pitch import SoccerPitch

logger = logging.getLogger(__name__)

# post-processes that read pitch positions, so they belong after this one
_READ_POSITIONS = ("track_ball", "join_tracks", "assign_roles")
# a scaled matrix this many times the norm of its stretch's median is left out
_NORM_LIMIT = 10.0


@POSTPROCESSES.register("smooth_homographies")
def smooth_homographies(
    result: PipelineResult, window: float = 0.36, order: int = 2
) -> PipelineResult:
    """Copy of ``result`` with its homographies smoothed over time and ``pitch_xy`` re-projected.

    Each frame's matrix becomes a centred Savitzky-Golay filter of the
    matrices of the frames around it: a polynomial of degree ``order`` is
    fitted, entry by entry, to the matrices within ``window`` seconds (9
    frames at 25 fps by default, the prototype's) and read at the frame. The
    window is centred, so a pan the polynomial follows (a steady one, or one
    speeding up) is not lagged; near the ends of a stretch the polynomial
    is fitted to its first or last frames. As a homography is defined up
    to scale, each matrix is first scaled so that it maps the frame centre
    with a homogeneous coordinate of 1 (the prototype divided by ``H[2, 2]``,
    the frame corner, which can sit near the horizon); the frame centre has
    to show the pitch. A pan along the pitch then moves the entries in a
    straight line.

    A frame without a homography keeps none and splits the smoothing: the
    stretches either side of it are smoothed apart. A frame whose matrix
    cannot be trusted keeps its matrix and splits them too: one that puts
    the frame centre more than a pitch length beyond the pitch (the centre
    near the horizon), or whose scaled matrix has over 10 times the norm of
    its stretch's median. A camera cut is split only where frames carry no
    homography: :class:`~tactifoot_vision.pitch.HomographyEstimator` reuses
    its last matrix for up to ``max_age`` frames, so a cut inside that span
    is smoothed across. A stretch shorter than the window is smoothed over
    itself, and one of ``order + 1`` frames or fewer is kept as it is. The
    frames are taken to be evenly spaced, as the pipeline leaves them, and
    the window is measured with their median time step, so a strided run
    covers the same time with fewer frames.

    Use it with ``HomographyEstimator(smoothing_window=1)``: the estimator's
    causal average lags a pan, and this filter cannot remove that lag.

    ``pitch_xy`` follows the new matrices (:func:`reproject`), the ball's
    included; the ball is not chosen again, so run
    :func:`~tactifoot_vision.pipeline.track_ball` and the post-processes that
    read pitch positions (``join_tracks``, ``assign_roles``) after this one.
    A warning is logged when one of them already ran. ``FrameResult.line_fit``
    still describes the pass's fit.
    """
    if not (math.isfinite(window) and window > 0):
        raise ValueError(f"window must be a positive number of seconds, got {window}")
    if order < 0:
        raise ValueError(f"order must be 0 or more, got {order}")
    times = np.array([f.timestamp for f in result.frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "smooth_homographies needs frames in order of increasing timestamps"
        )
    window_frames = _window_frames(times, window)
    if len(times) > 1 and order >= window_frames:
        raise ValueError(
            f"order must be below the {window_frames} frames {window} s cover, "
            f"got {order}"
        )
    applied = [p.name for p in result.postprocesses if p.name in _READ_POSITIONS]
    if applied:
        logger.warning(
            "smooth_homographies runs after %s, which read pitch positions; "
            "run it before them",
            ", ".join(applied),
        )
    centre = np.array([result.frame_size[0] / 2, result.frame_size[1] / 2, 1.0])
    scaled = _without_outliers(
        [_scaled(f.homography, centre, result.pitch) for f in result.frames]
    )
    smoothed = [f.homography for f in result.frames]
    for start, stop in _stretches(scaled):
        stretch = np.stack(scaled[start:stop])
        # odd, and no longer than the stretch
        span = min(window_frames, len(stretch) - (1 - len(stretch) % 2))
        if span > order + 1:
            stretch = savgol_filter(stretch, span, order, axis=0, mode="interp")
        smoothed[start:stop] = list(stretch)
    return reproject(result, smoothed)


def _window_frames(times: np.ndarray, window: float) -> int:
    """The odd number of frames ``window`` seconds cover at the median time step."""
    if len(times) < 2:
        return 1
    frames = round(window / float(np.median(np.diff(times))))
    return frames + 1 - frames % 2


def _scaled(
    matrix: np.ndarray | None, centre: np.ndarray, pitch: SoccerPitch
) -> np.ndarray | None:
    """``matrix`` scaled to map ``centre`` with a homogeneous coordinate of 1.

    ``None`` without a matrix or when the centre lands more than a pitch
    length beyond the pitch.
    """
    if matrix is None:
        return None
    w = (matrix @ centre)[2]
    if w == 0:
        return None
    scaled = np.asarray(matrix, dtype=float) / w
    x, y = (scaled @ centre)[:2]
    reach = pitch.length
    inside = -reach <= x <= pitch.length + reach and -reach <= y <= pitch.width + reach
    return scaled if inside else None


def _without_outliers(scaled: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """``scaled`` with ``None`` for each matrix far above its stretch's median norm."""
    scaled = list(scaled)
    for start, stop in _stretches(scaled):
        norms = np.array([np.linalg.norm(m) for m in scaled[start:stop]])
        for i in np.flatnonzero(norms > _NORM_LIMIT * np.median(norms)):
            scaled[start + i] = None
    return scaled


def _stretches(matrices: list[np.ndarray | None]) -> list[tuple[int, int]]:
    """``(start, stop)`` of each stretch of consecutive frames with a matrix."""
    stretches, start = [], None
    for i, matrix in enumerate([*matrices, None]):
        if matrix is not None and start is None:
            start = i
        elif matrix is None and start is not None:
            stretches.append((start, i))
            start = None
    return stretches
