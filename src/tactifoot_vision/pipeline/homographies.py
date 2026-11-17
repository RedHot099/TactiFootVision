"""Smoothing the homographies of a saved result over time without lag: two post-processes, one per estimator.

:class:`~tactifoot_vision.pitch.HomographyEstimator` (keypoints) is causal:
it sees only past frames, so averaging its fits makes the drawn lines trail a
pan. With the whole video at hand, :func:`smooth_homographies` fits each
frame's matrix from the frames on both sides of it, which removes the jitter
and not the motion, as the line-refinement prototype did.

:class:`~tactifoot_vision.pitch.LineHomographyEstimator` (the line network,
opt-in, ADR 0005) is causal too: :func:`frame_homography` makes a frame's
homography as the pipeline makes it. With the whole video at hand,
:func:`smooth_line_homographies` runs the same constant-velocity Kalman
filter forward over the frames' fresh fits (``FrameResult.homography_fit``,
fresh frames) and then the Rauch-Tung-Striebel pass backward. Each frame then
uses the frames on both sides of it, and no segment crosses a cut (ADR 0002,
ADR 0005).
"""

import dataclasses
import logging
import math

import numpy as np
from scipy.signal import savgol_filter

from tactifoot_vision.enums import PostProcessName
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, reproject
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.fit import LinePoints
from tactifoot_vision.pitch.line_homography import LineHomographyEstimator
from tactifoot_vision.pitch.markings import standard_to_pitch
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.pitch.temporal import FilterConfig, HomographyFilter

logger = logging.getLogger(__name__)


def frame_homography(
    line_model: Model, estimator: LineHomographyEstimator, frame: np.ndarray
) -> tuple[LinePoints, np.ndarray | None]:
    """One frame through the line model and the estimator: its line points and its matrix (frame -> pitch).

    The one place both the pipeline and
    :func:`~tactifoot_vision.evaluation.evaluate_lines` make a frame's
    homography, so the two cannot drift apart; ``estimator.fit`` then holds
    the frame's fit.
    """
    points = line_model.predict(frame)
    return points, estimator.update(points, (frame.shape[1], frame.shape[0]))


# post-processes that read pitch positions, so they belong after this one
_READ_POSITIONS = (
    PostProcessName.TRACK_BALL,
    PostProcessName.JOIN_TRACKS,
    PostProcessName.ASSIGN_ROLES,
)
# a scaled matrix this many times the norm of its stretch's median is left out
_NORM_LIMIT = 10.0
_FILTER_DEFAULTS = FilterConfig()


@POSTPROCESSES.register(PostProcessName.SMOOTH_HOMOGRAPHIES)
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


@POSTPROCESSES.register(PostProcessName.SMOOTH_LINE_HOMOGRAPHIES)
def smooth_line_homographies(
    result: PipelineResult,
    accel_m: float | None = None,
    meas_m: float | None = None,
    gate_chi2: float | None = None,
    reset_after: int | None = None,
    max_gap: int | None = None,
    rate_hz: float | None = None,
) -> PipelineResult:
    """Copy of ``result`` with its homographies smoothed over the whole video and ``pitch_xy`` re-projected.

    The settings are those of :class:`~tactifoot_vision.pitch.temporal.FilterConfig`:
    by default the ones the per-frame pass ran with, which the result
    records (``result.temporal_filter``, from the estimator's
    ``temporal_filter``), or the package defaults when the result records
    none (a result put together by hand). An argument given here overrides
    that one setting. A result whose pass ran without a filter
    (``temporal_filter=None``) has no filter to run backward: it comes back
    with its per-frame homographies and a warning, unless settings are given
    here, which then run the pass on the package defaults they override.

    The filter runs forward over the
    frames' fresh fits, segment by segment, each step the time between the
    frames (their indices at ``result.fps``, in steps of ``1 / rate_hz``
    seconds, so a strided run is filtered in physical time), and each frame of a segment gets
    the smoothed state, the frames after it included. A frame outside every
    segment (no fit yet, a gap longer than ``max_gap``, the rejected fits a
    reset hands to the new view) gets the estimator's matrix before the
    filter (``homography_fit.matrix``). A cut, or a view the filter loses, starts a new segment
    (``reset_after`` rejected fits in a row), so nothing is smoothed across
    it. A frame whose estimator held no matrix (none yet, or older than
    ``max_age_s``) ends the segment, as it does in the causal pass.

    The pass keeps, per frame, two filter states and two covariances
    (stored as their 2 x 2 blocks, losslessly): about 0.7 kB, so an hour at
    25 fps holds about 65 MB while it runs.

    It needs what the pass saved: each frame's matrix before the filter in
    ``FrameResult.homography_fit`` (the pipeline's line model and
    :class:`~tactifoot_vision.pitch.LineHomographyEstimator`). A result
    without fits (a fixed homography, a keypoint model, no model) comes back
    with its homographies unchanged, with a warning; a keypoint run is
    smoothed by :func:`smooth_homographies`.

    ``pitch_xy`` follows the new matrices (:func:`reproject`), the ball's
    included; the ball is not chosen again, so run
    :func:`~tactifoot_vision.pipeline.track_ball` and the post-processes that
    read pitch positions (``join_tracks``, ``assign_roles``) after this one.
    A warning is logged when one of them already ran.
    """
    given = {"accel_m": accel_m, "meas_m": meas_m, "gate_chi2": gate_chi2, "reset_after": reset_after,
             "max_gap": max_gap, "rate_hz": rate_hz}  # fmt: skip
    given = {k: v for k, v in given.items() if v is not None}
    recorded = (
        result.temporal_filter if result.temporal_filter_recorded else _FILTER_DEFAULTS
    )
    if recorded is None and not given:
        logger.warning(
            "The result's homographies were made without a temporal filter: "
            "no pass to run, the per-frame homographies are kept"
        )
        return reproject(result, [f.homography for f in result.frames])
    config = dataclasses.replace(recorded or _FILTER_DEFAULTS, **given)
    rate_hz = config.rate_hz
    times = np.array([f.timestamp for f in result.frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "smooth_line_homographies needs frames in order of increasing timestamps"
        )
    applied = [p.name for p in result.postprocesses if p.name in _READ_POSITIONS]
    if applied:
        logger.warning(
            "smooth_line_homographies runs after %s, which read pitch positions; "
            "run it before them",
            ", ".join(applied),
        )
    fits = [f.homography_fit for f in result.frames]
    if not any(fit is not None for fit in fits):
        logger.warning(
            "The result holds no per-frame fits (homography_fit): nothing to smooth"
        )
        return reproject(result, [f.homography for f in result.frames])
    to_pitch = standard_to_pitch(result.pitch)
    to_standard = np.linalg.inv(to_pitch)
    kalman = HomographyFilter(result.frame_size, config)
    # each step in physical time: frame index gaps at the source frame rate
    indices = [f.index for f in result.frames]
    steps = [1.0] + [
        (b - a) * rate_hz / result.fps
        for a, b in zip(indices, indices[1:], strict=False)
    ]
    for fit, dt in zip(fits, steps, strict=True):
        if fit is not None and fit.matrix is None:
            kalman.drop()  # the estimator held nothing (none yet, or past max_age_s): as the causal pass
        fresh = fit is not None and fit.fresh and fit.matrix is not None
        kalman.step(to_standard @ fit.matrix if fresh else None, dt)
    # outside every segment (before the first fit, after a long gap, the rejected frames
    # a reset moved to the new view): the estimator's matrix before the filter
    smoothed = [
        (frame.homography if fit is None else fit.matrix) if m is None else to_pitch @ m
        for frame, fit, m in zip(result.frames, fits, kalman.smoothed(), strict=True)
    ]
    return reproject(result, smoothed)
