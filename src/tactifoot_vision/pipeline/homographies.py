"""A frame's homography as the pipeline makes it (:func:`frame_homography`), and the ``smooth_homographies`` post-process: the temporal filter's RTS pass over a saved result.

:class:`~tactifoot_vision.pitch.HomographyEstimator` is causal: it sees only
past frames. With the whole video at hand, :func:`smooth_homographies` runs
the same constant-velocity Kalman filter forward over the frames' fresh fits
(``FrameResult.homography_fit``, fresh frames) and then the Rauch-Tung-Striebel
pass backward. Each frame then uses the frames on both sides of it, and no
segment crosses a cut (ADR 0002, ADR 0005).
"""

import dataclasses
import logging

import numpy as np

from tactifoot_vision.enums import PostProcessName
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, reproject
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.pitch.fit import LinePoints
from tactifoot_vision.pitch.homography import HomographyEstimator
from tactifoot_vision.pitch.markings import standard_to_pitch
from tactifoot_vision.pitch.temporal import FilterConfig, HomographyFilter

logger = logging.getLogger(__name__)


def frame_homography(
    line_model: Model, estimator: HomographyEstimator, frame: np.ndarray
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
_DEFAULTS = FilterConfig()


@POSTPROCESSES.register(PostProcessName.SMOOTH_HOMOGRAPHIES)
def smooth_homographies(
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
    estimator). A result without fits (a fixed homography, no line model)
    comes back unchanged, with a warning.

    ``pitch_xy`` follows the new matrices (:func:`reproject`), the ball's
    included; the ball is not chosen again, so run
    :func:`~tactifoot_vision.pipeline.track_ball` and the post-processes that
    read pitch positions (``join_tracks``, ``assign_roles``) after this one.
    A warning is logged when one of them already ran.
    """
    given = {"accel_m": accel_m, "meas_m": meas_m, "gate_chi2": gate_chi2, "reset_after": reset_after,
             "max_gap": max_gap, "rate_hz": rate_hz}  # fmt: skip
    given = {k: v for k, v in given.items() if v is not None}
    recorded = result.temporal_filter if result.temporal_filter_recorded else _DEFAULTS
    if recorded is None and not given:
        logger.warning(
            "The result's homographies were made without a temporal filter: "
            "no pass to run, the per-frame homographies are kept"
        )
        return reproject(result, [f.homography for f in result.frames])
    config = dataclasses.replace(recorded or _DEFAULTS, **given)
    rate_hz = config.rate_hz
    times = np.array([f.timestamp for f in result.frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "smooth_homographies needs frames in order of increasing timestamps"
        )
    applied = [p.name for p in result.postprocesses if p.name in _READ_POSITIONS]
    if applied:
        logger.warning(
            "smooth_homographies runs after %s, which read pitch positions; "
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
