"""The ``smooth_homographies`` post-process: smooth the homographies over time without lag.

:class:`~tactifoot_vision.pitch.HomographyEstimator` is causal: it sees only
past frames, so averaging its fits makes the drawn lines trail a pan. With
the whole video at hand, :func:`smooth_homographies` fits each frame's matrix
from the frames on both sides of it, which removes the jitter and not the
motion. The prototype's smoothing (``.scratch/line-refined-homography/prototype``).
"""

import numpy as np
from scipy.signal import savgol_filter

from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, reproject
from tactifoot_vision.pipeline.result import PipelineResult


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
    speeding up) is not lagged; near the ends of a run the polynomial is
    fitted to the run's first or last frames. As a homography is defined up
    to scale, each matrix is first scaled so that it maps the frame centre
    with a homogeneous coordinate of 1 (the prototype divided by ``H[2, 2]``,
    the frame corner, which can sit near the horizon); the frame centre has
    to show the pitch. A pan along the pitch then moves the entries in a
    straight line.

    A frame without a homography keeps none and splits the smoothing: the
    runs either side of it, which may be different camera shots, are
    smoothed apart. A run shorter than the window is smoothed over the whole
    run, and one of ``order + 1`` frames or fewer is kept as it is. The
    frames are taken to be evenly spaced, as the pipeline leaves them, and
    the window is measured with their median time step, so a strided run
    covers the same time with fewer frames.

    ``pitch_xy`` follows the new matrices (:func:`reproject`), the ball's
    included; the ball is not chosen again, so run
    :func:`~tactifoot_vision.pipeline.track_ball` and the post-processes that
    read pitch positions (``join_tracks``, ``assign_roles``) after this one.
    ``FrameResult.line_fit`` still describes the pass's fit.
    """
    if not window > 0:
        raise ValueError(f"window must be a positive number of seconds, got {window}")
    if order < 0:
        raise ValueError(f"order must be 0 or more, got {order}")
    matrices = [f.homography for f in result.frames]
    length = _window_frames(result, window)
    smoothed: list[np.ndarray | None] = list(matrices)
    for start, stop in _runs(matrices):
        run = _normalised(matrices[start:stop], result.frame_size)
        frames = min(length, len(run) - (1 - len(run) % 2))  # odd, within the run
        if frames > order + 1:
            run = savgol_filter(run, frames, order, axis=0, mode="interp")
        smoothed[start:stop] = list(run)
    return reproject(result, smoothed)


def _window_frames(result: PipelineResult, window: float) -> int:
    """The odd number of frames ``window`` seconds cover at the result's time step."""
    steps = np.diff([f.timestamp for f in result.frames])
    if len(steps) == 0:
        return 1
    frames = round(window / float(np.median(steps)))
    return frames + 1 - frames % 2


def _runs(matrices: list[np.ndarray | None]) -> list[tuple[int, int]]:
    """``(start, stop)`` of each run of consecutive frames with a homography."""
    runs, start = [], None
    for i, matrix in enumerate([*matrices, None]):
        if matrix is not None and start is None:
            start = i
        elif matrix is None and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _normalised(
    matrices: list[np.ndarray | None], frame_size: tuple[int, int]
) -> np.ndarray:
    """``(N, 3, 3)`` matrices scaled to map the frame centre with a homogeneous coordinate of 1."""
    stack = np.stack(matrices).astype(float)
    centre = np.array([frame_size[0] / 2, frame_size[1] / 2, 1.0])
    return stack / (stack @ centre)[:, 2, None, None]
