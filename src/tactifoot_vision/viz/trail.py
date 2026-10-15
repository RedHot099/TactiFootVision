"""Fading trails from earlier frames: the ball trail and the radar's motion tails."""

from collections.abc import Sequence

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.pipeline.result import FrameResult
from tactifoot_vision.viz.subpixel import SUBPIXEL, to_fixed

# Each segment between frames is cut into pieces of at most _PIECE_PX pixels,
# so the fade looks continuous; at most _PIECES, so long segments stay cheap.
_PIECES = 8
_PIECE_PX = 4.0


def trail_frames(
    history: Sequence[FrameResult], frame_result: FrameResult, seconds: float
) -> tuple[list[FrameResult], np.ndarray]:
    """The frames of ``history`` within ``seconds`` of ``frame_result``, then
    ``frame_result`` itself, and the age of each in seconds."""
    now = frame_result.timestamp
    # Within a microsecond: timestamps are index / fps, as in the window.
    recent = [f for f in history if 0 < now - f.timestamp <= seconds + 1e-6]
    frames = [*recent, frame_result]
    return frames, np.array([now - f.timestamp for f in frames])


def draw_trail(
    image: np.ndarray,
    xy: np.ndarray,
    valid: np.ndarray,
    ages: np.ndarray,
    seconds: float,
    color: sv.Color,
    thickness: int,
) -> None:
    """Blend a line through the pixels ``xy`` (oldest first) into ``image``.

    Opacity falls linearly with age along the line, from full at age 0 to
    none at ``seconds``. A point that is not ``valid`` breaks the line.
    """
    segments = np.flatnonzero(valid[:-1] & valid[1:])
    if len(segments) == 0:
        return
    # Opacity per pixel, only in the box around the line (clipped to the
    # image). Each segment is cut into pieces drawn at their mean age; newer
    # pieces are drawn last and cover older ones at the joints.
    size = (image.shape[1], image.shape[0])
    used = xy[np.union1d(segments, segments + 1)]
    x0, y0 = np.maximum(np.floor(used.min(axis=0)) - thickness - 1, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(used.max(axis=0)) + thickness + 2, size).astype(int)
    if x0 >= x1 or y0 >= y1:
        return
    alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    origin = np.array([x0, y0]) << SUBPIXEL
    lengths = np.hypot(*(xy[segments + 1] - xy[segments]).T)
    pieces = np.clip(np.ceil(lengths / _PIECE_PX), 1, _PIECES).astype(int)
    segment = np.repeat(segments, pieces)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    t0 = (np.arange(len(segment)) - first) / np.repeat(pieces, pieces)
    t1 = t0 + 1 / np.repeat(pieces, pieces)
    starts, ends = (
        to_fixed(xy[segment] + (xy[segment + 1] - xy[segment]) * t[:, None]) - origin
        for t in (t0, t1)
    )
    piece_ages = ages[segment] + (ages[segment + 1] - ages[segment]) * (t0 + t1) / 2
    opacity = np.round(255 * (1 - piece_ages / seconds)).astype(int)
    for start, end, value in zip(
        starts.tolist(), ends.tolist(), opacity.tolist(), strict=True
    ):
        cv2.line(alpha, start, end, value, thickness, cv2.LINE_AA, SUBPIXEL)
    drawn = alpha > 0
    if drawn.any():
        region = image[y0:y1, x0:x1]
        weight = alpha[drawn, None] / 255.0
        bgr = np.array(color.as_bgr(), dtype=float)
        region[drawn] = np.round((1 - weight) * region[drawn] + weight * bgr).astype(
            np.uint8
        )
