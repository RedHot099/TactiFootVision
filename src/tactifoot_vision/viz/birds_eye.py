"""Bird's-eye view: a video frame warped onto the pitch plane."""

import cv2
import numpy as np

from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.viz.annotate import project_to_frame
from tactifoot_vision.viz.radar import _STD_LENGTH, _STD_WIDTH


def warp_to_pitch(
    frame: np.ndarray,
    homography: np.ndarray,
    pitch: SoccerPitch | None = None,
    px_per_m: float = 10.0,
) -> np.ndarray:
    """Top-down image of the pitch as ``frame`` shows it, as BGRA ``uint8``.

    The image covers the whole pitch, ``105 * px_per_m`` by ``68 * px_per_m``
    pixels (a pitch in other units is a 105 x 68 m pitch, as the radar draws
    it), with ``x`` to the right and ``y`` down. The pixels tile the pitch:
    pixel ``(column, row)`` shows the pitch at ``((column + 0.5) / sx,
    (row + 0.5) / sy)``, where ``sx, sy`` are pixels per pitch unit.

    Where the frame does not show the pitch (outside the frame, or behind the
    camera beyond the horizon) the image is transparent and black, so nothing
    is invented. As everywhere in the package, the frame centre must lie on
    the pitch side of the horizon.

    Args:
        frame: BGR ``uint8`` video frame.
        homography: 3x3 frame -> pitch homography, e.g. ``FrameResult.homography``.
        pitch: pitch geometry and units of ``homography``.
        px_per_m: pixels per metre.
    """
    if px_per_m <= 0:
        raise ValueError(f"px_per_m must be > 0, got {px_per_m}")
    homography = np.asarray(homography, dtype=float)
    if homography.shape != (3, 3):
        raise ValueError(
            f"homography must be a 3 x 3 matrix, got shape {homography.shape}"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must be a BGR image (height x width x 3), got shape {frame.shape}"
        )
    pitch = pitch or SoccerPitch()
    width, height = round(_STD_LENGTH * px_per_m), round(_STD_WIDTH * px_per_m)
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    points = np.column_stack([
        (columns.ravel() + 0.5) * pitch.length / width,
        (rows.ravel() + 0.5) * pitch.width / height,
    ])  # fmt: skip
    frame_h, frame_w = frame.shape[:2]
    xy, in_front = project_to_frame(points, homography, (frame_w, frame_h))
    # The frame outline, as in the visible area: ``xy`` is continuous, with
    # pixel ``i`` covering ``[i, i + 1)``, so cv2 samples at ``xy - 0.5``.
    seen = (
        in_front
        & (xy[:, 0] >= 0)
        & (xy[:, 0] <= frame_w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= frame_h)
    )
    maps = np.where(seen[:, None], xy - 0.5, -2).astype(np.float32)
    bgr = cv2.remap(
        frame,
        maps.reshape(height, width, 2),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # the half pixel at the frame edges
    )
    seen = seen.reshape(height, width)
    bgr[~seen] = 0
    return np.dstack([bgr, np.where(seen, 255, 0).astype(np.uint8)])
