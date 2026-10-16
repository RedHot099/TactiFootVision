"""Bird's-eye view: a video frame warped onto the pitch plane."""

import cv2
import numpy as np

from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pitch.pitch import SoccerPitch

# Homographies take continuous frame coordinates (pixel ``i`` covers
# ``[i, i + 1)``); cv2 puts pixel centres at integers.
_TO_PIXEL_CENTRES = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])


def pitch_to_birds_eye(
    pitch: SoccerPitch | None = None, px_per_m: float = 10.0
) -> np.ndarray:
    """3x3 pitch -> pixel homography of :func:`warp_to_pitch`'s image.

    Pass it as ``pitch_to_image`` to draw on that image, e.g. with
    :func:`~tactifoot_vision.viz.draw_space_control`. Pixel centres are at
    integer coordinates (cv2's convention), so the pitch corner ``(0, 0)`` is
    the corner of pixel ``(0, 0)``, at ``(-0.5, -0.5)``.
    """
    pitch = pitch or SoccerPitch()
    sx, sy = _canvas_size(pitch, px_per_m) / [pitch.length, pitch.width]
    return _TO_PIXEL_CENTRES @ np.diag([sx, sy, 1.0])


def _canvas_size(pitch: SoccerPitch, px_per_m: float) -> np.ndarray:
    """``(width, height)`` of the image: the pitch at ``px_per_m`` per metre."""
    if px_per_m <= 0:
        raise ValueError(f"px_per_m must be > 0, got {px_per_m}")
    return np.round(metres_per_unit(pitch) * [pitch.length, pitch.width] * px_per_m)


def warp_to_pitch(
    frame: np.ndarray,
    homography: np.ndarray,
    pitch: SoccerPitch | None = None,
    px_per_m: float = 10.0,
) -> np.ndarray:
    """Top-down image of the pitch as ``frame`` shows it, as BGRA ``uint8``.

    The image covers the whole pitch, ``105 * px_per_m`` by ``68 * px_per_m``
    pixels (a pitch in other units is a 105 x 68 m pitch, as the radar draws
    it), with ``x`` to the right and ``y`` down. The pixels tile the pitch;
    :func:`pitch_to_birds_eye` gives the pitch -> pixel homography.

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
    homography = np.asarray(homography, dtype=float)
    if homography.shape != (3, 3):
        raise ValueError(
            f"homography must be a 3 x 3 matrix, got shape {homography.shape}"
        )
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            "frame must be a BGR uint8 image (height x width x 3), got "
            f"{frame.dtype} of shape {frame.shape}"
        )
    pitch = pitch or SoccerPitch()
    frame_to_image = pitch_to_birds_eye(pitch, px_per_m) @ homography
    size = tuple(int(n) for n in _canvas_size(pitch, px_per_m))
    nothing = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    frame_h, frame_w = frame.shape[:2]
    centre = frame_to_image @ [frame_w / 2, frame_h / 2, 1]
    try:
        image_to_frame = _TO_PIXEL_CENTRES @ np.linalg.inv(frame_to_image)
    except np.linalg.LinAlgError:  # a degenerate fit shows nothing
        return nothing
    if not np.isfinite(image_to_frame).all() or abs(centre[2]) < 1e-12:
        return nothing
    # In front of the camera: the image points whose ``w`` in the frame has
    # the sign of the frame centre's, a half-plane bounded by the horizon.
    # Points behind the camera would otherwise mirror into the frame.
    side = np.sign(image_to_frame[2] @ (centre / centre[2]))
    a, b, c = image_to_frame[2] * side
    in_front = a * np.arange(size[0]) + b * np.arange(size[1])[:, None] + c > 1e-9
    flags = cv2.WARP_INVERSE_MAP
    bgr = cv2.warpPerspective(
        frame, image_to_frame, size, flags=flags | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # the half pixel at the frame edges
    )  # fmt: skip
    in_frame = cv2.warpPerspective(
        np.full((frame_h, frame_w), 255, dtype=np.uint8), image_to_frame, size,
        flags=flags | cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )  # fmt: skip
    alpha = in_frame & (in_front.astype(np.uint8) * 255)
    return cv2.merge([*cv2.split(cv2.bitwise_and(bgr, bgr, mask=alpha)), alpha])
