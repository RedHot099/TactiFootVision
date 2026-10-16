"""Sub-pixel coordinates for OpenCV drawing calls."""

import numpy as np

SUBPIXEL = 4  # cv2 drawing "shift": coordinates carry 4 fractional bits


def to_fixed(points: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> int32 fixed point for cv2's ``shift`` argument."""
    return np.round(np.asarray(points) * (1 << SUBPIXEL)).astype(np.int32)


def clip_segment(
    a: np.ndarray, b: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray | None:
    """The ``(2, 2)`` part of segment ``a``-``b`` inside the box ``[low, high]``
    (Liang-Barsky), or ``None`` when it misses the box."""
    d = b - a
    start, end = 0.0, 1.0
    for p, q in zip((*-d, *d), (*(a - low), *(high - a)), strict=True):
        if p == 0:
            if q < 0:  # parallel to this side and outside it
                return None
        elif p < 0:
            start = max(start, q / p)
        else:
            end = min(end, q / p)
    return None if start > end else np.array([a + start * d, a + end * d])
