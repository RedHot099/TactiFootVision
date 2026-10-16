"""Sub-pixel coordinates for OpenCV drawing calls."""

import numpy as np

SUBPIXEL = 4  # cv2 drawing "shift": coordinates carry 4 fractional bits


def to_fixed(points: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> int32 fixed point for cv2's ``shift`` argument."""
    return np.round(np.asarray(points) * (1 << SUBPIXEL)).astype(np.int32)
