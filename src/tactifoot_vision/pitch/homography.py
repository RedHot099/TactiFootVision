"""Frame <-> pitch homography from detected pitch keypoints."""

import logging
from collections import deque

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.pitch.pitch import SoccerPitch

logger = logging.getLogger(__name__)


def frame_to_pitch(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` frame pixels to pitch coordinates with a frame->pitch homography."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def pitch_to_frame(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` pitch coordinates to frame pixels (inverse of :func:`frame_to_pitch`)."""
    return frame_to_pitch(points, np.linalg.inv(homography))


class HomographyEstimator:
    """Fits a frame->pitch homography per frame and smooths it over time.

    Keypoints below ``min_confidence`` are ignored; at least four are needed.
    Averaging the last ``smoothing_window`` fits steadies pitch positions but lags
    behind camera pans: on broadcast footage 3 frames cut position jitter in half
    versus no smoothing while adding about 2 px of line misalignment.
    When a frame yields no fit, the last good matrix is reused for up to
    ``max_age`` updates (processed frames; ``None`` = forever), then dropped.
    After such a gap the first new fit starts a fresh average, so matrices from
    before a camera cut are never blended with the new view.

    ``ransac_threshold`` is the RANSAC reprojection limit in pitch units (metres
    on the default pitch): keypoints further than this from the fitted model are
    ignored. The loose default favours smooth positions over outlier rejection.
    """

    def __init__(
        self,
        pitch: SoccerPitch | None = None,
        *,
        min_confidence: float = 0.5,
        smoothing_window: int = 3,
        ransac_threshold: float = 10.0,
        max_age: int | None = 50,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        if ransac_threshold <= 0:
            raise ValueError(f"ransac_threshold must be > 0, got {ransac_threshold}")
        if max_age is not None and max_age < 0:
            raise ValueError(f"max_age must be None or >= 0, got {max_age}")
        self.pitch = pitch or SoccerPitch()
        self.min_confidence = min_confidence
        self.ransac_threshold = ransac_threshold
        self.max_age = max_age
        self._history: deque[np.ndarray] = deque(maxlen=smoothing_window)
        self.matrix: np.ndarray | None = None
        self.used_indices: np.ndarray | None = None  # keypoints behind the last fit
        self._age = 0

    def reset(self) -> None:
        self._history.clear()
        self.matrix = None
        self.used_indices = None
        self._age = 0

    def update(self, keypoints: sv.KeyPoints | None) -> np.ndarray | None:
        """Fit on this frame's keypoints (first instance) and return the current matrix."""
        fitted = self._fit(keypoints)
        if fitted is not None:
            if self._age:
                self._history.clear()
            self._history.append(fitted)
            self.matrix = np.mean(self._history, axis=0)
            self._age = 0
        elif self.matrix is not None:
            self._age += 1
            if self.max_age is not None and self._age > self.max_age:
                self.reset()
        return self.matrix

    def _fit(self, keypoints: sv.KeyPoints | None) -> np.ndarray | None:
        vertices = self.pitch.vertices
        if keypoints is None or len(keypoints) == 0 or keypoints.confidence is None:
            return None
        xy, confidence = keypoints.xy[0], keypoints.confidence[0]
        if len(xy) != len(vertices):
            raise ValueError(
                f"The keypoint model predicts {len(xy)} keypoints but the pitch has "
                f"{len(vertices)} landmarks; use a pitch-landmark model"
            )
        indices = np.flatnonzero(confidence >= self.min_confidence)
        if len(indices) < 4:
            return None
        matrix, _ = cv2.findHomography(
            xy[indices].astype(np.float32),
            vertices[indices],
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
            return None
        self.used_indices = indices
        return matrix / matrix[2, 2]
