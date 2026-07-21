from collections import deque

import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.keypoints import KeypointSet
from tactifoot_vision.projection.pitch import PitchModel


def compute_homography(
    source_points: NDArray[np.float32], target_points: NDArray[np.float32]
) -> NDArray[np.float64] | None:
    if source_points.shape[0] < 4 or target_points.shape[0] < 4:
        return None
    import cv2

    matrix, _ = cv2.findHomography(
        source_points.astype(np.float32), target_points.astype(np.float32)
    )
    return np.asarray(matrix, dtype=np.float64) if matrix is not None else None


def apply_homography(
    points: NDArray[np.float32], matrix: NDArray[np.float64]
) -> NDArray[np.float32]:
    import cv2

    transformed = cv2.perspectiveTransform(
        points.astype(np.float32)[None, :, :], matrix
    )
    return np.asarray(transformed[0], dtype=np.float32)


class HomographyEstimator:
    def __init__(
        self,
        *,
        pitch: PitchModel | None = None,
        min_confidence: float = 0.5,
        min_keypoints: int = 4,
        smoothing_window: int = 15,
    ) -> None:
        self.pitch = pitch or PitchModel()
        self.min_confidence = min_confidence
        self.min_keypoints = min_keypoints
        self._matrices: deque[NDArray[np.float64]] = deque(maxlen=smoothing_window)
        self.current_homography: NDArray[np.float64] | None = None
        self.current_inverse_homography: NDArray[np.float64] | None = None

    def update(self, keypoints: KeypointSet) -> NDArray[np.float64] | None:
        source_points: list[tuple[float, float]] = []
        target_points: list[tuple[float, float]] = []
        targets = self.pitch.keypoint_targets
        for keypoint in keypoints:
            if keypoint.index not in targets:
                continue
            if (
                keypoint.confidence is not None
                and keypoint.confidence < self.min_confidence
            ):
                continue
            source_points.append((keypoint.x, keypoint.y))
            target = targets[keypoint.index]
            target_points.append((target.x, target.y))
        if len(source_points) < self.min_keypoints:
            return self.current_homography
        matrix = compute_homography(
            np.array(source_points, dtype=np.float32),
            np.array(target_points, dtype=np.float32),
        )
        if matrix is None:
            return self.current_homography
        self._matrices.append(matrix)
        self.current_homography = np.mean(np.array(self._matrices), axis=0)
        self.current_inverse_homography = np.linalg.pinv(self.current_homography)
        return self.current_homography
