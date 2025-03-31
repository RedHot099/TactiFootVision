# tactifoot_vision/geometry/view_transformer.py
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from config.models import GeometryConfig
from .pitch_definitions import SoccerPitchConfiguration

logger = logging.getLogger(__name__)


class ViewTransformer:
    def __init__(self, config: GeometryConfig):
        self.config = config
        # --- Instantiate pitch config with dimensions from GeometryConfig ---
        self.pitch_config = SoccerPitchConfiguration(
            length=config.target_pitch_length, width=config.target_pitch_width
        )
        # --------------------------------------------------------------------
        self.pitch_vertices = np.array(self.pitch_config.vertices, dtype=np.float32)
        self.min_confidence = config.min_keypoint_confidence_for_homography
        self.smoothing_window_size = config.homography_smoothing_window
        self.homography_matrices = deque(maxlen=self.smoothing_window_size)
        self.current_homography: Optional[np.ndarray] = None
        self.current_inverse_homography: Optional[np.ndarray] = None
        logger.info(
            f"Initialized ViewTransformer. Smoothing window: {self.smoothing_window_size}"
        )
        logger.info(
            f"Using target pitch dimensions (LxW): {self.pitch_config.length} x {self.pitch_config.width}"
        )
        if len(self.pitch_config.vertices) != len(self.pitch_config.labels):
            logger.warning(
                "Mismatch between number of vertices and labels in PitchConfiguration!"
            )

    def _update_inverse_homography(self):
        if self.current_homography is not None:
            try:
                self.current_inverse_homography = np.linalg.pinv(
                    self.current_homography
                )
            except np.linalg.LinAlgError:
                logger.error(
                    "Failed to calculate inverse homography (matrix likely singular)."
                )
                self.current_inverse_homography = None
        else:
            self.current_inverse_homography = None

    def update_homography(self, keypoints: sv.KeyPoints) -> bool:
        if (
            keypoints is None
            or keypoints.xy.size == 0
            or keypoints.confidence is None
            or keypoints.xy.shape[0] == 0
            or keypoints.xy.shape[1] != len(self.pitch_vertices)
        ):
            logger.warning(
                f"Received invalid or mismatched keypoints for homography update. "
                f"Expected {len(self.pitch_vertices)} keypoints, got shape {keypoints.xy.shape if keypoints else 'None'}."
            )
            self._update_inverse_homography()
            return self.current_homography is not None

        frame_xy = keypoints.xy[0]
        confidences = keypoints.confidence[0]

        valid_mask = confidences >= self.min_confidence
        frame_points_filtered = frame_xy[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        num_valid_points = len(frame_points_filtered)
        if num_valid_points < 4:
            logger.warning(
                f"Not enough confident keypoints ({num_valid_points} < 4) "
                f"to calculate homography. Threshold: {self.min_confidence:.2f}. "
                f"Confident indices: {valid_indices.tolist()}"
            )
            self._update_inverse_homography()
            return self.current_homography is not None

        pitch_points_filtered = self.pitch_vertices[valid_indices]

        logger.debug(
            f"Calculating homography with {num_valid_points} points. "
            f"Frame indices: {valid_indices.tolist()}. "
            f"Frame pts shape: {frame_points_filtered.shape}, "
            f"Pitch pts shape: {pitch_points_filtered.shape}"
        )

        try:
            homography_matrix, mask = cv2.findHomography(
                frame_points_filtered, pitch_points_filtered, cv2.RANSAC, 10.0
            )

            if homography_matrix is None:
                logger.warning("cv2.findHomography returned None.")
                self._update_inverse_homography()
                return self.current_homography is not None

            self.homography_matrices.append(homography_matrix)
            self.current_homography = np.mean(self.homography_matrices, axis=0)
            self._update_inverse_homography()
            logger.debug("Homography matrix updated successfully.")
            return True

        except cv2.error as cv_err:
            logger.error(f"OpenCV error during findHomography: {cv_err}", exc_info=True)
            self._update_inverse_homography()
            return self.current_homography is not None
        except Exception as e:
            logger.error(f"Unexpected error calculating homography: {e}", exc_info=True)
            self._update_inverse_homography()
            return self.current_homography is not None

    def transform_points(
        self, points_frame: np.ndarray, perspective: str = "frame_to_pitch"
    ) -> Optional[np.ndarray]:
        matrix = None
        if perspective == "frame_to_pitch":
            matrix = self.current_homography
            matrix_name = "Homography"
        elif perspective == "pitch_to_frame":
            matrix = self.current_inverse_homography
            matrix_name = "Inverse Homography"
        else:
            logger.error(f"Invalid perspective specified: {perspective}")
            return None

        if matrix is None:
            logger.warning(
                f"{matrix_name} matrix not available for transformation ({perspective})."
            )
            return None
        if not isinstance(points_frame, np.ndarray):
            logger.error(
                f"Invalid input type for transform_points: {type(points_frame)}. Expected np.ndarray."
            )
            return None
        if points_frame.ndim != 2 or points_frame.shape[1] != 2:
            if points_frame.size == 0:
                return np.empty((0, 2), dtype=np.float32)
            logger.error(
                f"Invalid input shape for transform_points: {points_frame.shape}. Expected (N, 2)."
            )
            return None

        points_frame_reshaped = points_frame.astype(np.float32).reshape(-1, 1, 2)
        try:
            transformed_points_reshaped = cv2.perspectiveTransform(
                points_frame_reshaped, matrix
            )
            if np.any(np.isnan(transformed_points_reshaped)) or np.any(
                np.isinf(transformed_points_reshaped)
            ):
                logger.warning(
                    f"NaN or Inf detected in transformed points ({perspective}). Homography might be unstable."
                )
                return None
            transformed_points = transformed_points_reshaped.reshape(-1, 2)
            return transformed_points
        except cv2.error as cv_err:
            logger.error(
                f"OpenCV error during perspectiveTransform ({perspective}): {cv_err}",
                exc_info=True,
            )
            return None
        except Exception:
            logger.exception(
                f"Unexpected error during point transformation ({perspective})."
            )
            return None

    def transform_frame_to_pitch(
        self, points_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        return self.transform_points(points_frame, perspective="frame_to_pitch")

    def transform_pitch_to_frame(
        self, points_pitch: np.ndarray
    ) -> Optional[np.ndarray]:
        return self.transform_points(points_pitch, perspective="pitch_to_frame")
