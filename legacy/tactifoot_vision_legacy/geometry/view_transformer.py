# tactifoot_vision/geometry/view_transformer.py
import logging
from collections import deque
from typing import Optional  # Added Set

import cv2
import numpy as np
import supervision as sv

from config.models import GeometryConfig
from .pitch_definitions import SoccerPitchConfiguration

logger = logging.getLogger(__name__)


class ViewTransformer:
    def __init__(self, config: GeometryConfig):
        self.config = config
        self.pitch_config = SoccerPitchConfiguration(
            length=config.target_pitch_length, width=config.target_pitch_width
        )
        self.pitch_vertices = np.array(self.pitch_config.vertices, dtype=np.float32)
        self.min_confidence = config.min_keypoint_confidence_for_homography
        self.smoothing_window_size = config.homography_smoothing_window
        self.homography_matrices = deque(maxlen=self.smoothing_window_size)
        self.current_homography: Optional[np.ndarray] = None
        self.current_inverse_homography: Optional[np.ndarray] = None
        # --- Store indices used in the last successful calculation ---
        self.last_used_indices: Optional[np.ndarray] = None
        # ----------------------------------------------------------
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
                logger.error("Failed to calculate inverse homography.", exc_info=True)
                self.current_inverse_homography = None
        else:
            self.current_inverse_homography = None

    def update_homography(self, keypoints: sv.KeyPoints) -> bool:
        # Reset last used indices for this frame attempt
        # self.last_used_indices = None # Option 1: Reset each time
        # Option 2: Keep last good ones until a new successful calc happens (current implementation)

        if (
            keypoints is None
            or keypoints.xy.size == 0
            or keypoints.confidence is None
            or keypoints.xy.shape[0] == 0
            or keypoints.xy.shape[1] != len(self.pitch_vertices)
        ):
            logger.warning(
                f"Invalid/mismatched keypoints. Expected {len(self.pitch_vertices)}, got shape {keypoints.xy.shape if keypoints else 'None'}."
            )
            # Don't update inverse or indices if input is bad, reuse existing H if available
            return self.current_homography is not None

        frame_xy = keypoints.xy[0]
        confidences = keypoints.confidence[0]
        valid_mask = confidences >= self.min_confidence
        frame_points_filtered = frame_xy[valid_mask]
        valid_indices = np.where(valid_mask)[0]  # Original indices of confident points

        num_valid_points = len(frame_points_filtered)
        if num_valid_points < 4:
            logger.warning(
                f"Not enough confident keypoints ({num_valid_points} < 4). Threshold: {self.min_confidence:.2f}."
            )
            # Don't update inverse or indices if not enough points, reuse existing H if available
            return self.current_homography is not None

        pitch_points_filtered = self.pitch_vertices[valid_indices]

        try:
            homography_matrix, ransac_mask = cv2.findHomography(
                frame_points_filtered, pitch_points_filtered, cv2.RANSAC, 10.0
            )

            if homography_matrix is None:
                logger.warning("cv2.findHomography returned None.")
                # Don't update inverse or indices, reuse existing H if available
                return self.current_homography is not None

            # --- Store the indices that were confident AND used by RANSAC ---
            # Note: ransac_mask corresponds to frame_points_filtered/valid_indices
            # inlier_indices_relative = np.where(ransac_mask.flatten() == 1)[0]
            # self.last_used_indices = valid_indices[inlier_indices_relative] # Indices of confident points that were RANSAC inliers
            # --- OR: Simpler - just store indices that were input to findHomography ---
            self.last_used_indices = (
                valid_indices  # Store confident indices used as input
            )
            # -------------------------------------------------------------------------

            self.homography_matrices.append(homography_matrix)
            self.current_homography = np.mean(self.homography_matrices, axis=0)
            self._update_inverse_homography()
            logger.debug(
                f"Homography updated successfully using {len(self.last_used_indices)} confident points."
            )
            return True

        except Exception as e:
            logger.error(f"Error calculating homography: {e}", exc_info=True)
            # Don't update inverse or indices on error, reuse existing H if available
            return self.current_homography is not None

    def transform_points(
        self, points: np.ndarray, perspective: str = "frame_to_pitch"
    ) -> Optional[np.ndarray]:
        matrix = None
        if perspective == "frame_to_pitch":
            matrix = self.current_homography
            matrix_name = "Homography"
        elif perspective == "pitch_to_frame":
            matrix = self.current_inverse_homography
            matrix_name = "Inverse Homography"
        else:
            logger.error(f"Invalid perspective: {perspective}")
            return None

        if matrix is None:
            logger.warning(f"{matrix_name} not available for {perspective}.")
            return None
        if not isinstance(points, np.ndarray):
            logger.error(f"Invalid input type: {type(points)}")
            return None
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            logger.error(f"Invalid input shape: {points.shape}")
            return None

        points_reshaped = points.astype(np.float32).reshape(-1, 1, 2)
        try:
            transformed_reshaped = cv2.perspectiveTransform(points_reshaped, matrix)
            if np.any(np.isnan(transformed_reshaped)) or np.any(
                np.isinf(transformed_reshaped)
            ):
                logger.warning(
                    f"NaN/Inf detected in transformed points ({perspective})."
                )
                return None
            return transformed_reshaped.reshape(-1, 2)
        except Exception:
            logger.exception(f"Error during point transformation ({perspective}).")
            return None

    def transform_frame_to_pitch(
        self, points_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        return self.transform_points(points_frame, perspective="frame_to_pitch")

    def transform_pitch_to_frame(
        self, points_pitch: np.ndarray
    ) -> Optional[np.ndarray]:
        return self.transform_points(points_pitch, perspective="pitch_to_frame")
