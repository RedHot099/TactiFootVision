# tactifoot_vision/tracking/ball_tracker.py
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BallTracker:
    def __init__(self):
        self._raw_path: List[np.ndarray] = []
        self._empty_pos = np.empty((0, 2), dtype=np.float32)
        logger.info("Raw Ball Position Collector Initialized.")

    def add_point(self, point: Optional[np.ndarray]):
        if point is not None and point.shape == (1, 2):
            self._raw_path.append(point.astype(np.float32))
        else:
            if point is not None and point.size > 0:
                logger.debug(
                    f"Invalid shape for ball point: {point.shape}. Storing as empty."
                )
            self._raw_path.append(self._empty_pos)

    def get_raw_path(self) -> List[np.ndarray]:
        return self._raw_path

    def get_cleaned_path(self, distance_threshold: float) -> List[np.ndarray]:
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")

        logger.info(
            f"Cleaning raw ball path with distance threshold: {distance_threshold:.2f}..."
        )
        last_valid_position: Optional[np.ndarray] = None
        cleaned_positions: List[np.ndarray] = []

        for i, position in enumerate(self._raw_path):
            if position.shape == (1, 2):
                current_pos_valid = position
                if last_valid_position is None:
                    cleaned_positions.append(current_pos_valid)
                    last_valid_position = current_pos_valid
                else:
                    distance = np.linalg.norm(current_pos_valid - last_valid_position)
                    if distance > distance_threshold:
                        logger.debug(
                            f"Outlier removed at frame {i}. Dist: {distance:.2f} > Threshold: {distance_threshold:.2f}."
                        )
                        cleaned_positions.append(self._empty_pos)
                    else:
                        cleaned_positions.append(current_pos_valid)
                        last_valid_position = current_pos_valid
            elif position.shape == (0, 2):
                cleaned_positions.append(self._empty_pos)
            else:
                logger.warning(
                    f"Unexpected position shape during cleaning at index {i}: {position.shape}. Treating as missing."
                )
                cleaned_positions.append(self._empty_pos)

        valid_points_count = sum(1 for p in cleaned_positions if p.size > 0)
        logger.info(
            f"Ball path cleaning complete. Found {valid_points_count} valid points."
        )
        return cleaned_positions
