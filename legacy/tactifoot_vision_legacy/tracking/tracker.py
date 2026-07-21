# tactifoot_vision/tracking/tracker.py
import logging

import supervision as sv

from config.models import TrackingConfig

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        # Use correct argument names based on provided ByteTrack signature
        self.tracker = sv.ByteTrack(
            track_activation_threshold=(
                config.track_activation_threshold
                if config.track_activation_threshold is not None
                else 0.25
            ),
            lost_track_buffer=(
                config.lost_track_buffer if config.lost_track_buffer is not None else 30
            ),
            minimum_matching_threshold=(
                config.minimum_matching_threshold
                if config.minimum_matching_threshold is not None
                else 0.8
            ),
            frame_rate=(config.frame_rate if config.frame_rate is not None else 30),
            minimum_consecutive_frames=(
                config.minimum_consecutive_frames
                if config.minimum_consecutive_frames is not None
                else 1
            ),
        )
        logger.info(f"Initialized ByteTrack tracker with config: {config.model_dump()}")
        if config.frame_rate is None:
            logger.warning("Tracking frame_rate not set in config, defaulting to 30.")

    def update(self, detections: sv.Detections) -> sv.Detections:
        if not self.config.enabled:
            return detections
        try:
            tracked_detections = self.tracker.update_with_detections(detections)
            return tracked_detections
        except Exception as e:
            logger.error(f"Error during tracker update: {e}", exc_info=True)
            return detections

    def reset(self):
        # Re-initialize with correct arguments
        self.tracker = sv.ByteTrack(
            track_activation_threshold=(
                self.config.track_activation_threshold
                if self.config.track_activation_threshold is not None
                else 0.25
            ),
            lost_track_buffer=(
                self.config.lost_track_buffer
                if self.config.lost_track_buffer is not None
                else 30
            ),
            minimum_matching_threshold=(
                self.config.minimum_matching_threshold
                if self.config.minimum_matching_threshold is not None
                else 0.8
            ),
            frame_rate=(
                self.config.frame_rate if self.config.frame_rate is not None else 30
            ),
            minimum_consecutive_frames=(
                self.config.minimum_consecutive_frames
                if self.config.minimum_consecutive_frames is not None
                else 1
            ),
        )
        logger.info("Tracker state reset.")
