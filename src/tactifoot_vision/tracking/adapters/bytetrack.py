from typing import Any

from tactifoot_vision.domain import DetectionSet, Frame, TrackSet
from tactifoot_vision.tracking.conversions import (
    detections_to_supervision,
    supervision_to_tracks,
)


class ByteTrackTracker:
    def __init__(
        self,
        *,
        frame_rate: int = 25,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        import supervision as sv

        self._kwargs: dict[str, Any] = {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "frame_rate": frame_rate,
            "minimum_consecutive_frames": minimum_consecutive_frames,
        }
        self._tracker = sv.ByteTrack(**self._kwargs)

    def update(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        tracked = self._tracker.update_with_detections(
            detections_to_supervision(detections)
        )
        return supervision_to_tracks(tracked, detections)

    def reset(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack(**self._kwargs)
