"""ByteTrack (from supervision) behind the :class:`Tracker` interface."""

import copy
import logging

import numpy as np
import supervision as sv

from tactifoot_vision.tracking.base import TRACKERS, Tracker

logger = logging.getLogger(__name__)


@TRACKERS.register("bytetrack")
class ByteTrackTracker(Tracker):
    """Box-only multi-object tracking with ``sv.ByteTrack``.

    Args:
        track_activation_threshold: detections above this confidence can start a track.
        lost_track_buffer: frames (at 30 fps; scaled to ``frame_rate``) a lost track is kept.
        minimum_matching_threshold: IoU-cost threshold for matching tracks and detections.
        frame_rate: fixed frame rate; ``None`` uses the fps passed to :meth:`reset`.
        minimum_consecutive_frames: frames a track must be seen before it gets an id.
    """

    name = "bytetrack"

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float | None = None,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.reset()

    def reset(self, fps: float | None = None) -> None:
        rate = self.frame_rate or fps or 30.0
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=max(1, round(rate)),
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        logger.debug("ByteTrack reset at %.2f fps", rate)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        # sv.ByteTrack writes tracker_id into its input, so hand it a shallow copy.
        tracker_input = copy.copy(detections)
        if tracker_input.confidence is None:
            if len(detections):
                raise ValueError("ByteTrack needs detection confidences")
            tracker_input.confidence = np.zeros(0, dtype=np.float32)
        # Called on empty frames too, so lost tracks age.
        tracked = self._tracker.update_with_detections(tracker_input)
        if len(tracked) == 0:
            # supervision returns a bare Detections.empty() here, which drops the
            # data keys (class_name, ...); return an empty slice of the input instead.
            tracked = detections[np.zeros(len(detections), dtype=bool)]
            tracked.tracker_id = np.zeros(0, dtype=int)
        return tracked
