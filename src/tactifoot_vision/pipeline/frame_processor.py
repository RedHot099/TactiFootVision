from tactifoot_vision.detection import Detector
from tactifoot_vision.domain import Frame, FrameResult, PitchProjection
from tactifoot_vision.tracking import Tracker


class FrameProcessor:
    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        projector: object | None = None,
        team_assigner: object | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.projector = projector
        self.team_assigner = team_assigner

    def process(self, frame: Frame) -> FrameResult:
        detections = self.detector.predict(frame)
        tracks = self.tracker.update(frame, detections)
        if self.team_assigner is not None and getattr(
            self.team_assigner, "is_fitted", False
        ):
            tracks = self.team_assigner.assign_tracks(frame, tracks)
        projection = PitchProjection.unavailable()
        if self.projector is not None and hasattr(self.projector, "project"):
            projection = self.projector.project(
                frame=frame, keypoints=None, tracks=tracks
            )
        return FrameResult(
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            detections=detections,
            tracks=tracks,
            projection=projection,
        )
