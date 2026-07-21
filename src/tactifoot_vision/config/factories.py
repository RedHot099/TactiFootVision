from typing import TYPE_CHECKING

from tactifoot_vision.config.schemas import PipelineConfig
from tactifoot_vision.detection.interfaces import Detector
from tactifoot_vision.domain import AdapterUnavailable, ConfigurationError
from tactifoot_vision.enums import DetectionBackend, TrackingBackend
from tactifoot_vision.tracking.interfaces import Tracker

if TYPE_CHECKING:
    from tactifoot_vision.pipeline import InferencePipeline
    from tactifoot_vision.projection import PitchProjector


def build_detector(config: PipelineConfig) -> Detector:
    det = config.detection
    if det.backend == DetectionBackend.FAKE:
        from tactifoot_vision.detection import FakeDetector

        return FakeDetector()
    if det.backend == DetectionBackend.YOLO:
        from tactifoot_vision.detection import YOLODetectionModel

        if det.checkpoint is None:
            raise ValueError("YOLO detection requires detection.checkpoint")
        return YOLODetectionModel.from_weights(det.checkpoint).as_detector(
            confidence=det.confidence,
            nms=det.nms,
            classes=det.classes,
            include_labels=det.include_labels,
            per_class_confidence=det.per_class_confidence,
        )
    if det.backend == DetectionBackend.RFDETR:
        from tactifoot_vision.detection import RFDETRDetectionModel

        if det.checkpoint is None:
            raise ValueError("RF-DETR detection requires detection.checkpoint")
        return RFDETRDetectionModel.from_weights(det.checkpoint).as_detector(
            confidence=det.confidence,
            nms=det.nms,
            classes=det.classes,
            include_labels=det.include_labels,
            per_class_confidence=det.per_class_confidence,
        )
    if det.checkpoint is None:
        raise ValueError("RF-DETR Seg detection requires detection.checkpoint")
    from tactifoot_vision.detection import RFDETRSegDetectionModel

    return RFDETRSegDetectionModel.from_weights(det.checkpoint).as_detector(
        confidence=det.confidence,
        nms=det.nms,
        classes=det.classes,
        include_labels=det.include_labels,
        per_class_confidence=det.per_class_confidence,
    )


def build_tracker(config: PipelineConfig) -> Tracker:
    tracking = config.tracking
    if tracking.backend == TrackingBackend.FAKE:
        from tactifoot_vision.tracking import FakeTracker

        return FakeTracker()
    if tracking.backend == TrackingBackend.BYTETRACK:
        from tactifoot_vision.tracking import ByteTrackTracker

        return ByteTrackTracker(
            frame_rate=tracking.frame_rate,
            track_activation_threshold=tracking.track_activation_threshold,
            lost_track_buffer=tracking.lost_track_buffer,
            minimum_matching_threshold=tracking.minimum_matching_threshold,
            minimum_consecutive_frames=tracking.minimum_consecutive_frames,
        )
    if tracking.backend == TrackingBackend.BOTSORT:
        raise AdapterUnavailable(
            "BoTSORT tracking is disabled until a stable production adapter is selected."
        )
    if tracking.sam2 is None:
        raise ValueError("SAM2 tracking requires tracking.sam2 config")
    from tactifoot_vision.tracking import SAM2Tracker

    return SAM2Tracker(tracking.sam2)


def build_pipeline(config: PipelineConfig) -> "InferencePipeline":
    if config.team_assignment.enabled:
        raise ConfigurationError(
            "team_assignment.enabled requires a fitted TeamAssigner. "
            "Use TeamClassificationExperimentRunner or pass a fitted assigner "
            "to InferencePipeline from Python."
        )
    from tactifoot_vision.pipeline import InferencePipeline

    return InferencePipeline(
        detector=build_detector(config),
        tracker=build_tracker(config),
        projector=build_projector(config),
    )


def build_projector(config: PipelineConfig) -> "PitchProjector | None":
    if not config.projection.enabled:
        return None
    from tactifoot_vision.projection import HomographyEstimator, PitchProjector

    keypoint_detector = None
    if config.keypoints.checkpoint is not None:
        from tactifoot_vision.keypoints import YOLOPoseKeypointModel

        keypoint_detector = YOLOPoseKeypointModel.from_weights(
            config.keypoints.checkpoint,
            confidence=config.keypoints.confidence,
        )
    return PitchProjector(
        keypoint_detector=keypoint_detector,
        estimator=HomographyEstimator(
            min_confidence=config.projection.min_keypoint_confidence,
            min_keypoints=config.projection.min_keypoints,
            smoothing_window=config.projection.smoothing_window,
        ),
        project_ball=config.projection.project_ball,
    )
