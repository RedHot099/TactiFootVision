__all__ = [
    "DetectionTrackingExperimentRunner",
    "HomographyComparisonRunner",
    "SoccerNetDetectionTrackingExperimentRunner",
    "TeamClassificationExperimentRunner",
    "VideoXgExperimentRunner",
]


def __getattr__(name: str) -> object:
    if name == "DetectionTrackingExperimentRunner":
        from tactifoot_vision.experiments.detection_tracking import (
            DetectionTrackingExperimentRunner,
        )

        return DetectionTrackingExperimentRunner
    if name == "HomographyComparisonRunner":
        from tactifoot_vision.experiments.homography_comparison import (
            HomographyComparisonRunner,
        )

        return HomographyComparisonRunner
    if name == "SoccerNetDetectionTrackingExperimentRunner":
        from tactifoot_vision.experiments.soccernet_detection_tracking import (
            SoccerNetDetectionTrackingExperimentRunner,
        )

        return SoccerNetDetectionTrackingExperimentRunner
    if name == "TeamClassificationExperimentRunner":
        from tactifoot_vision.experiments.team_classification import (
            TeamClassificationExperimentRunner,
        )

        return TeamClassificationExperimentRunner
    if name == "VideoXgExperimentRunner":
        from tactifoot_vision.experiments.video_xg import VideoXgExperimentRunner

        return VideoXgExperimentRunner
    raise AttributeError(
        f"module 'tactifoot_vision.experiments' has no attribute {name!r}"
    )
