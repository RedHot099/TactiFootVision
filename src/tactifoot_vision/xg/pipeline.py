from collections.abc import Sequence

from tactifoot_vision.ball import (
    BallTrajectoryReconstructor,
    LinearBallTrajectoryReconstructor,
)
from tactifoot_vision.domain import FrameResult, PipelineResult
from tactifoot_vision.shots import KinematicShotDetector, ShotDetector
from tactifoot_vision.xg.features import build_shot_features
from tactifoot_vision.xg.geometry import GeometryXgEstimator
from tactifoot_vision.xg.interfaces import XgEstimator
from tactifoot_vision.xg.results import VideoXgSummary


class VideoXgEstimator:
    def __init__(
        self,
        *,
        ball_reconstructor: BallTrajectoryReconstructor | None = None,
        shot_detector: ShotDetector | None = None,
        xg_estimator: XgEstimator | None = None,
        image_width: int = 1920,
        image_height: int = 1080,
        attacking_goal_x: float | None = None,
    ) -> None:
        self.ball_reconstructor = (
            ball_reconstructor or LinearBallTrajectoryReconstructor()
        )
        self.shot_detector = shot_detector or KinematicShotDetector()
        self.xg_estimator = xg_estimator or GeometryXgEstimator()
        self.image_width = image_width
        self.image_height = image_height
        self.attacking_goal_x = attacking_goal_x

    def run(
        self,
        result: PipelineResult,
        *,
        group_id: str | None = None,
    ) -> VideoXgSummary:
        trajectory = self.ball_reconstructor.reconstruct(result)
        candidates = self.shot_detector.detect(trajectory, result.frames)
        frames_by_index = _frames_by_index(result.frames)
        predictions = []
        for candidate in candidates:
            features = build_shot_features(
                candidate=candidate,
                trajectory=trajectory,
                frame_result=frames_by_index.get(candidate.frame_index),
                image_width=self.image_width,
                image_height=self.image_height,
                attacking_goal_x=self.attacking_goal_x,
            )
            predictions.append(self.xg_estimator.predict(features, candidate))
        return VideoXgSummary(tuple(predictions), group_id=group_id)


def _frames_by_index(frames: Sequence[FrameResult]) -> dict[int, FrameResult]:
    return {frame.frame_index: frame for frame in frames}
