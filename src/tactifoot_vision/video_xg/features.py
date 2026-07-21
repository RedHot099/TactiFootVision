from tactifoot_vision.ball import BallTrajectory
from tactifoot_vision.domain import FrameResult
from tactifoot_vision.shots import ShotCandidate
from tactifoot_vision.video_xg.results import VideoShotFeatures
from tactifoot_vision.xg.features import build_shot_features


def build_video_shot_features(
    *,
    shot_id: str,
    candidate: ShotCandidate,
    trajectory: BallTrajectory,
    frame_result: FrameResult | None,
    image_width: int = 1920,
    image_height: int = 1080,
    attacking_goal_x: float | None = None,
) -> VideoShotFeatures:
    features = build_shot_features(
        candidate=candidate,
        trajectory=trajectory,
        frame_result=frame_result,
        image_width=image_width,
        image_height=image_height,
        attacking_goal_x=attacking_goal_x,
    )
    goal_x = (
        attacking_goal_x
        if attacking_goal_x is not None
        else 0.0
        if features.shot_x < 105.0 / 2.0
        else 105.0
    )
    return VideoShotFeatures(
        shot_id=shot_id,
        frame_index=candidate.frame_index,
        shot_x=features.shot_x,
        shot_y=features.shot_y,
        goal_x=goal_x,
        goal_y=34.0,
        nearest_player_distance=features.nearest_player_distance,
        goalkeeper_distance=features.goalkeeper_distance,
        defender_count_in_cone=features.defender_count_in_cone,
        ball_speed=features.ball_speed,
        ball_direction_to_goal=None,
        shot_confidence=candidate.confidence,
    )
