import math
from typing import Protocol

from tactifoot_vision.enums import XgModelKind
from tactifoot_vision.video_xg.results import (
    VideoOnlyShotPrediction,
    VideoShotFeatures,
)

GOAL_WIDTH = 7.32


class VideoOnlyXgEstimator(Protocol):
    model_kind: XgModelKind

    def predict(self, features: VideoShotFeatures) -> VideoOnlyShotPrediction:
        pass


class VideoGeometryXgEstimator:
    model_kind = XgModelKind.VIDEO_GEOMETRY

    def predict(self, features: VideoShotFeatures) -> VideoOnlyShotPrediction:
        xg = _sigmoid(
            0.15
            - 0.13 * _distance_to_goal(features)
            + 1.05 * _shot_angle(features)
            + 0.25 * _centrality(features)
        )
        return _prediction(features, xg, self.model_kind)


class VideoFreezeContextXgEstimator:
    model_kind = XgModelKind.VIDEO_FREEZE_CONTEXT

    def predict(self, features: VideoShotFeatures) -> VideoOnlyShotPrediction:
        pressure = 0.0
        if features.nearest_player_distance is not None:
            pressure = max(0.0, (3.5 - features.nearest_player_distance) / 3.5)
        goalkeeper_pressure = 0.0
        if features.goalkeeper_distance is not None:
            goalkeeper_pressure = max(0.0, (8.0 - features.goalkeeper_distance) / 8.0)
        speed_bonus = 0.0
        if features.ball_speed is not None:
            speed_bonus = min(features.ball_speed / 80.0, 0.35)
        xg = _sigmoid(
            0.2
            - 0.12 * _distance_to_goal(features)
            + 1.18 * _shot_angle(features)
            + 0.35 * _centrality(features)
            + speed_bonus
            - 0.65 * pressure
            - 0.25 * goalkeeper_pressure
            - 0.07 * features.defender_count_in_cone
        )
        return _prediction(features, xg, self.model_kind)


class VideoKinematicContextXgEstimator:
    model_kind = XgModelKind.VIDEO_KINEMATIC_CONTEXT

    def predict(self, features: VideoShotFeatures) -> VideoOnlyShotPrediction:
        pressure = 0.0
        if features.nearest_player_distance is not None:
            pressure = max(0.0, (3.5 - features.nearest_player_distance) / 3.5)
        goalkeeper_pressure = 0.0
        if features.goalkeeper_distance is not None:
            goalkeeper_pressure = max(0.0, (8.0 - features.goalkeeper_distance) / 8.0)
        speed_bonus = 0.0
        if features.ball_speed is not None:
            speed_bonus = min(features.ball_speed / 45.0, 0.45)
        direction_bonus = 0.0
        if features.ball_direction_to_goal is not None:
            direction_bonus = 0.35 * max(0.0, min(features.ball_direction_to_goal, 1.0))
        xg = _sigmoid(
            0.1
            - 0.115 * _distance_to_goal(features)
            + 1.22 * _shot_angle(features)
            + 0.34 * _centrality(features)
            + speed_bonus
            + direction_bonus
            - 0.65 * pressure
            - 0.25 * goalkeeper_pressure
            - 0.07 * features.defender_count_in_cone
        )
        return _prediction(features, xg, self.model_kind)


def _prediction(
    features: VideoShotFeatures, xg: float, model_kind: XgModelKind
) -> VideoOnlyShotPrediction:
    return VideoOnlyShotPrediction(
        shot_id=features.shot_id,
        frame_index=features.frame_index,
        xg=min(max(float(xg), 0.001), 0.999),
        model_kind=model_kind,
        features=features,
    )


def _distance_to_goal(features: VideoShotFeatures) -> float:
    return math.hypot(
        features.goal_x - features.shot_x, features.goal_y - features.shot_y
    )


def _centrality(features: VideoShotFeatures) -> float:
    goal_y = max(abs(features.goal_y), 1e-9)
    return max(0.0, 1.0 - abs(features.shot_y - features.goal_y) / goal_y)


def _shot_angle(features: VideoShotFeatures) -> float:
    post_top_y = features.goal_y - GOAL_WIDTH / 2.0
    post_bottom_y = features.goal_y + GOAL_WIDTH / 2.0
    distance_top = math.hypot(
        features.goal_x - features.shot_x, post_top_y - features.shot_y
    )
    distance_bottom = math.hypot(
        features.goal_x - features.shot_x, post_bottom_y - features.shot_y
    )
    if distance_top == 0.0 or distance_bottom == 0.0:
        return math.pi
    value = min(max(GOAL_WIDTH / max(distance_top * distance_bottom, 1e-9), -1.0), 1.0)
    return math.asin(value)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
