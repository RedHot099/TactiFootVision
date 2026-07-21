import math

from tactifoot_vision.enums import ShotOutcome, XgModelKind
from tactifoot_vision.shots import ShotCandidate
from tactifoot_vision.xg.results import XgPrediction, XgShotFeatures


class GeometryXgEstimator:
    def __init__(self, *, penalty_xg: float = 0.76) -> None:
        self.penalty_xg = penalty_xg

    def predict(
        self, features: XgShotFeatures, candidate: ShotCandidate
    ) -> XgPrediction:
        if features.is_penalty or candidate.outcome == ShotOutcome.PENALTY:
            xg = self.penalty_xg
        else:
            xg = _sigmoid(_logit(features))
        return XgPrediction(
            candidate=candidate,
            xg=min(max(float(xg), 0.001), 0.999),
            features=features,
            model_kind=XgModelKind.GEOMETRY,
        )


def _logit(features: XgShotFeatures) -> float:
    pressure = 0.0
    if features.nearest_player_distance is not None:
        pressure = max(0.0, (3.0 - features.nearest_player_distance) / 3.0)
    goalkeeper_pressure = 0.0
    if features.goalkeeper_distance is not None:
        goalkeeper_pressure = max(0.0, (8.0 - features.goalkeeper_distance) / 8.0)
    speed_bonus = 0.0
    if features.ball_speed is not None:
        speed_bonus = min(features.ball_speed / 80.0, 0.35)
    return (
        0.6
        - 0.12 * features.distance_to_goal
        + 1.1 * features.angle_to_goal
        + 0.45 * features.centrality
        + speed_bonus
        - 0.75 * pressure
        - 0.35 * goalkeeper_pressure
        - 0.08 * features.defender_count_in_cone
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
