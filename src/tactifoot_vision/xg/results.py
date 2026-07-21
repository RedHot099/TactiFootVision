from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tactifoot_vision.enums import XgModelKind
from tactifoot_vision.shots import ShotCandidate


@dataclass(frozen=True, slots=True)
class XgShotFeatures:
    shot_x: float
    shot_y: float
    distance_to_goal: float
    angle_to_goal: float
    centrality: float
    ball_speed: float | None = None
    nearest_player_distance: float | None = None
    goalkeeper_distance: float | None = None
    defender_count_in_cone: int = 0
    is_penalty: bool = False


@dataclass(frozen=True, slots=True)
class XgPrediction:
    candidate: ShotCandidate
    xg: float
    features: XgShotFeatures
    model_kind: XgModelKind
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VideoXgSummary:
    predictions: tuple[XgPrediction, ...]
    group_id: str | None = None

    @property
    def shot_count(self) -> int:
        return len(self.predictions)

    @property
    def total_xg(self) -> float:
        return float(sum(prediction.xg for prediction in self.predictions))
