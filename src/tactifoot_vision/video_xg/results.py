from dataclasses import dataclass
from pathlib import Path

from tactifoot_vision.enums import XgModelKind


@dataclass(frozen=True, slots=True)
class VideoShotFeatures:
    shot_id: str
    frame_index: int
    shot_x: float
    shot_y: float
    goal_x: float = 105.0
    goal_y: float = 34.0
    nearest_player_distance: float | None = None
    goalkeeper_distance: float | None = None
    defender_count_in_cone: int = 0
    ball_speed: float | None = None
    ball_direction_to_goal: float | None = None
    shot_confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class VideoOnlyShotPrediction:
    shot_id: str
    frame_index: int
    xg: float
    model_kind: XgModelKind
    features: VideoShotFeatures


@dataclass(frozen=True, slots=True)
class VideoOnlyXgSummary:
    predictions: tuple[VideoOnlyShotPrediction, ...]
    group_id: str | None = None

    @property
    def shot_count(self) -> int:
        return len(self.predictions)

    @property
    def total_xg(self) -> float:
        return float(sum(prediction.xg for prediction in self.predictions))


@dataclass(frozen=True, slots=True)
class VideoTimelineSegment:
    part_index: int
    path: Path
    start_seconds: float
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VideoShotCandidate:
    shot_id: str
    global_frame_index: int
    global_seconds: float
    part_index: int
    part_frame_index: int
    score: float
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class VideoShotEvent:
    shot_id: str
    global_frame_index: int
    global_seconds: float
    part_index: int
    part_frame_index: int
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class VideoOnlyXgRunResult:
    output_dir: Path
    artifacts: tuple[Path, ...]
    metrics: dict[str, float]
