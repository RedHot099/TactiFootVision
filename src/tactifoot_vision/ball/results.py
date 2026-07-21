from collections.abc import Iterator
from dataclasses import dataclass

from tactifoot_vision.enums import BallTrajectorySource


@dataclass(frozen=True, slots=True)
class BallTrajectoryPoint:
    frame_index: int
    image_x: float | None
    image_y: float | None
    pitch_x: float | None = None
    pitch_y: float | None = None
    confidence: float = 0.0
    source: BallTrajectorySource = BallTrajectorySource.MISSING
    uncertainty: float | None = None

    @property
    def has_image_position(self) -> bool:
        return self.image_x is not None and self.image_y is not None

    @property
    def has_pitch_position(self) -> bool:
        return self.pitch_x is not None and self.pitch_y is not None


@dataclass(frozen=True, slots=True)
class BallTrajectory:
    points: tuple[BallTrajectoryPoint, ...]

    def __post_init__(self) -> None:
        indexes = [point.frame_index for point in self.points]
        if len(indexes) != len(set(indexes)):
            raise ValueError("BallTrajectory cannot contain duplicate frame indexes.")

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[BallTrajectoryPoint]:
        return iter(self.points)

    @property
    def frame_indexes(self) -> tuple[int, ...]:
        return tuple(point.frame_index for point in self.points)

    @property
    def observed_count(self) -> int:
        return sum(
            1 for point in self.points if point.source == BallTrajectorySource.OBSERVED
        )

    def point_at(self, frame_index: int) -> BallTrajectoryPoint | None:
        return self.by_frame().get(frame_index)

    def by_frame(self) -> dict[int, BallTrajectoryPoint]:
        return {point.frame_index: point for point in self.points}

    def available_points(self) -> tuple[BallTrajectoryPoint, ...]:
        return tuple(point for point in self.points if point.has_image_position)
