from collections.abc import Iterator
from dataclasses import dataclass

from tactifoot_vision.domain import BBox


@dataclass(frozen=True, slots=True)
class Keypoint:
    index: int
    x: float
    y: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class KeypointSet:
    keypoints: tuple[Keypoint, ...] = ()
    pitch_bbox: BBox | None = None

    def __len__(self) -> int:
        return len(self.keypoints)

    def __iter__(self) -> Iterator[Keypoint]:
        return iter(self.keypoints)

    @classmethod
    def empty(cls) -> "KeypointSet":
        return cls(())
