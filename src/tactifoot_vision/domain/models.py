from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    image: NDArray[np.uint8]
    timestamp_seconds: float | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_xyxy(cls, values: Iterable[float]) -> "BBox":
        x1, y1, x2, y2 = [float(value) for value in values]
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DetectionSet:
    detections: tuple[Detection, ...] = ()

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    @classmethod
    def empty(cls) -> "DetectionSet":
        return cls(())


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float | None = None
    team_id: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackSet:
    tracks: tuple[Track, ...] = ()

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self.tracks)

    @classmethod
    def empty(cls) -> "TrackSet":
        return cls(())


@dataclass(frozen=True, slots=True)
class PitchPoint:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PitchProjection:
    status: str
    points_by_track_id: Mapping[int, PitchPoint] = field(default_factory=dict)
    ball: PitchPoint | None = None
    homography: NDArray[np.floating[Any]] | None = None
    visible_area: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def unavailable(cls) -> "PitchProjection":
        return cls(status="unavailable")


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_seconds: float | None
    detections: DetectionSet
    tracks: TrackSet
    projection: PitchProjection | None = None


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    path: Path
    format: str
    rows: int | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    frames: tuple[FrameResult, ...]
    artifacts: tuple[ExportArtifact, ...] = ()

    def __post_init__(self) -> None:
        indexes = [frame.frame_index for frame in self.frames]
        if len(indexes) != len(set(indexes)):
            raise ValueError("PipelineResult cannot contain duplicate frame indexes.")

    def to_csv(self, path: str | Path) -> ExportArtifact:
        from tactifoot_vision.export.pipeline_csv import PipelineCsvExporter

        return PipelineCsvExporter().write(self, Path(path))

    def to_mot(self, path: str | Path) -> ExportArtifact:
        from tactifoot_vision.export.mot import MotExporter

        return MotExporter().write(self, Path(path))


@dataclass(frozen=True, slots=True)
class TrainingRun:
    model_name: str
    output_dir: Path | None = None
    best_checkpoint: Path | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    model_name: str
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    name: str
    artifacts: tuple[ExportArtifact, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
