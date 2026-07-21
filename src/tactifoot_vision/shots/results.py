from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tactifoot_vision.enums import ShotDetectorKind, ShotOutcome


@dataclass(frozen=True, slots=True)
class ShotWindow:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError("ShotWindow end_frame must be >= start_frame.")

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True, slots=True)
class ShotCandidate:
    frame_index: int
    window: ShotWindow
    confidence: float
    detector_kind: ShotDetectorKind
    outcome: ShotOutcome = ShotOutcome.UNKNOWN
    data: Mapping[str, Any] = field(default_factory=dict)
