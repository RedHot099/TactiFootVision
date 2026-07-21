from dataclasses import dataclass
from pathlib import Path

from tactifoot_vision.enums import DetectionBackend


@dataclass(frozen=True, slots=True)
class DetectionModelInfo:
    backend: DetectionBackend
    weights: Path
    task: str | None = None
