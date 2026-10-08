"""The interface shared by multi-object trackers, plus its registry."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import supervision as sv

from tactifoot_vision.registry import Registry

TRACKERS: "Registry[Tracker]" = Registry("tracker")


class Tracker(ABC):
    """Assigns stable ``tracker_id`` values to detections across frames."""

    name: ClassVar[str]

    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Consume one frame's detections; return the tracked subset with ``tracker_id`` set.

        ``data`` entries (e.g. ``class_name``) must survive tracking.
        """

    @abstractmethod
    def reset(self, fps: float | None = None) -> None:
        """Forget all tracks before a new video; ``fps`` lets motion models adapt."""


def create_tracker(name: str, **kwargs: Any) -> Tracker:
    """Create a registered tracker, e.g. ``create_tracker("bytetrack")``."""
    return TRACKERS.create(name, **kwargs)


def available_trackers() -> list[str]:
    return TRACKERS.names()
