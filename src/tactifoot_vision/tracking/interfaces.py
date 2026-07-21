from typing import Protocol

from tactifoot_vision.domain import DetectionSet, Frame, TrackSet


class Tracker(Protocol):
    def update(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
