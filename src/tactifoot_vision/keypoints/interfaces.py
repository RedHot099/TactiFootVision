from typing import Protocol

from tactifoot_vision.domain import Frame
from tactifoot_vision.keypoints.results import KeypointSet


class KeypointDetector(Protocol):
    def predict(self, frame: Frame) -> KeypointSet:
        raise NotImplementedError
