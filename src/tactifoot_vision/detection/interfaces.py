from pathlib import Path
from typing import Protocol

from tactifoot_vision.config.schemas import DetectionTrainingConfig
from tactifoot_vision.domain import DetectionSet, Frame, TrainingRun, ValidationReport


class Detector(Protocol):
    def predict(self, frame: Frame) -> DetectionSet:
        raise NotImplementedError


class TrainableDetectionModel(Protocol):
    def train(self, config: DetectionTrainingConfig) -> TrainingRun:
        raise NotImplementedError

    def validate(self, data: str | Path) -> ValidationReport:
        raise NotImplementedError

    def as_detector(self) -> Detector:
        raise NotImplementedError
