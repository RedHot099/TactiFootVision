from pathlib import Path
from typing import Any

from tactifoot_vision.config.schemas import DetectionTrainingConfig
from tactifoot_vision.detection.conversions import detections_from_ultralytics_result
from tactifoot_vision.detection.postprocess import filter_detections
from tactifoot_vision.domain import (
    DetectionSet,
    Frame,
    ModelArtifactNotFound,
    TrainingRun,
    ValidationReport,
)


class YOLODetectionModel:
    def __init__(self, weights: Path, *, task: str = "detect") -> None:
        self.weights = Path(weights)
        self.task = task
        self._model: Any | None = None
        if not self.weights.is_file():
            raise ModelArtifactNotFound(f"YOLO weights not found: {self.weights}")

    @classmethod
    def from_weights(cls, weights: str | Path) -> "YOLODetectionModel":
        return cls(Path(weights))

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights, task=self.task)
        return self._model

    def train(self, config: DetectionTrainingConfig) -> TrainingRun:
        model = self._load()
        args: dict[str, object] = {
            "data": str(config.data),
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.image_size,
            "project": str(config.output_dir) if config.output_dir else None,
            "name": config.run_name,
            "device": config.device,
        }
        if config.learning_rate is not None:
            args["lr0"] = config.learning_rate
        if config.early_stopping and config.early_stopping_patience is not None:
            args["patience"] = config.early_stopping_patience
        result = model.train(
            **{key: value for key, value in args.items() if value is not None}
        )
        save_dir = Path(getattr(result, "save_dir", config.output_dir or "."))
        best = save_dir / "weights" / "best.pt"
        return TrainingRun(
            model_name="yolo",
            output_dir=save_dir,
            best_checkpoint=best if best.exists() else None,
        )

    def validate(self, data: str | Path) -> ValidationReport:
        result = self._load().val(data=str(data))
        metrics = getattr(result, "results_dict", {}) or {}
        return ValidationReport(
            model_name="yolo",
            metrics={
                str(key): float(value)
                for key, value in metrics.items()
                if _is_number(value)
            },
        )

    def as_detector(self, **kwargs: Any) -> "YOLODetector":
        return YOLODetector(self, **kwargs)


class YOLODetector:
    def __init__(
        self,
        model: YOLODetectionModel,
        *,
        confidence: float = 0.3,
        nms: float = 0.5,
        classes: dict[str, int] | None = None,
        include_labels: tuple[str, ...] | None = None,
        per_class_confidence: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.confidence = confidence
        self.nms = nms
        self.classes = classes or {
            "ball": 0,
            "goalkeeper": 1,
            "player": 2,
            "referee": 3,
        }
        self.id_to_name = {value: key for key, value in self.classes.items()}
        self.include_labels = include_labels
        self.per_class_confidence = per_class_confidence

    def predict(self, frame: Frame) -> DetectionSet:
        result = self.model._load().predict(  # noqa: SLF001
            frame.image,
            conf=self.confidence,
            iou=self.nms,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionSet.empty()
        detections = detections_from_ultralytics_result(result, self.id_to_name)
        return filter_detections(
            detections,
            include_labels=self.include_labels,
            per_class_confidence=self.per_class_confidence,
            default_confidence=self.confidence,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float)
