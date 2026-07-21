from pathlib import Path
from typing import Any

from PIL import Image

from tactifoot_vision.config.schemas import DetectionTrainingConfig
from tactifoot_vision.detection.conversions import detections_from_supervision
from tactifoot_vision.detection.postprocess import filter_detections
from tactifoot_vision.detection.rfdetr_training import (
    build_rfdetr_train_args,
    copy_best_checkpoint_if_requested,
    find_best_rfdetr_checkpoint,
)
from tactifoot_vision.domain import (
    DetectionSet,
    Frame,
    ModelArtifactNotFound,
    TrainingRun,
    ValidationReport,
)


class RFDETRDetectionModel:
    model_name = "rfdetr"

    def __init__(self, weights: Path) -> None:
        self.weights = Path(weights)
        self._model: Any | None = None
        if not self.weights.is_file():
            raise ModelArtifactNotFound(f"RF-DETR weights not found: {self.weights}")

    @classmethod
    def from_weights(cls, weights: str | Path) -> "RFDETRDetectionModel":
        return cls(Path(weights))

    def _model_class(self) -> type[Any]:
        from rfdetr import RFDETRBase

        return RFDETRBase

    def _load(self) -> Any:
        if self._model is None:
            model_class = self._model_class()
            self._model = model_class(pretrain_weights=str(self.weights))
        return self._model

    def train(self, config: DetectionTrainingConfig) -> TrainingRun:
        output_dir = config.output_dir or Path("output") / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self._load().train(**build_rfdetr_train_args(config, output_dir))
        best = find_best_rfdetr_checkpoint(output_dir)
        if config.save_checkpoint_path is not None:
            best = copy_best_checkpoint_if_requested(
                best_checkpoint=best,
                destination=config.save_checkpoint_path,
            )
        return TrainingRun(
            model_name=self.model_name,
            output_dir=output_dir,
            best_checkpoint=best,
        )

    def validate(self, data: str | Path) -> ValidationReport:
        model = self._load()
        if hasattr(model, "val"):
            result = model.val(str(data))
        elif hasattr(model, "validate"):
            result = model.validate(str(data))
        else:
            return ValidationReport(
                model_name=self.model_name,
                metrics={"validation_supported": 0.0},
            )
        metrics = getattr(result, "results_dict", result)
        if not isinstance(metrics, dict):
            return ValidationReport(
                model_name=self.model_name,
                metrics={"validation_supported": 1.0},
            )
        return ValidationReport(
            model_name=self.model_name,
            metrics={
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, int | float)
            },
        )

    def as_detector(self, **kwargs: Any) -> "RFDETRDetector":
        return RFDETRDetector(self, **kwargs)


class RFDETRDetector:
    def __init__(
        self,
        model: RFDETRDetectionModel,
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
        import cv2

        image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        output = self.model._load().predict(
            Image.fromarray(image_rgb), threshold=self.confidence
        )  # noqa: SLF001
        detections = detections_from_supervision(output, self.id_to_name)
        return filter_detections(
            detections,
            include_labels=self.include_labels,
            per_class_confidence=self.per_class_confidence,
            default_confidence=self.confidence,
        )
