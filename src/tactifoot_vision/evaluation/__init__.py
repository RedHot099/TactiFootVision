"""Backend-independent evaluation: detection mAP, keypoint error and StatsBomb comparison."""

from typing import Any

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.evaluation.detection import evaluate_detector
from tactifoot_vision.evaluation.keypoints import evaluate_keypoints
from tactifoot_vision.evaluation.metrics import DetectionMetrics, KeypointMetrics
from tactifoot_vision.evaluation.statsbomb import compare_with_statsbomb
from tactifoot_vision.models.base import Model


def evaluate(
    model: Model, dataset: Dataset, split: str = "valid", **kwargs: Any
) -> DetectionMetrics | KeypointMetrics:
    """Score ``model`` on a dataset split with the metrics matching its task.

    Detection models go to :func:`evaluate_detector`, keypoint models to
    :func:`evaluate_keypoints`; ``kwargs`` are passed through
    (``max_images=50``, ``conf=0.05``, ``pck_threshold=0.02``, ...).
    """
    if model.task is Task.DETECT:
        return evaluate_detector(model, dataset, split=split, **kwargs)
    if model.task is Task.POSE:
        return evaluate_keypoints(model, dataset, split=split, **kwargs)
    raise ValueError(f"No evaluation for {model.task.value} models")


__all__ = [
    "DetectionMetrics",
    "KeypointMetrics",
    "compare_with_statsbomb",
    "evaluate",
    "evaluate_detector",
    "evaluate_keypoints",
]
