"""Backend-independent evaluation: detection mAP, keypoint error and StatsBomb comparison."""

import inspect
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
    (``max_images=50``, ``conf=0.05``, ``pck_threshold=0.02``, ...) and an
    option the evaluator does not take is a ``ValueError``.
    """
    evaluators = {Task.DETECT: evaluate_detector, Task.POSE: evaluate_keypoints}
    evaluator = evaluators.get(model.task)
    if evaluator is None:
        raise ValueError(f"No evaluation for {model.task.value} models")
    parameters = inspect.signature(evaluator).parameters.values()
    options = [
        p.name for p in parameters if p.name not in ("model", "dataset", "split")
    ]
    open_ended = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    unknown = sorted(set(kwargs) - set(options))
    if unknown and not open_ended:
        raise ValueError(
            f"Unknown evaluation option(s) {', '.join(unknown)} for "
            f"{evaluator.__name__}; valid options: {', '.join(options)}"
        )
    return evaluator(model, dataset, split=split, **kwargs)


__all__ = [
    "DetectionMetrics",
    "KeypointMetrics",
    "compare_with_statsbomb",
    "evaluate",
    "evaluate_detector",
    "evaluate_keypoints",
]
