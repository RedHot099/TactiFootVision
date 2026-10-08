"""Keypoint localisation metrics computed the same way for every pose backend."""

import logging

import numpy as np
import pandas as pd

from tactifoot_vision.data.annotations import NOT_LABELLED, Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.evaluation.metrics import KeypointMetrics
from tactifoot_vision.models.base import Model

logger = logging.getLogger(__name__)


def evaluate_keypoints(
    model: Model,
    dataset: Dataset,
    split: str = "valid",
    pck_threshold: float = 0.05,
    max_images: int | None = None,
) -> KeypointMetrics:
    """Pixel error and PCK of a keypoint model on one dataset split.

    For every image with a ground-truth instance, the model's best instance
    (``xy[0]``) is compared with ground-truth instance 0 on the keypoints that
    are labelled (visibility > 0), whatever confidence the model gives them.
    Error and PCK are measured on images where the model found an instance;
    ``detection_rate`` reports how often that happened.

    Args:
        model: a keypoint model (``model.task == Task.POSE``).
        dataset: a pose dataset with the model's keypoint layout.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        pck_threshold: a keypoint is correct when its error is below this
            fraction of the image diagonal.
        max_images: use only the first ``max_images`` images of the split.
    """
    if model.task is not Task.POSE:
        raise ValueError(
            f"evaluate_keypoints needs a keypoint model, got a {model.task.value} model"
        )
    num_keypoints = dataset.num_keypoints
    if num_keypoints is None:
        raise ValueError(f"{dataset!r} has no keypoints")
    samples = [
        s
        for s in dataset[split][:max_images]
        if len(s.annotations) and s.annotations.keypoints is not None
    ]
    if not samples:
        raise ValueError(f"Split {split!r} of {dataset!r} has no labelled instances")

    # One row per image, NaN where a keypoint is unlabelled or nothing was found.
    errors = np.full((len(samples), num_keypoints), np.nan)
    diagonals = np.array([np.hypot(s.width, s.height) for s in samples])
    detected = np.zeros(len(samples), dtype=bool)
    for row, sample in enumerate(samples):
        prediction = model.predict(sample.read_image())
        if len(prediction) == 0:
            continue
        if prediction.xy.shape[1] != num_keypoints:
            raise ValueError(
                f"Model predicts {prediction.xy.shape[1]} keypoints, "
                f"the dataset has {num_keypoints}"
            )
        detected[row] = True
        truth = sample.annotations.keypoints[0]
        labelled = truth[:, 2] > NOT_LABELLED
        distance = np.linalg.norm(prediction.xy[0] - truth[:, :2], axis=1)
        errors[row, labelled] = distance[labelled]

    scored = ~np.isnan(errors)
    correct = scored & (errors < pck_threshold * diagonals[:, None])
    counts = scored.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_keypoint = pd.DataFrame(
            {
                "mean_error_px": np.where(scored, errors, 0).sum(axis=0) / counts,
                "pck": correct.sum(axis=0) / counts,
                "count": counts,
            },
            index=pd.RangeIndex(num_keypoints, name="keypoint"),
        )
    total = int(scored.sum())
    if total == 0:
        logger.warning(
            "No labelled keypoint could be scored on %s/%s", dataset.name, split
        )
    metrics = KeypointMetrics(
        mean_error_px=float(errors[scored].mean()) if total else np.nan,
        pck=float(correct.sum() / total) if total else np.nan,
        pck_threshold=pck_threshold,
        detection_rate=float(detected.mean()),
        per_keypoint=per_keypoint,
        num_images=len(samples),
    )
    logger.info(
        "%s on %s/%s (%d images): mean error %.1f px, PCK@%.2f %.3f, detected %.0f%%",
        type(model).__name__,
        dataset.name,
        split,
        len(samples),
        metrics.mean_error_px,
        pck_threshold,
        metrics.pck,
        100 * metrics.detection_rate,
    )
    return metrics
