"""Box detection metrics computed the same way for every detector backend."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.evaluation.metrics import DetectionMetrics
from tactifoot_vision.models.base import Model

logger = logging.getLogger(__name__)


def evaluate_detector(
    model: Model,
    dataset: Dataset,
    split: str = "valid",
    conf: float = 0.01,
    max_images: int | None = None,
) -> DetectionMetrics:
    """COCO-style mAP of a detector on one dataset split (``supervision.metrics``).

    Predictions are matched to the dataset's classes by name
    (``data["class_name"]``), so models with a different class order score
    correctly; predicted classes the dataset does not have are ignored. The
    model runs with confidence threshold ``conf`` (restored afterwards) because
    mAP needs the low-confidence tail of the precision-recall curve.

    Args:
        model: a detection model (``model.task == Task.DETECT``).
        dataset: labelled images; its ``class_names`` define the classes scored.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        conf: confidence threshold used during evaluation.
        max_images: score only the first ``max_images`` images of the split.

    Note:
        Like ``supervision``'s metric, images without ground-truth objects do
        not contribute, so false positives on empty images are not penalised.
    """
    from supervision.metrics import MeanAveragePrecision

    if model.task is not Task.DETECT:
        raise ValueError(
            f"evaluate_detector needs a detection model, got a {model.task.value} model"
        )
    samples = dataset[split][:max_images]
    if not samples:
        raise ValueError(f"Split {split!r} of {dataset!r} has no images")

    metric = MeanAveragePrecision()
    previous_conf = model.conf
    model.conf = conf
    try:
        for sample in samples:
            predictions = model.predict(sample.read_image())
            metric.update(
                _to_dataset_classes(predictions, dataset.class_names),
                sample.annotations.to_detections(),
            )
    finally:
        model.conf = previous_conf
    result = metric.compute()

    class_ids = np.concatenate([s.annotations.class_ids for s in samples])
    instances = np.bincount(class_ids, minlength=len(dataset.class_names))
    ap_by_class = dict(
        zip(result.matched_classes.tolist(), result.ap_per_class, strict=True)
    )
    rows = []
    for class_id, name in enumerate(dataset.class_names):
        ap = ap_by_class.get(class_id)
        rows.append(
            {
                "class_name": name,
                "map50_95": float(ap.mean()) if ap is not None else np.nan,
                "map50": float(ap[0]) if ap is not None else np.nan,
                "map75": float(ap[5]) if ap is not None else np.nan,
                "instances": int(instances[class_id]),
            }
        )
    metrics = DetectionMetrics(
        map50_95=float(result.map50_95),
        map50=float(result.map50),
        map75=float(result.map75),
        per_class=pd.DataFrame(rows).set_index("class_name"),
        num_images=len(samples),
    )
    logger.info(
        "%s on %s/%s (%d images): mAP50-95 %.4f, mAP50 %.4f",
        type(model).__name__,
        dataset.name,
        split,
        len(samples),
        metrics.map50_95,
        metrics.map50,
    )
    return metrics


_warned: set[str] = set()  # unknown class names already reported


def _to_dataset_classes(
    predictions: sv.Detections, class_names: Sequence[str]
) -> sv.Detections:
    """Re-index predictions to ``class_names`` by name, dropping unknown classes."""
    if len(predictions) == 0:
        return sv.Detections.empty()
    names = predictions.data.get("class_name")
    if names is None:
        raise ValueError("Detector predictions must carry data['class_name']")
    if predictions.confidence is None:
        raise ValueError("Detector predictions must carry confidence scores")
    index = {name: i for i, name in enumerate(class_names)}
    class_id = np.array([index.get(str(name), -1) for name in names], dtype=int)
    keep = class_id >= 0
    if not keep.all():
        unknown = sorted({str(n) for n in np.asarray(names)[~keep]} - _warned)
        if unknown:
            _warned.update(unknown)
            logger.warning(
                "Ignoring predicted classes that the dataset does not have: %s (dataset: %s)",
                ", ".join(unknown),
                ", ".join(class_names),
            )
    return sv.Detections(
        xyxy=np.asarray(predictions.xyxy, dtype=np.float32)[keep],
        confidence=np.asarray(predictions.confidence, dtype=np.float32)[keep],
        class_id=class_id[keep],
    )
