"""Box detection metrics computed the same way for every detector backend."""

import logging
from collections.abc import Mapping, Sequence

import cv2
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
    operating_conf: float | None = None,
    class_conf: Mapping[str, float] | None = None,
    match_iou: float = 0.5,
    input_size: tuple[int, int] | None = None,
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
        operating_conf: confidence threshold of the precision and recall
            columns (the pipeline's operating point); ``None`` uses the
            model's own ``conf``, the one it was loaded with. Must be >= ``conf``.
        class_conf: per-class thresholds (by class name) replacing
            ``operating_conf`` for those classes, e.g. ``{"ball": 0.15}``.
        match_iou: IoU at which a kept prediction matches a ground-truth box
            for precision and recall (greedy, most confident first, per class).
        input_size: ``(width, height)`` the image is resized to (area
            interpolation) before the model sees it, e.g. ``(1280, 720)`` to
            score 1080p frames as 720p video; boxes are scaled back to the
            labels' pixels.

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

    thresholds = _operating_thresholds(
        dataset.class_names,
        model.conf if operating_conf is None else operating_conf,
        class_conf or {},
        conf,
    )
    metric = MeanAveragePrecision()
    counts = np.zeros((len(dataset.class_names), 3), dtype=np.int64)  # tp, fp, fn
    warned: set[str] = set()  # unknown class names already reported
    previous_conf = model.conf
    model.conf = conf
    try:
        for sample in samples:
            image = sample.read_image()
            if input_size is not None:
                image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
            predictions = _to_dataset_classes(
                model.predict(image), dataset.class_names, warned
            )
            if input_size is not None and len(predictions):
                scale = [sample.width / input_size[0], sample.height / input_size[1]]
                predictions.xyxy = predictions.xyxy * np.float32(scale * 2)
            targets = sample.annotations.to_detections()
            metric.update(predictions, targets)
            counts += _match_counts(predictions, targets, thresholds, match_iou)
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
        tp, fp, fn = counts[class_id]
        rows.append(
            {
                "class_name": name,
                "map50_95": float(ap.mean()) if ap is not None else np.nan,
                "map50": float(ap[0]) if ap is not None else np.nan,
                "map75": float(ap[5]) if ap is not None else np.nan,
                "instances": int(instances[class_id]),
                "conf": float(thresholds[class_id]),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
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


def _operating_thresholds(
    class_names: Sequence[str],
    operating_conf: float,
    class_conf: Mapping[str, float],
    conf: float,
) -> np.ndarray:
    """Per-class confidence threshold of the precision and recall columns."""
    unknown = sorted(set(class_conf) - set(class_names))
    if unknown:
        raise ValueError(
            f"class_conf names unknown class(es) {', '.join(unknown)}; "
            f"the dataset has {', '.join(class_names)}"
        )
    thresholds = np.array(
        [class_conf.get(name, operating_conf) for name in class_names], dtype=float
    )
    if (thresholds < conf).any():
        raise ValueError(
            f"Operating thresholds {thresholds.tolist()} must be >= conf={conf}, "
            "the evaluation threshold: lower conf"
        )
    return thresholds


def _match_counts(
    predictions: sv.Detections,
    targets: sv.Detections,
    thresholds: np.ndarray,
    match_iou: float,
) -> np.ndarray:
    """``(C, 3)`` true positives, false positives and missed targets per class.

    Predictions under their class's threshold are dropped; the rest are
    matched greedily, most confident first, to the unmatched target of their
    class with the highest IoU, if that IoU is >= ``match_iou``.
    """
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for class_id, threshold in enumerate(thresholds):
        target = targets.xyxy[targets.class_id == class_id]
        keep = (predictions.class_id == class_id) & (
            predictions.confidence >= threshold
        )
        order = np.argsort(-predictions.confidence[keep], kind="stable")
        boxes = predictions.xyxy[keep][order]
        tp = 0
        if len(boxes) and len(target):
            free = np.ones(len(target), dtype=bool)
            for row in sv.box_iou_batch(boxes, target):
                candidates = np.where(free, row, -1.0)
                best = int(np.argmax(candidates))
                if candidates[best] >= match_iou:
                    free[best] = False
                    tp += 1
        counts[class_id] = (tp, len(boxes) - tp, len(target) - tp)
    return counts


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _to_dataset_classes(
    predictions: sv.Detections, class_names: Sequence[str], warned: set[str]
) -> sv.Detections:
    """Re-index predictions to ``class_names`` by name, dropping unknown classes.

    Unknown names are logged once, then added to ``warned``.
    """
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
        unknown = sorted({str(n) for n in np.asarray(names)[~keep]} - warned)
        if unknown:
            warned.update(unknown)
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
