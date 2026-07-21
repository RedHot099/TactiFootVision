from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import BBox, Detection, DetectionSet


def detections_from_arrays(
    *,
    xyxy: NDArray[np.floating[Any]],
    confidence: NDArray[np.floating[Any]] | None,
    class_id: NDArray[np.integer[Any]],
    id_to_name: Mapping[int, str],
) -> DetectionSet:
    boxes = np.asarray(xyxy, dtype=np.float32)
    class_ids = np.asarray(class_id, dtype=int)
    confidences = (
        None if confidence is None else np.asarray(confidence, dtype=np.float32)
    )

    if boxes.size == 0:
        if boxes.reshape(-1, 4).shape[0] == 0 and class_ids.size == 0:
            return DetectionSet.empty()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("xyxy must have shape (N, 4)")
    if class_ids.ndim != 1:
        raise ValueError("class_id must have shape (N,)")
    if len(boxes) != len(class_ids):
        raise ValueError("xyxy and class_id lengths must match")
    if confidences is not None:
        if confidences.ndim != 1:
            raise ValueError("confidence must have shape (N,)")
        if len(confidences) != len(boxes):
            raise ValueError("confidence and xyxy lengths must match")

    detections: list[Detection] = []
    for index, box in enumerate(boxes):
        current_class_id = int(class_ids[index])
        detections.append(
            Detection(
                bbox=BBox.from_xyxy(box),
                class_id=current_class_id,
                class_name=id_to_name.get(
                    current_class_id, f"unknown_{current_class_id}"
                ),
                confidence=(None if confidences is None else float(confidences[index])),
            )
        )
    return DetectionSet(tuple(detections))


def detections_from_ultralytics_result(
    result: Any, id_to_name: Mapping[int, str]
) -> DetectionSet:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return DetectionSet.empty()
    xyxy = _to_numpy(boxes.xyxy)
    class_id = _to_numpy(boxes.cls).astype(int)
    confidence = None if getattr(boxes, "conf", None) is None else _to_numpy(boxes.conf)
    return detections_from_arrays(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        id_to_name=id_to_name,
    )


def detections_from_supervision(
    value: Any, id_to_name: Mapping[int, str]
) -> DetectionSet:
    if len(value) == 0:
        return DetectionSet.empty()
    xyxy = np.asarray(value.xyxy, dtype=np.float32)
    raw_class_id = getattr(value, "class_id", None)
    if raw_class_id is None:
        raise ValueError("class_id is required for supervision detections")
    raw_confidence = getattr(value, "confidence", None)
    return detections_from_arrays(
        xyxy=xyxy,
        confidence=(
            None
            if raw_confidence is None
            else np.asarray(raw_confidence, dtype=np.float32)
        ),
        class_id=np.asarray(raw_class_id, dtype=int),
        id_to_name=id_to_name,
    )


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)
