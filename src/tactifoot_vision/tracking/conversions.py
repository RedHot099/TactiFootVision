from typing import Any

import numpy as np

from tactifoot_vision.domain import BBox, Detection, DetectionSet, Track, TrackSet


def detections_to_supervision(detections: DetectionSet) -> Any:
    import supervision as sv

    if len(detections) == 0:
        return sv.Detections.empty()
    xyxy = np.array(
        [[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2] for det in detections],
        dtype=np.float32,
    )
    confidence_values = [det.confidence for det in detections]
    confidence = (
        None
        if all(value is None for value in confidence_values)
        else np.array(
            [value if value is not None else 0.0 for value in confidence_values],
            dtype=np.float32,
        )
    )
    class_id = np.array([det.class_id for det in detections], dtype=int)
    data: dict[str, Any] = {
        "class_name": np.array([det.class_name for det in detections])
    }
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, data=data)


def supervision_to_tracks(detections: Any, fallback: DetectionSet) -> TrackSet:
    if len(detections) == 0:
        return TrackSet.empty()
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None or np.asarray(xyxy).ndim != 2 or np.asarray(xyxy).shape[1] != 4:
        raise ValueError("Invalid xyxy shape for tracked detections.")
    data = getattr(detections, "data", None) or {}
    names = data.get("class_name")
    tracks: list[Track] = []
    for index in range(len(detections)):
        fallback_detection = _fallback_detection(fallback, index)
        tracker_id = (
            int(detections.tracker_id[index])
            if getattr(detections, "tracker_id", None) is not None
            else index + 1
        )
        class_id = (
            int(detections.class_id[index])
            if getattr(detections, "class_id", None) is not None
            else fallback_detection.class_id
        )
        class_name = _class_name(names, index, fallback_detection, class_id)
        confidence = (
            float(detections.confidence[index])
            if getattr(detections, "confidence", None) is not None
            else fallback_detection.confidence
        )
        tracks.append(
            Track(
                track_id=tracker_id,
                bbox=BBox.from_xyxy(xyxy[index]),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
            )
        )
    return TrackSet(tuple(tracks))


def _fallback_detection(fallback: DetectionSet, index: int) -> Detection:
    if index < len(fallback.detections):
        return fallback.detections[index]
    return Detection(
        bbox=BBox(0.0, 0.0, 0.0, 0.0),
        class_id=-1,
        class_name="unknown_-1",
        confidence=None,
    )


def _class_name(
    names: Any,
    index: int,
    fallback_detection: Detection,
    class_id: int,
) -> str:
    if names is not None and index < len(names):
        return str(names[index])
    if fallback_detection.class_id == class_id:
        return fallback_detection.class_name
    return f"unknown_{class_id}"
