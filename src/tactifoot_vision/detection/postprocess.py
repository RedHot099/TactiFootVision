from tactifoot_vision.domain import Detection, DetectionSet


def filter_detections(
    detections: DetectionSet,
    *,
    include_labels: tuple[str, ...] | None = None,
    per_class_confidence: dict[str, float] | None = None,
    default_confidence: float = 0.0,
) -> DetectionSet:
    kept: list[Detection] = []
    allowed = set(include_labels or ())
    for detection in detections:
        if allowed and detection.class_name not in allowed:
            continue
        threshold = (
            per_class_confidence.get(detection.class_name, default_confidence)
            if per_class_confidence
            else default_confidence
        )
        if detection.confidence is not None and detection.confidence < threshold:
            continue
        kept.append(detection)
    return DetectionSet(tuple(kept))
