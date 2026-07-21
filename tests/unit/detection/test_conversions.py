from pathlib import Path

import numpy as np

from tactifoot_vision.detection import (
    DetectionModelInfo,
    FakeDetector,
    RFDETRDetectionModel,
    RFDETRSegDetectionModel,
    YOLODetectionModel,
)
from tactifoot_vision.detection.conversions import (
    detections_from_arrays,
    detections_from_supervision,
    detections_from_ultralytics_result,
)
from tactifoot_vision.domain import DetectionSet
from tactifoot_vision.enums import DetectionBackend


class _TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _UltralyticsBoxes:
    def __init__(self):
        self.xyxy = _TensorLike([[1, 2, 11, 22]])
        self.conf = _TensorLike([0.75])
        self.cls = _TensorLike([99])

    def __len__(self):
        return 1


class _UltralyticsResult:
    boxes = _UltralyticsBoxes()


class _SupervisionLike:
    xyxy = np.array([[1, 2, 11, 22]], dtype=np.float32)
    confidence = None
    class_id = np.array([2], dtype=int)

    def __len__(self):
        return 1


def test_public_detection_imports_are_stable() -> None:
    assert YOLODetectionModel is not None
    assert RFDETRDetectionModel is not None
    assert RFDETRSegDetectionModel is not None
    assert FakeDetector is not None
    assert (
        DetectionModelInfo(
            backend=DetectionBackend.YOLO,
            weights=Path("models/yolo11m.pt"),
        ).backend
        is DetectionBackend.YOLO
    )


def test_empty_arrays_return_empty_detection_set() -> None:
    result = detections_from_arrays(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=None,
        class_id=np.empty((0,), dtype=int),
        id_to_name={2: "player"},
    )

    assert result == DetectionSet.empty()


def test_arrays_convert_to_detection_set() -> None:
    result = detections_from_arrays(
        xyxy=np.array([[1, 2, 11, 22]], dtype=np.float32),
        confidence=np.array([0.5], dtype=np.float32),
        class_id=np.array([2], dtype=int),
        id_to_name={2: "player"},
    )

    detection = result.detections[0]
    assert detection.class_name == "player"
    assert detection.confidence == np.float32(0.5)
    assert detection.bbox.xywh == (1.0, 2.0, 10.0, 20.0)


def test_unknown_class_id_uses_unknown_name() -> None:
    result = detections_from_arrays(
        xyxy=np.array([[1, 2, 11, 22]], dtype=np.float32),
        confidence=np.array([0.5], dtype=np.float32),
        class_id=np.array([99], dtype=int),
        id_to_name={2: "player"},
    )

    assert result.detections[0].class_name == "unknown_99"


def test_mismatched_lengths_raise_value_error() -> None:
    try:
        detections_from_arrays(
            xyxy=np.array([[1, 2, 11, 22]], dtype=np.float32),
            confidence=np.array([0.5], dtype=np.float32),
            class_id=np.array([2, 3], dtype=int),
            id_to_name={2: "player"},
        )
    except ValueError as exc:
        assert "xyxy and class_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_ultralytics_like_result_conversion() -> None:
    result = detections_from_ultralytics_result(
        _UltralyticsResult(), id_to_name={2: "player"}
    )

    assert result.detections[0].class_name == "unknown_99"
    assert result.detections[0].confidence == np.float32(0.75)


def test_supervision_like_result_conversion_with_missing_confidence() -> None:
    result = detections_from_supervision(_SupervisionLike(), id_to_name={2: "player"})

    assert result.detections[0].class_name == "player"
    assert result.detections[0].confidence is None
