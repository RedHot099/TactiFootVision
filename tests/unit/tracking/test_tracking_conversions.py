from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tactifoot_vision.domain import BBox, Detection, DetectionSet
from tactifoot_vision.tracking.conversions import (
    detections_to_supervision,
    supervision_to_tracks,
)


@dataclass
class FakeSupervisionDetections:
    xyxy: np.ndarray
    confidence: np.ndarray | None = None
    class_id: np.ndarray | None = None
    tracker_id: np.ndarray | None = None
    data: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.xyxy)


def test_empty_detection_set_converts_to_empty_supervision() -> None:
    result = detections_to_supervision(DetectionSet.empty())

    assert len(result) == 0


def test_detection_to_supervision_preserves_domain_fields() -> None:
    detections = DetectionSet(
        (
            Detection(
                bbox=BBox(1.0, 2.0, 5.0, 9.0),
                class_id=2,
                class_name="player",
                confidence=0.7,
            ),
        )
    )

    result = detections_to_supervision(detections)

    assert result.xyxy.tolist() == [[1.0, 2.0, 5.0, 9.0]]
    assert result.class_id.tolist() == [2]
    assert result.confidence.tolist() == pytest.approx([0.7])
    assert result.data["class_name"].tolist() == ["player"]


def test_missing_confidence_does_not_roundtrip_as_high_confidence() -> None:
    fallback = DetectionSet(
        (
            Detection(
                bbox=BBox(0.0, 0.0, 1.0, 1.0),
                class_id=3,
                class_name="referee",
                confidence=None,
            ),
        )
    )
    tracked = FakeSupervisionDetections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        class_id=np.array([3]),
        data={"class_name": np.array(["referee"])},
    )

    result = supervision_to_tracks(tracked, fallback)

    assert result.tracks[0].confidence is None


def test_missing_tracker_id_gets_deterministic_ids() -> None:
    tracked = FakeSupervisionDetections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
        class_id=np.array([0, 2]),
        data={"class_name": np.array(["ball", "player"])},
    )

    result = supervision_to_tracks(tracked, DetectionSet.empty())

    assert [track.track_id for track in result] == [1, 2]


def test_missing_class_names_fall_back_safely() -> None:
    tracked = FakeSupervisionDetections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]),
        class_id=np.array([99]),
    )

    result = supervision_to_tracks(tracked, DetectionSet.empty())

    assert result.tracks[0].class_id == 99
    assert result.tracks[0].class_name == "unknown_99"


def test_shorter_fallback_does_not_crash() -> None:
    tracked = FakeSupervisionDetections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
        tracker_id=np.array([10, 11]),
    )

    result = supervision_to_tracks(tracked, DetectionSet.empty())

    assert [track.track_id for track in result] == [10, 11]
    assert [track.class_name for track in result] == ["unknown_-1", "unknown_-1"]


def test_invalid_xyxy_shape_raises_named_error() -> None:
    tracked = FakeSupervisionDetections(xyxy=np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(ValueError, match="xyxy"):
        supervision_to_tracks(tracked, DetectionSet.empty())
