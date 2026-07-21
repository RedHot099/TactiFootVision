import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tactifoot_vision.domain import BBox, Detection, DetectionSet, Frame, TrackSet
from tactifoot_vision.tracking.adapters.bytetrack import ByteTrackTracker


class FakeDetections:
    @classmethod
    def empty(cls) -> "FakeDetections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=None,
            class_id=None,
            tracker_id=None,
            data={},
        )

    def __init__(
        self,
        *,
        xyxy: np.ndarray,
        confidence: np.ndarray | None = None,
        class_id: np.ndarray | None = None,
        tracker_id: np.ndarray | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = data or {}

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeByteTrack:
    instances: list["FakeByteTrack"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[FakeDetections] = []
        FakeByteTrack.instances.append(self)

    def update_with_detections(self, detections: FakeDetections) -> FakeDetections:
        self.calls.append(detections)
        if len(detections) == 0:
            return FakeDetections.empty()
        return FakeDetections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.arange(10, 10 + len(detections)),
            data=detections.data,
        )


@pytest.fixture(autouse=True)
def fake_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeByteTrack.instances.clear()
    module = ModuleType("supervision")
    module.ByteTrack = FakeByteTrack
    module.Detections = FakeDetections
    monkeypatch.setitem(sys.modules, "supervision", module)


def test_constructor_forwards_bytetrack_kwargs() -> None:
    ByteTrackTracker(
        frame_rate=30,
        track_activation_threshold=0.4,
        lost_track_buffer=12,
        minimum_matching_threshold=0.6,
        minimum_consecutive_frames=3,
    )

    assert FakeByteTrack.instances[0].kwargs == {
        "track_activation_threshold": 0.4,
        "lost_track_buffer": 12,
        "minimum_matching_threshold": 0.6,
        "frame_rate": 30,
        "minimum_consecutive_frames": 3,
    }


def test_update_calls_supervision_and_returns_tracks() -> None:
    tracker = ByteTrackTracker()
    frame = Frame(index=0, image=np.zeros((8, 8, 3), dtype=np.uint8))
    detections = DetectionSet(
        (
            Detection(
                bbox=BBox(1.0, 2.0, 5.0, 7.0),
                class_id=2,
                class_name="player",
                confidence=0.8,
            ),
        )
    )

    result = tracker.update(frame, detections)

    assert isinstance(result, TrackSet)
    assert len(FakeByteTrack.instances[0].calls) == 1
    assert result.tracks[0].track_id == 10
    assert result.tracks[0].class_name == "player"


def test_empty_detections_return_empty_tracks() -> None:
    tracker = ByteTrackTracker()
    frame = Frame(index=0, image=np.zeros((8, 8, 3), dtype=np.uint8))

    result = tracker.update(frame, DetectionSet.empty())

    assert len(result) == 0


def test_reset_recreates_tracker_with_same_kwargs() -> None:
    tracker = ByteTrackTracker(frame_rate=50)
    first = FakeByteTrack.instances[0]

    tracker.reset()

    assert FakeByteTrack.instances[0] is first
    assert FakeByteTrack.instances[1] is not first
    assert FakeByteTrack.instances[1].kwargs["frame_rate"] == 50
