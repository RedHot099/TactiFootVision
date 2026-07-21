from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tactifoot_vision.domain import Frame, ModelArtifactNotFound
from tactifoot_vision.keypoints.adapters.yolo_pose import YOLOPoseKeypointModel


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeKeypoints:
    def __init__(self, array: np.ndarray) -> None:
        self.data = FakeTensor(array)


class FakeBoxes:
    def __init__(self, xyxy: np.ndarray) -> None:
        self.xyxy = FakeTensor(xyxy)


class FakeRuntime:
    def __init__(self, result: object) -> None:
        self.result = result

    def predict(self, image, conf, verbose):
        self.call = (image, conf, verbose)
        return [self.result]


class FakeResult:
    def __init__(self, keypoints=None, boxes=None) -> None:
        self.keypoints = keypoints
        self.boxes = boxes


def test_yolo_pose_missing_weights_raise(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFound):
        YOLOPoseKeypointModel.from_weights(tmp_path / "missing.pt")


def test_yolo_pose_can_download_known_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_urls = []

    def fake_urlretrieve(url: str, filename: str | Path, *args: Any) -> None:
        requested_urls.append(url)
        Path(filename).write_bytes(b"weights")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    weights = tmp_path / "models" / "yolov8n-pose.pt"
    model = YOLOPoseKeypointModel.from_weights(weights)

    assert weights.read_bytes() == b"weights"
    assert model.weights == weights
    assert requested_urls


def test_yolo_pose_auto_download_can_be_disabled(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFound):
        YOLOPoseKeypointModel.from_weights(
            tmp_path / "yolov8n-pose.pt", auto_download=False
        )


def test_yolo_pose_converts_result(tmp_path: Path) -> None:
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"weights")
    model = YOLOPoseKeypointModel.from_weights(weights, confidence=0.4)
    runtime = FakeRuntime(
        FakeResult(
            keypoints=FakeKeypoints(np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]])),
            boxes=FakeBoxes(np.array([[0.0, 0.0, 10.0, 20.0]])),
        )
    )
    model._model = runtime

    result = model.predict(Frame(index=0, image=np.zeros((5, 5, 3), dtype=np.uint8)))

    assert len(result) == 2
    assert result.keypoints[0].confidence == pytest.approx(0.9)
    assert result.pitch_bbox is not None
    assert runtime.call[1] == 0.4


def test_yolo_pose_empty_result_returns_empty(tmp_path: Path) -> None:
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"weights")
    model = YOLOPoseKeypointModel.from_weights(weights)
    model._model = FakeRuntime(FakeResult(keypoints=None))

    result = model.predict(Frame(index=0, image=np.zeros((5, 5, 3), dtype=np.uint8)))

    assert len(result) == 0
