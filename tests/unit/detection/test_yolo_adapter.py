from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tactifoot_vision.config import DetectionTrainingConfig
from tactifoot_vision.detection.adapters.yolo import YOLODetectionModel
from tactifoot_vision.domain import Frame, ModelArtifactNotFound


class _TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self):
        self.xyxy = _TensorLike([[1, 2, 11, 22], [3, 4, 13, 24]])
        self.conf = _TensorLike([0.9, 0.2])
        self.cls = _TensorLike([2, 0])

    def __len__(self):
        return 2


class _Result:
    boxes = _Boxes()


class _FakeYoloRuntime:
    def __init__(self):
        self.predict_calls = []
        self.train_calls = []

    def predict(self, *args, **kwargs):
        self.predict_calls.append((args, kwargs))
        return [_Result()]

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return SimpleNamespace(save_dir=Path("runs/test"))

    def val(self, **kwargs):
        return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5, "ignored": "x"})


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    return path


def test_from_weights_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFound):
        YOLODetectionModel.from_weights(tmp_path / "missing.pt")


def test_predict_forwards_thresholds_and_filters(tmp_path: Path) -> None:
    model = YOLODetectionModel.from_weights(_weights(tmp_path))
    runtime = _FakeYoloRuntime()
    model._model = runtime
    detector = model.as_detector(
        confidence=0.3,
        nms=0.4,
        include_labels=("player",),
        per_class_confidence={"player": 0.8},
    )

    result = detector.predict(
        Frame(index=0, image=np.zeros((10, 10, 3), dtype=np.uint8))
    )

    assert runtime.predict_calls[0][1]["conf"] == 0.3
    assert runtime.predict_calls[0][1]["iou"] == 0.4
    assert runtime.predict_calls[0][1]["verbose"] is False
    assert len(result) == 1
    assert result.detections[0].class_name == "player"


def test_train_forwards_expected_args(tmp_path: Path) -> None:
    model = YOLODetectionModel.from_weights(_weights(tmp_path))
    runtime = _FakeYoloRuntime()
    model._model = runtime

    run = model.train(
        DetectionTrainingConfig(
            data=tmp_path / "data.yaml",
            epochs=3,
            batch_size=2,
            image_size=320,
            learning_rate=0.01,
            output_dir=tmp_path / "runs",
            run_name="unit",
            device="cpu",
            early_stopping=True,
            early_stopping_patience=4,
        )
    )

    args = runtime.train_calls[0]
    assert args["data"] == str(tmp_path / "data.yaml")
    assert args["epochs"] == 3
    assert args["batch"] == 2
    assert args["imgsz"] == 320
    assert args["lr0"] == 0.01
    assert args["patience"] == 4
    assert run.model_name == "yolo"


def test_validate_returns_numeric_metrics(tmp_path: Path) -> None:
    model = YOLODetectionModel.from_weights(_weights(tmp_path))
    model._model = _FakeYoloRuntime()

    report = model.validate(tmp_path / "data.yaml")

    assert report.model_name == "yolo"
    assert report.metrics == {"metrics/mAP50(B)": 0.5}
