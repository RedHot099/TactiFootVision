import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tactifoot_vision.config import DetectionTrainingConfig
from tactifoot_vision.detection.adapters.rfdetr import RFDETRDetectionModel
from tactifoot_vision.detection.adapters.rfdetr_seg import RFDETRSegDetectionModel
from tactifoot_vision.detection.rfdetr_training import (
    build_rfdetr_train_args,
    copy_best_checkpoint_if_requested,
    find_best_rfdetr_checkpoint,
    prepare_rfdetr_dataset,
)
from tactifoot_vision.domain import AdapterUnavailable, Frame, ModelArtifactNotFound
from tactifoot_vision.enums import DatasetFormat, DatasetSource


class _SupervisionLike:
    xyxy = np.array([[1, 2, 11, 22]], dtype=np.float32)
    confidence = np.array([0.7], dtype=np.float32)
    class_id = np.array([2], dtype=int)

    def __len__(self):
        return 1


class _FakeRFDETRRuntime:
    def __init__(self):
        self.train_calls = []

    def predict(self, image, threshold):
        self.predict_call = (image, threshold)
        return _SupervisionLike()

    def train(self, **kwargs):
        self.train_calls.append(kwargs)

    def val(self, data):
        return {"hota": 0.1}


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "model.pth"
    path.write_bytes(b"weights")
    return path


def test_missing_weights_raise(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFound):
        RFDETRDetectionModel.from_weights(tmp_path / "missing.pth")


def test_coco_train_args_are_built(tmp_path: Path) -> None:
    dataset = tmp_path / "coco"
    dataset.mkdir()
    output = tmp_path / "out"

    args = build_rfdetr_train_args(
        DetectionTrainingConfig(
            data=dataset,
            dataset_format=DatasetFormat.COCO,
            epochs=2,
            batch_size=3,
            learning_rate=0.001,
            num_workers=1,
            multi_scale=True,
            early_stopping=True,
            early_stopping_patience=5,
        ),
        output,
    )

    assert args["dataset_dir"] == str(dataset)
    assert args["coco_path"] == str(dataset)
    assert args["dataset_file"] == "roboflow"
    assert args["lr"] == 0.001
    assert args["num_workers"] == 1
    assert args["multi_scale"] is True
    assert args["early_stopping_patience"] == 5


def test_yolo_dataset_format_is_explicitly_deferred(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="YOLO to COCO conversion"):
        build_rfdetr_train_args(
            DetectionTrainingConfig(data=tmp_path / "data.yaml"),
            tmp_path / "out",
        )


def test_soccernet_dataset_source_converts_to_coco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted: list[Path] = []

    class FakeSoccerNetDataset:
        def __init__(self, root: Path) -> None:
            self.root = root

        def to_coco(self, output_dir: Path, **kwargs: object) -> object:
            _ = kwargs
            output_dir.mkdir(parents=True)
            converted.append(output_dir)
            return object()

    monkeypatch.setattr(
        "tactifoot_vision.detection.rfdetr_training.SoccerNetTrackingDataset",
        FakeSoccerNetDataset,
    )

    result = prepare_rfdetr_dataset(
        DetectionTrainingConfig(
            data=tmp_path / "soccernet",
            dataset_source=DatasetSource.SOCCERNET_TRACKING,
            converted_dataset_dir=tmp_path / "converted",
        ),
        tmp_path / "out",
    )

    assert result == tmp_path / "converted"
    assert converted == [tmp_path / "converted"]


def test_best_checkpoint_fallback_and_copy(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    fallback = output / "checkpoint_best_regular.pth"
    fallback.write_bytes(b"best")
    destination = tmp_path / "models" / "best.pth"

    best = find_best_rfdetr_checkpoint(output)
    copied = copy_best_checkpoint_if_requested(
        best_checkpoint=best,
        destination=destination,
    )

    assert best == fallback
    assert copied == destination
    assert destination.read_bytes() == b"best"


def test_predict_converts_supervision_like_output(tmp_path: Path) -> None:
    model = RFDETRDetectionModel.from_weights(_weights(tmp_path))
    runtime = _FakeRFDETRRuntime()
    model._model = runtime
    detector = model.as_detector(confidence=0.4)

    result = detector.predict(
        Frame(index=0, image=np.zeros((10, 10, 3), dtype=np.uint8))
    )

    assert runtime.predict_call[1] == 0.4
    assert result.detections[0].class_name == "player"
    assert result.detections[0].confidence == np.float32(0.7)


def test_rfdetr_train_uses_helper_and_checkpoint(tmp_path: Path) -> None:
    dataset = tmp_path / "coco"
    dataset.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    (output / "checkpoint.pth").write_bytes(b"checkpoint")
    model = RFDETRDetectionModel.from_weights(_weights(tmp_path))
    runtime = _FakeRFDETRRuntime()
    model._model = runtime

    run = model.train(
        DetectionTrainingConfig(
            data=dataset,
            dataset_format=DatasetFormat.COCO,
            output_dir=output,
        )
    )

    assert runtime.train_calls[0]["dataset_dir"] == str(dataset)
    assert run.best_checkpoint == output / "checkpoint.pth"


def test_seg_prefers_preview_class(monkeypatch) -> None:
    module = ModuleType("rfdetr")
    preview = type("RFDETRSegPreview", (), {})
    fallback = type("RFDETRSeg", (), {})
    module.RFDETRSegPreview = preview
    module.RFDETRSeg = fallback
    monkeypatch.setitem(sys.modules, "rfdetr", module)

    model = RFDETRSegDetectionModel.__new__(RFDETRSegDetectionModel)
    assert model._model_class() is preview


def test_seg_falls_back_to_rfdetr_seg(monkeypatch) -> None:
    module = ModuleType("rfdetr")
    fallback = type("RFDETRSeg", (), {})
    module.RFDETRSeg = fallback
    monkeypatch.setitem(sys.modules, "rfdetr", module)

    model = RFDETRSegDetectionModel.__new__(RFDETRSegDetectionModel)
    assert model._model_class() is fallback


def test_seg_raises_when_no_seg_class(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "rfdetr", ModuleType("rfdetr"))

    with pytest.raises(AdapterUnavailable):
        model = RFDETRSegDetectionModel.__new__(RFDETRSegDetectionModel)
        model._model_class()
