from __future__ import annotations

from pathlib import Path

from config.models import DetectionConfig, DetectionModelType, TrainingDetectionConfig
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler


class _FakeRFDETRModel:
    def __init__(self) -> None:
        self.train_kwargs: dict[str, object] | None = None

    def train(self, **kwargs) -> None:
        self.train_kwargs = kwargs


def test_rfdetr_handler_passes_training_output_and_resolution(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "coco_dataset"
    dataset_dir.mkdir()
    model = _FakeRFDETRModel()

    monkeypatch.setattr(
        RFDETRHandler,
        "_init_model_from_name",
        lambda self, model_name: model,
    )

    detection_config = DetectionConfig(
        model_type=DetectionModelType.RFDETR,
        checkpoint_path=None,
        classes={"person": 0},
        include_labels=["person"],
    )
    training_config = TrainingDetectionConfig(
        dataset_path=dataset_dir,
        dataset_format="coco",
        base_model="rf-detr-base.pth",
        epochs=3,
        batch_size=2,
        learning_rate=5e-4,
        imgsz=960,
        project_name=str(tmp_path / "training_outputs"),
        run_name="fullhd_person",
        device="cuda",
    )

    handler = RFDETRHandler(detection_config, training_config, model_dir=tmp_path / "models")
    handler.train()

    assert model.train_kwargs is not None
    assert model.train_kwargs["dataset_dir"] == str(dataset_dir)
    assert model.train_kwargs["epochs"] == 3
    assert model.train_kwargs["batch_size"] == 2
    assert model.train_kwargs["grad_accum_steps"] == 2
    assert model.train_kwargs["lr"] == 5e-4
    assert model.train_kwargs["output_dir"] == str(tmp_path / "training_outputs" / "fullhd_person")
    assert model.train_kwargs["project"] == str(tmp_path / "training_outputs")
    assert model.train_kwargs["run"] == "fullhd_person"
    assert model.train_kwargs["resolution"] == 960
    assert model.train_kwargs["class_names"] == ["person"]
    assert model.train_kwargs["num_classes"] == 1
    assert model.train_kwargs["device"] == "cuda"
    assert model.train_kwargs["run_test"] is False
