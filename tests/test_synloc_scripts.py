from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from config.synloc_loaders import load_synloc_config
import scripts.train_synloc_detector as train_synloc_detector


def test_synloc_scripts_expose_help() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "scripts/prepare_synloc_dataset.py",
        "scripts/prepare_synloc_pose_dataset.py",
        "scripts/train_synloc_detector.py",
        "scripts/train_synloc_point_regressor.py",
        "scripts/run_synloc_inference.py",
        "scripts/run_synloc_author_baseline.py",
        "scripts/import_author_baseline_results.py",
        "scripts/package_synloc_author_submission.py",
        "scripts/run_synloc_24h_plan.py",
        "scripts/evaluate_synloc_val.py",
        "scripts/build_synloc_submission.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(project_root / script), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_first_test_inference_configs_load_with_expected_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    val_config = load_synloc_config(project_root / "run_config/synloc_inference_val_first_test.yaml")
    test_config = load_synloc_config(project_root / "run_config/synloc_inference_test_first_test.yaml")

    assert val_config.dataset.root == (project_root / "data/SoccerNet/SpiideoSynLoc").resolve()
    assert test_config.dataset.root == (project_root / "data/SoccerNet/SpiideoSynLoc").resolve()
    assert val_config.detector.base_model == "yolo11m.pt"
    assert test_config.detector.base_model == "yolo11m.pt"
    assert val_config.detector.checkpoint_path is None
    assert test_config.detector.checkpoint_path is None
    assert val_config.dataset.split == "val"
    assert test_config.dataset.split == "test"
    assert val_config.dataset.auxiliary_tasks == []
    assert test_config.dataset.auxiliary_tasks == []
    assert val_config.submission.output_dir == (project_root / "results/synloc/val_first_test").resolve()
    assert test_config.submission.output_dir == (project_root / "results/synloc/test_first_test").resolve()


def test_rfdetr_fullhd_configs_load_with_expected_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_config_path = project_root / "run_config/synloc_detector_rfdetr_fullhd_person.yaml"
    val_config = load_synloc_config(project_root / "run_config/synloc_inference_val_rfdetr_fullhd.yaml")
    test_config = load_synloc_config(project_root / "run_config/synloc_inference_test_rfdetr_fullhd.yaml")
    train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))

    assert train_config["dataset_root"] == "data/SoccerNetFullHD/SpiideoSynLoc"
    assert train_config["model_type"] == "rfdetr"
    assert train_config["base_model"] == "rf-detr-base.pth"
    assert train_config["train_imgsz"] == 960

    assert val_config.dataset.root == (project_root / "data/SoccerNetFullHD/SpiideoSynLoc").resolve()
    assert test_config.dataset.root == (project_root / "data/SoccerNetFullHD/SpiideoSynLoc").resolve()
    assert val_config.dataset.image_version == "fullhd"
    assert test_config.dataset.image_version == "fullhd"
    assert val_config.detector.model_type == "rfdetr"
    assert test_config.detector.model_type == "rfdetr"
    checkpoint_path = (project_root / "results/synloc/training/rfdetr_fullhd_person/checkpoint_best_total.pth").resolve()
    assert val_config.detector.checkpoint_path == checkpoint_path
    assert test_config.detector.checkpoint_path == checkpoint_path
    assert val_config.submission.output_dir == (project_root / "results/synloc/rfdetr_fullhd_val").resolve()
    assert test_config.submission.output_dir == (project_root / "results/synloc/rfdetr_fullhd_test").resolve()


def test_rfdetr_fullhd_learned_offset_configs_load_with_expected_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    val_config = load_synloc_config(project_root / "run_config/synloc_inference_val_rfdetr_fullhd_learned.yaml")
    test_config = load_synloc_config(project_root / "run_config/synloc_inference_test_rfdetr_fullhd_learned.yaml")

    detector_checkpoint = (project_root / "results/synloc/training/rfdetr_fullhd_person/checkpoint_best_total.pth").resolve()
    regressor_checkpoint = (project_root / "results/synloc/point_regressor/rfdetr_fullhd_point_regressor.pt").resolve()

    assert val_config.dataset.root == (project_root / "data/SoccerNetFullHD/SpiideoSynLoc").resolve()
    assert test_config.dataset.root == (project_root / "data/SoccerNetFullHD/SpiideoSynLoc").resolve()
    assert val_config.detector.checkpoint_path == detector_checkpoint
    assert test_config.detector.checkpoint_path == detector_checkpoint
    assert val_config.projection.point_strategy == "learned_offset"
    assert test_config.projection.point_strategy == "learned_offset"
    assert val_config.projection.point_regressor_checkpoint == regressor_checkpoint
    assert test_config.projection.point_regressor_checkpoint == regressor_checkpoint
    assert val_config.submission.output_dir == (project_root / "results/synloc/rfdetr_fullhd_learned_val").resolve()
    assert test_config.submission.output_dir == (project_root / "results/synloc/rfdetr_fullhd_learned_test").resolve()


def test_train_synloc_detector_cli_overrides_yaml(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_root": "data/SoccerNetFullHD/SpiideoSynLoc",
                "prepared_dataset_dir": "data/SoccerNetFullHD/SpiideoSynLoc_detection",
                "model_type": "yolo",
                "base_model": "yolo11m.pt",
                "epochs": 20,
                "run_name": "yaml_run",
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeHandler:
        def __init__(self, detection_config, training_config, model_dir):
            captured["epochs"] = training_config.epochs
            captured["run_name"] = training_config.run_name

        def train(self) -> None:
            captured["trained"] = True

    monkeypatch.setattr(
        train_synloc_detector,
        "export_synloc_detection_dataset",
        lambda *args, **kwargs: {
            "coco_root": tmp_path / "coco",
            "yolo_yaml": tmp_path / "dataset.yaml",
        },
    )
    monkeypatch.setattr(train_synloc_detector, "YOLOHandler", FakeHandler)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_synloc_detector.py",
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--run-name",
            "cli_run",
        ],
    )

    train_synloc_detector.main()

    assert captured["epochs"] == 1
    assert captured["run_name"] == "cli_run"
    assert captured["trained"] is True
