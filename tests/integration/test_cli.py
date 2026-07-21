import json
from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.cli import main
from tactifoot_vision.detection import FakeDetector
from tactifoot_vision.enums import DatasetFormat, DatasetSource


def test_cli_infer_fake_config() -> None:
    assert (
        main(
            [
                "infer",
                "--config",
                "configs/pipeline/fake_bytetrack.yaml",
                "--max-frames",
                "1",
            ]
        )
        == 0
    )


def test_cli_detect_image_fake_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "tactifoot_vision.cli.build_detector", lambda config: FakeDetector()
    )

    assert (
        main(
            [
                "detect",
                "image",
                "--config",
                "configs/pipeline/fake_bytetrack.yaml",
                "--input",
                "data/soccernet_dummy/img1/frame_0001.jpg",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["detections"] == 2
    assert payload["classes"] == ["ball", "player"]


def test_cli_train_detector_builds_dataset_config(monkeypatch, tmp_path) -> None:
    configs = []

    class FakeTrainableModel:
        def train(self, config):
            configs.append(config)

    monkeypatch.setattr(
        "tactifoot_vision.cli._detection_model",
        lambda backend, weights: FakeTrainableModel(),
    )

    result = main(
        [
            "train",
            "detector",
            "--backend",
            "rfdetr",
            "--weights",
            str(tmp_path / "weights.pth"),
            "--data",
            str(tmp_path / "soccernet"),
            "--epochs",
            "2",
            "--dataset-format",
            "coco",
            "--dataset-source",
            "soccernet_tracking",
            "--converted-dataset-dir",
            str(tmp_path / "converted"),
            "--valid-fraction",
            "0.1",
            "--every-nth-frame",
            "5",
            "--max-sequences",
            "3",
            "--copy-images",
        ]
    )

    assert result == 0
    assert configs[0].dataset_format is DatasetFormat.COCO
    assert configs[0].dataset_source is DatasetSource.SOCCERNET_TRACKING
    assert configs[0].converted_dataset_dir == tmp_path / "converted"
    assert configs[0].valid_fraction == 0.1
    assert configs[0].every_nth_frame == 5
    assert configs[0].max_sequences == 3
    assert configs[0].symlink_images is False


def test_cli_runs_video_only_xg_from_features(tmp_path, capsys) -> None:
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "shot_id,frame_index,shot_x,shot_y,goalkeeper_distance",
                "s1,10,99,34,12",
            ]
        ),
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "\n".join(["shot_id,reference_xg,is_goal", "s1,0.3,1"]),
        encoding="utf-8",
    )

    result = main(
        [
            "video-xg",
            "from-features",
            "--features",
            str(features_path),
            "--reference",
            str(reference_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--group-id",
            "match-1",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["matched_shots"] == 1.0
    assert (tmp_path / "out" / "video_only_shots.csv").exists()


def test_cli_compares_video_only_xg_methods(tmp_path, capsys) -> None:
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "shot_id,frame_index,shot_x,shot_y,goalkeeper_distance",
                "s1,10,99,34,12",
            ]
        ),
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "\n".join(["shot_id,reference_xg,is_goal", "s1,0.3,1"]),
        encoding="utf-8",
    )

    result = main(
        [
            "video-xg",
            "compare-methods",
            "--features",
            str(features_path),
            "--reference",
            str(reference_path),
            "--output-dir",
            str(tmp_path / "experiment"),
            "--group-id",
            "match-1",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["methods"]) == 3
    assert (tmp_path / "experiment" / "method_metrics.csv").exists()


def test_cli_runs_video_xg_end_to_end_and_writes_report(tmp_path, capsys) -> None:
    video_path = tmp_path / "part1.mp4"
    _write_tiny_video(video_path)
    config_path = tmp_path / "video_xg_end_to_end.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_smoke",
                "video_parts:",
                f"  - {video_path}",
                "scan_fps: 2",
                f"output_dir: {output_dir}",
                "detector:",
                "  backend: fake",
                "tracking:",
                "  backend: fake",
            ]
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "video-xg",
            "end-to-end",
            "--config",
            str(config_path),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(output_dir)
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "09_predictions.csv").exists()
    assert (output_dir / "10_metrics.json").exists()


def test_cli_video_xg_end_to_end_stop_after_checkpoint(tmp_path, capsys) -> None:
    video_path = tmp_path / "part1.mp4"
    _write_tiny_video(video_path)
    config_path = tmp_path / "video_xg_end_to_end.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_stop_after",
                "video_parts:",
                f"  - {video_path}",
                "scan_fps: 2",
                f"output_dir: {output_dir}",
                "detector:",
                "  backend: fake",
                "tracking:",
                "  backend: fake",
            ]
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "video-xg",
            "end-to-end",
            "--config",
            str(config_path),
            "--stop-after",
            "01_sampled_frames",
        ]
    )

    assert result == 0
    _ = capsys.readouterr()
    assert (output_dir / "00_video_timeline.json").exists()
    assert (output_dir / "01_sampled_frames.parquet").exists()
    assert not (output_dir / "02_detections.parquet").exists()

    resumed = main(
        [
            "video-xg",
            "end-to-end",
            "--config",
            str(config_path),
            "--resume-from",
            "01_sampled_frames",
            "--stop-after",
            "02_detections",
        ]
    )

    assert resumed == 0
    _ = capsys.readouterr()
    assert (output_dir / "02_detections.parquet").exists()


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    for index in range(12):
        frame = np.full((48, 64, 3), index * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()
