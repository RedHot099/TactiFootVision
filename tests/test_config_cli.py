"""YAML configs and the ``tactifoot`` command, driven by fake registered models."""

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv
import yaml

from tactifoot_vision.cli import main
from tactifoot_vision.config import load_config
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.models import MODELS, Model
from tactifoot_vision.pipeline import PipelineResult
from tactifoot_vision.tracking import ByteTrackTracker


@MODELS.register("test_box_detector")
class BoxDetector(Model):
    """One player box that moves right by 4 px per frame (found by brightness)."""

    task = Task.DETECT
    name = "test_box_detector"

    def __init__(self, weights=None, *, conf=0.25, device="cpu", iou=0.5):
        self.iou = iou
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self):
        return ["ball", "player"]

    def predict(self, image):
        ys, xs = np.nonzero(image[..., 1] > 128)
        return sv.Detections(
            xyxy=np.array(
                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32
            ),
            confidence=np.array([0.9], np.float32),
            class_id=np.array([1]),
            data={"class_name": np.array(["player"])},
        )


def _video(path, frames=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(frames):
        image = np.zeros((48, 64, 3), np.uint8)
        image[20:30, 4 + 4 * i : 12 + 4 * i] = (0, 255, 0)
        writer.write(image)
    writer.release()
    return path


def _write_config(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_config_resolves_only_real_relative_paths(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"")
    config = load_config(
        _write_config(
            tmp_path / "pipeline.yaml",
            {
                "detector": {"type": "yolo", "weights": "weights.pt"},
                "keypoints": {"type": "yolo_pose", "weights": "yolov8n-pose.pt"},
                "tracker": {
                    "type": "sam2",
                    "options": {
                        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                        "checkpoint": "./ckpt.pt",
                    },
                },
            },
        )
    )
    assert config.detector.weights == str(tmp_path / "weights.pt")
    assert config.keypoints.weights == "yolov8n-pose.pt"  # downloaded by the backend
    assert config.tracker.options["config"] == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert config.tracker.options["checkpoint"] == str(tmp_path / "./ckpt.pt")


def test_load_config_rejects_unknown_sections(tmp_path):
    with pytest.raises(ValueError, match="detectr"):
        load_config(_write_config(tmp_path / "bad.yaml", {"detectr": {"type": "yolo"}}))


def test_config_builds_the_pipeline(tmp_path):
    config = load_config(
        _write_config(
            tmp_path / "pipeline.yaml",
            {
                "detector": {
                    "type": "test_box_detector",
                    "conf": 0.4,
                    "options": {"iou": 0.3, "conf": 0.1},
                },
                "tracker": {"type": "bytetrack", "options": {"lost_track_buffer": 5}},
                "pitch": {"length": 120, "width": 80},
            },
        )
    )
    pipeline = config.build()
    assert pipeline.detector.conf == 0.4  # the explicit field wins over options
    assert pipeline.detector.iou == 0.3
    assert isinstance(pipeline.tracker, ByteTrackTracker)
    assert pipeline.pitch.length == 120 and pipeline.homography.pitch == pipeline.pitch
    assert pipeline.team_classifier is None and pipeline.keypoint_model is None


def test_cli_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "yolo" in out and "bytetrack" in out and "siglip" in out


def test_cli_run_writes_all_outputs(tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml", {"detector": {"type": "test_box_detector"}}
    )
    video = _video(tmp_path / "clip.mp4")
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--video",
                str(video),
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    result = PipelineResult.load(out / "result.pkl")
    assert len(result) == 6 and result.track_ids == [1]
    tracks = pd.read_csv(out / "tracks.csv")
    assert (tracks["class_name"] == "player").all() and len(tracks) == 6
    assert (out / "freeze_frames.csv").is_file()
    assert (
        cv2.VideoCapture(str(out / "annotated.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) == 6
    )
