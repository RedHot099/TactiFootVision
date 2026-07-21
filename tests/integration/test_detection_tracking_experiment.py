from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.config import ExperimentConfig, PipelineConfig
from tactifoot_vision.domain import (
    BBox,
    DetectionSet,
    FrameResult,
    PipelineResult,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import ExperimentKind
from tactifoot_vision.experiments import DetectionTrackingExperimentRunner
from tactifoot_vision.experiments.soccernet_detection_tracking import _aggregate


def make_sequence(root: Path) -> None:
    sequence = root / "SNMOT-001"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "img1").mkdir()
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=SNMOT-001",
                "frameRate=25",
                "seqLength=1",
                "imWidth=20",
                "imHeight=10",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    (sequence / "gameinfo.ini").write_text(
        "\n".join(["[Sequence]", "num_tracklets=1", "trackletID_1=Player 1"]),
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,1,1,4,4,1,-1,-1,-1\n", encoding="utf-8"
    )
    cv2.imwrite(
        str(sequence / "img1" / "000001.jpg"), np.zeros((10, 20, 3), dtype=np.uint8)
    )


def test_detection_tracking_experiment_soccernet_smoke(tmp_path: Path) -> None:
    root = tmp_path / "soccernet"
    root.mkdir()
    make_sequence(root)
    config = ExperimentConfig(
        name="soccernet_smoke",
        kind=ExperimentKind.DETECTION_TRACKING,
        pipeline=PipelineConfig(),
        soccernet_root=root,
        max_sequences=1,
        max_frames=1,
        output_dir=tmp_path / "out",
    )

    report = DetectionTrackingExperimentRunner().run(config)

    assert report.metrics["sequences"] == 1.0
    assert (tmp_path / "out" / "metrics.json").is_file()


class _OneFramePipeline:
    def run_video(
        self, path: object, *, max_frames: int | None = None
    ) -> PipelineResult:
        _ = path, max_frames
        return PipelineResult(
            (
                FrameResult(
                    frame_index=0,
                    timestamp_seconds=None,
                    detections=DetectionSet.empty(),
                    tracks=TrackSet(
                        (
                            Track(
                                track_id=1,
                                bbox=BBox(1.0, 1.0, 5.0, 5.0),
                                class_id=2,
                                class_name="player",
                            ),
                        )
                    ),
                ),
            )
        )


def test_soccernet_experiment_evaluates_one_based_mot_export(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "soccernet"
    root.mkdir()
    make_sequence(root)
    monkeypatch.setattr(
        "tactifoot_vision.experiments.soccernet_detection_tracking.build_pipeline",
        lambda config: _OneFramePipeline(),
    )
    config = ExperimentConfig(
        name="soccernet_smoke",
        kind=ExperimentKind.DETECTION_TRACKING,
        pipeline=PipelineConfig(),
        soccernet_root=root,
        max_sequences=1,
        max_frames=1,
        output_dir=tmp_path / "out",
    )

    report = DetectionTrackingExperimentRunner().run(config)

    assert report.metrics["tp"] == 1.0
    assert report.metrics["fn"] == 0.0


def test_soccernet_aggregate_uses_global_counts_not_macro_average() -> None:
    metrics = _aggregate(
        {
            "a": {
                "tp": 100.0,
                "fp": 0.0,
                "fn": 0.0,
                "matches": 100.0,
                "frames_evaluated": 100.0,
                "id_switches": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "mean_iou": 0.8,
                "iou_threshold": 0.5,
            },
            "b": {
                "tp": 0.0,
                "fp": 1.0,
                "fn": 9.0,
                "matches": 0.0,
                "frames_evaluated": 9.0,
                "id_switches": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mean_iou": 0.0,
                "iou_threshold": 0.5,
            },
        }
    )

    assert metrics["tp"] == 100.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 9.0
    assert metrics["precision"] == 100.0 / 101.0
    assert metrics["recall"] == 100.0 / 109.0
    assert metrics["precision"] != 0.5
