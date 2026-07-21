from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tactifoot_vision.config import (
    ExperimentConfig,
    PipelineConfig,
    load_experiment_config,
)
from tactifoot_vision.domain import (
    AdapterUnavailable,
    BBox,
    DetectionSet,
    Frame,
    FrameResult,
    PipelineResult,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import (
    ExperimentKind,
    TeamAssignmentCropMethod,
)
from tactifoot_vision.experiments import TeamClassificationExperimentRunner


class _StaticPipeline:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result

    def run(self, frames: list[Frame]) -> PipelineResult:
        _ = frames
        return self.result


def test_team_classification_experiment_smoke(tmp_path: Path) -> None:
    config = load_experiment_config(
        "configs/experiments/team_classification_smoke.yaml"
    )
    config.output_dir = tmp_path / "team"

    report = TeamClassificationExperimentRunner().run(config)

    assert report.metrics["frames"] == 2.0
    assert report.metrics["samples"] > 0.0
    assert report.artifacts[0].path.is_file()
    assignments_path = tmp_path / "team" / "team_classification_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    players = assignments[assignments["class_name"] == "player"]
    assert assignments_path.is_file()
    assert players["team_id"].notna().any()
    assert {
        "crop_method",
        "embedding",
        "reducer",
        "clusterer",
        "crop_ratio",
        "has_crop",
        "true_team_id",
    }.issubset(assignments.columns)


def test_team_classification_opencv_mask_uses_configured_cropper(
    tmp_path: Path, monkeypatch
) -> None:
    frame = Frame(index=0, image=np.zeros((20, 20, 3), dtype=np.uint8))
    result = PipelineResult(
        (
            FrameResult(
                frame_index=0,
                timestamp_seconds=None,
                detections=DetectionSet.empty(),
                tracks=TrackSet(
                    (
                        Track(1, BBox(0, 0, 10, 10), 2, "player"),
                        Track(2, BBox(10, 0, 20, 10), 2, "player"),
                    )
                ),
            ),
        )
    )
    calls = []

    def fake_crop(image, bbox):
        calls.append((image, bbox))
        value = 255 if len(calls) == 1 else 0
        return np.full((8, 8, 3), value, dtype=np.uint8)

    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.build_pipeline",
        lambda config: _StaticPipeline(result),
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.read_frames",
        lambda path: iter([frame]),
    )
    monkeypatch.setattr(
        "tactifoot_vision.team_assignment.opencv_masks.opencv_mask_crop",
        fake_crop,
    )
    config = ExperimentConfig(
        name="team_opencv",
        kind=ExperimentKind.TEAM_CLASSIFICATION,
        pipeline=PipelineConfig(),
        output_dir=tmp_path / "team",
    )
    config.pipeline.paths.input = tmp_path
    config.pipeline.team_assignment.crop_method = TeamAssignmentCropMethod.OPENCV_MASK

    report = TeamClassificationExperimentRunner().run(config)

    assert len(calls) == 2
    assert report.metrics["valid_crops"] == 2.0
    assignments = pd.read_csv(tmp_path / "team" / "team_classification_assignments.csv")
    assert set(assignments["crop_method"]) == {"opencv_mask"}


def test_team_classification_sam2_mask_is_explicitly_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    frame = Frame(index=0, image=np.zeros((20, 20, 3), dtype=np.uint8))
    result = PipelineResult(
        (
            FrameResult(
                frame_index=0,
                timestamp_seconds=None,
                detections=DetectionSet.empty(),
                tracks=TrackSet((Track(1, BBox(0, 0, 10, 10), 2, "player"),)),
            ),
        )
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.build_pipeline",
        lambda config: _StaticPipeline(result),
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.read_frames",
        lambda path: iter([frame]),
    )
    config = ExperimentConfig(
        name="team_sam2",
        kind=ExperimentKind.TEAM_CLASSIFICATION,
        pipeline=PipelineConfig(),
        output_dir=tmp_path / "team",
    )
    config.pipeline.paths.input = tmp_path
    config.pipeline.team_assignment.crop_method = TeamAssignmentCropMethod.SAM2_MASK

    with pytest.raises(AdapterUnavailable, match="does not silently fall back"):
        TeamClassificationExperimentRunner().run(config)


def test_team_classification_no_players_writes_zero_sample_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    frame = Frame(index=0, image=np.zeros((20, 20, 3), dtype=np.uint8))
    result = PipelineResult(
        (
            FrameResult(
                frame_index=0,
                timestamp_seconds=None,
                detections=DetectionSet.empty(),
                tracks=TrackSet((Track(1, BBox(0, 0, 5, 5), 0, "ball"),)),
            ),
        )
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.build_pipeline",
        lambda config: _StaticPipeline(result),
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.read_frames",
        lambda path: iter([frame]),
    )
    config = ExperimentConfig(
        name="team_empty",
        kind=ExperimentKind.TEAM_CLASSIFICATION,
        pipeline=PipelineConfig(),
        output_dir=tmp_path / "team",
    )
    config.pipeline.paths.input = tmp_path

    report = TeamClassificationExperimentRunner().run(config)

    assert report.metrics["samples"] == 0.0
    assert report.metrics["valid_crops"] == 0.0
    assert (tmp_path / "team" / "team_classification_assignments.csv").is_file()


def test_team_classification_labeled_tracks_compute_purity(
    tmp_path: Path, monkeypatch
) -> None:
    frame = Frame(index=0, image=np.zeros((20, 20, 3), dtype=np.uint8))
    frame.image[:, :10] = (255, 0, 0)
    frame.image[:, 10:] = (0, 255, 0)
    result = PipelineResult(
        (
            FrameResult(
                frame_index=0,
                timestamp_seconds=None,
                detections=DetectionSet.empty(),
                tracks=TrackSet(
                    (
                        Track(
                            1, BBox(0, 0, 10, 10), 2, "player", data={"team_label": 0}
                        ),
                        Track(
                            2, BBox(10, 0, 20, 10), 2, "player", data={"team_label": 1}
                        ),
                    )
                ),
            ),
        )
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.build_pipeline",
        lambda config: _StaticPipeline(result),
    )
    monkeypatch.setattr(
        "tactifoot_vision.experiments.team_classification.read_frames",
        lambda path: iter([frame]),
    )
    config = ExperimentConfig(
        name="team_labeled",
        kind=ExperimentKind.TEAM_CLASSIFICATION,
        pipeline=PipelineConfig(),
        output_dir=tmp_path / "team",
    )
    config.pipeline.paths.input = tmp_path

    report = TeamClassificationExperimentRunner().run(config)

    assert report.metrics["labeled_samples"] == 2.0
    assert report.metrics["purity"] == 1.0
