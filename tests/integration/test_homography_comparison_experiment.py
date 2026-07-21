import json
from pathlib import Path

import pytest

from tactifoot_vision.config import ExperimentConfig
from tactifoot_vision.enums import ExperimentKind, HomographyMethod
from tactifoot_vision.experiments import HomographyComparisonRunner


def make_gsr_sequence(root: Path, name: str = "SNGS-001") -> Path:
    sequence = root / "valid" / name
    sequence.mkdir(parents=True)
    annotations = []
    for track_id, image_x, image_y, pitch_x, pitch_y in (
        (1, 0.0, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 2.0, 0.0),
        (3, 0.0, 1.0, 0.0, 2.0),
        (4, 1.0, 1.0, 2.0, 2.0),
    ):
        annotations.append(
            {
                "id": str(track_id),
                "image_id": "1001000001",
                "track_id": track_id,
                "supercategory": "object",
                "bbox_image": {
                    "x_center": image_x,
                    "y_center": image_y,
                    "w": 0.0,
                    "h": 0.0,
                },
                "bbox_pitch": {
                    "x_bottom_middle": pitch_x,
                    "y_bottom_middle": pitch_y,
                },
                "attributes": {"role": "player", "jersey": None, "team": "left"},
            }
        )
    payload = {
        "info": {"version": "1.3"},
        "images": [
            {
                "image_id": "1001000001",
                "file_name": "000001.jpg",
                "has_labeled_pitch": True,
                "has_labeled_camera": True,
                "has_labeled_person": True,
            }
        ],
        "annotations": annotations,
    }
    (sequence / "Labels-GameState.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return sequence


def test_homography_comparison_smoke_generates_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "SoccerNetGS"
    make_gsr_sequence(dataset_root)
    config = ExperimentConfig(
        name="homography_smoke",
        kind=ExperimentKind.HOMOGRAPHY_COMPARISON,
        soccernet_root=dataset_root,
        output_dir=tmp_path / "out",
        max_sequences=1,
    )
    config.homography_comparison.split = "valid"
    config.homography_comparison.methods = (
        HomographyMethod.CURRENT_YOLOPOSE_7PT,
        HomographyMethod.ORACLE_GSR_LINES_RANSAC,
    )
    config.homography_comparison.confidence_iterations = 0

    report = HomographyComparisonRunner().run(config)

    assert (tmp_path / "out" / "homographies.parquet").is_file()
    assert (tmp_path / "out" / "projections.parquet").is_file()
    assert (tmp_path / "out" / "metrics.json").is_file()
    assert (tmp_path / "out" / "failure_cases").is_dir()
    assert report.metrics["best_median_error_m"] == pytest.approx(0.0, abs=1e-12)
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["ranking"][0]["method"] == "oracle_gsr_lines_ransac"
    assert metrics["ranking"][0]["rankable"] == 1.0
