from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tactifoot_vision.synloc.author_baseline import (
    AUTHOR_BASELINE_BRANCH,
    AUTHOR_BASELINE_COMMIT,
    AUTHOR_BASELINE_CONFIG,
    build_author_baseline_sources_markdown,
    prepare_author_baseline_workspace,
)
from tactifoot_vision.synloc.camera import image_points_to_pitch
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.prediction_io import load_predictions_from_results_json
from config.synloc_models import SynLocDatasetConfig


def _identity_poly() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _write_minimal_synloc_split(root: Path, split: str = "val") -> Path:
    dataset_root = root / "SoccerNet" / "SpiideoSynLoc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / split).mkdir(parents=True, exist_ok=True)
    image_path = dataset_root / split / "scene_0001.jpg"
    image_path.write_bytes(b"fake-jpg")

    annotations = {
        "images": [
            {
                "id": 1,
                "file_name": "scene_0001.jpg",
                "width": 1920,
                "height": 1080,
                "camera_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 10.0],
                ],
                "dist_poly": _identity_poly(),
                "undist_poly": _identity_poly(),
            }
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "position_on_pitch": [2.0, 3.0, 0.0],
                "keypoints": [[120.0, 300.0, 1], [120.0, 360.0, 1]],
                "keypoints_3d": [[2.0, 3.0, 1.0, 1], [2.0, 3.0, 0.0, 1]],
                "bbox": [100.0, 200.0, 40.0, 160.0],
                "area": 6400.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_path = dataset_root / "annotations" / f"{split}.json"
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")
    return dataset_root


def test_load_predictions_from_results_json_projects_author_keypoints_to_world(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    split_data = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split="val"))
    image_record = split_data.images[0]
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.95,
                    "bbox": [100.0, 200.0, 40.0, 160.0],
                    "keypoints": [120.0, 300.0, 1.0, 120.0, 360.0, 1.0],
                }
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_predictions_from_results_json(
        results_path,
        split_data=split_data,
        position_from_keypoint_index=1,
    )

    expected_world = image_points_to_pitch(
        np.asarray([[120.0, 360.0]], dtype=np.float32),
        camera_matrix=image_record.camera_matrix,
        undist_poly=image_record.undist_poly,
        image_shape=image_record.image_shape,
    )[0]

    assert len(predictions) == 1
    assert predictions[0].image_point_xy == [120.0, 360.0]
    assert predictions[0].bbox_xyxy == [100.0, 200.0, 140.0, 360.0]
    np.testing.assert_allclose(predictions[0].position_on_pitch_xyz, expected_world, atol=1e-5)


def test_prepare_author_baseline_workspace_writes_manifest_and_override_config(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    repo_root = tmp_path / "mmpose"
    repo_root.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    prepared = prepare_author_baseline_workspace(
        dataset_root=dataset_root,
        output_dir=output_dir,
        official_repo_root=repo_root,
        split="val",
    )

    manifest = json.loads(prepared["manifest_path"].read_text(encoding="utf-8"))
    override_config = prepared["override_config_path"].read_text(encoding="utf-8")

    assert manifest["dataset_root"] == str(dataset_root.resolve())
    assert manifest["split"] == "val"
    assert manifest["official_repo_root"] == str(repo_root.resolve())
    assert manifest["official_config"] == AUTHOR_BASELINE_CONFIG
    assert manifest["position_from_keypoint_index"] == 1
    assert "annotations/val.json" in override_config
    assert "dataset['data_prefix'] = dict(img=image_dir)" in override_config
    assert str(dataset_root.resolve()) in override_config


def test_build_author_baseline_sources_markdown_mentions_pinned_official_assets() -> None:
    markdown = build_author_baseline_sources_markdown()

    assert AUTHOR_BASELINE_BRANCH in markdown
    assert AUTHOR_BASELINE_COMMIT in markdown
    assert AUTHOR_BASELINE_CONFIG in markdown
    assert "research.spiideo.com" in markdown
    assert "79.3" in markdown
