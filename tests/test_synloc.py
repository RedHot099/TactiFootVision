from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

from config.synloc_loaders import load_synloc_config
from config.synloc_models import (
    SynLocDatasetConfig,
    SynLocPrediction,
    SynLocSubmissionConfig,
)
from tactifoot_vision.synloc.camera import image_points_to_pitch, pitch_points_to_image
from tactifoot_vision.synloc.data import (
    export_synloc_detection_dataset,
    load_gamestate_split,
    load_synloc_split,
)
from tactifoot_vision.synloc.eval import compare_evaluation_backends, evaluate_predictions
from tactifoot_vision.synloc.postprocess import (
    merge_image_predictions,
    merge_world_predictions,
)
from tactifoot_vision.synloc.submission import build_submission_archive


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
                "bbox": [100.0, 200.0, 40.0, 160.0],
                "area": 6400.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_path = dataset_root / "annotations" / f"{split}.json"
    ann_path.write_text(json.dumps(annotations))
    return dataset_root


def _write_minimal_gamestate_split(
    root: Path,
    *,
    task: str = "gamestate-2024",
    split: str = "train",
    frames: tuple[str, ...] = ("000001",),
) -> Path:
    dataset_root = root / "SoccerNetGS" / task
    video_dir = dataset_root / split / "SNGS-001"
    image_dir = video_dir / "img1"
    image_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for index, frame_id in enumerate(frames, start=1):
        (image_dir / f"{frame_id}.jpg").write_bytes(b"fake-jpg")
        predictions.append(
            {
                "id": str(index),
                "image_id": frame_id,
                "video_id": "SNGS-001",
                "category_id": 1.0,
                "supercategory": "object",
                "track_id": index,
                "bbox_image": {"x": 20.0 * index, "y": 30.0, "w": 40.0, "h": 120.0},
                "bbox_pitch": {
                    "x_bottom_left": -2.0,
                    "y_bottom_left": 10.0,
                    "x_bottom_right": -1.0,
                    "y_bottom_right": 10.0,
                    "x_bottom_middle": -1.5 * index,
                    "y_bottom_middle": 9.5 + index,
                },
                "attributes": {"role": "player", "jersey": str(index), "team": "left"},
            }
        )
    predictions.append(
        {
            "id": "999",
            "image_id": frames[0],
            "video_id": "SNGS-001",
            "category_id": 1.0,
            "supercategory": "object",
            "track_id": 999,
            "bbox_image": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
            "bbox_pitch": {
                "x_bottom_left": 0.0,
                "y_bottom_left": 0.0,
                "x_bottom_right": 0.0,
                "y_bottom_right": 0.0,
                "x_bottom_middle": 0.0,
                "y_bottom_middle": 0.0,
            },
            "attributes": {"role": "referee", "jersey": "null", "team": "null"},
        }
    )
    (video_dir / "Labels-GameState.json").write_text(
        json.dumps({"info": {"version": "1.3"}, "predictions": predictions}),
        encoding="utf-8",
    )
    return dataset_root


def test_load_synloc_split_reads_images_and_annotations(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    config = SynLocDatasetConfig(root=dataset_root, split="val")

    split = load_synloc_split(config)

    assert split.annotation_path == dataset_root / "annotations" / "val.json"
    assert len(split.images) == 1
    assert split.images[0].image_id == 1
    assert split.images[0].file_path == dataset_root / "val" / "scene_0001.jpg"
    assert split.annotations_by_image[1][0].position_on_pitch_xyz == [2.0, 3.0, 0.0]


def test_load_synloc_split_extracts_missing_split_from_zip(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path, split="test")
    split_dir = dataset_root / "test"
    image_path = split_dir / "scene_0001.jpg"
    zip_path = dataset_root / "test.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(image_path, arcname="test/scene_0001.jpg")

    image_path.unlink()
    split_dir.rmdir()

    split = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split="test"))

    assert split.annotation_path == dataset_root / "annotations" / "test.json"
    assert split.images[0].file_path == dataset_root / "test" / "scene_0001.jpg"
    assert split.images[0].file_path.is_file()


def test_load_synloc_config_resolves_relative_dataset_root(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    config_path = tmp_path / "synloc.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "root": str(dataset_root.relative_to(tmp_path)),
                    "split": "val",
                },
                "projection": {"point_strategy": "bottom_center"},
                "submission": {"score_threshold": 0.4},
            }
        )
    )

    config = load_synloc_config(config_path)

    assert config.dataset.root == dataset_root.resolve()
    assert config.submission.score_threshold == 0.4


def test_load_synloc_config_resolves_auxiliary_paths_and_new_fields(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    auxiliary_root = _write_minimal_gamestate_split(tmp_path)
    config_path = tmp_path / "synloc-advanced.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "root": str(dataset_root.relative_to(tmp_path)),
                    "split": "val",
                    "image_version": "4K",
                    "auxiliary_roots": [str(auxiliary_root.relative_to(tmp_path))],
                    "auxiliary_tasks": ["gamestate-2024"],
                    "max_aux_images_per_split": 12,
                },
                "detector": {
                    "train_imgsz": 1280,
                    "inference_imgsz": 1280,
                    "tta_scales": [1024, 1280],
                    "max_detections": 250,
                    "class_filter": ["player"],
                },
                "projection": {
                    "point_strategy": "learned_offset",
                    "point_regressor_checkpoint": "models/point-regressor.pt",
                    "clip_margin_m": 1.25,
                    "behind_camera_policy": "drop",
                },
                "submission": {
                    "score_threshold": 0.4,
                    "topk_per_image": 123,
                    "zip_name": "best.zip",
                },
                "model_dir": "models",
            }
        ),
        encoding="utf-8",
    )

    config = load_synloc_config(config_path)

    assert config.dataset.image_version == "4K"
    assert config.dataset.auxiliary_roots == [auxiliary_root.resolve()]
    assert config.dataset.auxiliary_tasks == ["gamestate-2024"]
    assert config.dataset.max_aux_images_per_split == 12
    assert config.detector.train_imgsz == 1280
    assert config.detector.tta_scales == [1024, 1280]
    assert config.detector.class_filter == ["player"]
    assert config.projection.point_regressor_checkpoint == (tmp_path / "models/point-regressor.pt").resolve()
    assert config.projection.clip_margin_m == 1.25
    assert config.submission.topk_per_image == 123
    assert config.submission.zip_name == "best.zip"


def test_load_gamestate_split_reads_only_player_detections(tmp_path: Path) -> None:
    dataset_root = _write_minimal_gamestate_split(tmp_path)

    split = load_gamestate_split(dataset_root, split="train")

    assert len(split.images) == 1
    assert len(split.annotations) == 1
    assert split.images[0].file_path == dataset_root / "train" / "SNGS-001" / "img1" / "000001.jpg"
    assert split.annotations[0].position_on_pitch_xyz == [-1.5, 10.5, 0.0]
    assert split.annotations[0].bbox_xywh == [20.0, 30.0, 40.0, 120.0]


def test_export_synloc_detection_dataset_can_merge_auxiliary_gamestate(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path, split="train")
    _write_minimal_synloc_split(tmp_path, split="val")
    auxiliary_root = _write_minimal_gamestate_split(tmp_path, frames=("000001", "000002"))
    output_dir = tmp_path / "prepared"

    prepared = export_synloc_detection_dataset(
        dataset_root,
        output_dir,
        auxiliary_roots=(auxiliary_root,),
        auxiliary_tasks=("gamestate-2024",),
        max_aux_images_per_split=1,
    )

    coco_payload = json.loads((prepared["coco_root"] / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))

    assert len(coco_payload["images"]) == 2
    assert len(coco_payload["annotations"]) == 2
    assert (prepared["yolo_yaml"]).is_file()
    assert (prepared["coco_root"] / "test" / "_annotations.coco.json").is_file()

    test_payload = json.loads(
        (prepared["coco_root"] / "test" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    valid_payload = json.loads(
        (prepared["coco_root"] / "valid" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    assert len(test_payload["images"]) == len(valid_payload["images"])
    assert len(test_payload["annotations"]) == len(valid_payload["annotations"])


def test_camera_round_trip_restores_world_point() -> None:
    camera_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
    ]
    poly = _identity_poly()
    world = np.array([[2.0, 3.0, 0.0]], dtype=np.float32)

    image_points = pitch_points_to_image(
        world_points=world,
        camera_matrix=camera_matrix,
        dist_poly=poly,
    )
    recovered = image_points_to_pitch(
        image_points=image_points,
        camera_matrix=camera_matrix,
        undist_poly=poly,
    )

    assert recovered.shape == (1, 3)
    np.testing.assert_allclose(recovered[:, :2], world[:, :2], atol=1e-4)


def test_merge_predictions_deduplicates_in_image_and_world_space() -> None:
    preds = [
        SynLocPrediction(
            image_id=1,
            category_id=1,
            score=0.9,
            bbox_xyxy=[100.0, 100.0, 150.0, 250.0],
            image_point_xy=[125.0, 250.0],
            position_on_pitch_xyz=[1.0, 2.0, 0.0],
        ),
        SynLocPrediction(
            image_id=1,
            category_id=1,
            score=0.7,
            bbox_xyxy=[102.0, 102.0, 152.0, 252.0],
            image_point_xy=[126.0, 251.0],
            position_on_pitch_xyz=[1.1, 2.1, 0.0],
        ),
    ]

    merged_image = merge_image_predictions(preds, image_nms_iou=0.5)
    merged_world = merge_world_predictions(merged_image, world_nms_radius_m=0.5)

    assert len(merged_image) == 1
    assert len(merged_world) == 1
    assert merged_world[0].score == 0.9


def test_evaluate_predictions_returns_locsim_metrics(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    predictions = [
        SynLocPrediction(
            image_id=1,
            category_id=1,
            score=0.95,
            bbox_xyxy=[100.0, 200.0, 140.0, 360.0],
            image_point_xy=[120.0, 360.0],
            position_on_pitch_xyz=[2.0, 3.0, 0.0],
        )
    ]

    metrics = evaluate_predictions(
        annotation_path=dataset_root / "annotations" / "val.json",
        predictions=predictions,
    )

    assert metrics["map_locsim"] > 0.99
    assert 0.0 <= metrics["score_threshold"] <= 1.0
    assert metrics["frame_accuracy"] == 1.0


def test_evaluate_predictions_handles_empty_predictions(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)

    metrics = evaluate_predictions(
        annotation_path=dataset_root / "annotations" / "val.json",
        predictions=[],
    )

    assert metrics["map_locsim"] == 0.0


def test_evaluate_predictions_defaults_missing_iscrowd(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    annotation_path = dataset_root / "annotations" / "val.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    for annotation in payload["annotations"]:
        annotation.pop("iscrowd", None)
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    predictions = [
        SynLocPrediction(
            image_id=1,
            score=0.9,
            bbox_xyxy=[100.0, 200.0, 140.0, 360.0],
            image_point_xy=[120.0, 360.0],
            position_on_pitch_xyz=[2.0, 3.0, 0.0],
        )
    ]

    metrics = evaluate_predictions(
        annotation_path=annotation_path,
        predictions=predictions,
    )

    assert metrics["map_locsim"] > 0.99
    assert metrics["ap_50"] > 0.99
    assert metrics["recall"] > 0.99
    assert metrics["frame_accuracy"] == 1.0


def test_compare_evaluation_backends_returns_local_metrics_when_reference_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    predictions = [
        SynLocPrediction(
            image_id=1,
            category_id=1,
            score=0.95,
            bbox_xyxy=[100.0, 200.0, 140.0, 360.0],
            image_point_xy=[120.0, 360.0],
            position_on_pitch_xyz=[2.0, 3.0, 0.0],
        )
    ]
    monkeypatch.setattr("tactifoot_vision.synloc.eval._load_reference_cocoeval", lambda: None)

    comparison = compare_evaluation_backends(
        annotation_path=dataset_root / "annotations" / "val.json",
        predictions=predictions,
    )

    assert comparison["local"]["map_locsim"] > 0.99
    assert comparison["reference_available"] is False
    assert comparison["reference"] is None


def test_build_submission_archive_contains_required_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "submission"
    config = SynLocSubmissionConfig(
        split="challenge",
        output_dir=output_dir,
        score_threshold=0.55,
    )
    predictions = [
        SynLocPrediction(
            image_id=1,
            category_id=1,
            score=0.95,
            bbox_xyxy=[10.0, 20.0, 30.0, 40.0],
            image_point_xy=[20.0, 40.0],
            position_on_pitch_xyz=[1.0, 2.0, 0.0],
        )
    ]

    archive_path = build_submission_archive(predictions, config)

    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["metadata.json", "results.json"]
        metadata = json.loads(zf.read("metadata.json"))
        results = json.loads(zf.read("results.json"))

    assert metadata == {"score_threshold": 0.55}
    assert results[0]["position_on_pitch"] == [1.0, 2.0, 0.0]


def test_build_submission_archive_respects_topk_and_zip_name(tmp_path: Path) -> None:
    config = SynLocSubmissionConfig(
        split="challenge",
        output_dir=tmp_path / "submission",
        score_threshold=0.5,
        topk_per_image=1,
        zip_name="finalist.zip",
    )
    predictions = [
        SynLocPrediction(
            image_id=1,
            score=0.9,
            bbox_xyxy=[0.0, 0.0, 10.0, 20.0],
            image_point_xy=[5.0, 20.0],
            position_on_pitch_xyz=[1.0, 2.0, 0.0],
        ),
        SynLocPrediction(
            image_id=1,
            score=0.8,
            bbox_xyxy=[1.0, 1.0, 10.0, 20.0],
            image_point_xy=[5.0, 20.0],
            position_on_pitch_xyz=[1.2, 2.2, 0.0],
        ),
    ]

    archive_path = build_submission_archive(predictions, config)

    assert archive_path.name == "finalist.zip"
    with zipfile.ZipFile(archive_path) as zf:
        results = json.loads(zf.read("results.json"))
    assert len(results) == 1
