from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import supervision as sv

from config.synloc_models import (
    SynLocDatasetConfig,
    SynLocDetectorConfig,
    SynLocPrediction,
    SynLocProjectionConfig,
    SynLocSubmissionConfig,
)
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.data import SynLocImageRecord
from tactifoot_vision.synloc.eval import evaluate_predictions
from tactifoot_vision.synloc.inference import (
    DetectionCandidate,
    detect_image,
    project_candidates,
    run_inference_on_split,
)
from tactifoot_vision.synloc.point_regressor import (
    PointOffsetRegressor,
    build_regressor_examples_from_predictions,
)
from tactifoot_vision.synloc.projection import image_point_from_bbox, project_bbox_to_pitch
from tactifoot_vision.synloc.camera import image_points_to_pitch
from tactifoot_vision.synloc.submission import build_submission_archive
from tactifoot_vision.synloc.tiling import generate_tiles
from tactifoot_vision.synloc.visualize import render_prediction_debug


def _identity_poly() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _write_minimal_synloc_split(
    root: Path,
    split: str = "val",
    position_on_pitch: list[float] | None = None,
) -> Path:
    dataset_root = root / "SoccerNet" / "SpiideoSynLoc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / split).mkdir(parents=True, exist_ok=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_root / split / "scene_0001.jpg"), image)

    payload = {
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
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "position_on_pitch": position_on_pitch or [2.0, 3.0, 0.0],
                "bbox": [100.0, 200.0, 40.0, 160.0],
                "area": 6400.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (dataset_root / "annotations" / f"{split}.json").write_text(json.dumps(payload))
    return dataset_root


def test_generate_tiles_covers_edges() -> None:
    tiles = generate_tiles(1000, 800, tile_size=512, overlap=128)

    assert tiles[0].x1 == 0
    assert tiles[0].y1 == 0
    assert tiles[-1].x2 == 1000
    assert tiles[-1].y2 == 800
    assert len(tiles) >= 4


def test_project_bbox_to_pitch_uses_bottom_center() -> None:
    image_point, pitch_point = project_bbox_to_pitch(
        [100.0, 200.0, 140.0, 360.0],
        camera_matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
        ],
        undist_poly=_identity_poly(),
        image_shape=(3, 1080, 1920),
        point_strategy="bottom_center",
    )

    assert image_point == [120.0, 360.0]
    assert len(pitch_point) == 3


def test_image_point_from_bbox_respects_learned_offset() -> None:
    point = image_point_from_bbox(
        [10.0, 20.0, 30.0, 60.0],
        point_strategy="learned_offset",
        learned_offset_xy=[0.25, 0.5],
    )

    np.testing.assert_allclose(point, np.array([15.0, 40.0], dtype=np.float32))


def test_point_regressor_forward_shape() -> None:
    model = PointOffsetRegressor()

    output = model(torch.randn(2, 3, 96, 96))

    assert output.shape == (2, 2)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_project_candidates_drops_invalid_and_off_pitch_predictions(monkeypatch) -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image_record = load_synloc_split(
        SynLocDatasetConfig(root=_write_minimal_synloc_split(Path("/tmp")), split="val")
    ).images[0]
    candidates = [
        DetectionCandidate(
            image_id=1,
            score=0.9,
            bbox_xyxy=[100.0, 200.0, 140.0, 360.0],
            category_id=1,
            source_tile_xyxy=[0.0, 0.0, 512.0, 512.0],
            source_scale=1280,
        ),
        DetectionCandidate(
            image_id=1,
            score=0.8,
            bbox_xyxy=[200.0, 200.0, 240.0, 360.0],
            category_id=1,
            source_tile_xyxy=[0.0, 0.0, 512.0, 512.0],
            source_scale=1280,
        ),
        DetectionCandidate(
            image_id=1,
            score=0.7,
            bbox_xyxy=[300.0, 200.0, 340.0, 360.0],
            category_id=1,
            source_tile_xyxy=[0.0, 0.0, 512.0, 512.0],
            source_scale=1280,
        ),
    ]
    outputs = iter(
        [
            ([120.0, 360.0], [2.0, 3.0, 0.0]),
            ([220.0, 360.0], [float("nan"), 3.0, 0.0]),
            ([320.0, 360.0], [200.0, 3.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        "tactifoot_vision.synloc.inference.project_bbox_to_pitch",
        lambda *args, **kwargs: next(outputs),
    )

    predictions = project_candidates(
        image=image,
        image_record=image_record,
        candidates=candidates,
        projection_config=SynLocProjectionConfig(
            point_strategy="bottom_center",
            clip_margin_m=0.0,
            behind_camera_policy="drop",
        ),
    )

    assert len(predictions) == 1
    assert predictions[0].world_confidence == 0.9
    assert predictions[0].source_tile_xyxy == [0.0, 0.0, 512.0, 512.0]
    assert predictions[0].source_scale == 1280


def test_render_prediction_debug_writes_file(tmp_path: Path) -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), image)
    output_path = tmp_path / "debug.jpg"

    render_prediction_debug(
        image=image,
        predictions=[
            SynLocPrediction(
                image_id=1,
                score=0.9,
                bbox_xyxy=[20.0, 30.0, 60.0, 100.0],
                image_point_xy=[40.0, 100.0],
                position_on_pitch_xyz=[1.0, 2.0, 0.0],
            )
        ],
        output_path=output_path,
    )

    assert output_path.is_file()


def test_build_regressor_examples_from_detector_predictions(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    split_data = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split="val"))
    predictions = [
        SynLocPrediction(
            image_id=1,
            score=0.9,
            bbox_xyxy=[98.0, 198.0, 142.0, 362.0],
            image_point_xy=[120.0, 360.0],
            position_on_pitch_xyz=[2.0, 3.0, 0.0],
        )
    ]

    examples = build_regressor_examples_from_predictions(split_data, predictions)

    assert len(examples) == 1
    assert examples[0].bbox_xyxy == [98.0, 198.0, 142.0, 362.0]


def test_detect_image_keeps_rfdetr_person_detections_with_class_id_one(tmp_path: Path) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    image_record = load_synloc_split(
        SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False)
    ).images[0]
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class FakeDetector:
        def detect(self, image: np.ndarray) -> sv.Detections:
            return sv.Detections(
                xyxy=np.asarray([[100.0, 200.0, 140.0, 360.0]], dtype=np.float32),
                confidence=np.asarray([0.95], dtype=np.float32),
                class_id=np.asarray([1], dtype=np.int32),
            )

    candidates = detect_image(
        image=image,
        image_record=image_record,
        handler=FakeDetector(),
        use_tiles=False,
        tile_size=1280,
        tile_overlap=256,
        detector_config=SynLocDetectorConfig(model_type="rfdetr", base_model="rf-detr-base.pth"),
    )

    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(0.95)


def test_detect_image_rescales_boxes_to_annotation_resolution() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image_record = SynLocImageRecord(
        image_id=7,
        file_name="frame.jpg",
        file_path=Path("/tmp/frame.jpg"),
        width=3840,
        height=2160,
        camera_matrix=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 10.0]],
        dist_poly=_identity_poly(),
        undist_poly=_identity_poly(),
    )

    class FakeDetector:
        def detect(self, image: np.ndarray) -> sv.Detections:
            return sv.Detections(
                xyxy=np.asarray([[100.0, 200.0, 140.0, 360.0]], dtype=np.float32),
                confidence=np.asarray([0.95], dtype=np.float32),
                class_id=np.asarray([1], dtype=np.int32),
            )

    candidates = detect_image(
        image=image,
        image_record=image_record,
        handler=FakeDetector(),
        use_tiles=False,
        tile_size=1280,
        tile_overlap=256,
        detector_config=SynLocDetectorConfig(model_type="rfdetr", base_model="rf-detr-base.pth"),
    )

    assert len(candidates) == 1
    assert candidates[0].bbox_xyxy == pytest.approx([200.0, 400.0, 280.0, 720.0])


def test_end_to_end_smoke_pipeline_with_fake_detector(tmp_path: Path, monkeypatch) -> None:
    projected_position = image_points_to_pitch(
        np.asarray([[120.0, 360.0]], dtype=np.float32),
        camera_matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
        ],
        undist_poly=_identity_poly(),
        image_shape=(3, 1080, 1920),
    )[0].astype(float).tolist()
    dataset_root = _write_minimal_synloc_split(
        tmp_path,
        position_on_pitch=projected_position,
    )
    split_data = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False))

    class FakeDetector:
        def detect(self, image: np.ndarray) -> sv.Detections:
            return sv.Detections(
                xyxy=np.asarray([[100.0, 200.0, 140.0, 360.0]], dtype=np.float32),
                confidence=np.asarray([0.95], dtype=np.float32),
                class_id=np.asarray([0], dtype=np.int32),
            )

    monkeypatch.setattr(
        "tactifoot_vision.synloc.inference.build_detection_handler",
        lambda *args, **kwargs: FakeDetector(),
    )

    predictions = run_inference_on_split(
        split_data,
        dataset_config=SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False),
        detector_config=SynLocDetectorConfig(model_type="yolo", base_model="yolo11n.pt"),
        projection_config=SynLocProjectionConfig(point_strategy="bottom_center"),
    )
    metrics = evaluate_predictions(
        annotation_path=split_data.annotation_path,
        predictions=predictions,
    )
    archive_path = build_submission_archive(
        predictions,
        config=SynLocSubmissionConfig(
            split="val",
            output_dir=tmp_path / "submission",
            score_threshold=metrics["score_threshold"],
        ),
    )

    assert len(predictions) == 1
    assert metrics["map_locsim"] > 0.99
    assert archive_path.is_file()
