from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import supervision as sv

from config.synloc_models import (
    SynLocDatasetConfig,
    SynLocDetectorConfig,
    SynLocProjectionConfig,
)
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.inference import run_inference_on_split_with_diagnostics
from tactifoot_vision.synloc.plan24h import (
    ExperimentRecord,
    FinalistSummary,
    PhaseRunSelection,
    finalize_24h_run,
)


def _identity_poly() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _write_minimal_synloc_split(root: Path, split: str = "val") -> Path:
    dataset_root = root / "SoccerNet" / "SpiideoSynLoc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / split).mkdir(parents=True, exist_ok=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image_path = dataset_root / split / "scene_0001.jpg"
    import cv2

    cv2.imwrite(str(image_path), image)
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
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
                "position_on_pitch": [2.0, 3.0, 0.0],
                "bbox": [100.0, 200.0, 40.0, 160.0],
                "area": 6400.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (dataset_root / "annotations" / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return dataset_root


def test_run_inference_on_split_with_diagnostics_tracks_stage_counts(tmp_path: Path, monkeypatch) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
    split_data = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False))

    class FakeDetector:
        def detect(self, image: np.ndarray) -> sv.Detections:
            return sv.Detections(
                xyxy=np.asarray(
                    [
                        [100.0, 200.0, 140.0, 360.0],
                        [200.0, 200.0, 240.0, 360.0],
                    ],
                    dtype=np.float32,
                ),
                confidence=np.asarray([0.95, 0.80], dtype=np.float32),
                class_id=np.asarray([0, 0], dtype=np.int32),
            )

    outputs = iter(
        [
            ([120.0, 360.0], [2.0, 3.0, 0.0]),
            ([220.0, 360.0], [200.0, 3.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        "tactifoot_vision.synloc.inference.build_detection_handler",
        lambda *args, **kwargs: FakeDetector(),
    )
    monkeypatch.setattr(
        "tactifoot_vision.synloc.inference.project_bbox_to_pitch",
        lambda *args, **kwargs: next(outputs),
    )

    result = run_inference_on_split_with_diagnostics(
        split_data,
        dataset_config=SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False),
        detector_config=SynLocDetectorConfig(model_type="yolo", base_model="yolo11n.pt"),
        projection_config=SynLocProjectionConfig(point_strategy="bottom_center", behind_camera_policy="drop"),
    )

    assert len(result.predictions) == 1
    assert result.summary.aggregate.raw_detector_outputs == 2
    assert result.summary.aggregate.after_projection == 1
    assert result.summary.aggregate.after_image_nms == 1
    assert result.summary.aggregate.after_world_nms == 1
    assert result.summary.aggregate.after_final_filtering == 1
    assert result.summary.aggregate.non_empty_images == 1
    assert result.summary.per_image[0].image_id == 1


def test_run_inference_on_split_with_diagnostics_clip_policy_keeps_prediction(tmp_path: Path, monkeypatch) -> None:
    dataset_root = _write_minimal_synloc_split(tmp_path)
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
    monkeypatch.setattr(
        "tactifoot_vision.synloc.inference.project_bbox_to_pitch",
        lambda *args, **kwargs: ([120.0, 360.0], [200.0, 3.0, 0.0]),
    )

    result = run_inference_on_split_with_diagnostics(
        split_data,
        dataset_config=SynLocDatasetConfig(root=dataset_root, split="val", use_tiles=False),
        detector_config=SynLocDetectorConfig(model_type="yolo", base_model="yolo11n.pt"),
        projection_config=SynLocProjectionConfig(
            point_strategy="bottom_center",
            behind_camera_policy="clip",
            clip_margin_m=0.5,
        ),
    )

    assert len(result.predictions) == 1
    assert result.predictions[0].position_on_pitch_xyz[0] == 105.5
    assert result.summary.aggregate.after_projection == 1


def test_finalize_24h_run_writes_required_artifacts(tmp_path: Path) -> None:
    best_run_dir = tmp_path / "best-run"
    best_run_dir.mkdir()
    (best_run_dir / "results.json").write_text(json.dumps([{"image_id": 1, "score": 0.9}]), encoding="utf-8")
    (best_run_dir / "metadata.json").write_text(json.dumps({"score_threshold": 0.4}), encoding="utf-8")
    archive_path = best_run_dir / "submission.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("results.json", "[]")
        archive.writestr("metadata.json", "{}")

    one_drive_dir = tmp_path / "OneDrive" / "Uczelnia" / "Konferencje" / "synloc_24h_run"
    selection = PhaseRunSelection(
        run_name="baseline_full_val",
        run_dir=best_run_dir,
        archive_path=archive_path,
        results_path=best_run_dir / "results.json",
        metadata_path=best_run_dir / "metadata.json",
        raw_results_path=best_run_dir / "results.json",
        metrics={
            "map_locsim": 0.123,
            "precision": 0.4,
            "recall": 0.5,
            "f1": 0.444,
            "score_threshold": 0.4,
        },
        config_snapshot={"projection": {"point_strategy": "bottom_center"}},
        notes="best variant",
    )
    artifacts = finalize_24h_run(
        output_dir=one_drive_dir,
        best_run=selection,
        finalists=[
            FinalistSummary(
                run_name="baseline_full_val",
                phase="phase2",
                map_locsim=0.123,
                precision=0.4,
                recall=0.5,
                f1=0.444,
                score_threshold=0.4,
                final_predictions=1,
                archive_path=archive_path,
            )
        ],
        experiments=[
            ExperimentRecord(
                timestamp="2026-04-24T10:00:00Z",
                phase="phase2",
                run_name="baseline_full_val",
                detector_checkpoint="rf-detr-base.pth",
                point_strategy="bottom_center",
                confidence_threshold=0.2,
                image_nms_iou=0.6,
                world_nms_radius_m=0.75,
                behind_camera_policy="drop",
                clip_margin_m=0.0,
                tile_size=1280,
                tile_overlap=256,
                topk_per_image=250,
                aux_data_used="",
                val_map_locsim=0.123,
                val_recall=0.5,
                val_precision=0.4,
                val_f1=0.444,
                notes="best variant",
            )
        ],
        run_started_at="2026-04-24T10:00:00Z",
        run_finished_at="2026-04-24T12:00:00Z",
    )

    assert artifacts.best_submission_zip.is_file()
    assert artifacts.best_results_json.is_file()
    assert artifacts.best_metadata_json.is_file()
    assert artifacts.val_metrics_summary_json.is_file()
    assert artifacts.experiments_csv.is_file()
    assert artifacts.run_report_md.is_file()

    with zipfile.ZipFile(artifacts.best_submission_zip) as archive:
        assert sorted(archive.namelist()) == ["metadata.json", "results.json"]

    with artifacts.experiments_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_name"] == "baseline_full_val"
    assert set(rows[0]) >= {
        "timestamp",
        "phase",
        "run_name",
        "detector_checkpoint",
        "point_strategy",
        "confidence_threshold",
        "image_nms_iou",
        "world_nms_radius_m",
        "behind_camera_policy",
        "clip_margin_m",
        "tile_size",
        "tile_overlap",
        "topk_per_image",
        "aux_data_used",
        "val_map_locsim",
        "val_recall",
        "val_precision",
        "val_f1",
        "notes",
    }
    report_text = artifacts.run_report_md.read_text(encoding="utf-8")
    assert "baseline_full_val" in report_text
    assert str(artifacts.best_submission_zip) in report_text
