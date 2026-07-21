"""Integration-style test for running the detection and tracking pipeline on SoccerNet.

The test expects the user to provide the path to an extracted SoccerNet tracking
sequence (the ``img1`` directory with frames) through the environment variable
``SOCCERNET_TRACKING_SEQUENCE_DIR``. Results are written to a CSV file that can
be inspected later, along with a MOT-format text file and a small JSON summary.
"""

from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv
from scipy.optimize import linear_sum_assignment

from config.models import DetectionConfig, DetectionModelType, TrackingConfig, SAM2Config
from tactifoot_vision.detection.base_handler import BaseHandler
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.tracking.botsort_tracker import BoTSORTArgs, BoTSORTTracker
from tactifoot_vision.tracking.tracker import Tracker

try:  # pragma: no cover - optional dependency
    from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
except ImportError:  # pragma: no cover
    RFDETRHandler = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler
except ImportError:  # pragma: no cover
    RFDETRSegHandler = None  # type: ignore[assignment]

EXPECTED_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "confidence",
    "class_id",
    "class_name",
]

MOT_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "confidence",
    "x3",
    "y3",
    "z3",
]


@dataclass
class SoccerNetTestSettings:
    sequence_dir: Path
    detection_model: DetectionModelType
    checkpoint_path: Path
    max_frames: int
    output_dir: Path
    confidence_threshold: float
    nms_threshold: float
    include_labels: List[str] | None
    tracking_backend: str
    sam2_checkpoint: Path | None
    sam2_config: Path | None

    @classmethod
    def from_env(cls) -> "SoccerNetTestSettings":
        """Create settings for the test from environment variables."""
        seq_dir_env = os.environ.get("SOCCERNET_TRACKING_SEQUENCE_DIR")
        if not seq_dir_env:
            pytest.skip("Set SOCCERNET_TRACKING_SEQUENCE_DIR to run this test.")
        sequence_dir = Path(seq_dir_env).expanduser().resolve()
        if not sequence_dir.is_dir():
            pytest.skip(f"Sequence directory not found: {sequence_dir}")

        model_type_str = os.environ.get("TACTIFOOT_TEST_MODEL_TYPE", "rfdetr_seg")
        try:
            detection_model = DetectionModelType(model_type_str)
        except ValueError as exc:
            pytest.skip(f"Unsupported detection model '{model_type_str}': {exc}")

        checkpoint_env = os.environ.get("TACTIFOOT_TEST_CHECKPOINT")
        if checkpoint_env:
            checkpoint_path = Path(checkpoint_env).expanduser().resolve()
        else:
            default_weights = (
                Path(__file__).resolve().parents[1] / "rf-detr-seg-preview.pt"
            )
            if detection_model == DetectionModelType.RFDETR_SEG and default_weights.is_file():
                checkpoint_path = default_weights
            else:
                pytest.skip(
                    "Set TACTIFOOT_TEST_CHECKPOINT to a valid detection weights file."
                )
        if not checkpoint_path.is_file():
            pytest.skip(f"Detection checkpoint not found: {checkpoint_path}")

        max_frames = int(os.environ.get("TACTIFOOT_TEST_MAX_FRAMES", "40"))
        if max_frames <= 0:
            max_frames = 1

        output_dir = Path(
            os.environ.get("TACTIFOOT_TEST_OUTPUT_DIR", "results/soccernet_tracking")
        ).expanduser().resolve()

        confidence_threshold = float(os.environ.get("TACTIFOOT_TEST_CONFIDENCE", "0.3"))
        nms_threshold = float(os.environ.get("TACTIFOOT_TEST_NMS", "0.5"))

        include_labels_env = os.environ.get("TACTIFOOT_TEST_INCLUDE_LABELS")
        include_labels = None
        if include_labels_env:
            parsed = [label.strip() for label in include_labels_env.split(",")]
            include_labels = [label for label in parsed if label]
            if not include_labels:
                include_labels = None

        tracking_backend = os.environ.get(
            "TACTIFOOT_TEST_TRACKING_BACKEND", "bytetrack"
        ).strip().lower()

        sam2_checkpoint_env = os.environ.get("TACTIFOOT_TEST_SAM2_CHECKPOINT")
        sam2_config_env = os.environ.get("TACTIFOOT_TEST_SAM2_CONFIG")
        sam2_checkpoint_path: Path | None = (
            Path(sam2_checkpoint_env).expanduser().resolve()
            if sam2_checkpoint_env
            else None
        )
        sam2_config_path: Path | None = (
            Path(sam2_config_env).expanduser().resolve()
            if sam2_config_env
            else None
        )
        if tracking_backend == "sam2":
            if not sam2_checkpoint_path or not sam2_checkpoint_path.is_file():
                pytest.skip(
                    "SAM2 backend selected but TACTIFOOT_TEST_SAM2_CHECKPOINT is missing or invalid."
                )
            if not sam2_config_path or not sam2_config_path.is_file():
                pytest.skip(
                    "SAM2 backend selected but TACTIFOOT_TEST_SAM2_CONFIG is missing or invalid."
                )

        return cls(
            sequence_dir=sequence_dir,
            detection_model=detection_model,
            checkpoint_path=checkpoint_path,
            max_frames=max_frames,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            include_labels=include_labels,
            tracking_backend=tracking_backend,
            sam2_checkpoint=sam2_checkpoint_path,
            sam2_config=sam2_config_path,
        )


def _gather_frame_paths(sequence_dir: Path) -> List[Path]:
    """Collect frame image paths from the SoccerNet sequence."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = [
        item
        for item in sequence_dir.iterdir()
        if item.is_file() and item.suffix.lower() in valid_ext
    ]
    return sorted(frame_paths, key=lambda path: path.name)


def _infer_frame_rate(sequence_dir: Path) -> float:
    """Infer sequence FPS from seqinfo.ini if available; default to 25 FPS."""
    seqinfo_path = sequence_dir.parent / "seqinfo.ini"
    if not seqinfo_path.is_file():
        return 25.0
    parser = configparser.ConfigParser()
    try:
        parser.read(seqinfo_path)
    except configparser.Error:
        return 25.0
    try:
        if parser.has_option("Sequence", "frameRate"):
            return float(parser.get("Sequence", "frameRate"))
    except ValueError:
        return 25.0
    return 25.0


def _instantiate_detection_handler(det_config: DetectionConfig) -> BaseHandler:
    """Instantiate the correct detection handler based on the configuration."""
    model_dir = (
        det_config.checkpoint_path.parent
        if det_config.checkpoint_path is not None
        else Path.cwd()
    )
    if det_config.model_type == DetectionModelType.YOLO:
        return YOLOHandler(det_config, model_dir=model_dir)
    if det_config.model_type == DetectionModelType.RFDETR:
        if RFDETRHandler is None:
            pytest.skip("RF-DETR handler not available in this environment.")
        return RFDETRHandler(det_config, model_dir=model_dir)
    if det_config.model_type == DetectionModelType.RFDETR_SEG:
        if RFDETRSegHandler is None:
            pytest.skip("RF-DETR Seg handler not available in this environment.")
        return RFDETRSegHandler(det_config, model_dir=model_dir)
    pytest.skip(f"Unsupported detection model: {det_config.model_type}")


def _load_frames(frame_paths: Sequence[Path]) -> List[np.ndarray]:
    """Load image frames using OpenCV, skipping unreadable files."""
    frames: List[np.ndarray] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            frames.append(frame)
    return frames


def _run_tracking(
    frames: Sequence[np.ndarray],
    detection_handler: BaseHandler,
    detection_config: DetectionConfig,
    tracking_backend: str,
    tracker_config: TrackingConfig,
) -> pd.DataFrame:
    """Run detection + tracking over frames and collect MOT-style rows."""
    backend = tracking_backend.lower()
    class_lookup = {v: k for k, v in detection_config.classes.items()}
    ball_class_id = detection_config.classes.get("ball")

    tracker: Tracker | None = None
    botsort_tracker: BoTSORTTracker | None = None
    sam2_tracker = None
    sam2_initialized = False

    if backend == "sam2":
        try:
            from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker
        except ImportError as sam2_err:  # pragma: no cover
            pytest.skip(f"SAM2 tracker unavailable: {sam2_err}")
        sam2_tracker = SAM2Tracker(tracker_config)
    elif backend == "botsort":
        reid_model = str((Path(__file__).resolve().parents[1] / "yolo11n.pt").resolve())
        botsort_tracker = BoTSORTTracker(
            BoTSORTArgs(with_reid=True, model=reid_model),
            frame_rate=int(tracker_config.frame_rate or 30),
        )
    else:
        tracker = Tracker(tracker_config)

    rows: List[dict] = []

    for frame_idx, frame in enumerate(frames, start=1):
        detections = detection_handler.detect(frame)
        usable_detections = detections
        if (
            ball_class_id is not None
            and detections.class_id is not None
            and len(detections) > 0
        ):
            try:
                mask_players = detections.class_id != int(ball_class_id)
                usable_detections = detections[mask_players]
            except Exception:
                usable_detections = detections

        if backend == "sam2":
            if sam2_tracker is None:
                continue
            if not sam2_initialized:
                if len(usable_detections) == 0:
                    continue
                init_boxes = usable_detections.xyxy
                init_classes = (
                    usable_detections.class_id
                    if usable_detections.class_id is not None
                    else None
                )
                sam2_tracker.initialize(frame, init_boxes, init_classes)
                tracked = sam2_tracker.track(frame)
                sam2_initialized = True
            else:
                tracked = sam2_tracker.track(frame)
                if len(usable_detections) > 0:
                    det_boxes = usable_detections.xyxy.astype(np.float32)
                    tracked_boxes = (
                        tracked.xyxy.astype(np.float32)
                        if len(tracked) > 0
                        else np.empty((0, 4), dtype=np.float32)
                    )
                    if det_boxes.size > 0:
                        iou_matrix = (
                            sv.box_iou_batch(tracked_boxes, det_boxes)
                            if tracked_boxes.size and det_boxes.size
                            else np.zeros((len(tracked_boxes), len(det_boxes)))
                        )
                        matched_det = set()
                        if iou_matrix.size:
                            row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
                            for r, c in zip(row_ind, col_ind):
                                if iou_matrix[r, c] >= 0.5:
                                    matched_det.add(int(c))
                        new_indices = [
                            idx for idx in range(len(det_boxes)) if idx not in matched_det
                        ]
                        if new_indices:
                            new_boxes = det_boxes[new_indices]
                            new_classes = (
                                usable_detections.class_id[new_indices].astype(int)
                                if usable_detections.class_id is not None
                                else np.full(len(new_boxes), -1, dtype=int)
                            )
                            existing_boxes = (
                                tracked_boxes
                                if tracked_boxes.size
                                else np.empty((0, 4), dtype=np.float32)
                            )
                            existing_ids = (
                                tracked.tracker_id.astype(int)
                                if tracked.tracker_id is not None
                                else np.empty((0,), dtype=int)
                            )
                            new_ids = sam2_tracker.allocate_ids(len(new_boxes))
                            combined_boxes = (
                                np.vstack([existing_boxes, new_boxes])
                                if existing_boxes.size
                                else new_boxes
                            )
                            combined_classes = (
                                np.concatenate(
                                    [
                                        tracked.class_id.astype(int)
                                        if tracked.class_id is not None
                                        else np.full(len(existing_boxes), -1, dtype=int),
                                        new_classes,
                                    ]
                                )
                                if existing_boxes.size
                                else new_classes
                            )
                            combined_ids = (
                                np.concatenate([existing_ids, new_ids])
                                if existing_ids.size
                                else new_ids
                            )
                            refresh_result = sam2_tracker.refresh_prompts(
                                frame,
                                combined_boxes,
                                combined_classes,
                                combined_ids,
                            )
                            tracked = refresh_result
        else:
            if backend == "botsort":
                if botsort_tracker is None:
                    continue
                tracked = botsort_tracker.update(usable_detections, frame)
            elif tracker is None:
                continue
            else:
                tracked = tracker.update(usable_detections)

        if tracked is None or len(tracked) == 0:
            continue

        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for det_idx in range(len(tracked)):
            if det_idx >= len(tracked.xyxy):
                continue

            tracker_id = None
            if det_idx < len(tracker_ids):
                tracker_id = tracker_ids[det_idx]
            if tracker_id is None:
                continue

            bbox = tracked.xyxy[det_idx]
            x1, y1, x2, y2 = [float(val) for val in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            conf_value = None
            if tracked.confidence is not None and det_idx < len(tracked.confidence):
                candidate = tracked.confidence[det_idx]
                if candidate is not None:
                    try:
                        numeric = float(candidate)
                        if not np.isnan(numeric):
                            conf_value = numeric
                    except (TypeError, ValueError):
                        conf_value = None
            if conf_value is None and backend == "sam2":
                conf_value = 1.0

            class_value = -1
            if tracked.class_id is not None and det_idx < len(tracked.class_id):
                try:
                    class_value = int(tracked.class_id[det_idx])
                except (TypeError, ValueError):
                    class_value = -1

            rows.append(
                {
                    "frame": frame_idx,
                    "track_id": int(tracker_id),
                    "x": round(x1, 3),
                    "y": round(y1, 3),
                    "width": round(width, 3),
                    "height": round(height, 3),
                    "confidence": None if conf_value is None else round(conf_value, 4),
                    "class_id": class_value,
                    "class_name": class_lookup.get(class_value, "unknown"),
                }
            )

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _save_results(df: pd.DataFrame, output_stem: Path) -> Tuple[Path, Path]:
    """Save CSV with headers and MOT-compatible txt output."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    mot_df = df.copy()
    if "confidence" in mot_df.columns:
        mot_df["confidence"] = mot_df["confidence"].fillna(0.0).astype(float)
    else:
        mot_df["confidence"] = 0.0
    mot_df["x3"] = -1
    mot_df["y3"] = -1
    mot_df["z3"] = -1
    mot_path = output_stem.with_suffix(".mot.txt")
    mot_df[MOT_COLUMNS].to_csv(
        mot_path, header=False, index=False, float_format="%.3f"
    )
    return csv_path, mot_path


@pytest.mark.slow
def test_soccernet_tracking_pipeline():
    """Run detection + tracking on a SoccerNet clip and export results."""
    settings = SoccerNetTestSettings.from_env()

    frame_paths = _gather_frame_paths(settings.sequence_dir)
    if not frame_paths:
        pytest.skip(f"No frames found in {settings.sequence_dir}")
    frame_paths = frame_paths[: settings.max_frames]

    frames = _load_frames(frame_paths)
    if not frames:
        pytest.skip("No frames could be loaded for the provided sequence.")

    frame_rate = _infer_frame_rate(settings.sequence_dir)
    detection_config = DetectionConfig(
        model_type=settings.detection_model,
        checkpoint_path=settings.checkpoint_path,
        confidence_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
        classes={"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3},
        include_labels=settings.include_labels,
    )

    detection_handler = _instantiate_detection_handler(detection_config)
    tracking_backend = settings.tracking_backend
    sam2_cfg: SAM2Config | None = None
    if tracking_backend == "sam2":
        assert settings.sam2_checkpoint is not None
        assert settings.sam2_config is not None
        sam2_cfg = SAM2Config(
            checkpoint_path=settings.sam2_checkpoint,
            config_path=settings.sam2_config,
            mask_filter_distance=300.0,
            reseed_interval=45,
            reseed_iou_threshold=0.3,
        )
    tracker_config = TrackingConfig(
        enabled=True,
        backend=tracking_backend,
        frame_rate=int(max(1, round(frame_rate))),
        sam2=sam2_cfg,
    )

    detections_df = _run_tracking(
        frames,
        detection_handler,
        detection_config,
        tracking_backend,
        tracker_config,
    )
    sequence_name = settings.sequence_dir.parent.name
    model_stem = settings.checkpoint_path.stem
    output_stem = (
        settings.output_dir
        / f"{sequence_name}_{settings.detection_model.value}_{model_stem}"
    )
    csv_path, mot_path = _save_results(detections_df, output_stem)

    summary = {
        "sequence_dir": str(settings.sequence_dir),
        "frame_rate": frame_rate,
        "frames_requested": len(frame_paths),
        "frames_processed": len(frames),
        "detections": int(detections_df.shape[0]),
        "unique_tracks": int(detections_df["track_id"].nunique())
        if not detections_df.empty
        else 0,
        "detection_model": settings.detection_model.value,
        "checkpoint_path": str(settings.checkpoint_path),
        "confidence_threshold": settings.confidence_threshold,
        "nms_threshold": settings.nms_threshold,
        "tracking_backend": tracking_backend,
        "csv_path": str(csv_path),
        "mot_path": str(mot_path),
    }
    summary_path = output_stem.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    assert csv_path.is_file(), "Tracking CSV output was not created."
    assert mot_path.is_file(), "MOT-format output was not created."
    assert summary_path.is_file(), "Summary JSON output was not created."

    loaded_df = pd.read_csv(csv_path)
    assert list(loaded_df.columns) == EXPECTED_COLUMNS
    assert summary["frames_processed"] == len(frames)
