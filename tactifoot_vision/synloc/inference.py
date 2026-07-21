from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import supervision as sv

from config.models import DetectionConfig, DetectionModelType, TrainingDetectionConfig
from config.synloc_models import (
    SynLocDatasetConfig,
    SynLocDetectorConfig,
    SynLocPrediction,
    SynLocProjectionConfig,
)
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.synloc.data import SynLocImageRecord, SynLocSplitData
from tactifoot_vision.synloc.point_regressor import PointOffsetRegressor, predict_image_point
from tactifoot_vision.synloc.postprocess import merge_image_predictions, merge_world_predictions
from tactifoot_vision.synloc.projection import project_bbox_to_pitch
from tactifoot_vision.synloc.tiling import generate_tiles
from tactifoot_vision.synloc.visualize import render_pitch_debug, render_prediction_debug


@dataclass(frozen=True)
class DetectionCandidate:
    image_id: int
    score: float
    bbox_xyxy: list[float]
    category_id: int
    source_tile_xyxy: list[float] | None = None
    source_scale: int | None = None


@dataclass(frozen=True)
class InferenceImageDiagnostics:
    image_id: int
    raw_detector_outputs: int
    after_tile_merge: int
    after_projection: int
    after_image_nms: int
    after_world_nms: int
    after_final_filtering: int


@dataclass(frozen=True)
class InferenceAggregateDiagnostics:
    images_processed: int
    raw_detector_outputs: int
    after_tile_merge: int
    after_projection: int
    after_image_nms: int
    after_world_nms: int
    after_final_filtering: int
    non_empty_images: int


@dataclass(frozen=True)
class InferenceDiagnosticsSummary:
    per_image: list[InferenceImageDiagnostics]
    aggregate: InferenceAggregateDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "per_image": [asdict(item) for item in self.per_image],
            "aggregate": asdict(self.aggregate),
        }


@dataclass(frozen=True)
class InferenceRunResult:
    predictions: list[SynLocPrediction]
    summary: InferenceDiagnosticsSummary


def build_detection_handler(
    detector_config: SynLocDetectorConfig,
    *,
    model_dir: Path,
):
    detection_cfg = DetectionConfig(
        model_type=(
            DetectionModelType.YOLO
            if detector_config.model_type == "yolo"
            else DetectionModelType.RFDETR
        ),
        checkpoint_path=detector_config.checkpoint_path,
        confidence_threshold=detector_config.confidence_threshold,
        nms_threshold=detector_config.nms_threshold,
        classes={name: idx for idx, name in enumerate(detector_config.class_names)},
        include_labels=["person"],
    )
    training_cfg = None
    if detector_config.checkpoint_path is None and detector_config.base_model:
        training_cfg = TrainingDetectionConfig(
            dataset_path=Path("."),
            base_model=detector_config.base_model,
            dataset_format="coco",
        )

    if detector_config.model_type == "yolo":
        return YOLOHandler(detection_cfg, training_cfg, model_dir=model_dir)
    return RFDETRHandler(detection_cfg, training_cfg, model_dir=model_dir)


def run_inference_on_split(
    split_data: SynLocSplitData,
    *,
    dataset_config: SynLocDatasetConfig,
    detector_config: SynLocDetectorConfig,
    projection_config: SynLocProjectionConfig,
    model_dir: Path = Path("models"),
    point_regressor: PointOffsetRegressor | None = None,
    visualize_dir: Path | None = None,
) -> list[SynLocPrediction]:
    return run_inference_on_split_with_diagnostics(
        split_data,
        dataset_config=dataset_config,
        detector_config=detector_config,
        projection_config=projection_config,
        model_dir=model_dir,
        point_regressor=point_regressor,
        visualize_dir=visualize_dir,
    ).predictions


def run_inference_on_split_with_diagnostics(
    split_data: SynLocSplitData,
    *,
    dataset_config: SynLocDatasetConfig,
    detector_config: SynLocDetectorConfig,
    projection_config: SynLocProjectionConfig,
    model_dir: Path = Path("models"),
    point_regressor: PointOffsetRegressor | None = None,
    visualize_dir: Path | None = None,
) -> InferenceRunResult:
    handler = build_detection_handler(detector_config, model_dir=model_dir.resolve())
    all_predictions: list[SynLocPrediction] = []
    image_summaries: list[InferenceImageDiagnostics] = []
    for image_record in split_data.images:
        image = cv2.imread(str(image_record.file_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_record.file_path}")
        raw_candidates = detect_image(
            image=image,
            image_record=image_record,
            handler=handler,
            use_tiles=dataset_config.use_tiles,
            tile_size=detector_config.tile_size,
            tile_overlap=detector_config.tile_overlap,
            detector_config=detector_config,
        )
        merged_candidates = merge_detection_candidates(
            raw_candidates,
            image_id=image_record.image_id,
            iou_threshold=detector_config.nms_threshold,
        )
        predictions = project_candidates(
            image=image,
            image_record=image_record,
            candidates=merged_candidates,
            projection_config=projection_config,
            point_regressor=point_regressor,
        )
        projected_count = len(predictions)
        predictions = merge_image_predictions(
            predictions,
            image_nms_iou=projection_config.image_nms_iou,
        )
        image_nms_count = len(predictions)
        predictions = merge_world_predictions(
            predictions,
            world_nms_radius_m=projection_config.world_nms_radius_m,
        )
        world_nms_count = len(predictions)
        predictions = sorted(predictions, key=lambda item: item.score, reverse=True)[
            : detector_config.max_detections
        ]
        final_count = len(predictions)
        all_predictions.extend(predictions)
        image_summaries.append(
            InferenceImageDiagnostics(
                image_id=image_record.image_id,
                raw_detector_outputs=len(raw_candidates),
                after_tile_merge=len(merged_candidates),
                after_projection=projected_count,
                after_image_nms=image_nms_count,
                after_world_nms=world_nms_count,
                after_final_filtering=final_count,
            )
        )

        if visualize_dir is not None:
            render_prediction_debug(
                image=image,
                predictions=predictions,
                output_path=visualize_dir / "frames" / f"{image_record.image_id:06d}.jpg",
            )
            render_pitch_debug(
                predictions=predictions,
                output_path=visualize_dir / "pitch" / f"{image_record.image_id:06d}.jpg",
            )
    aggregate = InferenceAggregateDiagnostics(
        images_processed=len(image_summaries),
        raw_detector_outputs=sum(item.raw_detector_outputs for item in image_summaries),
        after_tile_merge=sum(item.after_tile_merge for item in image_summaries),
        after_projection=sum(item.after_projection for item in image_summaries),
        after_image_nms=sum(item.after_image_nms for item in image_summaries),
        after_world_nms=sum(item.after_world_nms for item in image_summaries),
        after_final_filtering=sum(item.after_final_filtering for item in image_summaries),
        non_empty_images=sum(1 for item in image_summaries if item.after_final_filtering > 0),
    )
    return InferenceRunResult(
        predictions=all_predictions,
        summary=InferenceDiagnosticsSummary(
            per_image=image_summaries,
            aggregate=aggregate,
        ),
    )


def detect_image(
    *,
    image: np.ndarray,
    image_record: SynLocImageRecord,
    handler,
    use_tiles: bool,
    tile_size: int,
    tile_overlap: int,
    detector_config: SynLocDetectorConfig,
) -> list[DetectionCandidate]:
    target_shape = (int(image_record.height), int(image_record.width))
    if not use_tiles:
        return _detect_at_scales(
            image=image,
            image_id=image_record.image_id,
            handler=handler,
            detector_config=detector_config,
            offset_xy=(0, 0),
            source_tile_xyxy=[0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
            target_shape=target_shape,
        )

    height, width = image.shape[:2]
    candidates: list[DetectionCandidate] = []
    for tile in generate_tiles(width, height, tile_size=tile_size, overlap=tile_overlap):
        tile_image = image[tile.y1 : tile.y2, tile.x1 : tile.x2]
        candidates.extend(
            _detect_at_scales(
                image=tile_image,
                image_id=image_record.image_id,
                handler=handler,
                detector_config=detector_config,
                offset_xy=(tile.x1, tile.y1),
                source_tile_xyxy=[float(tile.x1), float(tile.y1), float(tile.x2), float(tile.y2)],
                target_shape=target_shape,
            )
        )
    return candidates


def project_candidates(
    *,
    image: np.ndarray,
    image_record: SynLocImageRecord,
    candidates: Iterable[DetectionCandidate],
    projection_config: SynLocProjectionConfig,
    point_regressor: PointOffsetRegressor | None = None,
) -> list[SynLocPrediction]:
    predictions: list[SynLocPrediction] = []
    for candidate in candidates:
        learned_offset = None
        image_point = None
        if projection_config.point_strategy == "learned_offset" and point_regressor is not None:
            image_point = predict_image_point(point_regressor, image, candidate.bbox_xyxy)
            x1, y1, x2, y2 = candidate.bbox_xyxy
            learned_offset = [
                (image_point[0] - x1) / max(1e-6, x2 - x1),
                (image_point[1] - y1) / max(1e-6, y2 - y1),
            ]

        image_point_xy, pitch_xyz = project_bbox_to_pitch(
            candidate.bbox_xyxy,
            camera_matrix=image_record.camera_matrix,
            undist_poly=image_record.undist_poly,
            image_shape=image_record.image_shape,
            point_strategy=projection_config.point_strategy,
            learned_offset_xy=learned_offset,
        )
        prediction = SynLocPrediction(
                image_id=candidate.image_id,
                category_id=candidate.category_id,
                score=candidate.score,
                bbox_xyxy=candidate.bbox_xyxy,
                image_point_xy=image_point_xy,
                position_on_pitch_xyz=pitch_xyz,
                source_tile_xyxy=candidate.source_tile_xyxy,
                source_scale=candidate.source_scale,
                world_confidence=candidate.score,
            )
        normalized = _normalize_prediction(prediction, projection_config=projection_config)
        if normalized is not None:
            predictions.append(normalized)
    return predictions


def merge_detection_candidates(
    candidates: Iterable[DetectionCandidate],
    *,
    image_id: int,
    iou_threshold: float,
) -> list[DetectionCandidate]:
    grouped = [candidate for candidate in candidates if candidate.image_id == image_id]
    kept: list[DetectionCandidate] = []
    for candidate in sorted(grouped, key=lambda item: item.score, reverse=True):
        if all(_bbox_iou(candidate.bbox_xyxy, other.bbox_xyxy) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def _detect_at_scales(
    *,
    image: np.ndarray,
    image_id: int,
    handler,
    detector_config: SynLocDetectorConfig,
    offset_xy: tuple[int, int],
    source_tile_xyxy: list[float] | None,
    target_shape: tuple[int, int] | None = None,
) -> list[DetectionCandidate]:
    scales = detector_config.tta_scales or [max(image.shape[:2])]
    candidates: list[DetectionCandidate] = []
    for scale in scales:
        scale_factor = float(scale) / float(max(image.shape[:2]))
        if abs(scale_factor - 1.0) < 1e-6:
            resized = image
        else:
            new_width = max(1, int(round(image.shape[1] * scale_factor)))
            new_height = max(1, int(round(image.shape[0] * scale_factor)))
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        detections = handler.detect(resized)
        candidates.extend(
            _detections_to_candidates(
                detections=detections,
                image_id=image_id,
                offset_xy=offset_xy,
                detector_config=detector_config,
                scale_factor=scale_factor,
                source_tile_xyxy=source_tile_xyxy,
                source_scale=scale,
                input_shape=image.shape[:2],
                target_shape=target_shape,
            )
        )
    return candidates


def _detections_to_candidates(
    *,
    detections: sv.Detections,
    image_id: int,
    offset_xy: tuple[int, int],
    detector_config: SynLocDetectorConfig,
    scale_factor: float = 1.0,
    source_tile_xyxy: list[float] | None = None,
    source_scale: int | None = None,
    input_shape: tuple[int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
) -> list[DetectionCandidate]:
    if len(detections) == 0:
        return []
    allowed_ids = _filtered_class_ids(detector_config)
    class_ids = detections.class_id
    confidences = detections.confidence
    results: list[DetectionCandidate] = []
    scale_x, scale_y = _coordinate_scale(
        input_shape=input_shape,
        target_shape=target_shape,
    )
    for index, bbox in enumerate(detections.xyxy):
        class_id = int(class_ids[index]) if class_ids is not None else 0
        if allowed_ids and class_id not in allowed_ids:
            continue
        score = float(confidences[index]) if confidences is not None else 1.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        ox, oy = offset_xy
        results.append(
            DetectionCandidate(
                image_id=image_id,
                score=score,
                bbox_xyxy=[
                    (x1 / scale_factor + ox) * scale_x,
                    (y1 / scale_factor + oy) * scale_y,
                    (x2 / scale_factor + ox) * scale_x,
                    (y2 / scale_factor + oy) * scale_y,
                ],
                category_id=1,
                source_tile_xyxy=source_tile_xyxy,
                source_scale=source_scale,
            )
        )
    return results


def _coordinate_scale(
    *,
    input_shape: tuple[int, int] | None,
    target_shape: tuple[int, int] | None,
) -> tuple[float, float]:
    if input_shape is None or target_shape is None:
        return 1.0, 1.0
    input_h, input_w = input_shape
    target_h, target_w = target_shape
    if input_w <= 0 or input_h <= 0:
        return 1.0, 1.0
    return float(target_w) / float(input_w), float(target_h) / float(input_h)


def _filtered_class_ids(detector_config: SynLocDetectorConfig) -> set[int]:
    aliases = {"player": "person"}
    requested = {aliases.get(name, name) for name in detector_config.class_filter}
    available = {name: idx for idx, name in enumerate(detector_config.class_names)}
    matched = {available[name] for name in requested if name in available}
    if (
        detector_config.model_type == "rfdetr"
        and detector_config.class_names == ["person"]
        and matched == {0}
    ):
        matched = {0, 1}
    if matched:
        return matched
    fallback_ids = set(detector_config.person_class_ids)
    if (
        detector_config.model_type == "rfdetr"
        and fallback_ids == {0}
        and detector_config.class_names == ["person"]
    ):
        fallback_ids.add(1)
    return fallback_ids


def _normalize_prediction(
    prediction: SynLocPrediction,
    *,
    projection_config: SynLocProjectionConfig,
) -> SynLocPrediction | None:
    world = np.asarray(prediction.position_on_pitch_xyz, dtype=np.float32)
    image = np.asarray(prediction.image_point_xy, dtype=np.float32)
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(image)):
        return None

    if _is_prediction_valid(prediction, projection_config=projection_config):
        return prediction
    if projection_config.behind_camera_policy != "clip":
        return None
    clipped_position = _clip_pitch_position(
        prediction.position_on_pitch_xyz,
        clip_margin_m=projection_config.clip_margin_m,
    )
    clipped = prediction.model_copy(update={"position_on_pitch_xyz": clipped_position})
    if _is_prediction_valid(clipped, projection_config=projection_config):
        return clipped
    return None


def _is_prediction_valid(
    prediction: SynLocPrediction,
    *,
    projection_config: SynLocProjectionConfig,
) -> bool:
    x, y = float(prediction.position_on_pitch_xyz[0]), float(prediction.position_on_pitch_xyz[1])
    margin = float(projection_config.clip_margin_m)
    in_centered = -52.5 - margin <= x <= 52.5 + margin and -34.0 - margin <= y <= 34.0 + margin
    in_cornered = -margin <= x <= 105.0 + margin and -margin <= y <= 68.0 + margin
    return in_centered or in_cornered


def _clip_pitch_position(position_on_pitch_xyz: list[float], *, clip_margin_m: float) -> list[float]:
    x, y, z = [float(value) for value in position_on_pitch_xyz]
    bounds = [
        (-52.5 - clip_margin_m, 52.5 + clip_margin_m, -34.0 - clip_margin_m, 34.0 + clip_margin_m),
        (-clip_margin_m, 105.0 + clip_margin_m, -clip_margin_m, 68.0 + clip_margin_m),
    ]
    candidates: list[tuple[float, list[float]]] = []
    for min_x, max_x, min_y, max_y in bounds:
        clipped_x = min(max(x, min_x), max_x)
        clipped_y = min(max(y, min_y), max_y)
        distance = float((clipped_x - x) ** 2 + (clipped_y - y) ** 2)
        candidates.append((distance, [clipped_x, clipped_y, z]))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _bbox_iou(lhs: list[float], rhs: list[float]) -> float:
    ax1, ay1, ax2, ay2 = lhs
    bx1, by1, bx2, by2 = rhs
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
