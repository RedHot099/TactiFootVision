from __future__ import annotations

from collections import defaultdict
from math import hypot

from config.synloc_models import SynLocPrediction


def merge_image_predictions(
    predictions: list[SynLocPrediction],
    *,
    image_nms_iou: float,
) -> list[SynLocPrediction]:
    merged: list[SynLocPrediction] = []
    by_image: dict[int, list[SynLocPrediction]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction.image_id].append(prediction)

    for image_id in sorted(by_image):
        kept: list[SynLocPrediction] = []
        for candidate in sorted(by_image[image_id], key=lambda item: item.score, reverse=True):
            if all(_bbox_iou(candidate.bbox_xyxy, kept_item.bbox_xyxy) < image_nms_iou for kept_item in kept):
                kept.append(candidate)
        merged.extend(kept)
    return merged


def merge_world_predictions(
    predictions: list[SynLocPrediction],
    *,
    world_nms_radius_m: float,
) -> list[SynLocPrediction]:
    merged: list[SynLocPrediction] = []
    by_image: dict[int, list[SynLocPrediction]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction.image_id].append(prediction)

    for image_id in sorted(by_image):
        kept: list[SynLocPrediction] = []
        for candidate in sorted(by_image[image_id], key=lambda item: item.score, reverse=True):
            if all(_world_distance(candidate, kept_item) > world_nms_radius_m for kept_item in kept):
                kept.append(candidate)
        merged.extend(kept)
    return merged


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


def _world_distance(lhs: SynLocPrediction, rhs: SynLocPrediction) -> float:
    return hypot(
        float(lhs.position_on_pitch_xyz[0]) - float(rhs.position_on_pitch_xyz[0]),
        float(lhs.position_on_pitch_xyz[1]) - float(rhs.position_on_pitch_xyz[1]),
    )
