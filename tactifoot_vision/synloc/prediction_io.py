from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.synloc_models import SynLocPrediction
from tactifoot_vision.synloc.camera import image_points_to_pitch
from tactifoot_vision.synloc.data import SynLocSplitData


def load_predictions_from_results_json(
    results_path: Path,
    *,
    split_data: SynLocSplitData | None = None,
    position_from_keypoint_index: int | None = None,
) -> list[SynLocPrediction]:
    raw_results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return [
        result_item_to_prediction(
            item,
            split_data=split_data,
            position_from_keypoint_index=position_from_keypoint_index,
        )
        for item in raw_results
    ]


def result_item_to_prediction(
    item: dict[str, Any],
    *,
    split_data: SynLocSplitData | None = None,
    position_from_keypoint_index: int | None = None,
) -> SynLocPrediction:
    image_id = int(item["image_id"])
    bbox_xywh = [float(v) for v in item.get("bbox", [0.0, 0.0, 0.0, 0.0])]
    image_point_xy = _extract_image_point(item, position_from_keypoint_index=position_from_keypoint_index)
    position_on_pitch = _extract_position_on_pitch(
        item,
        image_id=image_id,
        image_point_xy=image_point_xy,
        split_data=split_data,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    return SynLocPrediction(
        image_id=image_id,
        category_id=int(item.get("category_id", 1)),
        score=float(item["score"]),
        bbox_xyxy=_xywh_to_xyxy(bbox_xywh),
        image_point_xy=image_point_xy,
        position_on_pitch_xyz=position_on_pitch,
    )


def _extract_image_point(
    item: dict[str, Any],
    *,
    position_from_keypoint_index: int | None,
) -> list[float]:
    keypoints = item.get("keypoints")
    if keypoints is None:
        bbox = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x, y, w, h = [float(v) for v in bbox]
        return [x + w / 2.0, y + h]

    flat = _flatten_keypoints(keypoints)
    if not flat:
        return [0.0, 0.0]
    keypoint_index = 1 if position_from_keypoint_index is None else int(position_from_keypoint_index)
    offset = keypoint_index * 3
    if len(flat) >= offset + 2:
        return [float(flat[offset]), float(flat[offset + 1])]
    return [0.0, 0.0]


def _extract_position_on_pitch(
    item: dict[str, Any],
    *,
    image_id: int,
    image_point_xy: list[float],
    split_data: SynLocSplitData | None,
    position_from_keypoint_index: int | None,
) -> list[float]:
    if "position_on_pitch" in item:
        values = [float(v) for v in item["position_on_pitch"]]
        if len(values) == 2:
            values.append(0.0)
        return values[:3]

    if split_data is not None and position_from_keypoint_index is not None:
        image_record = split_data.images_by_id.get(image_id)
        if image_record is not None:
            projected = image_points_to_pitch(
                [image_point_xy],
                camera_matrix=image_record.camera_matrix,
                undist_poly=image_record.undist_poly,
                image_shape=image_record.image_shape,
            )[0]
            return [float(projected[0]), float(projected[1]), float(projected[2])]

    return [0.0, 0.0, 0.0]


def _flatten_keypoints(raw_keypoints: Any) -> list[float]:
    if not isinstance(raw_keypoints, list):
        return []
    if raw_keypoints and isinstance(raw_keypoints[0], list):
        flattened: list[float] = []
        for triplet in raw_keypoints:
            if isinstance(triplet, list):
                flattened.extend(float(v) for v in triplet[:3])
        return flattened
    return [float(v) for v in raw_keypoints]


def _xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]
