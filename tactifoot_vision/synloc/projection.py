from __future__ import annotations

from typing import Sequence

import numpy as np

from config.synloc_models import SynLocPointStrategy
from tactifoot_vision.synloc.camera import image_points_to_pitch


def image_point_from_bbox(
    bbox_xyxy: Sequence[float],
    *,
    point_strategy: SynLocPointStrategy = "bottom_center",
    learned_offset_xy: Sequence[float] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if point_strategy == "learned_offset" and learned_offset_xy is not None:
        dx, dy = [float(v) for v in learned_offset_xy]
        return np.asarray([x1 + dx * (x2 - x1), y1 + dy * (y2 - y1)], dtype=np.float32)
    return np.asarray([(x1 + x2) / 2.0, y2], dtype=np.float32)


def project_bbox_to_pitch(
    bbox_xyxy: Sequence[float],
    *,
    camera_matrix: Sequence[Sequence[float]],
    undist_poly: Sequence[float],
    image_shape: tuple[int, int, int],
    point_strategy: SynLocPointStrategy = "bottom_center",
    learned_offset_xy: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    image_point = image_point_from_bbox(
        bbox_xyxy,
        point_strategy=point_strategy,
        learned_offset_xy=learned_offset_xy,
    )
    pitch_point = image_points_to_pitch(
        image_point.reshape(1, 2),
        camera_matrix,
        undist_poly,
        image_shape=image_shape,
    )[0]
    return image_point.astype(float).tolist(), pitch_point.astype(float).tolist()
