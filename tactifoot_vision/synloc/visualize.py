from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config.synloc_models import SynLocPrediction


def render_prediction_debug(
    *,
    image: np.ndarray,
    predictions: Iterable[SynLocPrediction],
    output_path: Path,
) -> Path:
    canvas = image.copy()
    for prediction in predictions:
        x1, y1, x2, y2 = [int(round(v)) for v in prediction.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        px, py = [int(round(v)) for v in prediction.image_point_xy]
        cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)
        label = f"{prediction.score:.2f} -> ({prediction.position_on_pitch_xyz[0]:.1f}, {prediction.position_on_pitch_xyz[1]:.1f})"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def render_pitch_debug(
    *,
    predictions: Iterable[SynLocPrediction],
    output_path: Path,
    canvas_size: tuple[int, int] = (840, 540),
    pitch_bounds: tuple[tuple[float, float], tuple[float, float]] = ((-55.0, 55.0), (-37.0, 37.0)),
) -> Path:
    width, height = canvas_size
    canvas = np.full((height, width, 3), (27, 83, 41), dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (width - 20, height - 20), (255, 255, 255), 2)
    x_bounds, y_bounds = pitch_bounds
    for prediction in predictions:
        x, y = [float(v) for v in prediction.position_on_pitch_xyz[:2]]
        px = int(np.interp(x, [x_bounds[0], x_bounds[1]], [20, width - 20]))
        py = int(np.interp(y, [y_bounds[1], y_bounds[0]], [20, height - 20]))
        cv2.circle(canvas, (px, py), 5, (0, 215, 255), -1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
