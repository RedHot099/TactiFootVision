import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import BBox
from tactifoot_vision.team_assignment.results import CropSample


def crop_bbox(
    image: NDArray[np.uint8], bbox: BBox, *, ratio: float = 1.0
) -> NDArray[np.uint8]:
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    width = bbox.width * ratio
    height = bbox.height * ratio
    x1 = max(0, int(round(cx - width / 2.0)))
    y1 = max(0, int(round(cy - height / 2.0)))
    x2 = min(image.shape[1], int(round(cx + width / 2.0)))
    y2 = min(image.shape[0], int(round(cy + height / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return np.empty(
            (0, 0, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype
        )
    return image[y1:y2, x1:x2]


def crop_sample(
    *,
    frame_index: int,
    track_id: int,
    class_name: str,
    image: NDArray[np.uint8],
    bbox: BBox,
    ratio: float = 1.0,
    team_label: int | None = None,
) -> CropSample:
    return CropSample(
        frame_index=frame_index,
        track_id=track_id,
        class_name=class_name,
        bbox=bbox,
        image=crop_bbox(image, bbox, ratio=ratio),
        team_label=team_label,
    )
