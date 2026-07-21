import cv2
import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import BBox
from tactifoot_vision.team_assignment.crops import crop_bbox


def opencv_mask_crop(
    image: NDArray[np.uint8],
    bbox: BBox,
    *,
    margin_ratio: float = 0.05,
    iter_count: int = 3,
) -> NDArray[np.uint8] | None:
    crop = crop_bbox(image, bbox, ratio=min(1.0, 1.0 + margin_ratio))
    if crop.size == 0:
        return None
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    rect = (1, 1, max(1, crop.shape[1] - 2), max(1, crop.shape[0] - 2))
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iter_count, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return crop
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
        "uint8"
    )
    return crop * foreground[:, :, None]
