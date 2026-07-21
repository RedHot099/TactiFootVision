from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def filter_segments_by_distance(
    mask: NDArray[np.bool_], distance_threshold: float
) -> NDArray[np.bool_]:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if num_labels <= 1:
        return mask
    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_centroid = centroids[main_label]
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        centroid = centroids[label]
        if label == main_label:
            filtered[labels == label] = True
            continue
        distance = float(np.linalg.norm(centroid - main_centroid))
        if distance <= distance_threshold:
            filtered[labels == label] = True
    return filtered


def postprocess_masks(
    masks: NDArray[np.bool_],
    *,
    open_kernel_size: int,
    close_kernel_size: int,
) -> NDArray[np.bool_]:
    masks_batch = _as_mask_batch(masks)
    if open_kernel_size <= 0 and close_kernel_size <= 0:
        return masks_batch
    open_kernel = _kernel(open_kernel_size)
    close_kernel = _kernel(close_kernel_size)
    processed: list[NDArray[np.bool_]] = []
    for mask in masks_batch:
        mask_uint8 = mask.astype(np.uint8)
        if open_kernel is not None:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
        if close_kernel is not None:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
        processed.append(mask_uint8.astype(bool))
    return np.array(processed, dtype=bool)


def tight_bbox_from_masks(
    masks: NDArray[np.bool_],
    *,
    min_mask_area: float,
) -> NDArray[np.float32]:
    masks_batch = _as_mask_batch(masks)
    if len(masks_batch) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes: list[list[float]] = []
    for mask in masks_batch:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= min_mask_area
        ]
        if not valid_contours:
            if mask.any():
                y_indices, x_indices = np.where(mask)
                boxes.append(
                    [
                        float(x_indices.min()),
                        float(y_indices.min()),
                        float(x_indices.max()),
                        float(y_indices.max()),
                    ]
                )
            else:
                boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest_contour)
        boxes.append([float(x), float(y), float(x + width), float(y + height)])
    return np.array(boxes, dtype=np.float32)


def _as_mask_batch(masks: NDArray[np.bool_]) -> NDArray[np.bool_]:
    array = np.asarray(masks, dtype=bool)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    if array.size == 0:
        return np.empty((0, 0, 0), dtype=bool)
    raise ValueError("SAM2 masks must be a 2D mask or a 3D mask batch.")


def _kernel(size: int) -> NDArray[Any] | None:
    if size <= 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
