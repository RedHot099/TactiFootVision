"""Cutting player crops out of frames for team classification."""

import numpy as np


def extract_crops(
    frame: np.ndarray,
    boxes: np.ndarray,
    scale: float = 0.8,
    center_ratio: float = 1.0,
) -> list[np.ndarray | None]:
    """Cut the central part of each box out of ``frame``.

    Each ``xyxy`` box is first shrunk around its centre to ``scale`` of its
    width and height (and clipped to the frame), then only the central
    ``center_ratio`` of that crop is kept. Both steps are centred, so the crop
    covers roughly ``scale * center_ratio`` of the box: the torso and shorts,
    with less grass and fewer neighbouring players than the full box.

    Returns one BGR crop (a copy) per box, or ``None`` for boxes that are
    degenerate or outside the frame.
    """
    if not 0 < scale <= 1 or not 0 < center_ratio <= 1:
        raise ValueError("scale and center_ratio must be in (0, 1]")
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return [_crop(frame, box, scale, center_ratio) for box in boxes]


def _crop(
    frame: np.ndarray, box: np.ndarray, scale: float, center_ratio: float
) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    cx, cy = x1 + w / 2, y1 + h / 2
    frame_h, frame_w = frame.shape[:2]
    left = int(max(0.0, cx - w * scale / 2))
    top = int(max(0.0, cy - h * scale / 2))
    right = int(min(frame_w, cx + w * scale / 2))
    bottom = int(min(frame_h, cy + h * scale / 2))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    if center_ratio < 1:
        ch, cw = crop.shape[:2]
        keep_w = max(1, round(cw * center_ratio))
        keep_h = max(1, round(ch * center_ratio))
        x0, y0 = (cw - keep_w) // 2, (ch - keep_h) // 2
        crop = crop[y0 : y0 + keep_h, x0 : x0 + keep_w]
    return crop.copy()
