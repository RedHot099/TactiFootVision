"""Labels of a single image, shared by data preparation, augmentation, training and plots."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import supervision as sv


class Task(StrEnum):
    """What a dataset or model is about."""

    DETECT = "detect"  # bounding boxes with classes (players, ball, ...)
    POSE = "pose"  # boxes plus a fixed set of keypoints per object (pitch landmarks)


# Keypoint visibility flags, identical in YOLO-pose and COCO.
NOT_LABELLED, OCCLUDED, VISIBLE = 0, 1, 2


@dataclass
class Annotations:
    """Ground-truth objects of one image, in pixel coordinates.

    Attributes:
        boxes: ``(N, 4)`` float32 ``xyxy`` boxes.
        class_ids: ``(N,)`` int64 indices into the dataset's ``class_names``.
        keypoints: ``(N, K, 3)`` float32 ``(x, y, visibility)`` or ``None`` for
            detection data. Visibility follows YOLO/COCO: 0 = not labelled,
            1 = labelled but occluded, 2 = visible.
        flip_idx: keypoint permutation to apply on a horizontal flip (left/right
            landmarks swap). Comes from the dataset; required to flip keypoints.
    """

    boxes: np.ndarray
    class_ids: np.ndarray
    keypoints: np.ndarray | None = None
    flip_idx: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.class_ids):
            raise ValueError(
                f"{len(self.boxes)} boxes but {len(self.class_ids)} class ids"
            )
        if self.keypoints is not None:
            kp = np.asarray(self.keypoints, dtype=np.float32)
            if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[0] != len(self.boxes):
                raise ValueError(
                    f"keypoints must have shape (N={len(self.boxes)}, K, 3), got {kp.shape}"
                )
            self.keypoints = kp
        if self.flip_idx is not None:
            self.flip_idx = tuple(int(i) for i in self.flip_idx)

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def num_keypoints(self) -> int | None:
        return None if self.keypoints is None else self.keypoints.shape[1]

    @classmethod
    def empty(
        cls, num_keypoints: int | None = None, flip_idx: Sequence[int] | None = None
    ) -> "Annotations":
        keypoints = (
            None
            if num_keypoints is None
            else np.zeros((0, num_keypoints, 3), dtype=np.float32)
        )
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
            keypoints=keypoints,
            flip_idx=tuple(flip_idx) if flip_idx is not None else None,
        )

    def select(self, index: np.ndarray | Sequence[int] | slice) -> "Annotations":
        """Keep a subset of objects (boolean mask or integer indices)."""
        return replace(
            self,
            boxes=self.boxes[index],
            class_ids=self.class_ids[index],
            keypoints=None if self.keypoints is None else self.keypoints[index],
        )

    def copy(self) -> "Annotations":
        return replace(
            self,
            boxes=self.boxes.copy(),
            class_ids=self.class_ids.copy(),
            keypoints=None if self.keypoints is None else self.keypoints.copy(),
        )

    def to_detections(self, class_names: Sequence[str] | None = None) -> sv.Detections:
        """Convert to ``sv.Detections`` (e.g. to reuse supervision annotators or metrics)."""
        detections = sv.Detections(
            xyxy=self.boxes.astype(np.float32),
            class_id=self.class_ids.astype(int),
            confidence=np.ones(len(self), dtype=np.float32),
        )
        if class_names is not None:
            detections.data["class_name"] = np.array(
                [class_names[i] for i in self.class_ids], dtype=str
            )
        return detections

    def to_keypoints(self) -> sv.KeyPoints:
        """Convert keypoints to ``sv.KeyPoints``; confidence is 1 for labelled points, else 0."""
        if self.keypoints is None:
            raise ValueError("These annotations have no keypoints")
        return sv.KeyPoints(
            xy=self.keypoints[..., :2].copy(),
            confidence=(self.keypoints[..., 2] > NOT_LABELLED).astype(np.float32),
            class_id=self.class_ids.astype(int),
        )
