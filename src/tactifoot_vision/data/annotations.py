"""Labels of a single image, shared by data preparation, augmentation, training and plots."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import supervision as sv

from tactifoot_vision.enums import DetectionKey, PitchView
from tactifoot_vision.enums import Task as Task  # re-exported from its old home


@dataclass
class Annotations:
    """Ground-truth objects of one image, in pixel coordinates.

    Attributes:
        boxes: ``(N, 4)`` float32 ``xyxy`` boxes.
        class_ids: ``(N,)`` int64 indices into the dataset's ``class_names``.
        lines: pitch line data (:data:`Task.LINES`, no objects): marking
            name -> ``(P, 2)`` float32 pixel points along its visible part;
            ``None`` for the other tasks.
        weight: the sample's weight in training (a line label interpolated
            between key frames weighs less).
        view: a :class:`~tactifoot_vision.enums.PitchView` (``"main"`` or
            ``"along"``, a view along the pitch): which
            marking names a horizontal flip swaps (left/right or top/bottom).
    """

    boxes: np.ndarray
    class_ids: np.ndarray
    lines: dict[str, np.ndarray] | None = None
    weight: float = 1.0
    view: PitchView | str = PitchView.MAIN

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.class_ids):
            raise ValueError(
                f"{len(self.boxes)} boxes but {len(self.class_ids)} class ids"
            )
        if self.lines is not None:
            self.lines = {
                str(k): np.asarray(v, dtype=np.float32).reshape(-1, 2)
                for k, v in self.lines.items()
            }
        self.view = PitchView.parse(self.view, "view")

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls) -> "Annotations":
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
        )

    def select(self, index: np.ndarray | Sequence[int] | slice) -> "Annotations":
        """Keep a subset of objects (boolean mask or integer indices)."""
        return replace(
            self,
            boxes=self.boxes[index],
            class_ids=self.class_ids[index],
        )

    def copy(self) -> "Annotations":
        return replace(
            self,
            boxes=self.boxes.copy(),
            class_ids=self.class_ids.copy(),
            lines=None
            if self.lines is None
            else {k: v.copy() for k, v in self.lines.items()},
        )

    def to_detections(self, class_names: Sequence[str] | None = None) -> sv.Detections:
        """Convert to ``sv.Detections`` (e.g. to reuse supervision annotators or metrics)."""
        detections = sv.Detections(
            xyxy=self.boxes.astype(np.float32),
            class_id=self.class_ids.astype(int),
            confidence=np.ones(len(self), dtype=np.float32),
        )
        if class_names is not None:
            detections.data[DetectionKey.CLASS_NAME] = np.array(
                [class_names[i] for i in self.class_ids], dtype=str
            )
        return detections
