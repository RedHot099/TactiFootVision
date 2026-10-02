"""Backend-independent metric containers returned by :func:`evaluate`."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DetectionMetrics:
    """COCO-style box metrics of a detector on one dataset split.

    ``per_class`` is indexed by class name with columns ``map50_95``, ``map50``,
    ``map75`` and ``instances`` (ground-truth objects of that class).
    """

    map50_95: float
    map50: float
    map75: float
    per_class: pd.DataFrame
    num_images: int

    def as_dict(self) -> dict[str, float]:
        return {"map50_95": self.map50_95, "map50": self.map50, "map75": self.map75}


@dataclass
class KeypointMetrics:
    """Keypoint localisation quality on one dataset split.

    Errors are measured on labelled keypoints of the best-matching instance.
    ``pck`` is the share of keypoints closer than ``pck_threshold`` times the
    image diagonal. ``per_keypoint`` is indexed by keypoint index with columns
    ``mean_error_px``, ``pck`` and ``count``.
    """

    mean_error_px: float
    pck: float
    pck_threshold: float
    detection_rate: float  # share of images where the model found an instance
    per_keypoint: pd.DataFrame
    num_images: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_error_px": self.mean_error_px,
            "pck": self.pck,
            "detection_rate": self.detection_rate,
        }
