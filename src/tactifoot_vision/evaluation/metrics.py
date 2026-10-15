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


@dataclass
class TrackingMetrics:
    """CLEAR MOT and identity metrics of a tracking run on one labelled clip.

    ``mota`` is ``1 - (false_negatives + false_positives + id_switches) /
    true boxes`` and ``idf1`` the F1 score of the boxes matched to the true
    identity of their track. The overall numbers add up the counts of every
    class. ``per_class`` is indexed by class name with columns ``idf1``,
    ``mota``, ``id_switches``, ``false_positives``, ``false_negatives`` and
    ``instances`` (true boxes in the scored frames).
    """

    idf1: float
    mota: float
    id_switches: int
    false_positives: int
    false_negatives: int
    per_class: pd.DataFrame
    num_frames: int  # frames scored: the processed frames

    def as_dict(self) -> dict[str, float]:
        return {
            "idf1": self.idf1,
            "mota": self.mota,
            "id_switches": self.id_switches,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }
