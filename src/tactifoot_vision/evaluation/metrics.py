"""Backend-independent metric containers returned by :func:`evaluate`."""

from dataclasses import dataclass

import pandas as pd

from tactifoot_vision.data.labelled_clip import Truth


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
    """HOTA, CLEAR MOT and identity metrics of a tracking run on one labelled clip.

    ``hota`` is TrackEval's HOTA: at each IoU threshold from 0.05 to 0.95 the
    geometric mean of detection accuracy (``deta``) and association accuracy
    (``assa``), and each of the three is the mean over the thresholds. All
    three are NaN when evaluated with ``hota=False``. ``mota`` is ``(matched
    boxes - false_positives - id_switches) / max(1, true boxes)`` as in
    TrackEval: ``1 - (false_negatives + false_positives + id_switches) / true
    boxes``, and ``-false_positives`` for a class without true boxes in the
    scored frames. ``idf1`` is the F1 score of the boxes matched to the true
    identity of their track. The overall numbers add up the counts (for HOTA,
    the detections) of every class. ``per_class`` is indexed by class name
    with columns ``hota``, ``deta``, ``assa``, ``idf1``, ``mota``,
    ``id_switches``, ``false_positives``, ``false_negatives`` and
    ``instances`` (true boxes in the scored frames).
    """

    hota: float
    deta: float
    assa: float
    idf1: float
    mota: float
    id_switches: int
    false_positives: int
    false_negatives: int
    per_class: pd.DataFrame
    num_frames: int  # frames scored: the processed frames

    def as_dict(self) -> dict[str, float]:
        return {
            "hota": self.hota,
            "deta": self.deta,
            "assa": self.assa,
            "idf1": self.idf1,
            "mota": self.mota,
            "id_switches": self.id_switches,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class HomographyMetrics:
    """Pitch position error of a run's frame -> pitch homographies, in metres.

    Each scored frame's error is the mean distance, on the pitch, between
    pitch points in view and where the run's homography puts the pixels that
    show them; a point the homography puts beyond its horizon is an infinite
    error. The points are the pitch's landmarks and a fixed grid over the
    pitch, so errors far from the landmarks are not hidden. ``median_error_m``,
    ``mean_error_m``, ``p90_error_m`` and ``under_1m``, ``under_2m``,
    ``under_5m`` (the share of scored frames whose error is below that) are
    over the frames with a homography; ``availability`` is their share of the
    frames with truth, so a missing homography lowers it and is not an error.
    ``jitter_m`` is the mean frame-to-frame change of the error at each point,
    between consecutive processed frames. The top-level numbers are on the
    grid; ``per_points`` is indexed by ``landmarks`` and ``grid`` with columns
    ``median_error_m``, ``mean_error_m``, ``p90_error_m``, ``under_1m``,
    ``under_2m``, ``under_5m``, ``jitter_m`` and ``frames`` (frames scored on
    those points). ``truth`` names the truth the errors are measured against.
    """

    median_error_m: float
    mean_error_m: float
    p90_error_m: float
    under_1m: float
    under_2m: float
    under_5m: float
    availability: float  # share of the frames with truth that have a homography
    jitter_m: float
    per_points: pd.DataFrame
    truth: Truth
    num_frames: int  # processed frames with truth

    def as_dict(self) -> dict[str, float | str]:
        return {
            "median_error_m": self.median_error_m,
            "mean_error_m": self.mean_error_m,
            "p90_error_m": self.p90_error_m,
            "under_1m": self.under_1m,
            "under_2m": self.under_2m,
            "under_5m": self.under_5m,
            "availability": self.availability,
            "jitter_m": self.jitter_m,
            "truth": self.truth.value,
        }
