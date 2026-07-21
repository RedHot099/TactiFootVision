from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class BoTSORTArgs:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = True
    model: str = "auto"


class BoTSORTTracker:
    def __init__(self, args: BoTSORTArgs, *, frame_rate: int = 30) -> None:
        from ultralytics.trackers import BOTSORT

        namespace = SimpleNamespace(**args.__dict__)
        self._tracker = BOTSORT(namespace, frame_rate=int(frame_rate))

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        from ultralytics.engine.results import Boxes

        if len(detections) == 0:
            empty = np.empty((0, 6), dtype=np.float32)
            boxes = Boxes(empty, orig_shape=frame.shape[:2])
            _ = self._tracker.update(boxes, img=frame)
            return sv.Detections.empty()

        xyxy = detections.xyxy.astype(np.float32)
        conf = (
            detections.confidence.astype(np.float32)
            if detections.confidence is not None
            else np.ones((len(xyxy),), dtype=np.float32)
        )
        cls = (
            detections.class_id.astype(np.float32)
            if detections.class_id is not None
            else np.full((len(xyxy),), -1, dtype=np.float32)
        )

        boxes_data = np.column_stack([xyxy, conf, cls]).astype(np.float32)
        boxes = Boxes(boxes_data, orig_shape=frame.shape[:2])
        tracked = self._tracker.update(boxes, img=frame)
        if tracked.size == 0:
            return sv.Detections.empty()

        tracked = np.asarray(tracked)
        tracked_xyxy = tracked[:, :4].astype(np.float32)
        tracker_id = tracked[:, 4].astype(int)
        score = tracked[:, 5].astype(np.float32)
        tracked_cls = tracked[:, 6].astype(int)
        return sv.Detections(
            xyxy=tracked_xyxy,
            confidence=score,
            class_id=tracked_cls,
            tracker_id=tracker_id,
        )

