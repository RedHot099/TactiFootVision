import numpy as np
import supervision as sv
import torch
import cv2
import sys
from pathlib import Path
from typing import Optional, Iterable

from config.models import TrackingConfig


def _filter_segments_by_distance(mask: np.ndarray, distance_threshold: float) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_centroid = centroids[main_label]
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        centroid = centroids[label]
        if label == main_label:
            filtered[labels == label] = True
            continue
        dist = float(np.linalg.norm(centroid - main_centroid))
        if dist <= distance_threshold:
            filtered[labels == label] = True
    return filtered


def _add_repo_to_path(config_path: Optional[Path]) -> Optional[Path]:
    if not config_path:
        return None
    resolved = config_path.resolve()
    repo_root = None
    for parent in resolved.parents:
        if parent.name == "segment-anything-2-real-time":
            repo_root = parent
            break
    if repo_root and str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    return repo_root


class SAM2Tracker:
    def __init__(self, config: TrackingConfig):
        if config.sam2 is None or not config.sam2.checkpoint_path or not config.sam2.config_path:
            raise ValueError("SAM2 tracker requires tracking.sam2.checkpoint_path and tracking.sam2.config_path")
        self.mask_filter_distance = float(config.sam2.mask_filter_distance)
        repo_root = _add_repo_to_path(config.sam2.config_path)
        config_path = Path(config.sam2.config_path).resolve()
        checkpoint_path = Path(config.sam2.checkpoint_path).resolve()
        config_name = config_path.stem
        if repo_root:
            configs_root = repo_root / "sam2" / "configs"
            if configs_root.is_dir():
                try:
                    rel = config_path.relative_to(configs_root)
                    config_name = str(rel.with_suffix("")).replace("\\", "/")
                except ValueError:
                    config_name = config_path.stem
        try:
            from sam2.build_sam import build_sam2_camera_predictor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SAM2 is not available. Install https://github.com/Gy920/segment-anything-2-real-time and ensure it is importable."
                f" Original error: {e}"
            ) from e

        self._predictor = build_sam2_camera_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
        )
        self._initialized = False
        self._id_to_class: dict[int, int] = {}
        self._next_id: int = 1
        self._last_boxes: dict[int, np.ndarray] = {}
        self._frame_index: int = 0

    def initialize(self, frame: np.ndarray, boxes_xyxy: np.ndarray, class_ids: Optional[np.ndarray] = None):
        boxes = boxes_xyxy.astype(np.float32) if boxes_xyxy is not None else np.empty((0, 4), dtype=np.float32)
        classes = (
            class_ids.astype(int)
            if class_ids is not None and len(class_ids) == len(boxes)
            else np.full(len(boxes), -1, dtype=int)
        )
        if len(boxes) == 0:
            self._reset_predictor(frame)
            self._initialized = True
            self._last_boxes = {}
            self._next_id = 1
            self._frame_index = 0
            return
        tracker_ids = np.arange(1, len(boxes) + 1, dtype=int)
        self.refresh_prompts(frame, boxes, classes, tracker_ids)

    def track(self, frame: np.ndarray) -> sv.Detections:
        if not self._initialized:
            raise RuntimeError("SAM2Tracker not initialized. Call initialize() with first frame and boxes.")
        if not self._id_to_class:
            self._frame_index += 1
            return sv.Detections.empty()
        return self._run_raw_track(frame)

    def allocate_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=int)
        ids = np.arange(self._next_id, self._next_id + count, dtype=int)
        self._next_id += count
        return ids

    def remove_ids(self, frame: np.ndarray, ids_to_remove: Iterable[int]) -> sv.Detections:
        remove_set = {int(tid) for tid in ids_to_remove if tid is not None}
        if not remove_set:
            return self._run_raw_track(frame)

        keep_entries: list[tuple[int, np.ndarray, int]] = []
        for tid, cls in self._id_to_class.items():
            if tid in remove_set:
                continue
            if tid not in self._last_boxes:
                continue
            keep_entries.append((tid, self._last_boxes[tid], cls))

        if not keep_entries:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_classes = np.empty((0,), dtype=int)
            empty_ids = np.empty((0,), dtype=int)
            return self.refresh_prompts(frame, empty_boxes, empty_classes, empty_ids)

        keep_entries.sort(key=lambda item: item[0])
        boxes = np.stack([box for _, box, _ in keep_entries], axis=0).astype(np.float32)
        classes = np.array([cls for _, _, cls in keep_entries], dtype=int)
        tracker_ids = np.array([tid for tid, _, _ in keep_entries], dtype=int)
        return self.refresh_prompts(frame, boxes, classes, tracker_ids)

    def refresh_prompts(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        tracker_ids: np.ndarray,
    ) -> sv.Detections:
        boxes = boxes.astype(np.float32)
        class_ids = (
            class_ids.astype(int)
            if len(class_ids) == len(boxes)
            else np.full(len(boxes), -1, dtype=int)
        )
        tracker_ids = (
            tracker_ids.astype(int)
            if len(tracker_ids) == len(boxes)
            else np.arange(self._next_id, self._next_id + len(boxes), dtype=int)
        )

        self._reset_predictor(frame)
        self._id_to_class.clear()
        for box, tid, cls in zip(boxes, tracker_ids, class_ids):
            bbox = np.array([box], dtype=np.float32)
            _ = self._predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(tid),
                bbox=bbox,
            )
            self._id_to_class[int(tid)] = int(cls)

        if len(boxes) == 0:
            self._last_boxes = {}
            self._next_id = 1
            self._frame_index = 0
            self._initialized = True
            return sv.Detections.empty()

        detections = self._run_raw_track(frame)
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            self._next_id = max(self._next_id, int(detections.tracker_id.max()) + 1)
        self._frame_index = 1
        self._initialized = True
        return detections

    def _run_raw_track(self, frame: np.ndarray) -> sv.Detections:
        use_cuda = torch.cuda.is_available()
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tracker_ids, mask_logits = self._predictor.track(frame)
            else:
                tracker_ids, mask_logits = self._predictor.track(frame)

        ids = np.array(tracker_ids, dtype=int)
        masks = (mask_logits > 0.0).detach().cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            self._last_boxes = {}
            self._frame_index += 1
            return sv.Detections.empty()

        if self.mask_filter_distance > 0:
            masks = np.array([
                _filter_segments_by_distance(m, self.mask_filter_distance) for m in masks
            ])

        xyxy = sv.mask_to_xyxy(masks=masks)
        conf = np.ones((len(xyxy),), dtype=float)
        class_ids = np.array([self._id_to_class.get(int(t), -1) for t in ids], dtype=int)
        self._last_boxes = {int(t): np.array(b, dtype=np.float32) for t, b in zip(ids, xyxy)}
        self._frame_index += 1
        return sv.Detections(xyxy=xyxy, mask=masks, confidence=conf, class_id=class_ids, tracker_id=ids)

    def _reset_predictor(self, frame: np.ndarray):
        try:
            self._predictor.reset_state()
        except Exception:
            pass
        self._predictor.load_first_frame(frame)
