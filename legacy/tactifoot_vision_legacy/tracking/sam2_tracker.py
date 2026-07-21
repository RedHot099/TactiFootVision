import numpy as np
import supervision as sv
import torch
import cv2
import sys
from pathlib import Path
from typing import Optional, Iterable

from loguru import logger

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
        self.mask_threshold = float(config.sam2.mask_threshold)
        self.mask_open = int(config.sam2.mask_open)
        self.mask_close = int(config.sam2.mask_close)
        self.max_side = int(config.sam2.max_side) if config.sam2.max_side is not None else None
        self.max_objects = int(config.sam2.max_objects) if config.sam2.max_objects is not None else None
        self.bbox_ema_alpha = float(config.sam2.bbox_ema_alpha) if hasattr(config.sam2, 'bbox_ema_alpha') else 0.0
        self._smoothed_boxes: dict[int, np.ndarray] = {}
        repo_root = _add_repo_to_path(config.sam2.config_path)
        config_path = Path(config.sam2.config_path).resolve()
        checkpoint_path = Path(config.sam2.checkpoint_path).resolve()
        config_name = config_path.stem
        config_dir = config_path.parent
        for parent in config_path.parents:
            if parent.name == "configs":
                config_dir = parent
                break
        try:
            from hydra.core.global_hydra import GlobalHydra  # type: ignore
            from hydra import initialize_config_dir  # type: ignore

            GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=str(config_dir), job_name="sam2_tracker")
        except Exception as hydra_exc:  # pragma: no cover
            logger.warning(f"Hydra initialization failed for SAM2 configs at {config_dir}: {hydra_exc}")
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

        device = str(config.sam2.device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = build_sam2_camera_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=device,
        )
        self._initialized = False
        self._id_to_class: dict[int, int] = {}
        self._next_id: int = 1
        self._last_boxes: dict[int, np.ndarray] = {}
        self._frame_index: int = 0
        self._scale: float = 1.0
        self._orig_hw: tuple[int, int] | None = None
        self._open_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_open, self.mask_open),
            )
            if self.mask_open > 0
            else None
        )
        self._close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_close, self.mask_close),
            )
            if self.mask_close > 0
            else None
        )

    def _ensure_scale(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        if self._orig_hw == (h, w):
            return self._scale
        self._orig_hw = (h, w)
        if self.max_side is None or self.max_side <= 0:
            self._scale = 1.0
            return self._scale
        max_dim = max(h, w)
        if max_dim <= int(self.max_side):
            self._scale = 1.0
            return self._scale
        self._scale = float(self.max_side) / float(max_dim)
        return self._scale

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        scale = self._ensure_scale(frame)
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _scale_boxes_to_model(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        scale = float(self._scale)
        if scale == 1.0:
            return boxes_xyxy.astype(np.float32)
        return (boxes_xyxy.astype(np.float32) * scale).astype(np.float32)

    def _unscale_boxes_to_original(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        scale = float(self._scale)
        if scale == 1.0:
            return boxes_xyxy.astype(np.float32)
        return (boxes_xyxy.astype(np.float32) / scale).astype(np.float32)

    def initialize(self, frame: np.ndarray, boxes_xyxy: np.ndarray, class_ids: Optional[np.ndarray] = None) -> sv.Detections:
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
            self._smoothed_boxes = {}
            self._next_id = 1
            self._frame_index = 0
            return sv.Detections.empty()
        tracker_ids = np.arange(1, len(boxes) + 1, dtype=int)
        return self.refresh_prompts(frame, boxes, classes, tracker_ids)

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
        self._ensure_scale(frame)
        boxes = boxes.astype(np.float32)
        if self.max_objects is not None and self.max_objects > 0 and len(boxes) > int(self.max_objects):
            limit = int(self.max_objects)
            boxes = boxes[:limit]
            class_ids = class_ids[:limit]
            tracker_ids = tracker_ids[:limit]
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
        boxes_scaled = self._scale_boxes_to_model(boxes)
        for box, tid, cls in zip(boxes_scaled, tracker_ids, class_ids):
            bbox = np.array([box], dtype=np.float32)
            _ = self._predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(tid),
                bbox=bbox,
            )
            self._id_to_class[int(tid)] = int(cls)

        if len(boxes) == 0:
            self._last_boxes = {}
            self._smoothed_boxes = {}
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
        frame_scaled = self._resize_frame(frame)
        use_cuda = torch.cuda.is_available()
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tracker_ids, mask_logits = self._predictor.track(frame_scaled)
            else:
                tracker_ids, mask_logits = self._predictor.track(frame_scaled)

        ids = np.array(tracker_ids, dtype=int)
        masks = (mask_logits > self.mask_threshold).detach().cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        
        # Calculate confidence based on logits
        # Logits -> Sigmoid -> Mean over the mask area
        logits_np = mask_logits.detach().cpu().numpy().squeeze()
        if logits_np.ndim == 2:
            logits_np = logits_np[None, ...]
            
        confidences = []
        for i, mask in enumerate(masks):
            if mask.any():
                # Extract probabilities for the mask region
                mask_probs = 1.0 / (1.0 + np.exp(-logits_np[i][mask]))
                mean_conf = float(np.mean(mask_probs))
                confidences.append(mean_conf)
            else:
                confidences.append(0.0)
        conf = np.array(confidences, dtype=float)

        if masks.size == 0:
            self._last_boxes = {}
            self._frame_index += 1
            return sv.Detections.empty()

        masks = self._postprocess_masks(masks)

        if self.mask_filter_distance > 0:
            threshold = float(self.mask_filter_distance) * float(self._scale)
            masks = np.array([
                _filter_segments_by_distance(m, threshold) for m in masks
            ])

        # Optimize bounding box generation from masks
        xyxy = self._tight_bbox_from_masks(masks)
        
        xyxy = self._unscale_boxes_to_original(xyxy)
        
        # Apply EMA smoothing if enabled
        if self.bbox_ema_alpha > 0:
            smoothed = []
            for tid, box in zip(ids, xyxy):
                tid_int = int(tid)
                if tid_int in self._smoothed_boxes:
                    prev = self._smoothed_boxes[tid_int]
                    # EMA: new = alpha * current + (1-alpha) * prev
                    box = self.bbox_ema_alpha * box + (1 - self.bbox_ema_alpha) * prev
                self._smoothed_boxes[tid_int] = box.copy()
                smoothed.append(box)
            xyxy = np.array(smoothed, dtype=np.float32)
            # Clean up old IDs not in current frame
            current_ids = set(int(t) for t in ids)
            self._smoothed_boxes = {k: v for k, v in self._smoothed_boxes.items() if k in current_ids}
        
        # conf is already calculated above
        class_ids = np.array([self._id_to_class.get(int(t), -1) for t in ids], dtype=int)
        self._last_boxes = {int(t): np.array(b, dtype=np.float32) for t, b in zip(ids, xyxy)}
        self._frame_index += 1
        return sv.Detections(xyxy=xyxy, mask=masks, confidence=conf, class_id=class_ids, tracker_id=ids)

    def _tight_bbox_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """
        Generates tight bounding boxes from masks, ignoring small artifacts.
        Replaces sv.mask_to_xyxy(masks).
        """
        boxes = []
        min_area = 100.0  # Minimum pixel area to consider a valid object part
        
        for mask in masks:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            
            if not valid_contours:
                # Fallback: check if mask has ANY pixels (maybe scattered)
                if mask.any():
                    # If scattered pixels but no coherent contour > min_area, we can either
                    # take the full extent or discard. Discarding (empty box) might be safer for precision.
                    # But to avoid breaking the pipeline, let's take the extent but treat it as low confidence downstream if possible.
                    # For now, let's stick to 'tight' extent of all pixels.
                    y_indices, x_indices = np.where(mask)
                    boxes.append([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()])
                else:
                    boxes.append([0, 0, 0, 0]) # Empty box
                continue

            # Find largest contour by area from VALID ones
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            boxes.append([x, y, x + w, y + h])
            
        return np.array(boxes, dtype=np.float32)

    def _postprocess_masks(self, masks: np.ndarray) -> np.ndarray:
        if self._open_kernel is None and self._close_kernel is None:
            return masks
        processed: list[np.ndarray] = []
        for mask in masks:
            mask_uint8 = mask.astype(np.uint8)
            if self._open_kernel is not None:
                mask_uint8 = cv2.morphologyEx(
                    mask_uint8,
                    cv2.MORPH_OPEN,
                    self._open_kernel,
                )
            if self._close_kernel is not None:
                mask_uint8 = cv2.morphologyEx(
                    mask_uint8,
                    cv2.MORPH_CLOSE,
                    self._close_kernel,
                )
            processed.append(mask_uint8.astype(bool))
        return np.array(processed, dtype=bool)

    def _reset_predictor(self, frame: np.ndarray):
        frame_scaled = self._resize_frame(frame)
        try:
            self._predictor.reset_state()
        except Exception:
            pass
        self._predictor.load_first_frame(frame_scaled)
