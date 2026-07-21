import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from tactifoot_vision.config.schemas import SAM2Config
from tactifoot_vision.domain import (
    AdapterUnavailable,
    BBox,
    DetectionSet,
    Frame,
    ModelArtifactNotFound,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import Device
from tactifoot_vision.tracking.sam2_masks import (
    filter_segments_by_distance,
    postprocess_masks,
    tight_bbox_from_masks,
)

PredictorFactory = Callable[[str, str, str], object]


class SAM2Runtime:
    def __init__(
        self,
        config: SAM2Config,
        *,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.config = config
        self._validate_artifacts(config)
        self._predictor = self._build_predictor(config, predictor_factory)
        self._initialized = False
        self._id_to_class: dict[int, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._id_to_confidence: dict[int, float | None] = {}
        self._next_id = 1
        self._last_boxes: dict[int, NDArray[np.float32]] = {}
        self._smoothed_boxes: dict[int, NDArray[np.float32]] = {}
        self._scale = 1.0
        self._orig_hw: tuple[int, int] | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_tracks(self) -> TrackSet:
        tracks: list[Track] = []
        for track_id, box in sorted(self._last_boxes.items()):
            class_id = self._id_to_class.get(track_id, -1)
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=BBox.from_xyxy(box),
                    class_id=class_id,
                    class_name=self._id_to_name.get(track_id, f"unknown_{class_id}"),
                    confidence=self._id_to_confidence.get(track_id),
                )
            )
        return TrackSet(tuple(tracks))

    def initialize(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        tracker_ids = self._allocate_ids(len(detections))
        return self._refresh_from_detections(frame, detections, tracker_ids)

    def refresh_prompts(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        tracker_ids = self._match_or_allocate_ids(detections)
        return self._refresh_from_detections(frame, detections, tracker_ids)

    def track(self, frame: Frame) -> TrackSet:
        if not self._initialized:
            raise RuntimeError("SAM2Runtime is not initialized.")
        if not self._id_to_class:
            return TrackSet.empty()
        return self._run_raw_track(frame)

    def reset(self) -> None:
        self._initialized = False
        self._id_to_class.clear()
        self._id_to_name.clear()
        self._id_to_confidence.clear()
        self._last_boxes.clear()
        self._smoothed_boxes.clear()
        self._next_id = 1
        try:
            self._predictor.reset_state()
        except Exception:
            pass

    def _refresh_from_detections(
        self,
        frame: Frame,
        detections: DetectionSet,
        tracker_ids: NDArray[np.int_],
    ) -> TrackSet:
        self._ensure_scale(frame.image)
        limited = list(detections)
        if self.config.max_objects is not None:
            limited = limited[: self.config.max_objects]
            tracker_ids = tracker_ids[: self.config.max_objects]
        self._reset_predictor(frame.image)
        self._id_to_class.clear()
        self._id_to_name.clear()
        self._id_to_confidence.clear()
        self._last_boxes.clear()
        self._smoothed_boxes.clear()
        if not limited:
            self._initialized = True
            self._next_id = 1
            return TrackSet.empty()
        boxes = np.array(
            [[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2] for det in limited],
            dtype=np.float32,
        )
        boxes_scaled = self._scale_boxes_to_model(boxes)
        for detection, box, track_id in zip(
            limited, boxes_scaled, tracker_ids, strict=True
        ):
            self._predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(track_id),
                bbox=np.array([box], dtype=np.float32),
            )
            self._id_to_class[int(track_id)] = detection.class_id
            self._id_to_name[int(track_id)] = detection.class_name
            self._id_to_confidence[int(track_id)] = detection.confidence
        self._initialized = True
        prompt_tracks = _tracks_from_detections(limited, tracker_ids)
        self._last_boxes = {
            track.track_id: np.array(
                [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2],
                dtype=np.float32,
            )
            for track in prompt_tracks
        }
        tracks = self._run_raw_track(frame)
        if len(tracks) > 0:
            self._next_id = max(
                self._next_id, max(track.track_id for track in tracks) + 1
            )
        return prompt_tracks

    def _run_raw_track(self, frame: Frame) -> TrackSet:
        frame_scaled = self._resize_frame(frame.image)
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tracker_ids, mask_logits = self._predictor.track(frame_scaled)
            else:
                tracker_ids, mask_logits = self._predictor.track(frame_scaled)
        ids = np.array(tracker_ids, dtype=int)
        if len(ids) == 0:
            self._last_boxes = {}
            return TrackSet.empty()
        logits = _normalize_logits(_to_numpy(mask_logits))
        masks = np.squeeze(logits > self.config.mask_threshold).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            self._last_boxes = {}
            return TrackSet.empty()
        masks = postprocess_masks(
            masks,
            open_kernel_size=self.config.mask_open,
            close_kernel_size=self.config.mask_close,
        )
        if self.config.mask_filter_distance > 0:
            threshold = self.config.mask_filter_distance * self._scale
            masks = np.array(
                [filter_segments_by_distance(mask, threshold) for mask in masks],
                dtype=bool,
            )
        boxes = tight_bbox_from_masks(masks, min_mask_area=self.config.min_mask_area)
        boxes = self._unscale_boxes_to_original(boxes)
        confidences = _mask_confidences(logits, masks)
        boxes = self._smooth_boxes(ids, boxes)
        self._last_boxes = {
            int(track_id): np.array(box, dtype=np.float32)
            for track_id, box in zip(ids, boxes, strict=True)
        }
        tracks: list[Track] = []
        for track_id, box, confidence in zip(ids, boxes, confidences, strict=True):
            track_id_int = int(track_id)
            class_id = self._id_to_class.get(track_id_int, -1)
            self._id_to_confidence[track_id_int] = float(confidence)
            tracks.append(
                Track(
                    track_id=track_id_int,
                    bbox=BBox.from_xyxy(box),
                    class_id=class_id,
                    class_name=self._id_to_name.get(
                        track_id_int, f"unknown_{class_id}"
                    ),
                    confidence=float(confidence),
                )
            )
        return TrackSet(tuple(tracks))

    def _match_or_allocate_ids(self, detections: DetectionSet) -> NDArray[np.int_]:
        ids: list[int] = []
        used: set[int] = set()
        for detection in detections:
            matched_id = None
            for track_id, box in self._last_boxes.items():
                if track_id in used:
                    continue
                if _iou(detection.bbox, BBox.from_xyxy(box)) >= self.config.reseed_iou:
                    matched_id = track_id
                    break
            if matched_id is None:
                matched_id = int(self._allocate_ids(1)[0])
            used.add(matched_id)
            ids.append(matched_id)
        return np.array(ids, dtype=int)

    def _allocate_ids(self, count: int) -> NDArray[np.int_]:
        if count <= 0:
            return np.empty((0,), dtype=int)
        ids = np.arange(self._next_id, self._next_id + count, dtype=int)
        self._next_id += count
        return ids

    def _ensure_scale(self, frame: NDArray[np.uint8]) -> float:
        height, width = frame.shape[:2]
        if self._orig_hw == (height, width):
            return self._scale
        self._orig_hw = (height, width)
        if self.config.max_side is None:
            self._scale = 1.0
            return self._scale
        max_dim = max(height, width)
        if max_dim <= self.config.max_side:
            self._scale = 1.0
            return self._scale
        self._scale = float(self.config.max_side) / float(max_dim)
        return self._scale

    def _resize_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        scale = self._ensure_scale(frame)
        if scale == 1.0:
            return frame
        height, width = frame.shape[:2]
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return np.asarray(resized, dtype=np.uint8)

    def _scale_boxes_to_model(
        self, boxes_xyxy: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        if self._scale == 1.0:
            return boxes_xyxy.astype(np.float32)
        return (boxes_xyxy.astype(np.float32) * self._scale).astype(np.float32)

    def _unscale_boxes_to_original(
        self, boxes_xyxy: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        if self._scale == 1.0:
            return boxes_xyxy.astype(np.float32)
        return (boxes_xyxy.astype(np.float32) / self._scale).astype(np.float32)

    def _smooth_boxes(
        self, ids: NDArray[np.int_], boxes: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        alpha = self.config.bbox_ema_alpha
        if alpha <= 0:
            return boxes
        smoothed: list[NDArray[np.float32]] = []
        for track_id, box in zip(ids, boxes, strict=True):
            track_id_int = int(track_id)
            if track_id_int in self._smoothed_boxes:
                box = (
                    alpha * box + (1.0 - alpha) * self._smoothed_boxes[track_id_int]
                ).astype(np.float32)
            self._smoothed_boxes[track_id_int] = box.copy()
            smoothed.append(box)
        current_ids = {int(track_id) for track_id in ids}
        self._smoothed_boxes = {
            track_id: box
            for track_id, box in self._smoothed_boxes.items()
            if track_id in current_ids
        }
        return np.array(smoothed, dtype=np.float32)

    def _reset_predictor(self, frame: NDArray[np.uint8]) -> None:
        frame_scaled = self._resize_frame(frame)
        try:
            self._predictor.reset_state()
        except Exception:
            pass
        self._predictor.load_first_frame(frame_scaled)

    @staticmethod
    def _validate_artifacts(config: SAM2Config) -> None:
        if not config.checkpoint.is_file():
            raise ModelArtifactNotFound(
                f"SAM2 checkpoint not found: {config.checkpoint}"
            )
        if not config.model_config_path.is_file():
            raise ModelArtifactNotFound(
                f"SAM2 config not found: {config.model_config_path}"
            )

    def _build_predictor(
        self, config: SAM2Config, predictor_factory: PredictorFactory | None
    ) -> object:
        config_name = _sam2_config_name(config.model_config_path)
        checkpoint = str(config.checkpoint.resolve())
        device = _resolve_device(config.device)
        if predictor_factory is not None:
            return predictor_factory(config_name, checkpoint, device)
        _add_repo_to_path(config.model_config_path)
        try:
            from sam2.build_sam import build_sam2_camera_predictor
        except Exception as exc:
            raise AdapterUnavailable(
                "SAM2 is not available. Install segment-anything-2-real-time "
                "and ensure it is importable."
            ) from exc
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(_sam2_config_dir(config.model_config_path).resolve()),
            job_name="sam2_tracker",
            version_base=None,
        ):
            return build_sam2_camera_predictor(
                config_file=config_name,
                ckpt_path=checkpoint,
                device=device,
            )


def _to_numpy(value: Any) -> NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_logits(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0, :, :]
    elif logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim == 2:
        logits = logits[None, ...]
    return logits


def _mask_confidences(
    logits: NDArray[np.float32], masks: NDArray[np.bool_]
) -> NDArray[np.float32]:
    confidences: list[float] = []
    for index, mask in enumerate(masks):
        if mask.any():
            probabilities = 1.0 / (1.0 + np.exp(-logits[index][mask]))
            confidences.append(float(np.mean(probabilities)))
        else:
            confidences.append(0.0)
    return np.array(confidences, dtype=np.float32)


def _tracks_from_detections(
    detections: list[Any], tracker_ids: NDArray[np.int_]
) -> TrackSet:
    tracks = []
    for detection, track_id in zip(detections, tracker_ids, strict=True):
        tracks.append(
            Track(
                track_id=int(track_id),
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
            )
        )
    return TrackSet(tuple(tracks))


def _iou(first: BBox, second: BBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.width * first.height + second.width * second.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _resolve_device(device: Device) -> str:
    if device == Device.AUTO:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device.value


def _add_repo_to_path(config_path: Path) -> None:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "segment-anything-2-real-time":
            if str(parent) not in sys.path:
                sys.path.append(str(parent))
            return


def _sam2_config_dir(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent
    return config_path.parent


def _sam2_config_name(config_path: Path) -> str:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "segment-anything-2-real-time":
            configs_root = parent / "sam2" / "configs"
            if configs_root.is_dir():
                try:
                    return str(
                        resolved.relative_to(configs_root).with_suffix("")
                    ).replace("\\", "/")
                except ValueError:
                    return resolved.stem
    return resolved.stem
