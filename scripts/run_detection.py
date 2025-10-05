# scripts/run_detection.py
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

# Ensure project root is discoverable when executed via `python scripts/...`.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import supervision as sv
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Optional
from collections import deque

from loguru import logger

from config.loaders import load_config
from config.models import DetectionModelType, KeypointModelType
from tactifoot_vision.data.video_loader import VideoLoader
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.tracking.tracker import Tracker
from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker
from tactifoot_vision.tracking.ball_tracker import BallTracker
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler
from tactifoot_vision.geometry.view_transformer import ViewTransformer
from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration
from tactifoot_vision.visualization.pitch_visualizer import PitchVisualizer
from tactifoot_vision.utils.logging_config import setup_logging
from tactifoot_vision.export.pipeline_exporter import PipelineExporter
from tactifoot_vision.team.classifier import TeamAssignmentManager, TeamClassifier

def _shrink_box(box: np.ndarray, scale: float, frame_shape: tuple[int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box.astype(float)
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return None
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    new_w = w * scale
    new_h = h * scale
    half_w = new_w / 2.0
    half_h = new_h / 2.0
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    new_x1 = max(0.0, cx - half_w)
    new_y1 = max(0.0, cy - half_h)
    new_x2 = min(frame_w - 1.0, cx + half_w)
    new_y2 = min(frame_h - 1.0, cy + half_h)
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=float)


def _extract_crop(frame: np.ndarray, box: np.ndarray, scale: float) -> Optional[np.ndarray]:
    shrunk = _shrink_box(box, scale, frame.shape)
    if shrunk is None:
        return None
    x1, y1, x2, y2 = shrunk.astype(int)
    return frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None


@dataclass
class PendingCandidate:
    box: np.ndarray
    class_id: int
    first_seen: int
    last_seen: int
    hits: int = 1

    def update(
        self, new_box: np.ndarray, frame_idx: int, momentum: float = 0.4
    ) -> None:
        # Smooth the box to reduce jitter before reseeding.
        blended = (1.0 - momentum) * self.box + momentum * new_box
        self.box = blended.astype(np.float32)
        self.last_seen = frame_idx
        self.hits += 1


def _single_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    iou_matrix = sv.box_iou_batch(box_a[np.newaxis, :], box_b[np.newaxis, :])
    return float(iou_matrix[0, 0]) if iou_matrix.size else 0.0


def main(config_path: Path):
    try:
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        config = load_config(config_path)

        setup_logging(level=config.logging_level)
        logger.info(f"Starting detection script: {config.project_name}")
        logger.debug(f"Config loaded from: {config_path}")

        if not config.detection.checkpoint_path:
            logger.error("Detection config must specify 'checkpoint_path'")
            sys.exit(1)

        logger.info(
            f"Initializing detection handler: {config.detection.model_type.value}"
        )
        model_dir_abs = (config_path.parent / config.paths.model_dir).resolve()
        detection_handler = None
        try:
            if config.detection.model_type == DetectionModelType.YOLO:
                detection_handler = YOLOHandler(
                    config.detection, model_dir=model_dir_abs
                )
            elif config.detection.model_type == DetectionModelType.RFDETR:
                from tactifoot_vision.detection.rfdetr_handler import (
                    RFDETRHandler,
                )
                detection_handler = RFDETRHandler(
                    config.detection, model_dir=model_dir_abs
                )
            else:
                raise ValueError(
                    f"Unsupported detection model type: {config.detection.model_type}"
                )
            if detection_handler.model is None:
                logger.error("Detection model failed to load.")
                sys.exit(1)
            logger.success("Detection handler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize detection handler: {e}", exc_info=True)
            sys.exit(1)

        keypoint_handler = None
        if config.keypoints.enabled:
            if not config.keypoints.checkpoint_path:
                logger.error(
                    "Keypoints config must specify 'checkpoint_path' when enabled"
                )
                sys.exit(1)
            logger.info(
                f"Initializing keypoint handler: {config.keypoints.model_type.value}"
            )
            try:
                if config.keypoints.model_type == KeypointModelType.YOLO_POSE:
                    keypoint_handler = YOLOPoseHandler(
                        config.keypoints, model_dir=model_dir_abs
                    )
                else:
                    raise ValueError(
                        f"Unsupported keypoint model type: {config.keypoints.model_type}"
                    )
                if keypoint_handler.model is None:
                    logger.error("Keypoint model failed to load.")
                    sys.exit(1)
                logger.success("Keypoint handler initialized.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize keypoint handler: {e}", exc_info=True
                )
                sys.exit(1)
        else:
            logger.info("Keypoint detection disabled in config.")

        view_transformer = ViewTransformer(config.geometry)
        logger.success("Geometry components initialized.")

        pitch_visualizer = PitchVisualizer(
            config=config.visualization,
            pitch_dims=(
                view_transformer.pitch_config.length,
                view_transformer.pitch_config.width,
            ),
        )
        logger.success("Pitch visualizer initialized.")

        video_path_abs = (config_path.parent / config.paths.input_video).resolve()
        if not video_path_abs.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path_abs}")
        video_info = sv.VideoInfo.from_video_path(str(video_path_abs))
        if config.tracking.frame_rate is None:
            logger.info(
                f"Tracker frame_rate not set, using video FPS: {video_info.fps:.2f}"
            )
            config.tracking.frame_rate = int(round(video_info.fps))
        backend = getattr(config.tracking, "backend", "bytetrack")
        player_tracker = None
        sam2_tracker = None
        match backend:
            case "sam2":
                try:
                    sam2_tracker = SAM2Tracker(config.tracking)
                    logger.success("SAM2 Tracker initialized.")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize SAM2 tracker: {e}", exc_info=True
                    )
                    sys.exit(1)
            case "bytetrack":
                player_tracker = Tracker(config.tracking)
                logger.success("ByteTrack Tracker initialized.")
            case other:
                logger.error(f"Unsupported tracking backend: {other}")
                sys.exit(1)

        video_loader = VideoLoader(video_path_abs)
        logger.info(
            f"Video Info: {video_info.width}x{video_info.height} @ {video_info.fps:.2f} FPS, Total Frames: {video_info.total_frames or 'Unknown'}"
        )

        ball_tracker = BallTracker()
        logger.success("Raw Ball Position Collector initialized.")

        id_to_name_map = {v: k for k, v in config.detection.classes.items()}
        pipeline_exporter = PipelineExporter(class_id_to_name=id_to_name_map)
        team_clf_cfg = getattr(config, "team_classification", None)
        team_classifier = None
        team_validator = None
        team_samples: List[np.ndarray] = []
        team_assignments: Dict[int, int] = {}
        sample_stride = 1
        max_samples = 0
        warmup_frames = 0
        crop_scale = 0.6
        consecutive_frames = 3
        if team_clf_cfg and team_clf_cfg.enabled:
            model_name = team_clf_cfg.embedding_model
            siglip_cfg = None
            if getattr(team_clf_cfg, "method", "resnet") == "siglip":
                siglip_cfg = team_clf_cfg.siglip.model_dump()
                model_name = team_clf_cfg.siglip.model_name
            team_classifier = TeamClassifier(
                device=team_clf_cfg.device,
                model_name=model_name,
                method=getattr(team_clf_cfg, "method", None),
                siglip_config=siglip_cfg,
            )
            team_validator = TeamAssignmentManager(team_clf_cfg.consecutive_frames)
            sample_stride = max(1, team_clf_cfg.sample_stride)
            max_samples = team_clf_cfg.max_samples
            warmup_frames = team_clf_cfg.warmup_frames
            crop_scale = team_clf_cfg.crop_scale
            consecutive_frames = team_clf_cfg.consecutive_frames

        team_colors = [
            sv.Color.from_hex(config.visualization.team_color_0),
            sv.Color.from_hex(config.visualization.team_color_1),
            sv.Color.from_hex(config.visualization.player_color_default),
        ]
        TEAM_DEFAULT_COLOR_IDX = len(team_colors) - 1
        team_palette = sv.ColorPalette(team_colors)

        box_annotator = sv.BoxAnnotator(thickness=2, color=team_palette)
        label_annotator = sv.LabelAnnotator(
            text_thickness=1, text_scale=0.5, color=team_palette
        )
        pitch_vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=3)
        mask_annotator = None
        if config.visualization.draw_segmentation_masks:
            mask_annotator = sv.MaskAnnotator(color=team_palette, opacity=0.45)
        ball_box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex(config.visualization.ball_color),
        )
        ball_class_id = config.detection.classes.get("ball", -1)
        pitch_config_instance = SoccerPitchConfiguration(
            length=config.geometry.target_pitch_length,
            width=config.geometry.target_pitch_width,
        )
        pitch_labels = pitch_config_instance.labels
        main_frame_edges_0_based = pitch_config_instance.edges
        kp_color_unused = sv.Color.RED
        kp_color_used = sv.Color.from_hex("#FFA500")
        kp_radius = 4
        kp_label_color = sv.Color.WHITE.as_bgr()
        kp_label_scale = 0.4
        kp_label_thickness = 1
        proj_label_color = sv.Color.YELLOW.as_bgr()
        proj_label_scale = 0.35
        proj_label_thickness = 1
        proj_line_color = sv.Color.WHITE.as_bgr()
        proj_line_thickness = 1

        output_video_path = (config_path.parent / config.paths.output_video).resolve()
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        save_output = True
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = video_info.fps
            output_w, output_h = video_info.width, video_info.height
            frame_size = (output_w, output_h)
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)
            logger.info(
                f"Output video will be saved to: {output_video_path} ({output_w}x{output_h})"
            )

        team_classifier_ready = not (
            team_classifier and team_clf_cfg and team_clf_cfg.enabled
        )
        buffered_frames: List[Dict[str, Any]] = []

        def clone_detections(detections: sv.Detections) -> sv.Detections:
            if detections is None or detections.xyxy.size == 0:
                return sv.Detections.empty()
            data_copy = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in detections.data.items()
            }
            return sv.Detections(
                xyxy=detections.xyxy.copy(),
                mask=detections.mask.copy() if detections.mask is not None else None,
                confidence=detections.confidence.copy()
                if detections.confidence is not None
                else None,
                class_id=detections.class_id.copy()
                if detections.class_id is not None
                else None,
                tracker_id=detections.tracker_id.copy()
                if detections.tracker_id is not None
                else None,
                data=data_copy,
            )

        def apply_team_classification(
            frame_local: np.ndarray, detections_local: sv.Detections
        ) -> Optional[np.ndarray]:
            nonlocal team_assignments
            if (
                not team_classifier
                or not team_classifier.is_fitted
                or team_validator is None
                or len(detections_local) == 0
            ):
                detections_local.data.pop("team_id", None)
                return None

            crops: List[np.ndarray] = []
            tracker_ids_local: List[int] = []
            for det_idx in range(len(detections_local)):
                if detections_local.tracker_id is None:
                    continue
                tracker_id = detections_local.tracker_id[det_idx]
                if tracker_id is None:
                    continue
                class_id = (
                    int(detections_local.class_id[det_idx])
                    if detections_local.class_id is not None
                    else None
                )
                class_name = (
                    id_to_name_map.get(class_id, "") if class_id is not None else ""
                )
                if class_name not in {"player", "goalkeeper"}:
                    continue
                crop = _extract_crop(
                    frame_local, detections_local.xyxy[det_idx], crop_scale
                )
                if crop is None:
                    continue
                crops.append(crop)
                tracker_ids_local.append(int(tracker_id))

            if not crops:
                detections_local.data.pop("team_id", None)
                return None

            try:
                predictions = team_classifier.predict(crops)
                validated = team_validator.update(tracker_ids_local, predictions)
                for tid, team_id in zip(tracker_ids_local, validated):
                    if team_id is not None:
                        team_assignments[int(tid)] = int(team_id)
            except Exception as pred_err:  # pragma: no cover
                logger.warning(f"Team classifier prediction failed: {pred_err}")
                detections_local.data.pop("team_id", None)
                return None

            active_ids = []
            if detections_local.tracker_id is not None:
                active_ids = [
                    int(tid) for tid in detections_local.tracker_id if tid is not None
                ]
            team_validator.prune(active_ids)
            team_assignments = {
                tid: team
                for tid, team in team_assignments.items()
                if tid in set(active_ids)
            }

            if detections_local.tracker_id is not None:
                team_array = np.array(
                    [
                        team_assignments.get(int(tid), -1) if tid is not None else -1
                        for tid in detections_local.tracker_id
                    ],
                    dtype=int,
                )
                detections_local.data["team_id"] = team_array
                return team_array

            detections_local.data.pop("team_id", None)
            return None

        def process_payload(payload: Dict[str, Any]) -> None:
            frame_local = payload["frame"].copy()
            tracked_local = payload["tracked"]
            ball_local = payload["ball"]
            player_pitch_coords_local = payload["player_pitch_coords"]
            ball_pitch_coords_local = payload["ball_pitch_coords"]
            frame_pitch_vertices_local = payload["frame_pitch_vertices_main"]
            visible_area_local = payload["visible_area_list"]
            current_timestamp_local = payload["timestamp_seconds"]
            frame_id_local = payload["frame_id"]
            period_local = payload["period"]
            homography_local = payload["homography"]

            team_array: Optional[np.ndarray] = None
            team_array = apply_team_classification(frame_local, tracked_local)
            pipeline_exporter.update_team_assignments(team_assignments)

            player_team_ids_for_vis = team_array if team_array is not None else None

            pitch_map_frame = None
            if pitch_visualizer and config.visualization.enabled:
                pitch_map_frame = pitch_visualizer.draw_frame(
                    player_coords=player_pitch_coords_local,
                    player_team_ids=player_team_ids_for_vis,
                    ball_coords=ball_pitch_coords_local
                    if ball_pitch_coords_local is not None
                    and hasattr(ball_pitch_coords_local, "size")
                    and ball_pitch_coords_local.size > 0
                    else None,
                )

            annotated_frame = frame_local.copy()
            team_indices = None
            if len(tracked_local) > 0:
                if team_array is not None:
                    team_indices = np.clip(team_array, 0, TEAM_DEFAULT_COLOR_IDX)
                    team_indices[team_array < 0] = TEAM_DEFAULT_COLOR_IDX
                else:
                    team_indices = np.full(
                        len(tracked_local), TEAM_DEFAULT_COLOR_IDX, dtype=int
                    )

            if (
                mask_annotator
                and tracked_local.mask is not None
                and tracked_local.mask.size > 0
            ):
                mask_kwargs = {}
                if team_indices is not None:
                    mask_kwargs["custom_color_lookup"] = team_indices
                annotated_frame = mask_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_local,
                    **mask_kwargs,
                )

            pipeline_exporter.add_frame_data(
                frame_id=frame_id_local,
                period=period_local,
                tracked_detections=tracked_local,
                pitch_coords=player_pitch_coords_local,
                ball_pitch_coords=ball_pitch_coords_local,
                homography=homography_local,
                visible_area=visible_area_local,
                timestamp_seconds=current_timestamp_local,
            )

            if config.visualization.draw_bounding_boxes and len(tracked_local) > 0:
                labels: List[str] = []
                for det_idx in range(len(tracked_local)):
                    tracker_id = (
                        tracked_local.tracker_id[det_idx]
                        if tracked_local.tracker_id is not None
                        else None
                    )
                    label_parts: List[str] = []
                    if tracker_id is not None:
                        label_parts.append(f"#{tracker_id}")
                    class_id = (
                        int(tracked_local.class_id[det_idx])
                        if tracked_local.class_id is not None
                        else None
                    )
                    if class_id is not None:
                        class_name = id_to_name_map.get(class_id, str(class_id))
                        if class_name:
                            label_parts.append(class_name)
                    if (
                        team_array is not None
                        and det_idx < len(team_array)
                        and team_array[det_idx] in (0, 1)
                    ):
                        label_parts.append(f"T{int(team_array[det_idx])}")
                    labels.append(" ".join(label_parts))

                box_kwargs = {}
                label_kwargs = {}
                if team_indices is not None:
                    box_kwargs["custom_color_lookup"] = team_indices
                    label_kwargs["custom_color_lookup"] = team_indices
                annotated_frame = box_annotator.annotate(
                    annotated_frame, tracked_local, **box_kwargs
                )
                if any(labels):
                    annotated_frame = label_annotator.annotate(
                        annotated_frame, tracked_local, labels, **label_kwargs
                    )

                if config.visualization.draw_bounding_boxes and len(ball_local) > 0:
                    annotated_frame = ball_box_annotator.annotate(
                        annotated_frame, ball_local
                    )

                if (
                    frame_pitch_vertices_local is not None
                    and frame_pitch_vertices_local.shape == (num_expected_vertices, 2)
                ):
                    if config.visualization.draw_projected_pitch:
                        for u, v in main_frame_edges_0_based:
                            if 0 <= u < len(frame_pitch_vertices_local) and 0 <= v < len(
                                frame_pitch_vertices_local
                            ):
                                pt1 = tuple(frame_pitch_vertices_local[u].astype(int))
                                pt2 = tuple(frame_pitch_vertices_local[v].astype(int))
                                cv2.line(
                                    annotated_frame,
                                    pt1,
                                    pt2,
                                    proj_line_color,
                                    proj_line_thickness,
                                )
                        key_points = sv.KeyPoints(
                            xy=frame_pitch_vertices_local[np.newaxis, ...]
                        )
                        annotated_frame = pitch_vertex_annotator.annotate(
                            scene=annotated_frame, key_points=key_points
                        )
                        for i, xy in enumerate(frame_pitch_vertices_local):
                            if i < len(pitch_labels):
                                center = tuple(xy.astype(int))
                                cv2.putText(
                                    annotated_frame,
                                    pitch_labels[i],
                                    (center[0] - 4, center[1] + kp_radius + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    proj_label_scale,
                                    proj_label_color,
                                    proj_label_thickness,
                                    cv2.LINE_AA,
                                )

            final_frame = annotated_frame
            if (
                pitch_map_frame is not None
                and config.visualization.overlay
                and pitch_map_frame.size > 0
            ):
                try:
                    overlay_w = int(
                        final_frame.shape[1]
                        * config.visualization.overlay_width_fraction
                    )
                    generated_h, generated_w = pitch_map_frame.shape[:2]
                    if generated_w == 0:
                        raise ValueError("Generated pitch map frame has zero width.")
                    generated_aspect_ratio = generated_h / generated_w
                    overlay_h = int(overlay_w * generated_aspect_ratio)
                    resized_pitch_map = cv2.resize(
                        pitch_map_frame, (overlay_w, overlay_h)
                    )
                    pad = config.visualization.overlay_padding
                    pos = config.visualization.overlay_position
                    max_y, max_x = final_frame.shape[:2]
                    if "bottom" in pos:
                        y_start = max_y - overlay_h - pad
                    elif "top" in pos:
                        y_start = pad
                    else:
                        y_start = (max_y - overlay_h) // 2
                    if "left" in pos:
                        x_start = pad
                    elif "right" in pos:
                        x_start = max_x - overlay_w - pad
                    else:
                        x_start = (max_x - overlay_w) // 2
                    y_start, x_start = max(0, y_start), max(0, x_start)
                    y_end, x_end = (
                        min(max_y, y_start + overlay_h),
                        min(max_x, x_start + overlay_w),
                    )
                    actual_overlay_h, actual_overlay_w = (
                        y_end - y_start,
                        x_end - x_start,
                    )
                    if actual_overlay_h != overlay_h or actual_overlay_w != overlay_w:
                        resized_pitch_map = cv2.resize(
                            resized_pitch_map, (actual_overlay_w, actual_overlay_h)
                        )
                    roi = final_frame[y_start:y_end, x_start:x_end]
                    alpha = config.visualization.overlay_alpha
                    if roi.shape == resized_pitch_map.shape:
                        blended_roi = cv2.addWeighted(
                            resized_pitch_map, alpha, roi, 1 - alpha, 0
                        )
                        final_frame[y_start:y_end, x_start:x_end] = blended_roi
                except Exception as overlay_e:  # pragma: no cover
                    logger.warning(
                        f"Could not overlay pitch map on frame {frame_id_local}: {overlay_e}",
                        exc_info=True,
                    )

            if writer:
                output_frame = final_frame
                if (
                    output_frame.shape[1] != frame_size[0]
                    or output_frame.shape[0] != frame_size[1]
                ):
                    output_frame = cv2.resize(output_frame, frame_size)
                writer.write(output_frame)

        logger.info("Starting frame processing...")
        frame_count = 0
        frame_generator = video_loader.frame_generator()
        total_frames = video_info.total_frames if video_info.total_frames else None
        num_expected_vertices = len(view_transformer.pitch_config.vertices)
        video_fps = video_info.fps if video_info.fps > 0 else 30.0
        # --- Read period config ---
        current_period = config.processing.period
        period_start_time_seconds = config.processing.period_start_time_seconds
        logger.info(
            f"Processing period {current_period}, starting time: {period_start_time_seconds:.3f}s"
        )
        # --------------------------

        reseed_interval = None
        reseed_iou_threshold = 0.3
        if backend == "sam2" and config.tracking.sam2 is not None:
            reseed_interval = config.tracking.sam2.reseed_interval
            reseed_iou_threshold = config.tracking.sam2.reseed_iou_threshold

        pending_candidates: Dict[int, PendingCandidate] = {}
        next_candidate_id = 1
        reseed_cooldown = reseed_interval if reseed_interval is not None else 30
        last_reseed_frame = -reseed_cooldown
        candidate_min_hits = 3
        candidate_timeout = max(reseed_cooldown, candidate_min_hits + 2)
        candidate_merge_iou = min(0.5, reseed_iou_threshold * 0.75)

        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame in frame_generator:
                detections = detection_handler.detect(frame)
                ball_detections = detections[detections.class_id == ball_class_id]
                other_detections = detections[detections.class_id != ball_class_id]

                if not config.tracking.enabled:
                    tracked_detections = other_detections
                else:
                    match backend:
                        case "sam2":
                            if frame_count == 0:
                                try:
                                    prompt_boxes = (
                                        other_detections.xyxy
                                        if len(other_detections) > 0
                                        else None
                                    )
                                    prompt_cls = (
                                        other_detections.class_id
                                        if len(other_detections) > 0
                                        else None
                                    )
                                    sam2_tracker.initialize(
                                        frame, prompt_boxes, prompt_cls
                                    )
                                except Exception as init_e:
                                    logger.error(
                                        f"SAM2 initialize failed: {init_e}",
                                        exc_info=True,
                                    )
                                    sys.exit(1)
                            try:
                                tracked_detections = sam2_tracker.track(frame)
                            except Exception as track_e:
                                logger.error(
                                    f"SAM2 track failed: {track_e}", exc_info=True
                                )
                                tracked_detections = sv.Detections.empty()
                        case "bytetrack":
                            tracked_detections = player_tracker.update(other_detections)
                        case _:
                            tracked_detections = other_detections

                if team_classifier and not team_classifier.is_fitted:
                    if frame_count <= warmup_frames and frame_count % sample_stride == 0:
                        candidate_boxes = (
                            other_detections.xyxy.astype(np.float32)
                            if len(other_detections) > 0
                            else np.empty((0, 4), dtype=np.float32)
                        )
                        candidate_classes = (
                            other_detections.class_id.astype(int)
                            if other_detections.class_id is not None
                            else np.full(len(candidate_boxes), -1, dtype=int)
                        )
                        if candidate_boxes.size > 0:
                            for box, cls in zip(candidate_boxes, candidate_classes):
                                class_name = id_to_name_map.get(int(cls), "")
                                if class_name not in {"player", "goalkeeper"}:
                                    continue
                                crop = _extract_crop(frame, box, crop_scale)
                                if crop is not None:
                                    team_samples.append(crop)
                                    if max_samples and len(team_samples) > max_samples:
                                        team_samples = team_samples[-max_samples:]
                    if (
                        len(team_samples) >= 2
                        and (frame_count >= warmup_frames or (max_samples and len(team_samples) >= max_samples))
                    ):
                        try:
                            team_classifier.fit(team_samples)
                            team_samples = []
                            logger.debug("Team classifier fitted successfully.")
                            team_classifier_ready = True
                            if buffered_frames:
                                for buffered_payload in buffered_frames:
                                    process_payload(buffered_payload)
                                buffered_frames.clear()
                        except Exception as clf_err:
                            logger.warning(f"Team classifier fitting failed: {clf_err}")

                if backend == "sam2" and sam2_tracker is not None:
                    tracked_boxes = (
                        tracked_detections.xyxy.astype(np.float32)
                        if len(tracked_detections) > 0
                        else np.empty((0, 4), dtype=np.float32)
                    )
                    tracked_ids = (
                        tracked_detections.tracker_id.astype(int)
                        if tracked_detections.tracker_id is not None
                        else np.empty((0,), dtype=int)
                    )
                    tracked_classes = (
                        tracked_detections.class_id.astype(int)
                        if tracked_detections.class_id is not None
                        else np.full(len(tracked_boxes), -1, dtype=int)
                    )

                    det_class_ids = (
                        other_detections.class_id.astype(int)
                        if other_detections.class_id is not None
                        else np.full(len(other_detections), -1, dtype=int)
                    )
                    candidate_boxes = np.empty((0, 4), dtype=np.float32)
                    candidate_classes = np.empty((0,), dtype=int)
                    if len(other_detections) > 0:
                        mask_players = np.array(
                            [
                                id_to_name_map.get(int(cid), "")
                                in {"player", "goalkeeper"}
                                for cid in det_class_ids
                            ]
                        )
                        if mask_players.any():
                            candidate_boxes = other_detections.xyxy[
                                mask_players
                            ].astype(np.float32)
                            candidate_classes = det_class_ids[mask_players]

                    for box, cls in zip(candidate_boxes, candidate_classes):
                        if tracked_boxes.size > 0:
                            iou_existing = sv.box_iou_batch(
                                box[np.newaxis, :], tracked_boxes
                            ).max()
                            if iou_existing >= reseed_iou_threshold:
                                continue

                        best_match = None
                        best_iou = 0.0
                        for cand_id, cand in pending_candidates.items():
                            cand_iou = _single_iou(box, cand.box)
                            if cand_iou > best_iou:
                                best_iou = cand_iou
                                best_match = cand_id

                        if best_match is not None and best_iou >= candidate_merge_iou:
                            pending_candidates[best_match].update(box, frame_count)
                        else:
                            pending_candidates[next_candidate_id] = PendingCandidate(
                                box=box.astype(np.float32),
                                class_id=int(cls),
                                first_seen=frame_count,
                                last_seen=frame_count,
                            )
                            next_candidate_id += 1

                    stale_ids = [
                        cand_id
                        for cand_id, cand in pending_candidates.items()
                        if frame_count - cand.last_seen >= candidate_timeout
                    ]
                    for cand_id in stale_ids:
                        pending_candidates.pop(cand_id, None)

                    cooldown_ok = (
                        reseed_cooldown <= 0
                        or frame_count - last_reseed_frame >= reseed_cooldown
                    )
                    promote_ids = [
                        cand_id
                        for cand_id, cand in pending_candidates.items()
                        if cand.hits >= candidate_min_hits and cooldown_ok
                    ]

                    if promote_ids:
                        new_boxes = np.vstack(
                            [
                                pending_candidates[cid].box.astype(np.float32)
                                for cid in promote_ids
                            ]
                        )
                        new_classes = np.array(
                            [pending_candidates[cid].class_id for cid in promote_ids],
                            dtype=int,
                        )
                        new_ids = sam2_tracker.allocate_ids(len(promote_ids))

                        if tracked_boxes.size > 0:
                            combined_boxes = np.concatenate(
                                [tracked_boxes, new_boxes], axis=0
                            )
                            combined_classes = np.concatenate(
                                [tracked_classes, new_classes], axis=0
                            )
                            combined_ids = np.concatenate(
                                [tracked_ids, new_ids], axis=0
                            )
                        else:
                            combined_boxes = new_boxes
                            combined_classes = new_classes
                            combined_ids = new_ids

                        refresh_result = sam2_tracker.refresh_prompts(
                            frame,
                            combined_boxes,
                            combined_classes,
                            combined_ids,
                        )
                        tracked_detections = refresh_result
                        last_reseed_frame = frame_count
                        for cid in promote_ids:
                            pending_candidates.pop(cid, None)

                keypoints_sv: Optional[sv.KeyPoints] = None
                pitch_bbox_xyxy: Optional[np.ndarray] = None
                if keypoint_handler:
                    kp_result = keypoint_handler.detect(frame)
                    if kp_result:
                        keypoints_sv, pitch_bbox_xyxy = kp_result
                homography_updated = False
                if view_transformer and keypoints_sv:
                    homography_updated = view_transformer.update_homography(
                        keypoints_sv
                    )

                player_pitch_coords, ball_pitch_coords, frame_pitch_vertices_main = (
                    None,
                    None,
                    None,
                )
                player_team_ids_for_vis = None
                current_ball_pitch_pos = None
                visible_area_list = None

                if team_validator and team_classifier and team_classifier.is_fitted and len(tracked_detections) > 0:
                    crops_for_prediction: List[np.ndarray] = []
                    tracker_ids_for_prediction: List[int] = []
                    for det_idx in range(len(tracked_detections)):
                        if tracked_detections.tracker_id is None:
                            continue
                        tracker_id = tracked_detections.tracker_id[det_idx]
                        if tracker_id is None:
                            continue
                        class_id = None
                        if tracked_detections.class_id is not None:
                            class_id = int(tracked_detections.class_id[det_idx])
                        class_name = id_to_name_map.get(class_id, "") if class_id is not None else ""
                        if class_name not in {"player", "goalkeeper"}:
                            continue
                        crop = _extract_crop(frame, tracked_detections.xyxy[det_idx], crop_scale)
                        if crop is None:
                            continue
                        crops_for_prediction.append(crop)
                        tracker_ids_for_prediction.append(int(tracker_id))
                    if crops_for_prediction:
                        try:
                            predictions = team_classifier.predict(crops_for_prediction)
                            validated = team_validator.update(tracker_ids_for_prediction, predictions)
                            for tid, team_id in zip(tracker_ids_for_prediction, validated):
                                if team_id is not None:
                                    team_assignments[tid] = int(team_id)
                        except Exception as pred_err:
                            logger.warning(f"Team classifier prediction failed: {pred_err}")

                    active_ids = set()
                    if tracked_detections.tracker_id is not None:
                        active_ids = {
                            int(tid)
                            for tid in tracked_detections.tracker_id
                            if tid is not None
                        }
                    team_validator.prune(active_ids)
                    team_assignments = {
                        tid: team for tid, team in team_assignments.items() if tid in active_ids
                    }
                    pipeline_exporter.update_team_assignments(team_assignments)
                    if tracked_detections.tracker_id is not None:
                        team_array = np.array(
                            [team_assignments.get(int(tid), -1) if tid is not None else -1 for tid in tracked_detections.tracker_id],
                            dtype=int,
                        )
                        player_team_ids_for_vis = team_array
                        tracked_detections.data["team_id"] = team_array
                elif team_validator:
                    tracked_detections.data.pop("team_id", None)

                if view_transformer.current_homography is not None:
                    if len(tracked_detections) > 0 and tracked_detections.xyxy.size > 0:
                        player_anchors_frame = (
                            tracked_detections.get_anchors_coordinates(
                                sv.Position.BOTTOM_CENTER
                            )
                        )
                        player_pitch_coords = view_transformer.transform_frame_to_pitch(
                            player_anchors_frame
                        )

                    if len(ball_detections) > 0 and ball_detections.xyxy.size > 0:
                        ball_anchor_frame = ball_detections.get_anchors_coordinates(
                            sv.Position.CENTER
                        )
                        if ball_anchor_frame.ndim == 1:
                            ball_anchor_frame = ball_anchor_frame.reshape(1, -1)
                        if ball_anchor_frame.shape[0] > 0:
                            ball_pitch_coords = (
                                view_transformer.transform_frame_to_pitch(
                                    ball_anchor_frame[0:1]
                                )
                            )
                            if (
                                ball_pitch_coords is not None
                                and ball_pitch_coords.shape == (1, 2)
                            ):
                                current_ball_pitch_pos = ball_pitch_coords

                    frame_pitch_vertices_main = (
                        view_transformer.transform_pitch_to_frame(
                            view_transformer.pitch_vertices
                        )
                    )

                    h, w = frame.shape[:2]
                    frame_corners = np.array(
                        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
                    )
                    visible_area_pitch = view_transformer.transform_frame_to_pitch(
                        frame_corners
                    )
                    visible_area_list = (
                        visible_area_pitch.tolist()
                        if visible_area_pitch is not None
                        else None
                    )

                ball_tracker.add_point(current_ball_pitch_pos)

                current_timestamp_seconds = period_start_time_seconds + (
                    frame_count / video_fps
                )

                # Buffer early frames until team classifier is ready
                if team_classifier and team_clf_cfg and team_clf_cfg.enabled and not team_classifier_ready:
                    buffered_frames.append(
                        {
                            "frame": frame,
                            "tracked": tracked_detections,
                            "ball": ball_detections,
                            "player_pitch_coords": player_pitch_coords,
                            "ball_pitch_coords": current_ball_pitch_pos,
                            "frame_pitch_vertices_main": frame_pitch_vertices_main,
                            "visible_area_list": visible_area_list,
                            "timestamp_seconds": current_timestamp_seconds,
                            "frame_id": frame_count,
                            "period": current_period,
                            "homography": view_transformer.current_homography,
                        }
                    )
                    frame_count += 1
                    pbar.update(1)
                    continue

                pipeline_exporter.add_frame_data(
                    frame_id=frame_count,
                    period=current_period,  # Pass period
                    tracked_detections=tracked_detections,
                    pitch_coords=player_pitch_coords,
                    ball_pitch_coords=current_ball_pitch_pos,
                    homography=view_transformer.current_homography,
                    visible_area=visible_area_list,
                    timestamp_seconds=current_timestamp_seconds,
                )

                pitch_map_frame = None
                if pitch_visualizer and config.visualization.enabled:
                    pitch_map_frame = pitch_visualizer.draw_frame(
                        player_coords=player_pitch_coords,
                        player_team_ids=player_team_ids_for_vis,
                        ball_coords=current_ball_pitch_pos
                        if current_ball_pitch_pos is not None
                        and current_ball_pitch_pos.size > 0
                        else None,
                    )

                annotated_frame = frame.copy()
                color_lookup_indices = None
                if (
                    tracked_detections.tracker_id is not None
                    and len(tracked_detections) > 0
                ):
                    color_lookup_indices = np.full(
                        len(tracked_detections), TEAM_DEFAULT_COLOR_IDX, dtype=int
                    )
                    for idx, tid in enumerate(tracked_detections.tracker_id):
                        if tid is None:
                            continue
                        assigned_team = team_assignments.get(int(tid))
                        if assigned_team in (0, 1):
                            color_lookup_indices[idx] = int(assigned_team)
                if (
                    mask_annotator
                    and tracked_detections.mask is not None
                    and tracked_detections.mask.size > 0
                ):
                    try:
                        kwargs = {}
                        if (
                            color_lookup_indices is not None
                            and color_lookup_indices.size == len(tracked_detections)
                        ):
                            kwargs["custom_color_lookup"] = color_lookup_indices
                        annotated_frame = mask_annotator.annotate(
                            scene=annotated_frame,
                            detections=tracked_detections,
                            **kwargs,
                        )
                    except Exception as mask_err:
                        logger.warning(
                            f"Frame {frame_count}: Failed to draw segmentation masks: {mask_err}"
                        )
                if (
                    config.visualization.draw_bounding_boxes
                    and len(tracked_detections) > 0
                ):
                    labels: list[str] = []
                    for det_idx in range(len(tracked_detections)):
                        tracker_id = (
                            tracked_detections.tracker_id[det_idx]
                            if tracked_detections.tracker_id is not None
                            else None
                        )
                        label_parts: List[str] = []
                        if tracker_id is not None:
                            label_parts.append(f"#{tracker_id}")
                        class_id = (
                            int(tracked_detections.class_id[det_idx])
                            if tracked_detections.class_id is not None
                            else None
                        )
                        if class_id is not None:
                            class_name = id_to_name_map.get(class_id, str(class_id))
                            if class_name:
                                label_parts.append(class_name)
                        if tracker_id is not None and tracker_id in team_assignments:
                            label_parts.append(f"T{team_assignments[int(tracker_id)]}")
                        label = " ".join(label_parts)
                        labels.append(label)
                    box_kwargs = {}
                    if (
                        color_lookup_indices is not None
                        and color_lookup_indices.size == len(tracked_detections)
                    ):
                        box_kwargs["custom_color_lookup"] = color_lookup_indices
                    annotated_frame = box_annotator.annotate(
                        annotated_frame, tracked_detections, **box_kwargs
                    )
                    if any(labels):
                        label_kwargs = {}
                        if (
                            color_lookup_indices is not None
                            and color_lookup_indices.size == len(tracked_detections)
                        ):
                            label_kwargs["custom_color_lookup"] = color_lookup_indices
                        annotated_frame = label_annotator.annotate(
                            annotated_frame, tracked_detections, labels, **label_kwargs
                        )
                if (
                    config.visualization.draw_bounding_boxes
                    and len(ball_detections) > 0
                ):
                    annotated_frame = ball_box_annotator.annotate(
                        annotated_frame, ball_detections
                    )
                pipeline_exporter.add_frame_data(
                    frame_id=frame_count,
                    period=current_period,
                    tracked_detections=tracked_detections,
                    pitch_coords=player_pitch_coords,
                    ball_pitch_coords=current_ball_pitch_pos,
                    homography=view_transformer.current_homography,
                    visible_area=visible_area_list,
                    timestamp_seconds=current_timestamp_seconds,
                )
                if pitch_bbox_xyxy is not None:
                    bbox_int = pitch_bbox_xyxy.astype(int)
                    cv2.rectangle(
                        annotated_frame,
                        (bbox_int[0], bbox_int[1]),
                        (bbox_int[2], bbox_int[3]),
                        (255, 0, 255),
                        1,
                    )

                if keypoints_sv is not None and keypoints_sv.xy.size > 0:
                    draw_threshold = config.keypoints.confidence_threshold
                    kpts_xy, kpts_conf = keypoints_sv.xy[0], keypoints_sv.confidence[0]
                    used_indices_set = (
                        set(view_transformer.last_used_indices)
                        if view_transformer.last_used_indices is not None
                        else set()
                    )
                    for i, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
                        if conf >= draw_threshold:
                            center = tuple(xy.astype(int))
                            is_used = i in used_indices_set
                            draw_color = (
                                kp_color_used.as_bgr()
                                if is_used
                                else kp_color_unused.as_bgr()
                            )
                            cv2.circle(
                                annotated_frame, center, kp_radius, draw_color, -1
                            )
                            label_text = (
                                pitch_labels[i] if i < len(pitch_labels) else str(i)
                            )
                            text_org = (center[0] - 5, center[1] - kp_radius - 2)
                            cv2.putText(
                                annotated_frame,
                                label_text,
                                text_org,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                kp_label_scale,
                                kp_label_color,
                                kp_label_thickness,
                                cv2.LINE_AA,
                            )

                if (
                    frame_pitch_vertices_main is not None
                    and frame_pitch_vertices_main.shape == (num_expected_vertices, 2)
                ):
                    try:
                        if config.visualization.draw_projected_pitch:
                            if main_frame_edges_0_based:
                                for u, v in main_frame_edges_0_based:
                                    if 0 <= u < len(
                                        frame_pitch_vertices_main
                                    ) and 0 <= v < len(frame_pitch_vertices_main):
                                        pt1 = tuple(
                                            frame_pitch_vertices_main[u].astype(int)
                                        )
                                        pt2 = tuple(
                                            frame_pitch_vertices_main[v].astype(int)
                                        )
                                        cv2.line(
                                            annotated_frame,
                                            pt1,
                                            pt2,
                                            proj_line_color,
                                            proj_line_thickness,
                                        )
                            frame_all_key_points_sv = sv.KeyPoints(
                                xy=frame_pitch_vertices_main[np.newaxis, ...]
                            )
                            annotated_frame = pitch_vertex_annotator.annotate(
                                scene=annotated_frame,
                                key_points=frame_all_key_points_sv,
                            )
                            for i, xy in enumerate(frame_pitch_vertices_main):
                                if i < len(pitch_labels):
                                    center = tuple(xy.astype(int))
                                    label_text = pitch_labels[i]
                                    text_org = (
                                        center[0] - 4,
                                        center[1] + kp_radius + 8,
                                    )
                                    cv2.putText(
                                        annotated_frame,
                                        label_text,
                                        text_org,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        proj_label_scale,
                                        proj_label_color,
                                        proj_label_thickness,
                                        cv2.LINE_AA,
                                    )
                        else:
                            logger.debug(
                                "Projected pitch drawing disabled; vertices computed but not rendered."
                            )
                    except Exception as vis_err:
                        logger.warning(
                            f"Frame {frame_count}: Error drawing projected pitch: {vis_err}"
                        )
                elif view_transformer.current_homography is not None:
                    logger.warning(
                        f"Frame {frame_count}: Skipping projected pitch drawing due to invalid vertices."
                    )

                final_frame = annotated_frame
                if pitch_map_frame is not None and config.visualization.overlay:
                    try:
                        overlay_w = int(
                            final_frame.shape[1]
                            * config.visualization.overlay_width_fraction
                        )
                        generated_h, generated_w = pitch_map_frame.shape[:2]
                        if generated_w == 0:
                            raise ValueError(
                                "Generated pitch map frame has zero width."
                            )
                        generated_aspect_ratio = generated_h / generated_w
                        overlay_h = int(overlay_w * generated_aspect_ratio)
                        resized_pitch_map = cv2.resize(
                            pitch_map_frame, (overlay_w, overlay_h)
                        )
                        pad, pos = (
                            config.visualization.overlay_padding,
                            config.visualization.overlay_position,
                        )
                        max_y, max_x = final_frame.shape[:2]
                        if "bottom" in pos:
                            y_start = max_y - overlay_h - pad
                        elif "top" in pos:
                            y_start = pad
                        else:
                            y_start = (max_y - overlay_h) // 2
                        if "left" in pos:
                            x_start = pad
                        elif "right" in pos:
                            x_start = max_x - overlay_w - pad
                        else:
                            x_start = (max_x - overlay_w) // 2
                        y_start, x_start = max(0, y_start), max(0, x_start)
                        y_end, x_end = (
                            min(max_y, y_start + overlay_h),
                            min(max_x, x_start + overlay_w),
                        )
                        actual_overlay_h, actual_overlay_w = (
                            y_end - y_start,
                            x_end - x_start,
                        )
                        if (
                            actual_overlay_h != overlay_h
                            or actual_overlay_w != overlay_w
                        ):
                            resized_pitch_map = cv2.resize(
                                resized_pitch_map, (actual_overlay_w, actual_overlay_h)
                            )
                        roi = final_frame[y_start:y_end, x_start:x_end]
                        alpha = config.visualization.overlay_alpha
                        if roi.shape == resized_pitch_map.shape:
                            blended_roi = cv2.addWeighted(
                                resized_pitch_map, alpha, roi, 1 - alpha, 0
                            )
                            final_frame[y_start:y_end, x_start:x_end] = blended_roi
                        else:
                            logger.warning(
                                f"Frame {frame_count}: ROI shape {roi.shape} mismatch with overlay shape {resized_pitch_map.shape}. Skipping blend."
                            )
                    except Exception as overlay_e:
                        logger.warning(
                            f"Could not overlay pitch map on frame {frame_count}: {overlay_e}",
                            exc_info=True,
                        )

                if writer:
                    if (
                        final_frame.shape[1] != frame_size[0]
                        or final_frame.shape[0] != frame_size[1]
                    ):
                        final_frame = cv2.resize(final_frame, frame_size)
                    writer.write(final_frame)

                frame_count += 1
                pbar.update(1)

        if writer:
            writer.release()
            logger.info(f"Output video saved. Processed {frame_count} frames.")

        abs_threshold = (
            config.geometry.ball_outlier_threshold_percent / 100.0
        ) * view_transformer.pitch_config.width
        cleaned_ball_path = ball_tracker.get_cleaned_path(abs_threshold)

        export_file_path = (
            output_video_path.parent
            / f"{output_video_path.stem}_pipelinedata_p{current_period}.csv"
        )  # Add period to filename
        pipeline_exporter.save(export_file_path)

        # Optional: Save cleaned path
        # try:
        #     path_save_file = output_video_path.parent / f"{output_video_path.stem}_ball_path_cleaned_p{current_period}.npy"
        #     np.save(path_save_file, cleaned_ball_path, allow_pickle=True)
        #     logger.info(f"Cleaned ball path saved to: {path_save_file}")
        # except Exception as save_err: logger.error(f"Failed to save cleaned ball path: {save_err}")

        logger.success("Detection script finished.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or value error: {e}", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during detection.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TactiFoot Vision Detection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "default_config.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
