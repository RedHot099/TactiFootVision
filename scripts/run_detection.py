# scripts/run_detection.py
import argparse
import sys
from pathlib import Path
import cv2
import supervision as sv
from tqdm import tqdm
import numpy as np
from typing import Optional

from loguru import logger

from config.loaders import load_config
from config.models import DetectionModelType, KeypointModelType
from tactifoot_vision.data.video_loader import VideoLoader
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.tracking.tracker import Tracker
from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker
from tactifoot_vision.tracking.ball_tracker import BallTracker
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler
from tactifoot_vision.geometry.view_transformer import ViewTransformer
from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration
from tactifoot_vision.visualization.pitch_visualizer import PitchVisualizer
from tactifoot_vision.utils.logging_config import setup_logging
from tactifoot_vision.export.pipeline_exporter import PipelineExporter


project_root = Path(__file__).resolve().parents[1]


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

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        pitch_vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=3)
        mask_annotator = None
        if config.visualization.draw_segmentation_masks:
            mask_annotator = sv.MaskAnnotator(
                color=sv.Color.WHITE,
                opacity=0.45,
                color_lookup=sv.ColorLookup.TRACK,
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
                        if "class_id" in tracked_detections.data:
                            player_team_ids_for_vis = tracked_detections.class_id

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
                if (
                    mask_annotator
                    and tracked_detections.mask is not None
                    and tracked_detections.mask.size > 0
                ):
                    try:
                        annotated_frame = mask_annotator.annotate(
                            scene=annotated_frame, detections=tracked_detections
                        )
                    except Exception as mask_err:
                        logger.warning(
                            f"Frame {frame_count}: Failed to draw segmentation masks: {mask_err}"
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
                        label = f"#{tracker_id}" if tracker_id is not None else ""
                        labels.append(label)
                    annotated_frame = box_annotator.annotate(
                        annotated_frame, tracked_detections
                    )
                    if any(labels):
                        annotated_frame = label_annotator.annotate(
                            annotated_frame, tracked_detections, labels
                        )
                if (
                    config.visualization.draw_bounding_boxes
                    and len(ball_detections) > 0
                ):
                    annotated_frame = box_annotator.annotate(
                        annotated_frame, ball_detections
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
