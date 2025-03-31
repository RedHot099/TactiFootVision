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
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler
from tactifoot_vision.geometry.view_transformer import ViewTransformer
from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration
from tactifoot_vision.visualization.pitch_visualizer import PitchVisualizer
from tactifoot_vision.utils.logging_config import setup_logging

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
        tracker = Tracker(config.tracking)
        logger.success("Tracker initialized.")

        video_loader = VideoLoader(video_path_abs)
        logger.info(
            f"Video Info: {video_info.width}x{video_info.height} @ {video_info.fps:.2f} FPS, Total Frames: {video_info.total_frames or 'Unknown'}"
        )

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        # --- Removed pitch line/vertex annotators ---
        # pitch_line_annotator = sv.EdgeAnnotator(color=sv.Color.WHITE, thickness=2, edges=view_transformer.pitch_config.edges)
        # pitch_vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=5)
        # --------------------------------------------
        id_to_name = {v: k for k, v in config.detection.classes.items()}
        ball_class_id = config.detection.classes.get("ball", -1)
        pitch_labels = SoccerPitchConfiguration(
            length=config.geometry.target_pitch_length,
            width=config.geometry.target_pitch_width,
        ).labels
        kp_color = sv.Color.RED
        kp_radius = 4
        kp_label_color = sv.Color.WHITE.as_bgr()
        kp_label_scale = 0.4
        kp_label_thickness = 1

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
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame in frame_generator:
                detections = detection_handler.detect(frame)
                ball_detections = detections[detections.class_id == ball_class_id]
                other_detections = detections[detections.class_id != ball_class_id]
                if config.tracking.enabled:
                    tracked_detections = tracker.update(other_detections)
                else:
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
                    if not homography_updated and frame_count == 0:
                        logger.warning("Failed to compute initial homography matrix.")
                    elif homography_updated and frame_count == 0:
                        logger.info("Initial homography computed successfully.")

                player_pitch_coords, ball_pitch_coords, frame_pitch_vertices_main = (
                    None,
                    None,
                    None,
                )
                player_team_ids_for_vis = None

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
                        if player_pitch_coords is None:
                            logger.warning(
                                f"Frame {frame_count}: Player transformation failed."
                            )

                    if len(ball_detections) > 0 and ball_detections.xyxy.size > 0:
                        ball_anchor_frame = ball_detections.get_anchors_coordinates(
                            sv.Position.CENTER
                        )
                        if ball_anchor_frame.ndim == 1:
                            ball_anchor_frame = ball_anchor_frame.reshape(1, -1)
                        ball_pitch_coords = view_transformer.transform_frame_to_pitch(
                            ball_anchor_frame
                        )
                        if ball_pitch_coords is None:
                            logger.warning(
                                f"Frame {frame_count}: Ball transformation failed."
                            )

                    # Calculation might still be needed elsewhere, but drawing is removed
                    # frame_pitch_vertices_main = (
                    #     view_transformer.transform_pitch_to_frame(
                    #         view_transformer.pitch_vertices
                    #     )
                    # )
                    # if frame_pitch_vertices_main is None: logger.warning(f"Frame {frame_count}: Pitch vertex inverse transformation failed.") # Optional warning

                pitch_map_frame = None
                if pitch_visualizer and config.visualization.enabled:
                    pitch_map_frame = pitch_visualizer.draw_frame(
                        player_coords=player_pitch_coords,
                        player_team_ids=player_team_ids_for_vis,
                        ball_coords=(
                            ball_pitch_coords
                            if ball_pitch_coords is not None
                            and ball_pitch_coords.size > 0
                            else None
                        ),
                    )

                annotated_frame = frame.copy()
                labels = []
                if len(tracked_detections) > 0:
                    for det_idx in range(len(tracked_detections)):
                        class_id = tracked_detections.class_id[det_idx]
                        tracker_id = (
                            tracked_detections.tracker_id[det_idx]
                            if tracked_detections.tracker_id is not None
                            else None
                        )
                        confidence = tracked_detections.confidence[det_idx]
                        class_name = id_to_name.get(class_id, f"CLS-{class_id}")
                        label = (
                            f"#{tracker_id} {class_name} {confidence:.2f}"
                            if tracker_id is not None
                            else f"{class_name} {confidence:.2f}"
                        )
                        labels.append(label)
                    annotated_frame = box_annotator.annotate(
                        annotated_frame, tracked_detections
                    )
                    if labels:
                        annotated_frame = label_annotator.annotate(
                            annotated_frame, tracked_detections, labels
                        )

                if len(ball_detections) > 0:
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
                    for i, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
                        if conf >= draw_threshold:
                            center = tuple(xy.astype(int))
                            cv2.circle(
                                annotated_frame,
                                center,
                                kp_radius,
                                kp_color.as_bgr(),
                                -1,
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

                # --- Removed drawing of projected pitch lines/vertices ---
                # if frame_pitch_vertices_main is not None and frame_pitch_vertices_main.size > 0:
                #     frame_all_key_points_sv = sv.KeyPoints(xy=frame_pitch_vertices_main[np.newaxis, ...])
                #     # annotated_frame = pitch_vertex_annotator.annotate(annotated_frame, frame_all_key_points_sv) # REMOVED
                #     # if view_transformer.pitch_config.edges: annotated_frame = pitch_line_annotator.annotate(annotated_frame, frame_all_key_points_sv) # REMOVED
                # ---------------------------------------------------------

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
