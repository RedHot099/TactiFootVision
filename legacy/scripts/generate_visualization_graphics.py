# scripts/generate_visualization_graphics.py

import argparse
import sys
from pathlib import Path
import cv2
import supervision as sv
import numpy as np
from typing import Optional, Tuple

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from loguru import logger

from config.loaders import load_config
from config.models import DetectionModelType, KeypointModelType
from tactifoot_vision.data.video_loader import VideoLoader
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler
from tactifoot_vision.geometry.view_transformer import ViewTransformer
from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration
from tactifoot_vision.visualization.pitch_visualizer import PitchVisualizer
from tactifoot_vision.utils.logging_config import setup_logging

# --- Configuration ---
# Choose a frame number that likely has good visibility of pitch lines and players
# You might need to experiment to find a good frame
TARGET_FRAME_NUMBER = 500
OUTPUT_DIR = project_root / "paper_graphics"
# ---------------------


# --- Helper Function for Schematic Drawing ---
def _scale_standard_point_schematic(
    standard_point: Tuple[float, float],
    canvas_dims: Tuple[int, int],
    pitch_dims_standard: Tuple[float, float],
    padding: int,
) -> Optional[Tuple[int, int]]:
    """Scales a point from standard pitch coords (e.g., 105x68) to canvas pixels."""
    canvas_w, canvas_h = canvas_dims
    pitch_len, pitch_wid = pitch_dims_standard
    draw_width = canvas_w - 2 * padding
    draw_height = canvas_h - 2 * padding

    if pitch_len <= 0 or pitch_wid <= 0:
        return None

    scale_x = draw_width / pitch_len
    scale_y = draw_height / pitch_wid

    x_std, y_std = standard_point
    # Invert y-axis for standard pitch drawing (0,0 top-left)
    px = int(x_std * scale_x) + padding
    py = int((pitch_wid - y_std) * scale_y) + padding  # Invert Y
    # Clamp to canvas boundaries just in case
    px = max(padding, min(canvas_w - padding - 1, px))
    py = max(padding, min(canvas_h - padding - 1, py))
    return px, py


# -------------------------------------------


def main(config_path: Path):
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        config = load_config(config_path)

        setup_logging(level=config.logging_level)
        logger.info(f"Starting graphic generation script: {config.project_name}")
        logger.debug(f"Config loaded from: {config_path}")

        # --- 1. Initialize Components ---
        logger.info("Initializing components...")

        # Video Loader
        video_path_abs = config.paths.input_video  # Already resolved by loader
        if not video_path_abs.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path_abs}")
        video_loader = VideoLoader(video_path_abs)
        video_info = video_loader.get_info()
        logger.info(f"Video loaded: {video_info.width}x{video_info.height}")

        # Keypoint Detector (Required for Homography Graphic)
        if not config.keypoints or not config.keypoints.enabled:
            raise ValueError(
                "Keypoint detection must be enabled in config for homography graphic."
            )
        if not config.keypoints.checkpoint_path:
            raise ValueError("Keypoints config must specify 'checkpoint_path'.")
        logger.info(
            f"Initializing keypoint handler: {config.keypoints.model_type.value}"
        )
        keypoint_handler = None
        try:
            if config.keypoints.model_type == KeypointModelType.YOLO_POSE:
                keypoint_handler = YOLOPoseHandler(
                    config.keypoints, model_dir=config.paths.model_dir
                )
            else:
                raise ValueError(
                    f"Unsupported keypoint model type: {config.keypoints.model_type}"
                )
            if keypoint_handler.model is None:
                raise RuntimeError("Keypoint model failed to load.")
            logger.success("Keypoint handler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize keypoint handler: {e}", exc_info=True)
            sys.exit(1)

        # Object Detector (Required for Detection/Mapping Graphic)
        if not config.detection or not config.detection.checkpoint_path:
            raise ValueError(
                "Detection config must specify 'checkpoint_path' for detection graphic."
            )
        logger.info(
            f"Initializing detection handler: {config.detection.model_type.value}"
        )
        detection_handler = None
        try:
            if config.detection.model_type == DetectionModelType.YOLO:
                detection_handler = YOLOHandler(
                    config.detection, model_dir=config.paths.model_dir
                )
            elif config.detection.model_type == DetectionModelType.RFDETR:
                detection_handler = RFDETRHandler(
                    config.detection, model_dir=config.paths.model_dir
                )
            elif config.detection.model_type == DetectionModelType.RFDETR_SEG:
                detection_handler = RFDETRSegHandler(
                    config.detection, model_dir=config.paths.model_dir
                )
            else:
                raise ValueError(
                    f"Unsupported detection model type: {config.detection.model_type}"
                )
            if detection_handler.model is None:
                raise RuntimeError("Detection model failed to load.")
            logger.success("Detection handler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize detection handler: {e}", exc_info=True)
            sys.exit(1)

        # View Transformer & Pitch Definition
        view_transformer = ViewTransformer(config.geometry)
        # Use standard pitch dimensions for the schematic
        pitch_def_standard = SoccerPitchConfiguration(
            length=105.0,
            width=68.0,  # Standard dimensions
        )
        pitch_labels = pitch_def_standard.labels
        logger.success("Geometry components initialized.")

        # Annotators
        kp_point_annotator = sv.DotAnnotator(color=sv.Color.RED, radius=5) # Changed PointAnnotator to DotAnnotator
        kp_label_annotator = sv.LabelAnnotator(
            color=sv.Color.WHITE,
            text_color=sv.Color.BLACK,
            text_scale=0.4,
            text_thickness=1,
        )
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        id_to_name_map = {v: k for k, v in config.detection.classes.items()}

        # Pitch Visualizer (for drawing base pitch)
        pitch_visualizer = PitchVisualizer(
            config=config.visualization,
            pitch_dims=(
                view_transformer.pitch_config.length,  # Use target dims for mapping
                view_transformer.pitch_config.width,
            ),
        )

        # --- 2. Load Target Frame ---
        logger.info(f"Loading target frame: {TARGET_FRAME_NUMBER}")
        frame = None
        for i, f in enumerate(video_loader.frame_generator()):
            if i == TARGET_FRAME_NUMBER:
                frame = f
                break
        if frame is None:
            raise ValueError(f"Could not load frame {TARGET_FRAME_NUMBER} from video.")
        logger.info(f"Frame {TARGET_FRAME_NUMBER} loaded successfully.")

        # --- 3. Generate Homography Visualization ---
        logger.info("Generating Homography Visualization graphic...")

        # Detect Keypoints
        kp_result = keypoint_handler.detect(frame)
        if not kp_result:
            logger.warning(
                f"No keypoints detected in frame {TARGET_FRAME_NUMBER}. Skipping homography graphic."
            )
            return  # Or handle differently

        keypoints_sv, _ = kp_result
        if keypoints_sv.xy.size == 0:
            logger.warning(
                f"Empty keypoints detected in frame {TARGET_FRAME_NUMBER}. Skipping homography graphic."
            )
            return

        # Update Homography (to know which points were used)
        homography_updated = view_transformer.update_homography(keypoints_sv)
        if not homography_updated or view_transformer.current_homography is None:
            logger.warning(
                f"Homography could not be calculated for frame {TARGET_FRAME_NUMBER}. Skipping homography graphic."
            )
            # Continue to next graphic if possible, but homography might be needed

        # Draw Keypoints on Frame
        frame_with_keypoints = frame.copy()
        kpts_xy = keypoints_sv.xy[0]  # Assuming single detection (the pitch)
        kpts_conf = keypoints_sv.confidence[0]
        used_indices_set = (
            set(view_transformer.last_used_indices)
            if view_transformer.last_used_indices is not None
            else set()
        )
        kp_color_unused = sv.Color.RED.as_bgr()
        kp_color_used = sv.Color.GREEN.as_bgr()  # Green for used points
        kp_radius = 5
        kp_label_color = sv.Color.WHITE.as_bgr()
        kp_label_scale = 0.4
        kp_label_thickness = 1

        for i, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
            if conf >= config.keypoints.confidence_threshold:  # Use config threshold
                center = tuple(xy.astype(int))
                is_used = i in used_indices_set
                draw_color = kp_color_used if is_used else kp_color_unused
                cv2.circle(frame_with_keypoints, center, kp_radius, draw_color, -1)
                label_text = pitch_labels[i] if i < len(pitch_labels) else str(i)
                text_org = (center[0] + kp_radius, center[1] - kp_radius)
                cv2.putText(
                    frame_with_keypoints,
                    label_text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    kp_label_scale,
                    kp_label_color,
                    kp_label_thickness,
                    cv2.LINE_AA,
                )

        output_path_frame_kp = OUTPUT_DIR / "homography_vis_frame_with_keypoints.png"
        cv2.imwrite(str(output_path_frame_kp), frame_with_keypoints)
        logger.info(f"Saved frame with keypoints to: {output_path_frame_kp}")

        # Draw Pitch Schematic with Keypoints
        schematic_width = 800
        schematic_height = int(
            schematic_width * (pitch_def_standard.width / pitch_def_standard.length)
        )
        schematic_padding = 30
        pitch_schematic = np.zeros(
            (schematic_height, schematic_width, 3), dtype=np.uint8
        )
        pitch_schematic[:, :] = sv.Color.from_hex("#22312b").as_bgr()  # Pitch color
        line_color_schematic = sv.Color.WHITE.as_bgr()
        line_thick_schematic = 2

        # Draw lines
        for u, v in pitch_def_standard.edges:
            pt1_std = pitch_def_standard.vertices[u]
            pt2_std = pitch_def_standard.vertices[v]
            pt1_scaled = _scale_standard_point_schematic(
                pt1_std,
                (schematic_width, schematic_height),
                (pitch_def_standard.length, pitch_def_standard.width),
                schematic_padding,
            )
            pt2_scaled = _scale_standard_point_schematic(
                pt2_std,
                (schematic_width, schematic_height),
                (pitch_def_standard.length, pitch_def_standard.width),
                schematic_padding,
            )
            if pt1_scaled and pt2_scaled:
                cv2.line(
                    pitch_schematic,
                    pt1_scaled,
                    pt2_scaled,
                    line_color_schematic,
                    line_thick_schematic,
                )

        # Draw center circle, penalty spots, arcs (simplified from PitchVisualizer)
        center_std = (pitch_def_standard.length / 2.0, pitch_def_standard.width / 2.0)
        center_scaled = _scale_standard_point_schematic(
            center_std,
            (schematic_width, schematic_height),
            (pitch_def_standard.length, pitch_def_standard.width),
            schematic_padding,
        )
        if center_scaled:
            radius_std = pitch_def_standard.centre_circle_radius
            # Approximate radius in pixels (use average scaling)
            avg_scale = (
                (schematic_width - 2 * schematic_padding) / pitch_def_standard.length
                + (schematic_height - 2 * schematic_padding) / pitch_def_standard.width
            ) / 2.0
            radius_px = int(radius_std * avg_scale)
            cv2.circle(
                pitch_schematic,
                center_scaled,
                radius_px,
                line_color_schematic,
                line_thick_schematic,
            )
            cv2.circle(
                pitch_schematic,
                center_scaled,
                max(1, line_thick_schematic // 2),
                line_color_schematic,
                -1,
            )  # Center dot

        # Mark Keypoints on Schematic
        for i, conf in enumerate(kpts_conf):
            if (
                conf >= config.keypoints.confidence_threshold
            ):  # Only show confident ones
                if i < len(pitch_def_standard.vertices):
                    pt_std = pitch_def_standard.vertices[i]
                    pt_scaled = _scale_standard_point_schematic(
                        pt_std,
                        (schematic_width, schematic_height),
                        (pitch_def_standard.length, pitch_def_standard.width),
                        schematic_padding,
                    )
                    if pt_scaled:
                        is_used = i in used_indices_set
                        draw_color = kp_color_used if is_used else kp_color_unused
                        cv2.circle(
                            pitch_schematic, pt_scaled, kp_radius, draw_color, -1
                        )
                        label_text = (
                            pitch_labels[i] if i < len(pitch_labels) else str(i)
                        )
                        text_org = (pt_scaled[0] + kp_radius, pt_scaled[1] - kp_radius)
                        cv2.putText(
                            pitch_schematic,
                            label_text,
                            text_org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            kp_label_scale,
                            kp_label_color,
                            kp_label_thickness,
                            cv2.LINE_AA,
                        )

        output_path_schematic = OUTPUT_DIR / "homography_vis_pitch_schematic.png"
        cv2.imwrite(str(output_path_schematic), pitch_schematic)
        logger.info(f"Saved pitch schematic with keypoints to: {output_path_schematic}")
        logger.success("Homography Visualization graphic generated.")

        # --- 4. Generate Detection and Mapping Visualization ---
        logger.info("Generating Detection and Mapping graphic...")

        if view_transformer.current_homography is None:
            logger.error(
                "Homography matrix is not available. Cannot generate mapping graphic."
            )
            sys.exit(1)

        # Detect Objects
        detections = detection_handler.detect(frame)
        if detections.is_empty():
            logger.warning(
                f"No objects detected in frame {TARGET_FRAME_NUMBER}. Skipping detection/mapping graphic."
            )
            return

        # Draw Detections on Frame
        frame_with_detections = frame.copy()
        labels = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            conf = detections.confidence[i]
            class_name = id_to_name_map.get(class_id, f"CLS-{class_id}")
            labels.append(f"{class_name} {conf:.2f}")

        frame_with_detections = box_annotator.annotate(
            frame_with_detections, detections
        )
        frame_with_detections = label_annotator.annotate(
            frame_with_detections, detections, labels
        )

        output_path_frame= OUTPUT_DIR / "base_frame.png"
        cv2.imwrite(str(output_path_frame), frame)
        output_path_frame_det = OUTPUT_DIR / "detection_map_frame_with_detections.png"
        cv2.imwrite(str(output_path_frame_det), frame_with_detections)
        logger.info(f"Saved frame with detections to: {output_path_frame_det}")

        # Map Detections to Pitch
        player_coords_pitch = None
        ball_coords_pitch = None
        ball_class_id = config.detection.classes.get("ball", -1)

        player_detections = detections[
            (detections.class_id != ball_class_id)
            & (
                detections.class_id != config.detection.classes.get("referee", -2)
            )  # Exclude referee if needed
            & (
                detections.class_id != config.detection.classes.get("goalkeeper", -3)
            )  # Exclude GK if needed
        ]
        ball_detections = detections[detections.class_id == ball_class_id]
        # Add other categories (ref, gk) if you want to map them too

        if not player_detections.is_empty():
            player_anchors_frame = player_detections.get_anchors_coordinates(
                sv.Position.BOTTOM_CENTER
            )
            player_coords_pitch = view_transformer.transform_frame_to_pitch(
                player_anchors_frame
            )

        if not ball_detections.is_empty():
            # Take the first ball detection if multiple
            ball_anchor_frame = ball_detections.get_anchors_coordinates(
                sv.Position.CENTER
            )[0:1]
            ball_coords_pitch = view_transformer.transform_frame_to_pitch(
                ball_anchor_frame
            )

        # Draw Mapped Objects on Pitch Canvas
        # Use PitchVisualizer's canvas size and drawing logic for consistency
        pitch_map_canvas = np.full(
            (pitch_visualizer.canvas_height_px, pitch_visualizer.canvas_width_px, 3),
            pitch_visualizer.pitch_color_bgr,
            dtype=np.uint8,
        )
        pitch_visualizer._draw_base_pitch(pitch_map_canvas)  # Draw the lines etc.

        # Draw mapped players (using default color for simplicity)
        pitch_visualizer._draw_points(
            pitch_map_canvas,
            player_coords_pitch,
            pitch_visualizer.config.player_dot_radius,
            pitch_visualizer.player_color_default_bgr,
            # team_ids=player_detections.class_id # Could add team colors if needed
        )
        # Draw mapped ball
        pitch_visualizer._draw_points(
            pitch_map_canvas,
            ball_coords_pitch,
            pitch_visualizer.config.ball_dot_radius,
            pitch_visualizer.ball_color_bgr,
        )

        output_path_pitch_map = OUTPUT_DIR / "detection_map_pitch_with_objects.png"
        cv2.imwrite(str(output_path_pitch_map), pitch_map_canvas)
        logger.info(f"Saved pitch map with objects to: {output_path_pitch_map}")
        logger.success("Detection and Mapping graphic generated.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or value error: {e}", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during graphic generation.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualization graphics for the TactiFoot Vision paper."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root
        / "config"
        / "default_config.yaml",  # Adjust if your default config is elsewhere
        help="Path to the main configuration YAML file.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    main(args.config)
