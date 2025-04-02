# tactifoot_vision/visualization/pitch_visualizer.py
import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from config.models import PitchVisualizerConfig
from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration

logger = logging.getLogger(__name__)

STANDARD_PITCH_LENGTH = 105.0
STANDARD_PITCH_WIDTH = 68.0


class PitchVisualizer:
    def __init__(self, config: PitchVisualizerConfig, pitch_dims: Tuple[float, float]):
        self.config = config
        self.logical_length, self.logical_width = pitch_dims

        try:
            self.pitch_def_standard = SoccerPitchConfiguration(
                length=STANDARD_PITCH_LENGTH, width=STANDARD_PITCH_WIDTH
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize standard SoccerPitchConfiguration: {e}",
                exc_info=True,
            )
            raise

        try:
            self.pitch_color_bgr = sv.Color.from_hex(config.pitch_color).as_bgr()
            self.line_color_bgr = sv.Color.from_hex(config.line_color).as_bgr()
            self.path_color_bgr = sv.Color.from_hex(config.path_color).as_bgr()
            self.ball_color_bgr = sv.Color.from_hex(config.ball_color).as_bgr()
            self.player_color_default_bgr = sv.Color.from_hex(
                config.player_color_default
            ).as_bgr()
            self.team_color_0_bgr = sv.Color.from_hex(config.team_color_0).as_bgr()
            self.team_color_1_bgr = sv.Color.from_hex(config.team_color_1).as_bgr()
        except Exception as e:
            logger.error(
                f"Failed to parse color hex strings from config: {e}", exc_info=True
            )
            raise ValueError("Invalid color format in PitchVisualizerConfig") from e

        self.padding_px = config.canvas_padding_px
        self.canvas_width_px = config.canvas_width_px
        standard_aspect_ratio = (
            STANDARD_PITCH_WIDTH / STANDARD_PITCH_LENGTH
            if STANDARD_PITCH_LENGTH
            else 1.0
        )
        self.canvas_height_px = int(self.canvas_width_px * standard_aspect_ratio)

        draw_width = self.canvas_width_px - 2 * self.padding_px
        draw_height = self.canvas_height_px - 2 * self.padding_px
        if draw_width <= 0 or draw_height <= 0:
            raise ValueError(
                "Canvas padding is too large for the specified canvas width/height."
            )

        self.scale_x = draw_width / self.logical_length if self.logical_length else 1.0
        self.scale_y = draw_height / self.logical_width if self.logical_width else 1.0

        logger.info(
            f"OpenCV PitchVisualizer initialized. Canvas: {self.canvas_width_px}x{self.canvas_height_px}px. "
            f"Logical Input Dims: {self.logical_length}x{self.logical_width}. Scale: ({self.scale_x:.2f}, {self.scale_y:.2f})"
        )

    def _scale_point(self, logical_point: np.ndarray) -> Optional[Tuple[int, int]]:
        if logical_point is None or logical_point.shape != (1, 2):
            return None
        x_logic, y_logic = logical_point[0]
        px = int(x_logic * self.scale_x) + self.padding_px
        py = int(y_logic * self.scale_y) + self.padding_px
        return px, py

    def _scale_standard_point(
        self, standard_point: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        x_logic = (standard_point[0] / STANDARD_PITCH_LENGTH) * self.logical_length
        y_logic = (
            (STANDARD_PITCH_WIDTH - standard_point[1]) / STANDARD_PITCH_WIDTH
        ) * self.logical_width
        return self._scale_point(np.array([[x_logic, y_logic]]))

    def _calculate_arc_angles(
        self, penalty_spot_x: float, box_line_x: float, radius: float, center_y: float
    ) -> Optional[Tuple[float, float]]:
        delta_x = box_line_x - penalty_spot_x
        delta_x_sq = delta_x * delta_x
        radius_sq = radius * radius
        if radius_sq < delta_x_sq:
            return None
        delta_y = math.sqrt(radius_sq - delta_x_sq)
        y_intersect1_std = center_y - delta_y
        y_intersect2_std = center_y + delta_y
        angle1_rad = math.atan2(
            y_intersect1_std - center_y, box_line_x - penalty_spot_x
        )
        angle2_rad = math.atan2(
            y_intersect2_std - center_y, box_line_x - penalty_spot_x
        )
        angle1_deg = math.degrees(angle1_rad)
        angle2_deg = math.degrees(angle2_rad)
        return min(angle1_deg, angle2_deg), max(angle1_deg, angle2_deg)

    def _draw_base_pitch(self, canvas: np.ndarray):
        line_thick = self.config.line_thickness
        # Uses the corrected edges from self.pitch_def_standard
        for edge_indices in self.pitch_def_standard.edges:
            idx1, idx2 = edge_indices
            if not (
                0 <= idx1 < len(self.pitch_def_standard.vertices)
                and 0 <= idx2 < len(self.pitch_def_standard.vertices)
            ):
                continue
            pt1_std = self.pitch_def_standard.vertices[idx1]
            pt2_std = self.pitch_def_standard.vertices[idx2]
            pt1_scaled = self._scale_standard_point(pt1_std)
            pt2_scaled = self._scale_standard_point(pt2_std)
            if pt1_scaled and pt2_scaled:
                cv2.line(
                    canvas, pt1_scaled, pt2_scaled, self.line_color_bgr, line_thick
                )

        center_std = (STANDARD_PITCH_LENGTH / 2.0, STANDARD_PITCH_WIDTH / 2.0)
        center_scaled = self._scale_standard_point(center_std)
        if center_scaled:
            radius_px = int(
                self.pitch_def_standard.centre_circle_radius
                * self.scale_y
                * (self.logical_width / STANDARD_PITCH_WIDTH)
            )
            cv2.circle(
                canvas, center_scaled, radius_px, self.line_color_bgr, line_thick
            )
            cv2.circle(
                canvas, center_scaled, max(1, line_thick), self.line_color_bgr, -1
            )

        spot_radius_px = max(1, line_thick)
        spot1_std = (
            self.pitch_def_standard.penalty_spot_distance,
            STANDARD_PITCH_WIDTH / 2.0,
        )
        spot2_std = (
            STANDARD_PITCH_LENGTH - self.pitch_def_standard.penalty_spot_distance,
            STANDARD_PITCH_WIDTH / 2.0,
        )
        spot1_scaled = self._scale_standard_point(spot1_std)
        spot2_scaled = self._scale_standard_point(spot2_std)
        if spot1_scaled:
            cv2.circle(canvas, spot1_scaled, spot_radius_px, self.line_color_bgr, -1)
        if spot2_scaled:
            cv2.circle(canvas, spot2_scaled, spot_radius_px, self.line_color_bgr, -1)

        arc_radius_std = self.pitch_def_standard.centre_circle_radius
        arc_radius_px_x = int(
            arc_radius_std
            * self.scale_x
            * (self.logical_length / STANDARD_PITCH_LENGTH)
        )
        arc_radius_px_y = int(
            arc_radius_std * self.scale_y * (self.logical_width / STANDARD_PITCH_WIDTH)
        )
        axes = (arc_radius_px_x, arc_radius_px_y)

        if spot1_scaled:  # Left Arc
            angles_left = self._calculate_arc_angles(
                self.pitch_def_standard.penalty_spot_distance,
                self.pitch_def_standard.penalty_box_length,
                arc_radius_std,
                STANDARD_PITCH_WIDTH / 2.0,
            )
            if angles_left:
                start_angle_deg, end_angle_deg = angles_left
                cv2.ellipse(
                    canvas,
                    center=spot1_scaled,
                    axes=axes,
                    angle=0,
                    startAngle=start_angle_deg,
                    endAngle=end_angle_deg,
                    color=self.line_color_bgr,
                    thickness=line_thick,
                )
        if spot2_scaled:  # Right Arc
            angles_right = self._calculate_arc_angles(
                STANDARD_PITCH_LENGTH - self.pitch_def_standard.penalty_spot_distance,
                STANDARD_PITCH_LENGTH - self.pitch_def_standard.penalty_box_length,
                arc_radius_std,
                STANDARD_PITCH_WIDTH / 2.0,
            )
            if angles_right:
                start_angle_deg, end_angle_deg = angles_right
                cv2.ellipse(
                    canvas,
                    center=spot2_scaled,
                    axes=axes,
                    angle=0,
                    startAngle=start_angle_deg,
                    endAngle=end_angle_deg,
                    color=self.line_color_bgr,
                    thickness=line_thick,
                )

    def _draw_points(
        self,
        canvas: np.ndarray,
        logical_coords: Optional[np.ndarray],
        radius: int,
        default_color: Tuple,
        team_ids: Optional[np.ndarray] = None,
    ):
        if logical_coords is None or logical_coords.size == 0:
            return
        if logical_coords.ndim == 1:
            logical_coords = logical_coords.reshape(1, -1)
        if logical_coords.shape[1] != 2:
            return
        use_team_colors = team_ids is not None and len(team_ids) == len(logical_coords)
        for i, point_logic_arr in enumerate(logical_coords):
            point_logic = point_logic_arr.reshape(1, 2)
            scaled_point = self._scale_point(point_logic)
            if scaled_point:
                color = default_color
                if use_team_colors:
                    team_id = team_ids[i]
                    if team_id == 0:
                        color = self.team_color_0_bgr
                    elif team_id == 1:
                        color = self.team_color_1_bgr
                cv2.circle(canvas, scaled_point, radius, color, -1)

    # Removed _draw_path method

    def draw_frame(
        self,
        player_coords: Optional[np.ndarray] = None,
        player_team_ids: Optional[np.ndarray] = None,
        ball_coords: Optional[np.ndarray] = None,
        # Removed ball_path argument
    ) -> Optional[np.ndarray]:
        if not self.config.enabled:
            return None
        canvas = np.full(
            (self.canvas_height_px, self.canvas_width_px, 3),
            self.pitch_color_bgr,
            dtype=np.uint8,
        )
        self._draw_base_pitch(canvas)
        # Removed call to _draw_path
        self._draw_points(
            canvas,
            player_coords,
            self.config.player_dot_radius,
            self.player_color_default_bgr,
            player_team_ids,
        )
        self._draw_points(
            canvas, ball_coords, self.config.ball_dot_radius, self.ball_color_bgr
        )
        return canvas
