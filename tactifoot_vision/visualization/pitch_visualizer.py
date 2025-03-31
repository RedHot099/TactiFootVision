# tactifoot_vision/visualization/pitch_visualizer.py
import logging
from typing import Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from config.models import PitchVisualizerConfig

try:
    matplotlib.use("Agg")
except ImportError:
    logging.warning("Matplotlib Agg backend could not be set.")

logger = logging.getLogger(__name__)


class PitchVisualizer:
    def __init__(self, config: PitchVisualizerConfig, pitch_dims: Tuple[float, float]):
        self.config = config
        self.target_pitch_length, self.target_pitch_width = pitch_dims

        try:
            pitch_length_arg = (
                self.target_pitch_length if config.pitch_type == "custom" else None
            )
            pitch_width_arg = (
                self.target_pitch_width if config.pitch_type == "custom" else None
            )

            self.pitch = Pitch(
                pitch_type=config.pitch_type,
                pitch_length=pitch_length_arg,
                pitch_width=pitch_width_arg,
                pitch_color=config.pitch_color,
                line_color=config.line_color,
                goal_type=config.goal_type,
                linewidth=config.linewidth,
                line_zorder=2,
            )
            # Access dimensions correctly after initialization
            logger.info(
                f"mplsoccer Pitch initialized. Type: {self.pitch.pitch_type}, "
                f"Length Used: {self.pitch.dim.pitch_length}, Width Used: {self.pitch.dim.pitch_width}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize mplsoccer Pitch: {e}", exc_info=True)
            raise RuntimeError("Could not initialize mplsoccer Pitch") from e

        self.ball_color = config.ball_color
        self.player_color_default = config.player_color_default
        self.team_color_0 = config.team_color_0
        self.team_color_1 = config.team_color_1
        self.player_dot_size = config.player_dot_size
        self.ball_dot_size = config.ball_dot_size

        self.fig_width = 6
        # Use dimensions from self.pitch.dim for aspect ratio calculation
        aspect_ratio = (
            self.pitch.dim.pitch_width / self.pitch.dim.pitch_length
            if self.pitch.dim.pitch_length
            else 1.0
        )
        self.fig_height = self.fig_width * aspect_ratio
        self.dpi = 100

    def _convert_fig_to_cv2(self, fig: plt.Figure) -> np.ndarray:
        try:
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img_array_rgba = np.asarray(buf)
            img_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGBA2BGR)
            return img_bgr
        except Exception as e:
            logger.exception(f"Failed to convert matplotlib figure to CV2 image: {e}")
            h = int(self.fig_height * self.dpi)
            w = int(self.fig_width * self.dpi)
            logger.error(f"Returning blank image ({w}x{h}) due to conversion error.")
            return np.zeros((h, w, 3), dtype=np.uint8)
        finally:
            plt.close(fig)

    def draw_frame(
        self,
        player_coords: Optional[np.ndarray] = None,
        player_team_ids: Optional[np.ndarray] = None,
        ball_coords: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not self.config.enabled:
            return None

        fig, ax = self.pitch.draw(figsize=(self.fig_width, self.fig_height))
        fig.set_facecolor(self.config.pitch_color)
        ax.set_facecolor(self.config.pitch_color)

        # Get pitch boundaries correctly from self.pitch.dim
        x_min, x_max = self.pitch.dim.left, self.pitch.dim.right
        y_min, y_max = (
            min(self.pitch.dim.bottom, self.pitch.dim.top),
            max(self.pitch.dim.bottom, self.pitch.dim.top),
        )

        try:
            if player_coords is not None and player_coords.size > 0:
                if player_coords.ndim == 1:
                    if player_coords.shape == (2,):
                        player_coords = player_coords.reshape(1, -1)
                    else:
                        logger.warning(
                            f"Invalid player_coords shape: {player_coords.shape}. Skipping."
                        )
                        player_coords = None

                if player_coords is not None:
                    # --- CORRECTED bounds check using self.pitch.dim attributes ---
                    valid_player_mask = (
                        (player_coords[:, 0] >= x_min)
                        & (player_coords[:, 0] <= x_max)
                        & (player_coords[:, 1] >= y_min)
                        & (player_coords[:, 1] <= y_max)
                    )
                    # -------------------------------------------------------------
                    player_coords_valid = player_coords[valid_player_mask]
                    # Ensure team IDs are also filtered if they exist
                    player_team_ids_valid = None
                    if player_team_ids is not None and len(player_team_ids) == len(
                        player_coords
                    ):
                        player_team_ids_valid = player_team_ids[valid_player_mask]

                    if player_coords_valid.size > 0:
                        use_team_colors = player_team_ids_valid is not None and len(
                            player_team_ids_valid
                        ) == len(player_coords_valid)
                        if use_team_colors:
                            mask_team0 = player_team_ids_valid == 0
                            mask_team1 = player_team_ids_valid == 1
                            mask_other = ~np.isin(player_team_ids_valid, [0, 1])
                            if np.any(mask_team0):
                                self.pitch.scatter(
                                    player_coords_valid[mask_team0, 0],
                                    player_coords_valid[mask_team0, 1],
                                    ax=ax,
                                    c=self.team_color_0,
                                    s=self.player_dot_size,
                                    zorder=3,
                                    edgecolors="black",
                                    linewidth=0.5,
                                )
                            if np.any(mask_team1):
                                self.pitch.scatter(
                                    player_coords_valid[mask_team1, 0],
                                    player_coords_valid[mask_team1, 1],
                                    ax=ax,
                                    c=self.team_color_1,
                                    s=self.player_dot_size,
                                    zorder=3,
                                    edgecolors="black",
                                    linewidth=0.5,
                                )
                            if np.any(mask_other):
                                self.pitch.scatter(
                                    player_coords_valid[mask_other, 0],
                                    player_coords_valid[mask_other, 1],
                                    ax=ax,
                                    c=self.player_color_default,
                                    s=self.player_dot_size,
                                    zorder=3,
                                    edgecolors="black",
                                    linewidth=0.5,
                                )
                        else:
                            self.pitch.scatter(
                                player_coords_valid[:, 0],
                                player_coords_valid[:, 1],
                                ax=ax,
                                c=self.player_color_default,
                                s=self.player_dot_size,
                                zorder=3,
                                edgecolors="black",
                                linewidth=0.5,
                            )

            if ball_coords is not None and ball_coords.size > 0:
                if ball_coords.ndim == 1:
                    if ball_coords.shape == (2,):
                        ball_coords = ball_coords.reshape(1, -1)
                    else:
                        logger.warning(
                            f"Invalid ball_coords shape: {ball_coords.shape}. Skipping."
                        )
                        ball_coords = None

                if ball_coords is not None:
                    ball_coord_valid = ball_coords[0]
                    # --- CORRECTED bounds check using self.pitch.dim attributes ---
                    if (
                        x_min <= ball_coord_valid[0] <= x_max
                        and y_min <= ball_coord_valid[1] <= y_max
                    ):
                        self.pitch.scatter(
                            ball_coord_valid[0],
                            ball_coord_valid[1],
                            ax=ax,
                            c=self.ball_color,
                            s=self.ball_dot_size,
                            zorder=4,
                            marker="o",
                            edgecolors="black",
                            linewidth=0.5,
                        )
                    # -------------------------------------------------------------

            pitch_image_cv2 = self._convert_fig_to_cv2(fig)
            return pitch_image_cv2

        except Exception as e:
            logger.error(f"Error during mplsoccer drawing: {e}", exc_info=True)
            plt.close(fig)
            return None
