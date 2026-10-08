"""Top-down pitch view ("radar") of players and ball, and pasting it onto frames."""

from collections.abc import Sequence

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.pipeline.result import FrameResult
from tactifoot_vision.pitch.pitch import SoccerPitch

type ColorLike = str | sv.Color | tuple[int, int, int] | list[int]
"""A hex string (``"#FF1493"``), an ``sv.Color`` or a BGR tuple or list (YAML gives lists)."""

TEAM_COLORS: tuple[str, ...] = ("#00BFFF", "#FF1493")
DEFAULT_COLOR = "#FFFFFF"  # people without a team (referees, unassigned tracks)
BALL_COLOR = "#FFFF00"
PITCH_COLOR = "#22312B"
LINE_COLOR = "#FFFFFF"

# Real-world proportions, so circles stay round on pitches with other units
# (StatsBomb's 120 x 80 is not 105:68).
_STD_LENGTH, _STD_WIDTH = 105.0, 68.0
_SUBPIXEL = 4  # cv2 drawing "shift": coordinates carry 4 fractional bits


def as_color(color: ColorLike) -> sv.Color:
    """Parse a hex string, ``sv.Color`` or BGR tuple/list into an ``sv.Color``."""
    if isinstance(color, sv.Color):
        return color
    if isinstance(color, str):
        return sv.Color.from_hex(color)
    b, g, r = (int(c) for c in color)
    return sv.Color(r=r, g=g, b=b)


def text_color_for(color: ColorLike) -> sv.Color:
    """Black or white, whichever has the higher WCAG contrast on ``color``."""
    c = as_color(color)
    linear = [
        v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        for v in (c.r / 255, c.g / 255, c.b / 255)
    ]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    # contrast with white (1.05 / (L + 0.05)) vs black ((L + 0.05) / 0.05)
    return sv.Color.BLACK if (luminance + 0.05) ** 2 > 0.0525 else sv.Color.WHITE


def team_color_index(team_ids: np.ndarray, n_teams: int) -> np.ndarray:
    """Palette index per object: the team id, or ``n_teams`` (default colour) when unknown."""
    team_ids = np.asarray(team_ids, dtype=int)
    return np.where((team_ids >= 0) & (team_ids < n_teams), team_ids, n_teams)


def pitch_markings(
    pitch: SoccerPitch, samples: int = 64
) -> tuple[list[np.ndarray], np.ndarray]:
    """Painted markings in pitch units.

    Returns ``(lines, spots)``: a list of ``(P, 2)`` polylines (straight lines,
    centre circle, penalty arcs) and the ``(3, 2)`` centre and penalty spots.
    """
    lines = [pitch.vertices[[a, b]].astype(float) for a, b in pitch.edges]
    rx, ry = pitch.circle_radii
    cx, cy = pitch.length / 2, pitch.width / 2
    t = np.linspace(0, 2 * np.pi, samples)
    lines.append(np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)]))
    spot = pitch.penalty_spot_distance
    reach = pitch.penalty_box_length - spot  # spot -> edge of the penalty box
    if reach < rx:  # the "D": the part of the circle around the spot outside the box
        half = np.arccos(reach / rx)
        t = np.linspace(-half, half, samples // 2)
        for x0, direction in ((spot, 1.0), (pitch.length - spot, -1.0)):
            lines.append(
                np.column_stack([x0 + direction * rx * np.cos(t), cy + ry * np.sin(t)])
            )
    spots = np.array([[cx, cy], [spot, cy], [pitch.length - spot, cy]])
    return lines, spots


def _fixed(points: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> int32 fixed point for cv2's ``shift`` argument."""
    return np.round(np.asarray(points) * (1 << _SUBPIXEL)).astype(np.int32)


class PitchRadar:
    """Renders a top-down pitch with players (coloured by team) and the ball.

    ``x`` runs left to right and ``y`` top to bottom, so the far touchline of a
    broadcast view is at the top, as on screen.

    >>> radar = PitchRadar()
    >>> image = radar.draw(result[0])          # BGR uint8
    >>> frame = overlay(frame, image)          # paste onto the video frame
    """

    def __init__(
        self,
        pitch: SoccerPitch | None = None,
        width_px: int = 640,
        padding_px: int = 24,
        pitch_color: ColorLike = PITCH_COLOR,
        line_color: ColorLike = LINE_COLOR,
        team_colors: Sequence[ColorLike] = TEAM_COLORS,
        default_color: ColorLike = DEFAULT_COLOR,
        ball_color: ColorLike = BALL_COLOR,
        player_radius: int | None = None,
        ball_radius: int | None = None,
        draw_ids: bool = False,
        line_thickness: int | None = None,
    ) -> None:
        """
        Args:
            pitch: pitch geometry and units of the positions to draw.
            width_px: image width; the height follows the real pitch proportions.
            padding_px: margin around the touch/goal lines.
            player_radius, ball_radius, line_thickness: pixels (at least 1);
                ``None`` scales them with ``width_px``.
            draw_ids: write each player's track id inside its dot.
        """
        draw_width = width_px - 2 * padding_px
        if padding_px < 0 or draw_width < 50:
            raise ValueError(
                f"width_px={width_px} leaves no room for the pitch with padding_px={padding_px}"
            )
        _check_pixels("player_radius", player_radius)
        _check_pixels("ball_radius", ball_radius)
        _check_pixels("line_thickness", line_thickness)
        self.pitch = pitch or SoccerPitch()
        self.width_px = int(width_px)
        self.padding_px = int(padding_px)
        draw_height = round(draw_width * _STD_WIDTH / _STD_LENGTH)
        self.height_px = draw_height + 2 * self.padding_px
        self._scale = np.array(
            [draw_width / self.pitch.length, draw_height / self.pitch.width]
        )
        self.pitch_color = as_color(pitch_color)
        self.line_color = as_color(line_color)
        self.team_colors = [as_color(c) for c in team_colors]
        self.default_color = as_color(default_color)
        self.ball_color = as_color(ball_color)
        if player_radius is None:
            player_radius = max(3, round(width_px / 80))
        if ball_radius is None:
            ball_radius = max(2, round(player_radius * 0.75))
        if line_thickness is None:
            line_thickness = max(1, round(width_px / 480))
        self.player_radius = player_radius
        self.ball_radius = ball_radius
        self.line_thickness = line_thickness
        self.draw_ids = draw_ids
        self._background = self._draw_background()

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the rendered image in pixels."""
        return self.width_px, self.height_px

    def to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` pitch coordinates to (float) radar pixels."""
        return (
            np.asarray(xy, dtype=float).reshape(-1, 2) * self._scale + self.padding_px
        )

    def draw(self, frame_result: FrameResult | None = None) -> np.ndarray:
        """Radar image of one frame: players coloured by team, then the ball.

        ``None`` gives the empty pitch. Objects without pitch coordinates (NaN)
        are skipped.
        """
        image = self._background.copy()
        if frame_result is None:
            return image
        index = team_color_index(frame_result.team_ids, len(self.team_colors))
        palette = [*self.team_colors, self.default_color]
        tracker_id = frame_result.detections.tracker_id
        labels = (
            [str(int(t)) for t in tracker_id]
            if self.draw_ids and tracker_id is not None
            else None
        )
        # Unassigned people first so team colours stay on top where dots overlap.
        order = np.argsort(index != len(self.team_colors), kind="stable")
        self._draw_dots(
            image,
            frame_result.pitch_xy[order],
            [palette[i] for i in index[order]],
            self.player_radius,
            [labels[i] for i in order] if labels else None,
        )
        ball_xy = frame_result.ball_xy
        self._draw_dots(
            image, ball_xy, [self.ball_color] * len(ball_xy), self.ball_radius
        )
        return image

    def draw_points(
        self,
        xy: np.ndarray,
        colors: ColorLike | Sequence[ColorLike] | None = None,
        radius: int | None = None,
        image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw arbitrary pitch points, e.g. StatsBomb freeze-frame positions.

        Args:
            xy: ``(N, 2)`` pitch coordinates (NaN rows are skipped).
            colors: one colour for all points or one per point; ``None`` uses
                ``default_color``.
            radius: dot radius in pixels, at least 1 (default ``player_radius``).
            image: radar image to draw on (not modified); default an empty pitch.
        """
        _check_pixels("radius", radius)
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if colors is None:
            palette = [self.default_color] * len(xy)
        elif _is_single_color(colors):
            palette = [as_color(colors)] * len(xy)  # type: ignore[arg-type]
        else:
            palette = [as_color(c) for c in colors]  # type: ignore[union-attr]
            if len(palette) != len(xy):
                raise ValueError(f"Got {len(palette)} colours for {len(xy)} points")
        out = self._background.copy() if image is None else image.copy()
        radius = self.player_radius if radius is None else radius
        self._draw_dots(out, xy, palette, radius)
        return out

    # ------------------------------------------------------------- drawing
    def _draw_background(self) -> np.ndarray:
        image = np.full(
            (self.height_px, self.width_px, 3),
            self.pitch_color.as_bgr(),
            dtype=np.uint8,
        )
        lines, spots = pitch_markings(self.pitch)
        color = self.line_color.as_bgr()
        polylines = [_fixed(self.to_pixels(line)) for line in lines]
        cv2.polylines(
            image, polylines, False, color, self.line_thickness, cv2.LINE_AA, _SUBPIXEL
        )
        spot_radius = max(2.0, self.line_thickness * 1.5)
        for center in _fixed(self.to_pixels(spots)):
            cv2.circle(
                image, tuple(center), int(spot_radius * (1 << _SUBPIXEL)), color,
                -1, cv2.LINE_AA, _SUBPIXEL,
            )  # fmt: skip
        return image

    def _draw_dots(
        self,
        image: np.ndarray,
        xy: np.ndarray,
        colors: Sequence[sv.Color],
        radius: int,
        labels: Sequence[str] | None = None,
    ) -> None:
        pixels = self.to_pixels(xy)
        radius = int(round(radius))
        ring = max(1, round(radius / 4))
        font_scale = radius / 22
        for i, (point, color) in enumerate(zip(pixels, colors, strict=True)):
            if not np.isfinite(point).all():
                continue
            center = tuple(_fixed(point))
            # A ring in the pitch colour keeps overlapping dots apart.
            cv2.circle(
                image, center, (radius + ring) << _SUBPIXEL, self.pitch_color.as_bgr(),
                -1, cv2.LINE_AA, _SUBPIXEL,
            )  # fmt: skip
            cv2.circle(
                image,
                center,
                radius << _SUBPIXEL,
                color.as_bgr(),
                -1,
                cv2.LINE_AA,
                _SUBPIXEL,
            )
            if labels:
                (tw, th), _ = cv2.getTextSize(
                    labels[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                org = (round(point[0] - tw / 2), round(point[1] + th / 2))
                cv2.putText(
                    image, labels[i], org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color_for(color).as_bgr(), 1, cv2.LINE_AA,
                )  # fmt: skip


def _check_pixels(name: str, value: float | None) -> None:
    """A size in pixels must be at least 1 (``None``: the default size)."""
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1 pixel, got {value}")


def _is_single_color(colors: object) -> bool:
    if isinstance(colors, str | sv.Color):
        return True
    return (
        isinstance(colors, tuple | list)
        and len(colors) == 3
        and all(isinstance(c, int | np.integer) for c in colors)
    )


_VERTICAL = ("top", "center", "bottom")
_HORIZONTAL = ("left", "center", "right")


def check_overlay(
    position: str | None = None,
    width_fraction: float | None = None,
    alpha: float | None = None,
    padding: int | None = None,
) -> None:
    """Raise ``ValueError`` for an :func:`overlay` setting it would reject.

    Only the settings given are checked, so callers can validate before they
    have a frame (``render_video`` does, before it opens the output).
    """
    if position is not None:
        vertical, _, horizontal = position.partition("-")
        if vertical not in _VERTICAL or (horizontal or "center") not in _HORIZONTAL:
            raise ValueError(
                f"Unknown position {position!r}; use "
                f"'<{'|'.join(_VERTICAL)}>-<{'|'.join(_HORIZONTAL)}>' or 'center'"
            )
    if width_fraction is not None and not 0 < width_fraction <= 1:
        raise ValueError(f"width_fraction must be in (0, 1], got {width_fraction}")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if padding is not None and padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")


def overlay(
    frame: np.ndarray,
    image: np.ndarray,
    position: str = "bottom-center",
    width_fraction: float = 0.25,
    alpha: float = 0.8,
    padding: int = 10,
) -> np.ndarray:
    """Blend ``image`` (e.g. a radar) onto a copy of ``frame``.

    Args:
        position: ``"<top|center|bottom>-<left|center|right>"``, or ``"center"``.
        width_fraction: overlay width as a fraction of the frame width (the
            height follows the image's aspect ratio and is capped at the frame).
        alpha: opacity of the overlay (1 = opaque).
        padding: distance in pixels from the frame border.
    """
    check_overlay(position, width_fraction, alpha, padding)
    vertical, _, horizontal = position.partition("-")
    horizontal = horizontal or "center"
    frame_h, frame_w = frame.shape[:2]
    image_h, image_w = image.shape[:2]
    scale = min(frame_w * width_fraction / image_w, frame_h / image_h)
    w, h = max(1, round(image_w * scale)), max(1, round(image_h * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    y = {"top": padding, "center": (frame_h - h) // 2, "bottom": frame_h - h - padding}[
        vertical
    ]
    x = {"left": padding, "center": (frame_w - w) // 2, "right": frame_w - w - padding}[
        horizontal
    ]
    x, y = int(np.clip(x, 0, frame_w - w)), int(np.clip(y, 0, frame_h - h))
    out = frame.copy()
    roi = out[y : y + h, x : x + w]
    out[y : y + h, x : x + w] = cv2.addWeighted(resized, alpha, roi, 1 - alpha, 0)
    return out
