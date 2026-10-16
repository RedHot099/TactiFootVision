"""Top-down pitch view ("radar") of players and ball, and pasting it onto frames."""

from collections.abc import Sequence

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.analysis.space import SpaceControl
from tactifoot_vision.pipeline.postprocess import metres_per_unit
from tactifoot_vision.pipeline.result import FrameResult
from tactifoot_vision.pitch.homography import visible_area
from tactifoot_vision.pitch.pitch import SoccerPitch, pitch_markings
from tactifoot_vision.viz.birds_eye import _warp_frame
from tactifoot_vision.viz.subpixel import SUBPIXEL, clip_segment, to_fixed
from tactifoot_vision.viz.trail import draw_trail, trail_frames

type ColorLike = str | sv.Color | tuple[int, int, int] | list[int]
"""A hex string (``"#FF1493"``), an ``sv.Color`` or a BGR tuple or list (YAML gives lists)."""

TEAM_COLORS: tuple[str, ...] = ("#00BFFF", "#FF1493")
DEFAULT_COLOR = "#FFFFFF"  # people without a team (referees, unassigned tracks)
BALL_COLOR = "#FFFF00"
VISIBLE_AREA_COLOR = "#FFD700"  # apart from the white markings
PITCH_COLOR = "#22312B"
LINE_COLOR = "#FFFFFF"

# Real-world proportions, so circles stay round on pitches with other units
# (StatsBomb's 120 x 80 is not 105:68).
_STD_LENGTH, _STD_WIDTH = 105.0, 68.0


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


class PitchRadar:
    """Renders a top-down pitch with players (coloured by team) and the ball,
    optionally over the video frame warped onto the pitch and a space-control map.

    ``x`` runs left to right and ``y`` top to bottom, so the far touchline of a
    broadcast view is at the top, as on screen. With ``trail_seconds``, each
    person's movement over that many seconds is drawn as a tail from the
    ``history`` passed to :meth:`draw` (ADR 0003). With ``draw_visible_area``,
    the part of the pitch the frame shows is outlined. With
    ``velocity_seconds``, each person's velocity, passed to :meth:`draw` from
    :func:`tv.analysis.velocities <tactifoot_vision.analysis.velocities>`, is
    an arrow from their dot.

    >>> radar = PitchRadar()
    >>> image = radar.draw(result[0])          # BGR uint8
    >>> frame = overlay(frame, image)          # paste onto the video frame
    >>> tails = PitchRadar(trail_seconds=2.0).draw(
    ...     result.frame(index), history=result.window(index, 2.0)
    ... )
    >>> visible = PitchRadar(draw_visible_area=True).draw(
    ...     result.frame(index), frame_size=result.frame_size
    ... )
    >>> velocities = tv.analysis.velocities(result)  # per frame of result.frames
    >>> arrows = PitchRadar(velocity_seconds=1.0).draw(
    ...     result.frames[i], velocities=velocities[i]
    ... )
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
        space_control_alpha: float = 0.5,
        trail_seconds: float | None = None,
        draw_visible_area: bool = False,
        visible_area_color: ColorLike = VISIBLE_AREA_COLOR,
        velocity_seconds: float | None = None,
    ) -> None:
        """
        Args:
            pitch: pitch geometry and units of the positions to draw.
            width_px: image width; the height follows the real pitch proportions.
            padding_px: margin around the touch/goal lines.
            player_radius, ball_radius, line_thickness: pixels (at least 1);
                ``None`` scales them with ``width_px``.
            draw_ids: write each player's track id inside its dot.
            space_control_alpha: opacity of the team colours a space-control
                map tints the pitch with (1 = opaque).
            trail_seconds: draw each person's path over the last this many
                seconds under the dots, in their colour fading to nothing at
                that age; ``None``: no tails.
            draw_visible_area: outline the frame's visible area
                (:func:`~tactifoot_vision.pitch.visible_area`) in
                ``visible_area_color`` under the players; nothing without a
                homography or when part of the frame lies beyond the horizon.
            velocity_seconds: draw each person's velocity as an arrow in their
                colour, from their dot to where it takes them in this many
                seconds, under the dots (an arrow shorter than the dot hides
                under it); ``None``: no arrows.
        """
        draw_width = width_px - 2 * padding_px
        if padding_px < 0 or draw_width < 50:
            raise ValueError(
                f"width_px={width_px} leaves no room for the pitch with padding_px={padding_px}"
            )
        _check_pixels("player_radius", player_radius)
        _check_pixels("ball_radius", ball_radius)
        _check_pixels("line_thickness", line_thickness)
        _check_alpha("space_control_alpha", space_control_alpha)
        if trail_seconds is not None and not trail_seconds > 0:
            raise ValueError(f"trail_seconds must be > 0 or None, got {trail_seconds}")
        if velocity_seconds is not None and not velocity_seconds > 0:
            raise ValueError(
                f"velocity_seconds must be > 0 or None, got {velocity_seconds}"
            )
        self.pitch = pitch or SoccerPitch()
        self.width_px = int(width_px)
        self.padding_px = int(padding_px)
        draw_height = round(draw_width * _STD_WIDTH / _STD_LENGTH)
        self.height_px = draw_height + 2 * self.padding_px
        self._scale = np.array(
            [draw_width / self.pitch.length, draw_height / self.pitch.width]
        )
        self._pitch_to_image = np.array([
            [self._scale[0], 0, self.padding_px],
            [0, self._scale[1], self.padding_px],
            [0, 0, 1],
        ])  # fmt: skip
        self.pitch_color = as_color(pitch_color)
        self.line_color = as_color(line_color)
        self.team_colors = [as_color(c) for c in team_colors]
        self.default_color = as_color(default_color)
        self.ball_color = as_color(ball_color)
        self.visible_area_color = as_color(visible_area_color)
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
        self.space_control_alpha = space_control_alpha
        self.trail_seconds = trail_seconds
        self.draw_visible_area = draw_visible_area
        self.velocity_seconds = velocity_seconds
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

    def draw(
        self,
        frame_result: FrameResult | None = None,
        space_control: SpaceControl | None = None,
        frame: np.ndarray | None = None,
        history: Sequence[FrameResult] | None = None,
        frame_size: tuple[int, int] | None = None,
        velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Radar image of one frame: players coloured by team, then the ball.

        ``None`` gives the empty pitch. Objects without pitch coordinates (NaN)
        are skipped. A ``space_control`` map (from
        :func:`tactifoot_vision.analysis.space_control`, on this radar's pitch)
        tints the pitch under the lines and players: team 0's colour where team
        0 controls the space, team 1's where team 1 does, a mix in between.

        ``frame``, the BGR video frame of ``frame_result``, is the background:
        warped onto the pitch through ``frame_result.homography`` (the
        bird's-eye view of :func:`~tactifoot_vision.viz.warp_to_pitch`, drawn
        at the radar's scale), the pitch colour where the frame shows nothing
        or there is no homography.

        ``history``: the results of the frames before ``frame_result``, oldest
        first, e.g. ``result.window(index, trail_seconds)``; only the motion
        tails use it, and they ignore frames older than ``trail_seconds``.

        ``frame_size``: ``(width, height)`` of the video frame, e.g.
        ``result.frame_size``, which the visible area needs; default the size
        of ``frame``.

        ``velocities``: ``(N, 2)`` m/s, row for row with
        ``frame_result.detections``, e.g. this frame's array of
        :func:`tv.analysis.velocities <tactifoot_vision.analysis.velocities>`;
        only the arrows use it, and a person with a NaN row gets none.
        """
        if frame is not None and frame_result is None:
            raise ValueError(
                "A frame needs its frame_result: the radar warps the frame "
                "with frame_result.homography"
            )
        homography = (
            None if frame is None or frame_result is None else frame_result.homography
        )
        if space_control is None and homography is None:
            image = self._background.copy()
        else:
            image = self._draw_background(space_control, frame, homography)
        if frame_result is None:
            return image
        if self.draw_visible_area:
            if frame_size is None and frame is not None:
                frame_size = (frame.shape[1], frame.shape[0])
            if frame_size is None:
                raise ValueError(
                    "draw_visible_area needs the frame size: pass "
                    "frame_size=result.frame_size, or the frame"
                )
            self._draw_visible_area(image, frame_result.homography, frame_size)
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
        colors = [palette[i] for i in index[order]]
        if self.trail_seconds and history and tracker_id is not None:
            self._draw_tails(image, frame_result, history, tracker_id[order], colors)
        if self.velocity_seconds and velocities is not None:
            velocities = np.asarray(velocities, dtype=float)
            if velocities.shape != (len(frame_result.detections), 2):
                raise ValueError(
                    f"velocities has shape {velocities.shape} but the frame has "
                    f"{len(frame_result.detections)} people; pass one (x, y) row "
                    "per detection, as tv.analysis.velocities gives"
                )
            self._draw_arrows(
                image, frame_result.pitch_xy[order], velocities[order], colors
            )
        self._draw_dots(
            image,
            frame_result.pitch_xy[order],
            colors,
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
    def _draw_background(
        self,
        space_control: SpaceControl | None = None,
        frame: np.ndarray | None = None,
        homography: np.ndarray | None = None,
    ) -> np.ndarray:
        image = np.full(
            (self.height_px, self.width_px, 3),
            self.pitch_color.as_bgr(),
            dtype=np.uint8,
        )
        if frame is not None and homography is not None:
            # Warped straight into this canvas, so the frame and the dots share
            # the radar's pitch -> pixel mapping.
            view = _warp_frame(frame, self._pitch_to_image @ homography, self.size)
            seen = view[..., 3] > 0
            image[seen] = view[seen, :3]
        if space_control is not None:
            if space_control.pitch != self.pitch:
                raise ValueError(
                    "The space control covers a "
                    f"{space_control.pitch.length:g} x {space_control.pitch.width:g} "
                    f"pitch, the radar a {self.pitch.length:g} x {self.pitch.width:g} "
                    "pitch; compute it on a result with the radar's pitch"
                )
            image = draw_space_control(
                image,
                space_control,
                self._pitch_to_image,
                self.team_colors,
                self.space_control_alpha,
            )
        lines, spots = pitch_markings(self.pitch)
        color = self.line_color.as_bgr()
        polylines = [to_fixed(self.to_pixels(line)) for line in lines]
        cv2.polylines(
            image, polylines, False, color, self.line_thickness, cv2.LINE_AA, SUBPIXEL
        )
        spot_radius = max(2.0, self.line_thickness * 1.5)
        for center in to_fixed(self.to_pixels(spots)):
            cv2.circle(
                image, tuple(center), int(spot_radius * (1 << SUBPIXEL)), color,
                -1, cv2.LINE_AA, SUBPIXEL,
            )  # fmt: skip
        return image

    def _draw_visible_area(
        self,
        image: np.ndarray,
        homography: np.ndarray | None,
        frame_size: tuple[int, int],
    ) -> None:
        """The visible area's outline, cut to the image: near the horizon its
        corners lie too far away for cv2's fixed-point coordinates."""
        area = visible_area(homography, frame_size)
        if area is None:
            return
        corners = self.to_pixels(area)
        margin = self.line_thickness + 2  # cut ends stay off the image
        low, high = np.array([-margin, -margin]), np.add(self.size, margin)
        edges = [
            clip_segment(a, b, low, high)
            for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True)
        ]
        cv2.polylines(
            image, [to_fixed(e) for e in edges if e is not None], False,
            self.visible_area_color.as_bgr(), self.line_thickness, cv2.LINE_AA,
            SUBPIXEL,
        )  # fmt: skip

    def _draw_tails(
        self,
        image: np.ndarray,
        frame_result: FrameResult,
        history: Sequence[FrameResult],
        tracker_ids: np.ndarray,
        colors: Sequence[sv.Color],
    ) -> None:
        """Each person's earlier pitch positions, joined by ``tracker_id``.

        A frame where the person is missing, or has no pitch position, breaks
        their tail; see :func:`~tactifoot_vision.viz.trail.draw_trail`.
        """
        seconds = self.trail_seconds or 0.0  # set: draw checked it
        frames, ages = trail_frames(history, frame_result, seconds)
        positions = [_positions_by_track(f) for f in frames]
        thickness = max(1, round(self.player_radius / 2))
        for track, color in zip(tracker_ids, colors, strict=True):
            points = np.array([p.get(int(track), (np.nan, np.nan)) for p in positions])
            valid = np.isfinite(points).all(axis=1)
            draw_trail(
                image, self.to_pixels(points), valid, ages, seconds, color, thickness
            )

    def _draw_arrows(
        self,
        image: np.ndarray,
        xy: np.ndarray,
        velocities: np.ndarray,
        colors: Sequence[sv.Color],
    ) -> None:
        """An arrow from each position along its velocity (m/s), as far as it
        goes in ``velocity_seconds``; none where either is NaN."""
        seconds = self.velocity_seconds or 0.0  # set: draw checked it
        tips = xy + velocities / metres_per_unit(self.pitch) * seconds
        thickness = max(1, round(self.player_radius / 4))
        for start, end, color in zip(
            self.to_pixels(xy), self.to_pixels(tips), colors, strict=True
        ):
            length = float(np.hypot(*(end - start)))
            if not np.isfinite(length) or length == 0:
                continue
            cv2.arrowedLine(
                image, tuple(to_fixed(start)), tuple(to_fixed(end)), color.as_bgr(),
                thickness, cv2.LINE_AA, SUBPIXEL, min(0.5, self.player_radius / length),
            )  # fmt: skip

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
            center = tuple(to_fixed(point))
            # A ring in the pitch colour keeps overlapping dots apart.
            cv2.circle(
                image, center, (radius + ring) << SUBPIXEL, self.pitch_color.as_bgr(),
                -1, cv2.LINE_AA, SUBPIXEL,
            )  # fmt: skip
            cv2.circle(
                image,
                center,
                radius << SUBPIXEL,
                color.as_bgr(),
                -1,
                cv2.LINE_AA,
                SUBPIXEL,
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


def _positions_by_track(frame_result: FrameResult) -> dict[int, np.ndarray]:
    """Pitch position per tracker id (empty without tracker ids)."""
    tracker_id = frame_result.detections.tracker_id
    if tracker_id is None:
        return {}
    return dict(zip((int(t) for t in tracker_id), frame_result.pitch_xy, strict=True))


def draw_space_control(
    image: np.ndarray,
    space_control: SpaceControl,
    pitch_to_image: np.ndarray,
    team_colors: Sequence[ColorLike] = TEAM_COLORS,
    alpha: float = 0.5,
) -> np.ndarray:
    """Tint the pitch in a copy of a top-down ``image`` with a space-control map.

    Team 0's colour where team 0 controls the space, team 1's where team 1
    does, a mix in between; outside the pitch the image is unchanged. Draw
    lines and players afterwards to keep them on top (``PitchRadar.draw`` does).

    Args:
        image: ``uint8`` image of the pitch seen from above: a BGR radar, or the
            BGRA bird's-eye view of :func:`~tactifoot_vision.viz.warp_to_pitch`,
            tinted only where it is not transparent, its alpha kept.
        space_control: from :func:`tactifoot_vision.analysis.space_control`.
        pitch_to_image: 3x3 homography from pitch coordinates (of
            ``space_control.pitch``) to pixels of ``image``, pixel centres at
            integers; for ``warp_to_pitch``'s image it is
            :func:`~tactifoot_vision.viz.pitch_to_birds_eye`.
        team_colors: team 0's colour, then team 1's.
        alpha: opacity of the tint (1 = opaque).
    """
    _check_alpha("alpha", alpha)
    if len(team_colors) < 2:
        raise ValueError(
            "Drawing space control needs two team colours; give team_colors "
            "a colour for team 0 and one for team 1"
        )
    pitch_to_image = np.asarray(pitch_to_image, dtype=float)
    if pitch_to_image.shape != (3, 3):
        raise ValueError(
            f"pitch_to_image must be a 3 x 3 homography, got shape {pitch_to_image.shape}"
        )
    pitch, (rows, columns) = space_control.pitch, space_control.control.shape
    cell_to_pitch = np.array([
        [pitch.length / columns, 0, 0.5 * pitch.length / columns],
        [0, pitch.width / rows, 0.5 * pitch.width / rows],
        [0, 0, 1],
    ])  # fmt: skip
    height, width = image.shape[:2]
    control = cv2.warpPerspective(
        space_control.control.astype(np.float32),
        pitch_to_image @ cell_to_pitch,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # out to the touchlines, not faded
    )[..., None]
    corners = cv2.perspectiveTransform(
        np.array(
            [
                [
                    [0, 0],
                    [pitch.length, 0],
                    [pitch.length, pitch.width],
                    [0, pitch.width],
                ]
            ],
            float,
        ),
        pitch_to_image,
    )[0]
    on_pitch = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(on_pitch, to_fixed(corners), 1, cv2.LINE_8, SUBPIXEL)
    team0, team1 = (np.array(as_color(c).as_bgr(), np.float32) for c in team_colors[:2])
    colors = control * team0 + (1 - control) * team1
    out = image.copy()
    inside = on_pitch.astype(bool)
    if image.shape[2] == 4:  # a bird's-eye view: tint only what the frame shows
        inside &= image[..., 3] > 0
    tinted = out[inside, :3] * (1 - alpha) + colors[inside] * alpha
    out[inside, :3] = np.round(tinted).astype(np.uint8)
    return out


def _check_alpha(name: str, alpha: float) -> None:
    if not 0 <= alpha <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {alpha}")


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
