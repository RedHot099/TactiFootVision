import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from numpy.typing import NDArray

from tactifoot_vision.domain import ExportArtifact
from tactifoot_vision.io import read_frames
from tactifoot_vision.projection import PitchModel

BgrColor = tuple[int, int, int]
CLASS_COLOR_ORDER = ("ball", "goalkeeper", "player", "referee")


@dataclass(frozen=True, slots=True)
class OverlayStyle:
    class_colors: dict[str, BgrColor] = field(
        default_factory=lambda: {
            "ball": (40, 220, 255),
            "goalkeeper": (255, 190, 40),
            "player": (80, 220, 80),
            "referee": (220, 80, 220),
        }
    )
    team_colors: dict[int, BgrColor] = field(
        default_factory=lambda: {
            0: (255, 80, 80),
            1: (80, 170, 255),
            2: (255, 230, 80),
            3: (180, 110, 255),
        }
    )
    fallback_color: BgrColor = (255, 255, 255)
    minimap_origin: tuple[int, int] = (18, 18)
    minimap_size: tuple[int, int] = (1200, 700)
    minimap_background: BgrColor = (20, 90, 35)
    minimap_line_color: BgrColor = (220, 220, 220)
    minimap_border_color: BgrColor = (245, 245, 245)
    minimap_padding: int = 50
    minimap_alpha: float = 0.65
    minimap_point_alpha: float = 0.95
    minimap_player_radius: int = 12
    minimap_ball_radius: int = 8
    radar_width_ratio: float = 0.4
    radar_height_ratio: float = 0.4
    radar_bottom_margin: int = 0
    bbox_thickness: int = 2
    label_font_scale: float = 0.45
    label_text_color: BgrColor = (245, 245, 245)

    def color_for_class(self, class_name: str) -> BgrColor:
        return self.class_colors.get(class_name, self.fallback_color)

    def color_for_row(self, row: object) -> BgrColor:
        class_name = str(getattr(row, "class_name", ""))
        team_id = getattr(row, "team_id", np.nan)
        if class_name in {"player", "goalkeeper"} and not pd.isna(team_id):
            return self.team_colors.get(int(team_id), self.color_for_class(class_name))
        return self.color_for_class(class_name)


class TrackOverlayRenderer:
    def __init__(self, *, style: OverlayStyle | None = None) -> None:
        self.style = style or OverlayStyle()
        self._palette = _supervision_palette(self.style)
        self._ellipse = sv.EllipseAnnotator(
            color=self._palette,
            thickness=2,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self._label = sv.LabelAnnotator(
            color=self._palette,
            text_color=_sv_color(self.style.label_text_color),
            text_padding=5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self._ball = sv.TriangleAnnotator(
            color=self._palette,
            base=14,
            height=14,
            position=sv.Position.TOP_CENTER,
            color_lookup=sv.ColorLookup.INDEX,
            outline_thickness=1,
            outline_color=sv.Color(0, 0, 0),
        )

    def draw(self, frame: NDArray[np.uint8], rows: pd.DataFrame) -> None:
        detections, color_lookup = _rows_to_supervision(rows, self.style)
        if len(detections) == 0:
            return
        class_ids = np.asarray(detections.class_id, dtype=np.int_)
        ball_mask = class_ids == _class_color_index("ball")
        player_mask = ~ball_mask
        if np.any(player_mask):
            player_detections = cast(sv.Detections, detections[player_mask])
            player_lookup = color_lookup[player_mask]
            player_labels = _tracker_id_labels(player_detections)
            self._ellipse.annotate(
                frame,
                player_detections,
                custom_color_lookup=player_lookup,
            )
            self._label.annotate(
                frame,
                player_detections,
                labels=player_labels,
                custom_color_lookup=player_lookup,
            )
        if np.any(ball_mask):
            self._ball.annotate(
                frame,
                detections[ball_mask],
                custom_color_lookup=color_lookup[ball_mask],
            )


class PitchMinimapRenderer:
    def __init__(
        self,
        *,
        pitch: PitchModel | None = None,
        style: OverlayStyle | None = None,
        fallback_to_image_position: bool = False,
    ) -> None:
        self.pitch = pitch or PitchModel()
        self.style = style or OverlayStyle()
        self.fallback_to_image_position = fallback_to_image_position

    def draw(self, frame: NDArray[np.uint8], rows: pd.DataFrame) -> None:
        pitch = SoccerPitchMinimap(
            style=self.style,
            pitch=self.pitch,
            fallback_to_image_position=self.fallback_to_image_position,
        ).draw(rows, frame_shape=frame.shape)
        frame_height, frame_width = frame.shape[:2]
        target_size = (
            max(1, int(frame_width * self.style.radar_width_ratio)),
            max(1, int(frame_height * self.style.radar_height_ratio)),
        )
        radar = sv.resize_image(
            pitch,
            target_size,
            keep_aspect_ratio=True,
        )
        radar_height, radar_width = radar.shape[:2]
        x0 = frame_width // 2 - radar_width // 2
        y0 = frame_height - radar_height - self.style.radar_bottom_margin
        if x0 < 0 or y0 < 0:
            return
        sv.draw_image(
            scene=frame,
            image=radar,
            opacity=self.style.minimap_alpha,
            rect=sv.Rect(
                x=x0,
                y=y0,
                width=radar_width,
                height=radar_height,
            ),
        )


class SoccerPitchMinimap:
    def __init__(
        self,
        *,
        style: OverlayStyle | None = None,
        pitch: PitchModel | None = None,
        fallback_to_image_position: bool = False,
    ) -> None:
        self.style = style or OverlayStyle()
        self.pitch = pitch or PitchModel()
        self.fallback_to_image_position = fallback_to_image_position
        self.geometry = SoccerPitchGeometry(
            length=self.pitch.length,
            width=self.pitch.width,
        )

    def draw(
        self,
        rows: pd.DataFrame,
        *,
        frame_shape: tuple[int, ...] | None = None,
    ) -> NDArray[np.uint8]:
        width, height = self.style.minimap_size
        canvas = np.full(
            (height, width, 3),
            self.style.minimap_background,
            dtype=np.uint8,
        )
        self._draw_pitch(canvas)
        self._draw_points(canvas, rows, frame_shape=frame_shape)
        return canvas

    def _draw_pitch(self, canvas: NDArray[np.uint8]) -> None:
        thickness = 1
        for edge in self.geometry.edges:
            cv2.line(
                canvas,
                self._scale_standard_point(self.geometry.vertices[edge[0]]),
                self._scale_standard_point(self.geometry.vertices[edge[1]]),
                self.style.minimap_line_color,
                thickness,
                cv2.LINE_AA,
            )

        center = (self.pitch.length / 2.0, self.pitch.width / 2.0)
        center_px = self._scale_standard_point(center)
        radius = int(self.geometry.centre_circle_radius * self._scale_y)
        cv2.circle(canvas, center_px, radius, self.style.minimap_line_color, thickness)
        cv2.circle(canvas, center_px, 2, self.style.minimap_line_color, -1)

        spot_radius = 2
        left_spot = (self.geometry.penalty_spot_distance, self.pitch.width / 2.0)
        right_spot = (
            self.pitch.length - self.geometry.penalty_spot_distance,
            self.pitch.width / 2.0,
        )
        cv2.circle(
            canvas,
            self._scale_standard_point(left_spot),
            spot_radius,
            self.style.minimap_line_color,
            -1,
        )
        cv2.circle(
            canvas,
            self._scale_standard_point(right_spot),
            spot_radius,
            self.style.minimap_line_color,
            -1,
        )
        self._draw_penalty_arcs(canvas)

    def _draw_points(
        self,
        canvas: NDArray[np.uint8],
        rows: pd.DataFrame,
        *,
        frame_shape: tuple[int, ...] | None,
    ) -> None:
        for row in rows.itertuples(index=False):
            point = _row_pitch_point(row)
            if point is None:
                if not self.fallback_to_image_position:
                    continue
                if frame_shape is None:
                    continue
                point = _row_image_pitch_point(row, frame_shape, self.pitch)
                if point is None:
                    continue
            class_name = str(getattr(row, "class_name", ""))
            radius = (
                self.style.minimap_ball_radius
                if class_name == "ball"
                else self.style.minimap_player_radius
            )
            scaled_point = self._scale_logical_point(point)
            point_overlay = canvas.copy()
            cv2.circle(
                point_overlay,
                scaled_point,
                radius,
                self.style.color_for_row(row),
                -1,
                cv2.LINE_AA,
            )
            cv2.addWeighted(
                point_overlay,
                self.style.minimap_point_alpha,
                canvas,
                1.0 - self.style.minimap_point_alpha,
                0,
                canvas,
            )
            cv2.circle(canvas, scaled_point, radius, (10, 10, 10), 2, cv2.LINE_AA)

    def _draw_penalty_arcs(self, canvas: NDArray[np.uint8]) -> None:
        arc_radius = self.geometry.centre_circle_radius
        delta_x = abs(
            self.geometry.penalty_box_length - self.geometry.penalty_spot_distance
        )
        if arc_radius <= delta_x:
            return
        alpha = math.degrees(math.acos(delta_x / arc_radius))
        axes = (int(arc_radius * self._scale_x), int(arc_radius * self._scale_y))
        left_spot = self._scale_standard_point(
            (self.geometry.penalty_spot_distance, self.pitch.width / 2.0)
        )
        right_spot = self._scale_standard_point(
            (
                self.pitch.length - self.geometry.penalty_spot_distance,
                self.pitch.width / 2.0,
            )
        )
        cv2.ellipse(
            canvas,
            left_spot,
            axes,
            0,
            -alpha,
            alpha,
            self.style.minimap_line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            canvas,
            right_spot,
            axes,
            0,
            180.0 - alpha,
            180.0 + alpha,
            self.style.minimap_line_color,
            1,
            cv2.LINE_AA,
        )

    @property
    def _inner_width(self) -> int:
        return self.style.minimap_size[0] - 2 * self.style.minimap_padding

    @property
    def _inner_height(self) -> int:
        return self.style.minimap_size[1] - 2 * self.style.minimap_padding

    @property
    def _scale_x(self) -> float:
        return self._inner_width / self.pitch.length

    @property
    def _scale_y(self) -> float:
        return self._inner_height / self.pitch.width

    def _scale_standard_point(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            int(round(x * self._scale_x)) + self.style.minimap_padding,
            int(round((self.pitch.width - y) * self._scale_y))
            + self.style.minimap_padding,
        )

    def _scale_logical_point(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            int(round(np.clip(x, 0.0, self.pitch.length) * self._scale_x))
            + self.style.minimap_padding,
            int(round(np.clip(y, 0.0, self.pitch.width) * self._scale_y))
            + self.style.minimap_padding,
        )


@dataclass(frozen=True, slots=True)
class SoccerPitchGeometry:
    length: float
    width: float

    @property
    def penalty_box_length(self) -> float:
        return self.length * (16.5 / 105.0)

    @property
    def penalty_box_width(self) -> float:
        return self.width * (40.32 / 68.0)

    @property
    def goal_box_length(self) -> float:
        return self.length * (5.5 / 105.0)

    @property
    def goal_box_width(self) -> float:
        return self.width * (18.32 / 68.0)

    @property
    def centre_circle_radius(self) -> float:
        return self.width * (9.15 / 68.0)

    @property
    def penalty_spot_distance(self) -> float:
        return self.length * (11.0 / 105.0)

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        half_width = self.width / 2.0
        half_length = self.length / 2.0
        half_penalty_width = self.penalty_box_width / 2.0
        half_goal_width = self.goal_box_width / 2.0
        return (
            (0.0, 0.0),
            (0.0, half_width - half_penalty_width),
            (0.0, half_width - half_goal_width),
            (0.0, half_width + half_goal_width),
            (0.0, half_width + half_penalty_width),
            (0.0, self.width),
            (self.goal_box_length, half_width - half_goal_width),
            (self.goal_box_length, half_width + half_goal_width),
            (self.penalty_spot_distance, half_width),
            (self.penalty_box_length, half_width - half_penalty_width),
            (self.penalty_box_length, half_width - half_goal_width),
            (self.penalty_box_length, half_width + half_goal_width),
            (self.penalty_box_length, half_width + half_penalty_width),
            (half_length, 0.0),
            (half_length, half_width - self.centre_circle_radius),
            (half_length, half_width + self.centre_circle_radius),
            (half_length, self.width),
            (self.length - self.penalty_box_length, half_width - half_penalty_width),
            (self.length - self.penalty_box_length, half_width - half_goal_width),
            (self.length - self.penalty_box_length, half_width + half_goal_width),
            (self.length - self.penalty_box_length, half_width + half_penalty_width),
            (self.length - self.penalty_spot_distance, half_width),
            (self.length - self.goal_box_length, half_width - half_goal_width),
            (self.length - self.goal_box_length, half_width + half_goal_width),
            (self.length, 0.0),
            (self.length, half_width - half_penalty_width),
            (self.length, half_width - half_goal_width),
            (self.length, half_width + half_goal_width),
            (self.length, half_width + half_penalty_width),
            (self.length, self.width),
        )

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (9, 10),
            (10, 11),
            (11, 12),
            (13, 16),
            (17, 18),
            (18, 19),
            (19, 20),
            (22, 23),
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (0, 13),
            (1, 9),
            (2, 6),
            (3, 7),
            (4, 12),
            (5, 16),
            (13, 24),
            (17, 25),
            (22, 26),
            (23, 27),
            (20, 28),
            (16, 29),
        )


class PipelineFrameAnnotator:
    def __init__(
        self,
        *,
        track_renderer: TrackOverlayRenderer | None = None,
        minimap_renderer: PitchMinimapRenderer | None = None,
    ) -> None:
        self.track_renderer = track_renderer or TrackOverlayRenderer()
        self.minimap_renderer = minimap_renderer or PitchMinimapRenderer()

    def annotate(
        self, frame: NDArray[np.uint8], rows: pd.DataFrame
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        self.track_renderer.draw(output, rows)
        self.minimap_renderer.draw(output, rows)
        return output


class PipelineVideoRenderer:
    def __init__(
        self,
        *,
        annotator: PipelineFrameAnnotator | None = None,
        default_fps: float = 25.0,
        codec: str = "mp4v",
    ) -> None:
        self.annotator = annotator or PipelineFrameAnnotator()
        self.default_fps = default_fps
        self.codec = codec

    def render(
        self,
        *,
        source: str | Path,
        annotations: pd.DataFrame,
        output_path: str | Path,
        max_frames: int | None = None,
        fps: float | None = None,
    ) -> ExportArtifact:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames = read_frames(source)
        first_frame = next(frames, None)
        if first_frame is None:
            raise ValueError(f"No readable frames found in source: {source}")

        height, width = first_frame.image.shape[:2]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*self.codec),
            fps or _source_fps(Path(source), self.default_fps),
            (width, height),
        )
        rows_by_frame = (
            {
                int(frame_index): rows
                for frame_index, rows in annotations.groupby("frame", sort=False)
            }
            if "frame" in annotations.columns
            else {}
        )
        count = 0
        try:
            frame_rows = rows_by_frame.get(first_frame.index, _empty_annotations())
            writer.write(self.annotator.annotate(first_frame.image, frame_rows))
            count += 1
            for frame in frames:
                if max_frames is not None and count >= max_frames:
                    break
                frame_rows = rows_by_frame.get(frame.index, _empty_annotations())
                writer.write(self.annotator.annotate(frame.image, frame_rows))
                count += 1
        finally:
            writer.release()
        return ExportArtifact(output, "annotated_video", count)

    def render_csv(
        self,
        *,
        source: str | Path,
        annotations_csv: str | Path,
        output_path: str | Path,
        max_frames: int | None = None,
        fps: float | None = None,
    ) -> ExportArtifact:
        return self.render(
            source=source,
            annotations=pd.read_csv(annotations_csv),
            output_path=output_path,
            max_frames=max_frames,
            fps=fps,
        )


def _row_box(row: object) -> tuple[int, int, int, int] | None:
    values = [
        getattr(row, "x", np.nan),
        getattr(row, "y", np.nan),
        getattr(row, "width", np.nan),
        getattr(row, "height", np.nan),
    ]
    if any(pd.isna(value) for value in values):
        return None
    x, y, width, height = [int(round(float(value))) for value in values]
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _rows_to_supervision(
    rows: pd.DataFrame, style: OverlayStyle
) -> tuple[sv.Detections, NDArray[np.int_]]:
    xyxy: list[list[float]] = []
    class_ids: list[int] = []
    tracker_ids: list[int] = []
    color_lookup: list[int] = []
    for row in rows.itertuples(index=False):
        box = _row_box(row)
        if box is None:
            continue
        x, y, width, height = box
        class_name = str(getattr(row, "class_name", ""))
        track_id = _safe_int(getattr(row, "track_id", len(tracker_ids)))
        xyxy.append([x, y, x + width, y + height])
        class_ids.append(_class_color_index(class_name))
        tracker_ids.append(track_id)
        color_lookup.append(_row_color_index(row, style))
    if not xyxy:
        return sv.Detections.empty(), np.empty((0,), dtype=np.int_)
    return (
        sv.Detections(
            xyxy=np.asarray(xyxy, dtype=np.float32),
            class_id=np.asarray(class_ids, dtype=np.int_),
            tracker_id=np.asarray(tracker_ids, dtype=np.int_),
        ),
        np.asarray(color_lookup, dtype=np.int_),
    )


def _tracker_id_labels(detections: sv.Detections) -> list[str]:
    if detections.tracker_id is None:
        return []
    tracker_ids = np.asarray(detections.tracker_id, dtype=np.int_)
    return [str(int(tracker_id)) for tracker_id in tracker_ids]


def _row_color_index(row: object, style: OverlayStyle) -> int:
    class_name = str(getattr(row, "class_name", ""))
    team_id = getattr(row, "team_id", np.nan)
    if class_name in {"player", "goalkeeper"} and not pd.isna(team_id):
        team_id_int = int(team_id)
        if team_id_int in style.team_colors:
            return len(style.class_colors) + sorted(style.team_colors).index(
                team_id_int
            )
    if class_name in CLASS_COLOR_ORDER:
        return _class_color_index(class_name)
    return len(CLASS_COLOR_ORDER) + len(style.team_colors)


def _class_color_index(class_name: str) -> int:
    if class_name in CLASS_COLOR_ORDER:
        return CLASS_COLOR_ORDER.index(class_name)
    return len(CLASS_COLOR_ORDER)


def _supervision_palette(style: OverlayStyle) -> sv.ColorPalette:
    return sv.ColorPalette([_sv_color(color) for color in _bgr_palette(style)])


def _bgr_palette(style: OverlayStyle) -> list[BgrColor]:
    colors = [
        style.class_colors.get(class_name, style.fallback_color)
        for class_name in CLASS_COLOR_ORDER
    ]
    colors.extend(style.team_colors[key] for key in sorted(style.team_colors))
    colors.append(style.fallback_color)
    return colors


def estimate_team_colors_from_crops(
    crops: list[NDArray[np.uint8]],
    labels: NDArray[np.int_],
    *,
    fallback_colors: dict[int, BgrColor] | None = None,
) -> dict[int, BgrColor]:
    fallback = fallback_colors or OverlayStyle().team_colors
    colors: dict[int, BgrColor] = {}
    for label in sorted({int(value) for value in labels.tolist()}):
        label_crops = [
            crop
            for crop, crop_label in zip(crops, labels, strict=True)
            if int(crop_label) == label
        ]
        colors[label] = _dominant_kit_color(label_crops) or fallback.get(
            label, OverlayStyle().fallback_color
        )
    return colors


def _dominant_kit_color(crops: list[NDArray[np.uint8]]) -> BgrColor | None:
    pixels = []
    for crop in crops:
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            continue
        height, width = crop.shape[:2]
        upper = crop[: max(1, int(height * 0.65)), :]
        center = upper[:, max(0, int(width * 0.2)) : max(1, int(width * 0.8))]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        non_grass = ~((35 <= hue) & (hue <= 85) & (saturation > 45))
        mask = (saturation > 35) & (value > 35) & non_grass
        selected = center[mask]
        if selected.size:
            pixels.append(selected)
    if not pixels:
        return None
    merged = np.concatenate(pixels, axis=0)
    color = np.median(merged, axis=0)
    return (int(color[0]), int(color[1]), int(color[2]))


def _sv_color(color: BgrColor) -> sv.Color:
    blue, green, red = color
    return sv.Color(red, green, blue)


def _safe_int(value: object) -> int:
    if pd.isna(value):
        return -1
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return -1


def _row_pitch_point(row: object) -> tuple[float, float] | None:
    pitch_x = getattr(row, "pitch_x", np.nan)
    pitch_y = getattr(row, "pitch_y", np.nan)
    if pd.isna(pitch_x) or pd.isna(pitch_y):
        return None
    return float(pitch_x), float(pitch_y)


def _row_image_pitch_point(
    row: object, frame_shape: tuple[int, ...], pitch: PitchModel
) -> tuple[float, float] | None:
    box = _row_box(row)
    if box is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None
    x, y, width, height = box
    class_name = str(getattr(row, "class_name", ""))
    image_x = x + width / 2.0
    image_y = y + height / 2.0 if class_name == "ball" else y + height
    return (
        float(np.clip(image_x / frame_width, 0.0, 1.0) * pitch.length),
        float(np.clip(image_y / frame_height, 0.0, 1.0) * pitch.width),
    )


def _source_fps(source: Path, default: float) -> float:
    if source.is_file():
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps and np.isfinite(fps):
                return float(fps)
        finally:
            capture.release()
    return default


def _empty_annotations() -> pd.DataFrame:
    return pd.DataFrame()
