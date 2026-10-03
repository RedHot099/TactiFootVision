"""Drawing pipeline results and dataset labels onto images."""

import logging
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.data.annotations import NOT_LABELLED, Annotations
from tactifoot_vision.pipeline.result import FrameResult, ObjectMasks
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.viz.radar import (
    BALL_COLOR,
    DEFAULT_COLOR,
    TEAM_COLORS,
    ColorLike,
    as_color,
    pitch_markings,
    team_color_index,
    text_color_for,
)

# Ground-truth class colours: bright hues that stand out on grass, checked for
# colour-blind separation (labels always accompany them).
CLASS_COLORS = ("#FFD400", "#FF6B00", "#00BFFF", "#FF1493", "#A259FF", "#FF4040")
KEYPOINT_COLOR = "#FFA500"  # confident keypoint that agrees with the homography
KEYPOINT_OUTLIER_COLOR = "#FF0000"  # confident keypoint the homography disagrees with
PITCH_LINE_COLOR = "#FFFFFF"
MASK_OPACITY = 0.45

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled_sizes(image: np.ndarray) -> tuple[int, float]:
    """Line thickness and font scale for an image: 2 px and 0.5 at 1080p."""
    short_side = min(image.shape[:2])
    return max(1, round(short_side / 540)), max(0.35, short_side / 2160)


def _put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: sv.Color,
    thickness: int = 1,
) -> None:
    """Text with a dark outline so it stays legible on any background."""
    cv2.putText(image, text, org, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, _FONT, scale, color.as_bgr(), thickness, cv2.LINE_AA)


def _draw_dot(
    image: np.ndarray, center: np.ndarray, radius: int, color: sv.Color
) -> None:
    point = (round(float(center[0])), round(float(center[1])))
    cv2.circle(image, point, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, point, radius, color.as_bgr(), -1, cv2.LINE_AA)


def project_to_frame(
    points: np.ndarray, homography: np.ndarray, frame_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project pitch points into the frame with a frame->pitch ``homography``.

    Returns ``(xy, valid)``: pixel coordinates and a mask of points that lie in
    front of the camera and not absurdly far outside the frame (points beyond
    the horizon would otherwise wrap around and draw garbage).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    width, height = frame_size
    # The side of the horizon the camera looks at: where the frame centre lands.
    centre = homography @ np.array([width / 2, height / 2, 1.0])
    inverse = np.linalg.inv(homography)
    projected = np.column_stack([points, np.ones(len(points))]) @ inverse.T
    w = projected[:, 2] * np.sign(centre[2])
    valid = w > 1e-9
    xy = projected[:, :2] / np.where(valid, projected[:, 2], 1.0)[:, None]
    limit = 4 * max(width, height)
    valid &= (np.abs(xy) < limit).all(axis=1)
    return xy, valid


class FrameAnnotator:
    """Draws one :class:`FrameResult` onto its video frame.

    People are coloured by ``team_id`` (``team_colors[i]`` for team ``i``,
    ``default_color`` otherwise, e.g. referees), labelled with their track id,
    and drawn as boxes (``style="standard"``) or as ellipses at their feet
    (``style="video_game"``, with a triangle over the ball). With a homography,
    the pitch model is projected into the frame; confident pitch keypoints are
    orange when the homography agrees with them and red when it does not.

    >>> annotator = FrameAnnotator(style="video_game")
    >>> image = annotator.annotate(frame, result.frame(index))
    """

    def __init__(
        self,
        style: Literal["standard", "video_game"] = "standard",
        team_colors: Sequence[ColorLike] = TEAM_COLORS,
        default_color: ColorLike = DEFAULT_COLOR,
        ball_color: ColorLike = BALL_COLOR,
        draw_boxes: bool = True,
        draw_labels: bool = True,
        draw_masks: bool = False,
        draw_keypoints: bool = True,
        draw_pitch_lines: bool = True,
        keypoint_threshold: float = 0.5,
        pitch: SoccerPitch | None = None,
        pitch_line_color: ColorLike = PITCH_LINE_COLOR,
    ) -> None:
        """
        Args:
            style: ``"standard"`` boxes or ``"video_game"`` ellipses and ball triangle.
            team_colors: one colour per team id.
            default_color: people without a team.
            draw_boxes: boxes/ellipses of people and the ball marker.
            draw_labels: ``#<track id>`` tags.
            draw_masks: segmentation masks, kept by ``Pipeline(keep_masks=True)``
                with a mask tracker (SAM2); warns once if a result has none.
            draw_keypoints: pitch keypoints with confidence ``>= keypoint_threshold``.
            draw_pitch_lines: pitch markings projected with the inverse homography.
            pitch: pitch model the homography maps to (default ``SoccerPitch()``).
        """
        if style not in ("standard", "video_game"):
            raise ValueError(f"Unknown style {style!r}; use 'standard' or 'video_game'")
        self.style = style
        self.team_colors = [as_color(c) for c in team_colors]
        self.default_color = as_color(default_color)
        self.ball_color = as_color(ball_color)
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.draw_masks = draw_masks
        self.draw_keypoints = draw_keypoints
        self.draw_pitch_lines = draw_pitch_lines
        self.keypoint_threshold = keypoint_threshold
        self.pitch = pitch or SoccerPitch()
        self.pitch_line_color = as_color(pitch_line_color)
        colors = [*self.team_colors, self.default_color]
        self._palette = sv.ColorPalette(colors)
        self._text_palette = sv.ColorPalette([text_color_for(c) for c in colors])
        self._markings = pitch_markings(self.pitch)
        self._warned_no_masks = False

    def annotate(self, frame: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        """Return an annotated copy of ``frame`` (BGR ``uint8``)."""
        image = frame.copy()
        thickness, text_scale = _scaled_sizes(image)
        homography = frame_result.homography
        if self.draw_pitch_lines and homography is not None:
            self._draw_pitch(image, homography, thickness)
        if self.draw_keypoints and frame_result.keypoints is not None:
            self._draw_keypoints(image, frame_result.keypoints, homography, text_scale)

        people = frame_result.detections
        lookup = team_color_index(frame_result.team_ids, len(self.team_colors))
        if len(people):
            if self.draw_masks:
                self._draw_masks(image, frame_result.masks, lookup)
            if self.draw_boxes:
                image = self._people_annotator(thickness).annotate(
                    image, people, custom_color_lookup=lookup
                )
            if self.draw_labels:
                image = self._label_annotator(thickness, text_scale).annotate(
                    image, people, labels=_labels(people), custom_color_lookup=lookup
                )
        if self.draw_boxes and len(frame_result.ball):
            image = self._ball_annotator(thickness).annotate(image, frame_result.ball)
        return image

    # ----------------------------------------------------------- annotators
    def _people_annotator(
        self, thickness: int
    ) -> sv.BoxAnnotator | sv.EllipseAnnotator:
        if self.style == "video_game":
            return sv.EllipseAnnotator(color=self._palette, thickness=thickness)
        return sv.BoxAnnotator(color=self._palette, thickness=thickness)

    def _label_annotator(self, thickness: int, text_scale: float) -> sv.LabelAnnotator:
        position = (
            sv.Position.BOTTOM_CENTER
            if self.style == "video_game"
            else sv.Position.TOP_LEFT
        )
        return sv.LabelAnnotator(
            color=self._palette,
            text_color=self._text_palette,
            text_scale=text_scale,
            text_thickness=max(1, thickness // 2),
            text_padding=max(2, round(6 * text_scale)),
            text_position=position,
            border_radius=max(0, thickness),
        )

    def _ball_annotator(self, thickness: int) -> sv.BoxAnnotator | sv.TriangleAnnotator:
        if self.style == "video_game":
            size = 10 * thickness
            return sv.TriangleAnnotator(
                color=self.ball_color,
                base=size,
                height=round(size * 0.85),
                outline_thickness=max(1, thickness // 2),
            )
        return sv.BoxAnnotator(color=self.ball_color, thickness=thickness)

    def _draw_masks(
        self, image: np.ndarray, masks: ObjectMasks | None, lookup: np.ndarray
    ) -> None:
        if masks is None:
            if not self._warned_no_masks:
                logger.warning(
                    "draw_masks=True but the result has no masks; run the pipeline "
                    "with a mask tracker (sam2) and keep_masks=True"
                )
                self._warned_no_masks = True
            return
        tints = np.array([c.as_bgr() for c in self._palette.colors], dtype=np.float32)
        for (x, y), crop, index in zip(masks.origins, masks.crops, lookup, strict=True):
            region = image[y : y + crop.shape[0], x : x + crop.shape[1]]
            region[crop] = (
                (1 - MASK_OPACITY) * region[crop] + MASK_OPACITY * tints[index]
            ).astype(np.uint8)

    # --------------------------------------------------------------- pitch
    def _draw_pitch(
        self, image: np.ndarray, homography: np.ndarray, thickness: int
    ) -> None:
        lines, spots = self._markings
        size = (image.shape[1], image.shape[0])
        color = self.pitch_line_color.as_bgr()
        polylines = []
        for line in lines:
            dense = _densify(line, samples=32)
            xy, valid = project_to_frame(dense, homography, size)
            polylines.extend(xy[run] for run in _runs(valid) if len(run) > 1)
        if polylines:
            cv2.polylines(
                image, [np.round(p * 16).astype(np.int32) for p in polylines], False,
                color, thickness, cv2.LINE_AA, 4,
            )  # fmt: skip
        xy, valid = project_to_frame(spots, homography, size)
        for point in xy[valid]:
            _draw_dot(image, point, thickness + 1, self.pitch_line_color)

    def _draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: sv.KeyPoints,
        homography: np.ndarray | None,
        text_scale: float,
    ) -> None:
        if len(keypoints) == 0:
            return
        xy = keypoints.xy[0]
        confidence = (
            keypoints.confidence[0]
            if keypoints.confidence is not None
            else np.ones(len(xy))
        )
        agrees = np.ones(len(xy), dtype=bool)
        if homography is not None and len(xy) == self.pitch.num_keypoints:
            size = (image.shape[1], image.shape[0])
            expected, valid = project_to_frame(self.pitch.vertices, homography, size)
            tolerance = 0.01 * float(np.hypot(*size))
            agrees = valid & (np.linalg.norm(xy - expected, axis=1) <= tolerance)
        radius = max(3, round(8 * text_scale))
        used, outlier = as_color(KEYPOINT_COLOR), as_color(KEYPOINT_OUTLIER_COLOR)
        drawable = (confidence >= self.keypoint_threshold) & np.isfinite(xy).all(axis=1)
        for i in np.flatnonzero(drawable):
            color = used if agrees[i] else outlier
            _draw_dot(image, xy[i], radius, color)
            org = (round(float(xy[i, 0])) + radius + 2, round(float(xy[i, 1])) - radius)
            _put_text(image, str(i), org, text_scale * 0.8, sv.Color.WHITE)


def _labels(detections: sv.Detections) -> list[str]:
    if detections.tracker_id is not None:
        return [f"#{int(t)}" for t in detections.tracker_id]
    names = detections.data.get("class_name")
    return [str(n) for n in names] if names is not None else [""] * len(detections)


def _densify(line: np.ndarray, samples: int) -> np.ndarray:
    """Subdivide straight segments so the horizon can cut them (curves are already dense)."""
    if len(line) != 2:
        return line
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return line[0] * (1 - t) + line[1] * t


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Indices of consecutive ``True`` stretches."""
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return []
    return np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)


def draw_annotations(
    image: np.ndarray,
    annotations: Annotations,
    class_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw dataset ground truth on a copy of ``image``.

    Boxes are coloured by class and tagged with the class name (or id);
    labelled keypoints (visibility > 0) are drawn with their index, unlabelled
    ones are hidden.
    """
    out = image.copy()
    thickness, text_scale = _scaled_sizes(out)
    if len(annotations):
        palette = sv.ColorPalette.from_hex(list(CLASS_COLORS))
        text_palette = sv.ColorPalette([text_color_for(c) for c in palette.colors])
        detections = annotations.to_detections()
        names = [
            class_names[c]
            if class_names is not None and c < len(class_names)
            else str(c)
            for c in annotations.class_ids
        ]
        out = sv.BoxAnnotator(color=palette, thickness=thickness).annotate(
            out, detections
        )
        out = sv.LabelAnnotator(
            color=palette,
            text_color=text_palette,
            text_scale=text_scale,
            text_thickness=max(1, thickness // 2),
            text_padding=max(2, round(6 * text_scale)),
        ).annotate(out, detections, labels=names)
    if annotations.keypoints is not None:
        radius = max(3, round(8 * text_scale))
        color = as_color(KEYPOINT_COLOR)
        for instance in annotations.keypoints:
            for k, (x, y, visibility) in enumerate(instance):
                if visibility <= NOT_LABELLED or not np.isfinite([x, y]).all():
                    continue
                _draw_dot(out, np.array([x, y]), radius, color)
                org = (round(float(x)) + radius + 2, round(float(y)) - radius)
                _put_text(out, str(k), org, text_scale * 0.8, sv.Color.WHITE)
    return out
