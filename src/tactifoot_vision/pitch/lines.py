"""Refine a frame -> pitch homography by fitting the pitch markings to the painted lines of the frame.

The pitch model's markings are sampled densely, projected into the frame and
pulled onto the nearest white-line pixel (a chamfer fit on the distance
transform of a line mask). The unknowns are the pixel offsets of four control
points spread over the frame, which keeps the problem well conditioned; a
robust least squares fits them coarse to fine. The image thresholds are tuned
on 1080p broadcast footage.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from tactifoot_vision.pitch.pitch import SoccerPitch, pitch_markings
from tactifoot_vision.pitch.projection import project_points

# Grass in OpenCV HSV (hue 0-180): green, saturated, not dark.
_GRASS_LOW, _GRASS_HIGH = (30, 60, 40), (90, 255, 255)
_TOPHAT_SIZE = 15  # px; lines narrower than this stand out of the grass
_MIN_TOPHAT = 22  # grey levels brighter than the grass around
_MAX_LINE_SATURATION = 110  # lines are white, kits often are not
# (distance cap in px, robust loss scale in px) per stage, coarse to fine: the
# first stage pulls lines up to 40 px off, the last one ignores what is beyond 6 px.
_STAGES = ((40.0, 8.0), (12.0, 2.5), (6.0, 1.2))
LINE_DISTANCE_CAP_PX = _STAGES[-1][0]  # the fit's distances are measured up to this
_GOOD_INLIER_SHARE = 0.85  # a fit this good from the first start needs no second
_MODEL_STEP = 0.4  # pitch units between model points along the markings
_MAX_EVALUATIONS = 200  # per stage
# Control points as fractions of the frame (width, height), below the stands.
_CONTROLS = np.array([[0.15, 0.35], [0.85, 0.35], [0.85, 0.9], [0.15, 0.9]])


@dataclass(frozen=True)
class LineFit:
    """How well a refined homography lays the pitch markings on its frame's painted lines.

    ``points`` model points along the markings were visible on the grass;
    ``inlier_share`` of them lie within the estimator's ``line_inlier_px`` of
    a painted-line pixel, and ``median_px`` is their median distance to one
    (capped at :data:`LINE_DISTANCE_CAP_PX`). A low share means the fit found few lines to hold on to.
    """

    points: int
    inlier_share: float
    median_px: float


class LineRefiner:
    """Fits frame -> pitch homographies to the painted lines of their frames.

    Holds the model points of ``pitch`` and the thresholds of
    :class:`~tactifoot_vision.pitch.HomographyEstimator`'s ``refine_with_lines``,
    which validates them.
    """

    def __init__(
        self, pitch: SoccerPitch, *, min_points: int, inlier_px: float
    ) -> None:
        self.model = model_points(pitch)
        self.min_points = min_points
        self.inlier_px = inlier_px

    def refine(
        self,
        frame: np.ndarray,
        starts: Sequence[np.ndarray | None],
        boxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, LineFit] | None:
        """The fit with the largest inlier share among those from each of ``starts``.

        ``None`` starts are skipped, and so are the rest once a fit is good
        enough. Returns ``None`` when no start gives a fit (too few model points
        in view, or no painted line).
        """
        line, seen = line_masks(frame, boxes)
        best = None
        for start in starts:
            if start is None:
                continue
            refined = refine_homography(
                start,
                self.model,
                line,
                seen,
                min_points=self.min_points,
                inlier_px=self.inlier_px,
            )
            if refined is not None and (
                best is None or refined[1].inlier_share > best[1].inlier_share
            ):
                best = refined
            if best is not None and best[1].inlier_share > _GOOD_INLIER_SHARE:
                break
        return best


def model_points(pitch: SoccerPitch) -> np.ndarray:
    """Points every ``_MODEL_STEP`` pitch units along the painted lines, circle and arcs of ``pitch``."""
    lines, _ = pitch_markings(pitch, samples=256)
    points = []
    for line in lines:
        segments = np.diff(line, axis=0)
        for start, segment in zip(line[:-1], segments, strict=True):
            n = max(1, int(np.ceil(np.hypot(*segment) / _MODEL_STEP)))
            points.append(start + np.arange(n)[:, None] / n * segment)
    return np.vstack(points)


def line_masks(
    frame: np.ndarray, boxes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The painted lines of a BGR frame and where the pitch can be seen, as uint8 0/255 masks.

    Returns ``(line, seen)``: ``line`` holds thin, bright, unsaturated pixels on
    the grass, ``seen`` the grass away from its edges, where a marking would
    show if it were there. Both leave out the people ``boxes`` (xyxy, frame
    pixels) with a margin for limbs and shadow, so white kits are no lines and
    the markings they hide count for nothing.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = _grass_mask(hsv)
    people = np.zeros(grass.shape, np.uint8)
    boxes = np.asarray(boxes if boxes is not None else [], int).reshape(-1, 4)
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        # Clipped at both ends: a negative end would count from the far edge.
        top, bottom = max(0, y1 - h // 4), max(0, y2 + h // 6)
        left, right = max(0, x1 - w // 3), max(0, x2 + w // 3)
        people[top:bottom, left:right] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_TOPHAT_SIZE,) * 2)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, kernel)
    line = (tophat > _MIN_TOPHAT) & (hsv[..., 1] < _MAX_LINE_SATURATION)
    line = line.astype(np.uint8) * 255 & cv2.erode(grass, np.ones((9, 9), np.uint8))
    seen = cv2.erode(grass, np.ones((15, 15), np.uint8))
    return line & ~people, seen & ~people


def _grass_mask(hsv: np.ndarray) -> np.ndarray:
    """The pitch's grass, lines included: the largest green region of an HSV frame."""
    grass = cv2.inRange(hsv, _GRASS_LOW, _GRASS_HIGH)
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if count <= 1:
        return closed
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def refine_homography(
    homography: np.ndarray,
    model: np.ndarray,
    line: np.ndarray,
    seen: np.ndarray,
    *,
    min_points: int,
    inlier_px: float,
) -> tuple[np.ndarray, LineFit] | None:
    """Fit the frame -> pitch ``homography`` to a frame's painted lines.

    ``model`` holds the :func:`model_points` of the pitch, ``line`` and
    ``seen`` the :func:`line_masks` of the frame. Returns the refined matrix
    (normalised, frame -> pitch) and its fit, or ``None`` when no painted line
    is in view or fewer than ``min_points`` model points fall where the pitch
    is seen at some stage.
    """
    height, width = line.shape
    if not line.any():
        return None
    image_to_pitch = np.asarray(homography, float)
    centre = (width / 2, height / 2)
    controls, valid = project_points(
        _CONTROLS * (width, height), image_to_pitch, centre
    )
    if not valid.all():
        return None
    pitch_to_image = np.linalg.inv(image_to_pitch)
    front = tuple(project_points([centre], image_to_pitch, centre)[0][0])
    controls_px, _ = project_points(controls, pitch_to_image, front)
    on_grass = seen > 0

    def to_image(offsets: np.ndarray) -> np.ndarray:
        return _four_point_homography(controls, controls_px + offsets.reshape(4, 2))

    def visible(matrix: np.ndarray) -> np.ndarray:
        xy, ahead = project_points(model, matrix, front)
        inside = ahead & (xy[:, 0] >= 2) & (xy[:, 0] < width - 2)
        inside &= (xy[:, 1] >= 2) & (xy[:, 1] < height - 2)
        index = np.flatnonzero(inside)
        # Frame pixel i covers [i, i + 1).
        cells = np.floor(xy[index]).astype(int)
        return index[on_grass[cells[:, 1], cells[:, 0]]]

    def distances(offsets: np.ndarray, points: np.ndarray, field: np.ndarray):
        xy, _ = project_points(points, to_image(offsets), front)
        # Pixel centres sit at i + 0.5 in frame coordinates, at i in the array.
        return map_coordinates(
            field, [xy[:, 1] - 0.5, xy[:, 0] - 0.5], order=1, mode="nearest"
        )

    offsets = np.zeros(8)
    raw_distance = cv2.distanceTransform(255 - line, cv2.DIST_L2, 5)
    for cap, scale in _STAGES:
        field = np.minimum(raw_distance, cap).astype(np.float32)
        points = model[visible(to_image(offsets))]
        if len(points) < min_points:
            return None
        offsets = least_squares(
            distances,
            offsets,
            loss="soft_l1",
            f_scale=scale,
            x_scale=5.0,
            max_nfev=_MAX_EVALUATIONS,
            args=(points, field),
        ).x
    residuals = distances(offsets, points, field)
    refined = np.linalg.inv(to_image(offsets))
    fit = LineFit(
        points=len(points),
        inlier_share=float(np.mean(residuals < inlier_px)),
        median_px=float(np.median(residuals)),
    )
    return refined / refined[2, 2], fit


def _four_point_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The homography taking four ``src`` points to ``dst``, in float64.

    ``cv2.getPerspectiveTransform`` takes float32, which would round away the
    small steps the least squares differentiates with.
    """
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    h = np.linalg.solve(np.array(rows), dst.reshape(-1))
    return np.append(h, 1.0).reshape(3, 3)
