"""Refine a frame -> pitch homography by fitting the pitch markings to the painted lines of the frame.

The pitch model's markings are sampled densely, projected into the frame and
pulled onto the nearest white-line pixel (a chamfer fit on the distance
transform of a line mask). The unknowns are the pixel offsets of four control
points spread over the frame, which keeps the problem well conditioned; a
robust least squares fits them coarse to fine. The image thresholds, in
:class:`LineRefinerConfig`, are tuned on 1080p broadcast footage.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from tactifoot_vision.pitch.pitch import SoccerPitch, pitch_markings
from tactifoot_vision.pitch.projection import project_points

# Control points as fractions of the frame (width, height), below the stands.
_CONTROLS = np.array([[0.15, 0.35], [0.85, 0.35], [0.85, 0.9], [0.15, 0.9]])


@dataclass(frozen=True)
class LineRefinerConfig:
    """Image thresholds of the line refinement, in pixels unless stated; defaults tuned on 1080p.

    Attributes:
        grass_hsv_low, grass_hsv_high: grass in OpenCV HSV (hue 0-180):
            green, saturated, not dark.
        grass_open_px, grass_close_px: square kernels that drop specks from
            the grass mask, then close it over lines and players; the largest
            region left is the pitch.
        tophat_px: lines narrower than this stand out of the grass (top-hat
            kernel on the HSV value).
        min_tophat: grey levels a line pixel is brighter than the grass around.
        max_line_saturation: lines are white, kits often are not.
        line_erode_px: line pixels are kept this far inside the grass.
        seen_erode_px: markings are looked for this far inside the grass.
        box_margin_side, box_margin_top, box_margin_bottom: the margin cut
            around each person box, as fractions of its width (side) and
            height (top, bottom), for the limbs and shadow the box misses.
        stages: ``(distance cap, robust loss scale)`` per least-squares stage,
            coarse to fine: the first pulls lines up to its cap off, the last
            ignores what is beyond its cap, which bounds ``line_inlier_px``.
        max_evaluations: residual evaluations per stage.
        good_inlier_share: a fit from the first start at least this good
            needs no second start.
        model_step: pitch units between model points along the markings.
    """

    grass_hsv_low: tuple[int, int, int] = (30, 60, 40)
    grass_hsv_high: tuple[int, int, int] = (90, 255, 255)
    grass_open_px: int = 5
    grass_close_px: int = 31
    tophat_px: int = 15
    min_tophat: float = 22
    max_line_saturation: float = 110
    line_erode_px: int = 9
    seen_erode_px: int = 15
    box_margin_side: float = 1 / 3
    box_margin_top: float = 1 / 4
    box_margin_bottom: float = 1 / 6
    stages: tuple[tuple[float, float], ...] = ((40.0, 8.0), (12.0, 2.5), (6.0, 1.2))
    max_evaluations: int = 200
    good_inlier_share: float = 0.85
    model_step: float = 0.4

    def __post_init__(self) -> None:
        # Run files give lists; keep the config hashable and immutable.
        for name in ("grass_hsv_low", "grass_hsv_high"):
            value = tuple(getattr(self, name))
            if len(value) != 3:
                raise ValueError(f"{name} must be 3 HSV values, got {value}")
            object.__setattr__(self, name, value)
        stages = tuple(tuple(float(v) for v in stage) for stage in self.stages)
        if not stages or any(len(s) != 2 or min(s) <= 0 for s in stages):
            raise ValueError(
                f"stages must be (cap, scale) pairs above 0, got {self.stages}"
            )
        object.__setattr__(self, "stages", stages)
        for name in (
            "grass_open_px", "grass_close_px", "tophat_px", "line_erode_px",
            "seen_erode_px", "max_evaluations",
        ):  # fmt: skip
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("box_margin_side", "box_margin_top", "box_margin_bottom"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.min_tophat < 0 or self.max_line_saturation <= 0:
            raise ValueError("min_tophat must be >= 0 and max_line_saturation > 0")
        if not 0 < self.good_inlier_share <= 1:
            raise ValueError(
                f"good_inlier_share must be in (0, 1], got {self.good_inlier_share}"
            )
        if self.model_step <= 0:
            raise ValueError(f"model_step must be > 0, got {self.model_step}")

    @property
    def distance_cap_px(self) -> float:
        """The distance the final stage measures up to: the fit's distances are capped there."""
        return self.stages[-1][0]


@dataclass(frozen=True)
class LineFit:
    """How well a refined homography lays the pitch markings on its frame's painted lines.

    ``points`` model points along the markings were visible on the grass;
    ``inlier_share`` of them lie within the estimator's ``line_inlier_px`` of
    a painted-line pixel, and ``median_px`` is their median distance to one
    (capped at the final stage's cap; a point the fit moved out of view
    counts as further off). A low share means the fit found few lines to hold on to.
    """

    points: int
    inlier_share: float
    median_px: float


class LineRefiner:
    """Fits frame -> pitch homographies to the painted lines of their frames.

    Holds the model points of ``pitch`` and the settings of
    :class:`~tactifoot_vision.pitch.HomographyEstimator`'s ``refine_with_lines``,
    which validates them.
    """

    def __init__(
        self,
        pitch: SoccerPitch,
        config: LineRefinerConfig,
        *,
        min_points: int,
        inlier_px: float,
        min_inlier_share: float,
    ) -> None:
        self.config = config
        self.model = model_points(pitch, config.model_step)
        self.min_points = min_points
        self.inlier_px = inlier_px
        self.min_inlier_share = min_inlier_share

    def refine(
        self,
        frame: np.ndarray,
        starts: Sequence[np.ndarray | None],
        boxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, LineFit] | None:
        """The best fit among those from each of ``starts``: most inliers, then most points.

        ``None`` starts are skipped, and so are the rest once a fit reaches
        ``good_inlier_share``. Returns ``None`` when no start gives a fit with
        at least ``min_inlier_share`` inliers (see :func:`refine_homography`
        for the other ways a start gives none).
        """
        line, seen = line_masks(frame, boxes, self.config)
        best = None
        for start in starts:
            if start is None:
                continue
            refined = refine_homography(
                start,
                self.model,
                line,
                seen,
                self.config,
                min_points=self.min_points,
                inlier_px=self.inlier_px,
            )
            if refined is None or refined[1].inlier_share < self.min_inlier_share:
                continue
            if best is None or _rank(refined[1]) > _rank(best[1]):
                best = refined
            if best[1].inlier_share >= self.config.good_inlier_share:
                break
        return best


def _rank(fit: LineFit) -> tuple[float, int]:
    return fit.inlier_share, fit.points


def model_points(pitch: SoccerPitch, step: float) -> np.ndarray:
    """Points every ``step`` pitch units along the painted lines, circle and arcs of ``pitch``."""
    lines, _ = pitch_markings(pitch, samples=256)
    points = []
    for line in lines:
        segments = np.diff(line, axis=0)
        for start, segment in zip(line[:-1], segments, strict=True):
            n = max(1, int(np.ceil(np.hypot(*segment) / step)))
            points.append(start + np.arange(n)[:, None] / n * segment)
    return np.vstack(points)


def line_masks(
    frame: np.ndarray,
    boxes: np.ndarray | None = None,
    config: LineRefinerConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The painted lines of a BGR frame and where the pitch can be seen, as uint8 0/255 masks.

    Returns ``(line, seen)``: ``line`` holds thin, bright, unsaturated pixels on
    the grass, ``seen`` the grass away from its edges, where a marking would
    show if it were there. Both leave out the people ``boxes`` (xyxy, frame
    pixels) with a margin for limbs and shadow, so white kits are no lines and
    the markings they hide count for nothing.
    """
    config = config or LineRefinerConfig()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = _grass_mask(hsv, config)
    people = np.zeros(grass.shape, np.uint8)
    boxes = np.asarray(boxes if boxes is not None else [], float).reshape(-1, 4)
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        # Clipped at both ends: a negative end would count from the far edge.
        top = max(0, int(y1 - h * config.box_margin_top))
        bottom = max(0, int(y2 + h * config.box_margin_bottom))
        left = max(0, int(x1 - w * config.box_margin_side))
        right = max(0, int(x2 + w * config.box_margin_side))
        people[top:bottom, left:right] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.tophat_px,) * 2)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, kernel)
    line = (tophat > config.min_tophat) & (hsv[..., 1] < config.max_line_saturation)
    line = line.astype(np.uint8) * 255 & _erode(grass, config.line_erode_px)
    seen = _erode(grass, config.seen_erode_px)
    return line & ~people, seen & ~people


def _erode(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.erode(mask, np.ones((size, size), np.uint8))


def _grass_mask(hsv: np.ndarray, config: LineRefinerConfig) -> np.ndarray:
    """The pitch's grass, lines included: the largest green region of an HSV frame."""
    grass = cv2.inRange(hsv, config.grass_hsv_low, config.grass_hsv_high)
    open_kernel = np.ones((config.grass_open_px,) * 2, np.uint8)
    close_kernel = np.ones((config.grass_close_px,) * 2, np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, close_kernel)
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
    config: LineRefinerConfig,
    *,
    min_points: int,
    inlier_px: float,
) -> tuple[np.ndarray, LineFit] | None:
    """Fit the frame -> pitch ``homography`` to a frame's painted lines.

    ``model`` holds the :func:`model_points` of the pitch, ``line`` and
    ``seen`` the :func:`line_masks` of the frame. Returns the refined matrix
    (normalised, frame -> pitch) and its fit, or ``None`` when:

    * no painted line is in view;
    * the start puts part of the control points (from 35 % of the frame
      height down) beyond the horizon, as on a view looking far up the pitch;
    * fewer than ``min_points`` model points fall where the pitch is seen at
      some stage;
    * the fit degenerates: it folds the control quadrilateral over or puts a
      control point beyond the horizon.

    Model points the fit moves beyond the horizon count as the stage's
    distance cap, and those it moves out of the frame as the distance at the
    frame's edge plus how far out they are, so the fit cannot gain by pushing
    the markings out of view.
    """
    height, width = line.shape
    if not line.any():
        return None
    image_to_pitch = np.asarray(homography, float)
    centre = (width / 2, height / 2)
    frame_controls = _CONTROLS * (width, height)
    controls, valid = project_points(frame_controls, image_to_pitch, centre)
    if not valid.all():
        return None
    pitch_to_image = np.linalg.inv(image_to_pitch)
    front = tuple(project_points([centre], image_to_pitch, centre)[0][0])
    controls_px, _ = project_points(controls, pitch_to_image, front)
    on_grass = seen > 0

    def to_image(offsets: np.ndarray) -> np.ndarray | None:
        try:
            return _four_point_homography(controls, controls_px + offsets.reshape(4, 2))
        except np.linalg.LinAlgError:
            return None

    def in_frame(xy: np.ndarray, margin: float) -> np.ndarray:
        return (
            (xy[:, 0] >= margin)
            & (xy[:, 0] < width - margin)
            & (xy[:, 1] >= margin)
            & (xy[:, 1] < height - margin)
        )

    def visible(matrix: np.ndarray) -> np.ndarray:
        xy, ahead = project_points(model, matrix, front)
        index = np.flatnonzero(ahead & in_frame(xy, 2))
        # Frame pixel i covers [i, i + 1).
        cells = np.floor(xy[index]).astype(int)
        return index[on_grass[cells[:, 1], cells[:, 0]]]

    def distances(
        offsets: np.ndarray, points: np.ndarray, field: np.ndarray, cap: float
    ) -> np.ndarray:
        # float32 like the distance field: float64 residuals change where the
        # finite-difference steps land and with them the fit.
        result = np.full(len(points), cap, np.float32)
        matrix = to_image(offsets)
        if matrix is None:
            return result
        xy, ahead = project_points(points, matrix, front)
        xy = xy[ahead]
        # Pixel centres sit at i + 0.5 in frame coordinates, at i in the array.
        # A point off the frame scores the edge's distance plus how far off it
        # is, which keeps the cost continuous as the fit pushes points out.
        inside = np.clip(xy, 0, [width - 1e-6, height - 1e-6])
        off = np.hypot(*(xy - inside).T)
        sampled = map_coordinates(
            field, [inside[:, 1] - 0.5, inside[:, 0] - 0.5], order=1, mode="nearest"
        )
        result[ahead] = np.minimum(sampled + off, cap)
        return result

    offsets = np.zeros(8)
    raw_distance = cv2.distanceTransform(255 - line, cv2.DIST_L2, 5)
    for cap, scale in config.stages:
        field = np.minimum(raw_distance, cap).astype(np.float32)
        matrix = to_image(offsets)
        if matrix is None:
            return None
        points = model[visible(matrix)]
        if len(points) < min_points:
            return None
        offsets = least_squares(
            distances,
            offsets,
            loss="soft_l1",
            f_scale=scale,
            x_scale=5.0,
            max_nfev=config.max_evaluations,
            args=(points, field, cap),
        ).x
    matrix = to_image(offsets)
    if matrix is None or _folded(controls_px, controls_px + offsets.reshape(4, 2)):
        return None
    refined = np.linalg.inv(matrix)
    if not project_points(frame_controls, refined, centre)[1].all():
        return None
    residuals = distances(offsets, points, field, config.stages[-1][0])
    fit = LineFit(
        points=len(points),
        inlier_share=float(np.mean(residuals < inlier_px)),
        median_px=float(np.median(residuals)),
    )
    return refined / refined[2, 2], fit


def _folded(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether the quadrilateral ``after`` no longer turns the way ``before`` does at every corner."""

    def turns(quad: np.ndarray) -> np.ndarray:
        edges = np.roll(quad, -1, axis=0) - quad
        following = np.roll(edges, -1, axis=0)
        return np.sign(edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0])

    return not np.array_equal(turns(before), turns(after))


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
