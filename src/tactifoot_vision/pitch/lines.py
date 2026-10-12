"""Refining a frame -> pitch homography on the painted lines of the frame.

A keypoint homography can be metres off: pitch-landmark models place their
keypoints tens of pixels from the landmarks on footage unlike their training
data. The painted lines are visible in most frames, so the fit is refined by
pulling the projected pitch markings onto them (a chamfer fit):

* a white-line mask: thin bright structures (a top-hat on the HSV value) of
  low saturation, on the grass, with the people's boxes removed;
* its distance transform, capped, gives every pixel its distance to the
  nearest painted line;
* samples along every marking of the pitch are projected into the frame, and
  the eight pixel offsets of four control points (which define the
  homography) move them onto the lines by robust least squares, coarse to
  fine.

Only samples inside the frame and on the grass count. With fewer than
``min_points`` of them the start is kept. So it is when the result puts fewer
than ``min_inliers`` of the samples on a line, or no more than the start: a
fit that ends between lines has snapped partly onto the wrong ones (on
SoccerNet frames such fits had a worse scale than the keypoint fit they
started from). So it is, too, when the result is no camera's view of the
grass: a mirror image (far and near touchline swapped), which fits the
symmetric markings perfectly, or a view whose horizon cuts the grass, where
the fit shrank the markings into a corner of the frame. :class:`LineFit` reports the share of samples within
``inlier_px`` of a painted line (``inliers``), before and after.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from tactifoot_vision.pitch.homography import project_points
from tactifoot_vision.pitch.pitch import SoccerPitch, pitch_markings

# Pixel sizes are for a 720-pixel-high frame and scale with the frame height.
_REFERENCE_HEIGHT = 720
_FINEST_PX = 5.0  # distance cap of the last stage
_STAGE_RATIO = 2.5  # caps shrink by about this much per stage
# Frame positions (fractions of width and height) of the four control points.
_CONTROL = ((0.15, 0.35), (0.85, 0.35), (0.85, 0.9), (0.15, 0.9))


@dataclass(frozen=True)
class LineFit:
    """Outcome of :func:`refine_homography`.

    ``matrix`` is the refined frame -> pitch homography, or the start when the
    refinement was not used (``refined`` False). ``inliers`` is the share of
    the visible marking samples within ``inlier_px`` of a painted line under
    ``matrix``, ``start_inliers`` the same for the start (NaN when fewer than
    ``min_points`` samples are visible); ``points`` counts those samples.
    """

    matrix: np.ndarray
    refined: bool
    inliers: float
    start_inliers: float
    points: int


def marking_samples(pitch: SoccerPitch, step: float) -> np.ndarray:
    """``(N, 2)`` points every ``step`` pitch units along every painted marking."""
    lines, _ = pitch_markings(pitch, samples=256)
    points = []
    for line in lines:
        for a, b in zip(line[:-1], line[1:], strict=True):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
            points.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.vstack(points)


def grass_mask(frame: np.ndarray) -> np.ndarray:
    """``uint8`` mask (0/255) of the pitch: green pixels closed over lines and players, largest region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if count <= 1:
        return closed
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def line_mask(
    frame: np.ndarray, boxes: np.ndarray | None = None, grass: np.ndarray | None = None
) -> np.ndarray:
    """``uint8`` mask (0/255) of painted-line pixels, without the area around ``boxes`` (people)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, kernel)
    mask = ((tophat > 22) & (hsv[..., 1] < 110)).astype(np.uint8) * 255
    grass = grass_mask(frame) if grass is None else grass
    mask &= cv2.erode(grass, np.ones((9, 9), np.uint8))
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
            w, h = x2 - x1, y2 - y1
            top, bottom = int(max(0, y1 - h / 4)), int(max(0, y2 + h / 6))
            left, right = int(max(0, x1 - w / 3)), int(max(0, x2 + w / 3))
            mask[top:bottom, left:right] = 0
    return mask


def refine_homography(
    frame: np.ndarray,
    homography: np.ndarray,
    pitch: SoccerPitch | None = None,
    boxes: np.ndarray | None = None,
    *,
    min_points: int = 80,
    min_inliers: float = 0.5,
    inlier_px: float = 3.0,
    search_px: float = 120.0,
    step: float = 0.4,
) -> LineFit:
    """Refine a frame -> pitch ``homography`` on the painted lines of ``frame`` (BGR).

    Args:
        pitch: pitch model (units of ``homography``); defaults to ``SoccerPitch()``.
        boxes: ``(N, 4)`` people boxes in frame pixels, removed from the line mask
            (white kits are not lines).
        min_points: fewest visible marking samples to refine on.
        min_inliers: least share of samples on a line for the refined matrix
            to replace the start.
        inlier_px: distance to a painted line (pixels at 720 p, scaled with the
            frame height) under which a sample counts as on the line.
        search_px: how far (pixels at 720 p) the projected markings may be from
            their lines. The fit runs in stages from this distance cap down to
            5 px, each with a robust loss scaled to its cap; a wider search
            corrects starts that are further off but can snap onto a wrong line.
        step: spacing of the marking samples in pitch units.
    """
    if not 0 <= min_inliers <= 1:
        raise ValueError(f"min_inliers must be in [0, 1], got {min_inliers}")
    for name, value in (
        ("inlier_px", inlier_px),
        ("search_px", search_px),
        ("step", step),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    pitch = pitch or SoccerPitch()
    height, width = frame.shape[:2]
    scale = height / _REFERENCE_HEIGHT
    grass = grass_mask(frame)
    lines = line_mask(frame, boxes, grass)
    field = cv2.erode(grass, np.ones((15, 15), np.uint8)) > 0
    samples = marking_samples(pitch, step)
    start = np.asarray(homography, dtype=float)
    start = start / start[2, 2]

    distance = cv2.distanceTransform(255 - lines, cv2.DIST_L2, 5)

    def visible(to_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(indices, frame pixels)`` of the samples on the visible grass under a pitch -> frame matrix."""
        none = np.zeros(0, int), np.zeros((0, 2))
        to_pitch = _invert(to_frame)
        if to_pitch is None:
            return none
        centre = to_pitch @ np.array([width / 2, height / 2, 1.0])
        if abs(centre[2]) < 1e-12:
            return none
        xy, front = project_points(samples, to_frame, tuple(centre[:2] / centre[2]))
        inside = (
            front
            & (xy[:, 0] > 2)
            & (xy[:, 0] < width - 3)
            & (xy[:, 1] > 2)
            & (xy[:, 1] < height - 3)
        )
        index = np.flatnonzero(inside)
        on_grass = field[xy[index, 1].astype(int), xy[index, 0].astype(int)]
        return index[on_grass], xy[index[on_grass]]

    def inliers(to_frame: np.ndarray) -> tuple[float, int]:
        _, xy = visible(to_frame)
        if len(xy) < min_points:
            return np.nan, len(xy)
        near = _sample(distance, xy) <= inlier_px * scale
        return float(near.mean()), len(xy)

    to_frame = _invert(start)
    if to_frame is None:
        return LineFit(start, False, np.nan, np.nan, 0)
    start_inliers, points = inliers(to_frame)
    unchanged = LineFit(start, False, start_inliers, start_inliers, points)
    if points < min_points:
        return unchanged

    # The homography as four control points: frame positions of four pitch
    # points spread over the view, moved by eight pixel offsets.
    control_px = np.array(_CONTROL) * np.array([width, height])
    control = _project(control_px, start)

    def matrix(offsets: np.ndarray) -> np.ndarray | None:
        moved, _ = cv2.findHomography(control, control_px + offsets.reshape(4, 2))
        return moved

    stages = 1 + int(np.ceil(np.log(search_px / _FINEST_PX) / np.log(_STAGE_RATIO)))
    offsets = np.zeros(8)
    for cap in np.geomspace(max(search_px, _FINEST_PX), _FINEST_PX, max(stages, 1)):
        loss_scale = cap / 5  # residuals well inside the cap count fully
        capped = np.minimum(distance, cap * scale).astype(np.float32)
        current = matrix(offsets)
        if current is None:
            return unchanged
        index, _ = visible(current)
        if len(index) < min_points:
            return unchanged

        def residuals(d: np.ndarray, points=samples[index], dist=capped) -> np.ndarray:
            moved = matrix(d)
            if moved is None:
                return np.full(len(points), float(dist.max()))
            return _sample(dist, _project(points, moved))

        offsets = _least_squares(residuals, offsets, loss_scale * scale)

    refined_to_frame = matrix(offsets)
    refined = None if refined_to_frame is None else _invert(refined_to_frame)
    if refined is None:
        return unchanged
    refined_inliers, refined_points = inliers(refined_to_frame)
    if not (refined_inliers >= min_inliers and refined_inliers > start_inliers):
        return unchanged
    if not _faces_camera(refined, grass):
        return unchanged
    return LineFit(refined, True, refined_inliers, start_inliers, refined_points)


def _faces_camera(to_pitch: np.ndarray, grass: np.ndarray, every: int = 16) -> bool:
    """Whether a frame -> pitch homography is a camera's view of all the grass.

    With the far touchline at y = 0 (the keypoint dataset's landmark order,
    which :class:`SoccerPitch` follows) a camera above the pitch sees x and y
    as the frame does, so the homography keeps the orientation at every grass
    pixel. It reverses it everywhere for a mirror image, which fits the
    symmetric markings perfectly, and beyond its horizon, which a fit that
    shrinks the markings into a corner of the view puts inside the grass. The
    Jacobian determinant of a homography at a point has the sign of
    ``det(H) * w``, ``w`` the point's homogeneous scale; it is checked on a
    grid of grass pixels, ``every`` pixels apart.
    """
    rows, cols = np.nonzero(grass[::every, ::every])
    if len(rows) == 0:
        return True
    points = np.column_stack([cols * every, rows * every, np.ones(len(rows))])
    w = points @ to_pitch[2]
    return bool((np.linalg.det(to_pitch) * w > 0).all())


def _least_squares(residuals, start: np.ndarray, loss_scale: float) -> np.ndarray:
    # SciPy is not a direct dependency yet; scikit-learn requires it.
    from scipy.optimize import least_squares

    fit = least_squares(
        residuals, start, loss="soft_l1", f_scale=loss_scale, x_scale=5.0, max_nfev=200
    )
    return fit.x


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``image`` at ``(N, 2)`` pixel positions (clamped to the image)."""
    from scipy.ndimage import map_coordinates

    return map_coordinates(image, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def _invert(matrix: np.ndarray) -> np.ndarray | None:
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all() or abs(inverse[2, 2]) < 1e-12:
        return None
    return inverse / inverse[2, 2]
