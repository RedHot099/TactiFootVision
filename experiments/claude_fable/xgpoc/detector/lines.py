"""Label-free homography check: do the projected pitch lines fall on the painted lines of the frame? (xg-detector 08)

The pitch markings (`tactifoot_vision.pitch.pitch_markings`: straight lines, centre circle, penalty arcs) are sampled
every `STEP` metres and projected into the frame with the inverse of a frame -> pitch homography. The painted lines are
a white-line mask: a top-hat on the HSV value (thin bright structures), low saturation, inside the grass. A projected
sample *agrees* when a mask pixel lies within `tol` pixels of it.

Per frame (`score`):
- `points`: projected samples inside the frame and on the grass (samples in the stands or beyond the horizon are not
  scored, but `off_grass` counts the in-frame ones that miss the grass);
- `agreement`: the share of `points` that agree (precision of the projected lines);
- `recall`: the share of mask pixels within `tol` pixels of a projected line (how much of the painted lines the model
  explains; below 1 even for a perfect fit, as the mask also catches kits, logos and shadows).

A frame is scorable with at least `MIN_POINTS` points spread at least `MIN_THICKNESS` pixels across their main
direction: a degenerate homography that maps the whole pitch onto one painted line would otherwise agree perfectly. No StatsBomb data, labels or detections are used: the check reads
the frame and the homography only. The sanity tests (`test_xgd_lines.py`) render the markings through a known
homography: the true matrix scores 1 and a known shift on the pitch lowers it.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from tactifoot_vision.pitch import SoccerPitch, pitch_markings
from tactifoot_vision.pitch.homography import project_points

STEP = 0.25          # metres between samples along a marking
TOL = 5.0            # pixels, at 1280 x 720: about a painted line's width at mid distance
MIN_POINTS = 100
MIN_THICKNESS = 10.0  # pixels: projected markings thinner than this across their main direction are one line


def marking_samples(pitch: SoccerPitch | None = None, step: float = STEP) -> np.ndarray:
    """`(N, 2)` points every `step` metres along every painted marking of `pitch`."""
    lines, _ = pitch_markings(pitch or SoccerPitch(), samples=256)
    points = []
    for line in lines:
        for a, b in zip(line[:-1], line[1:], strict=True):
            k = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
            points.append(a + (b - a) * (np.arange(k)[:, None] / k))
    points.append(lines[-1][-1:])
    return np.vstack(points)


SAMPLES = marking_samples()


def grass_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of the pitch: green pixels, closed over the painted lines and players, largest component only."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 50, 30), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n <= 1:
        return closed > 0
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def line_mask(image: np.ndarray, grass: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of painted-line pixels: thin bright low-saturation structures on the (eroded) grass."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    grass = grass_mask(image) if grass is None else grass
    inside = cv2.erode(grass.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return (tophat > 22) & (hsv[..., 1] < 110) & inside


def usable(homography) -> bool:
    """A homography that maps the frame onto the plane: given and not (numerically) singular."""
    if homography is None:
        return False
    condition = np.linalg.cond(np.asarray(homography, dtype=float))
    return bool(np.isfinite(condition) and condition < 1e12)


@dataclass(frozen=True)
class LineView:
    """What the score needs from one frame: the grass, the line mask and the distance (px) to the nearest line pixel."""

    grass: np.ndarray
    lines: np.ndarray
    distance: np.ndarray

    @classmethod
    def of(cls, image: np.ndarray) -> "LineView":
        grass = grass_mask(image)
        lines = line_mask(image, grass)
        distance = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5)
        return cls(grass, lines, distance)


def projected_samples(homography: np.ndarray, shape: tuple[int, int], samples: np.ndarray = SAMPLES) -> np.ndarray:
    """Pixel positions of the marking samples inside a frame of `shape` (h, w), for a frame -> pitch `homography`."""
    h, w = shape
    to_frame = np.linalg.inv(np.asarray(homography, dtype=float))
    # A pitch point is on the visible side of the horizon when it maps to the same side as the frame centre's pitch point.
    centre = project_points(np.array([[w / 2, h / 2]]), homography, (w / 2, h / 2))[0][0]
    xy, valid = project_points(samples, to_frame, tuple(centre))
    inside = valid & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    return xy[inside]


def score(view: LineView, homography: np.ndarray | None, tol: float = TOL) -> dict[str, float]:
    """`points`, `off_grass`, `agreement` and `recall` of one frame's homography (NaN scores without a homography or
    with fewer than `MIN_POINTS` points)."""
    nan = {"points": 0.0, "off_grass": np.nan, "agreement": np.nan, "recall": np.nan}
    if not usable(homography):
        return nan                                    # none, or singular (e.g. collinear keypoints): no frame image
    shape = view.lines.shape
    xy = projected_samples(homography, shape)
    if len(xy) == 0:
        return nan
    col, row = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
    on_grass = view.grass[row, col]
    points = int(on_grass.sum())
    off_grass = float(1 - on_grass.mean())
    if points < MIN_POINTS or thickness(xy[on_grass]) < MIN_THICKNESS:
        return nan | {"points": float(points), "off_grass": off_grass}
    agreement = float((view.distance[row[on_grass], col[on_grass]] <= tol).mean())
    drawn = np.zeros(shape, np.uint8)
    drawn[row, col] = 1
    near_model = cv2.distanceTransform(1 - drawn, cv2.DIST_L2, 5) <= tol
    recall = float(near_model[view.lines].mean()) if view.lines.any() else np.nan
    return {"points": float(points), "off_grass": off_grass, "agreement": agreement, "recall": recall}


def thickness(xy: np.ndarray) -> float:
    """Standard deviation (pixels) of `(N, 2)` points across their main direction; 0 for fewer than 3."""
    if len(xy) < 3:
        return 0.0
    return float(np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)[1] / np.sqrt(len(xy)))


def render(homography: np.ndarray, shape: tuple[int, int] = (720, 1280), width_px: int = 3) -> np.ndarray:
    """A synthetic BGR frame: grass with the pitch markings painted through a frame -> pitch `homography`."""
    h, w = shape
    image = np.zeros((h, w, 3), np.uint8)
    image[:] = (40, 130, 50)                                    # grass green (BGR)
    xy = projected_samples(homography, shape, marking_samples(step=0.05))
    for x, y in np.round(xy).astype(int):
        cv2.circle(image, (int(x), int(y)), width_px // 2, (235, 235, 235), -1)
    return image


def draw(image: np.ndarray, homography: np.ndarray | None, color=(0, 0, 255)) -> np.ndarray:
    """`image` with the projected marking samples drawn on it (for overlays)."""
    out = image.copy()
    if usable(homography):
        for x, y in np.round(projected_samples(homography, image.shape[:2])).astype(int):
            cv2.circle(out, (int(x), int(y)), 1, color, -1)
    return out


# ------------------------------------------------------------- checks that use no lines
PLAYER_HEIGHT_M = 1.8
PITCH_MARGIN_M = 3.0


def players_on_pitch(homography: np.ndarray | None, boxes: np.ndarray, pitch: SoccerPitch | None = None) -> float:
    """Share of player boxes whose feet map inside the pitch (with a `PITCH_MARGIN_M` margin); NaN without a homography
    or players. Independent of the line mask: a homography metres off puts players outside the pitch."""
    pitch = pitch or SoccerPitch()
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not usable(homography) or len(boxes) == 0:
        return np.nan
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    xy = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), np.asarray(homography, dtype=float)).reshape(-1, 2)
    inside = ((xy[:, 0] > -PITCH_MARGIN_M) & (xy[:, 0] < pitch.length + PITCH_MARGIN_M)
              & (xy[:, 1] > -PITCH_MARGIN_M) & (xy[:, 1] < pitch.width + PITCH_MARGIN_M))
    return float(inside.mean())


def height_ratio(homography: np.ndarray | None, boxes: np.ndarray) -> float:
    """Median over player boxes of box height / (1.8 m x the homography's horizontal pixels per metre at the feet).

    Independent of the line mask: a homography of the wrong scale makes players too tall or too short for the pitch
    around them. About 1 on a broadcast view (above 1 by the box margin and the camera tilt), the same bias for every
    homography of the frame; NaN without a homography or players."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not usable(homography) or len(boxes) == 0:
        return np.nan
    homography = np.asarray(homography, dtype=float)
    to_frame = np.linalg.inv(homography)
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    on_pitch = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), homography).reshape(-1, 2)
    ratios = []
    for (x, y), box in zip(on_pitch, boxes, strict=True):
        # One metre along each pitch axis around the feet, in frame pixels; the larger horizontal extent is the scale.
        ends = np.array([[x - 0.5, y], [x + 0.5, y], [x, y - 0.5], [x, y + 0.5]])
        px = cv2.perspectiveTransform(ends.reshape(-1, 1, 2), to_frame).reshape(-1, 2)
        scale = max(abs(px[1, 0] - px[0, 0]), abs(px[3, 0] - px[2, 0]))
        if scale > 0:
            ratios.append((box[3] - box[1]) / (PLAYER_HEIGHT_M * scale))
    return float(np.median(ratios)) if ratios else np.nan


def mirrored(homography: np.ndarray | None, shape: tuple[int, int]) -> float:
    """1.0 when the frame -> pitch homography is a mirror image at the bottom centre of the frame, else 0.0 (NaN
    without a homography). With the pitch's far touchline at y = 0, a camera above the pitch never sees a mirror image
    (image and pitch both have y growing towards the camera); keypoints of one end labelled as the other end's give one.
    The sign of the Jacobian determinant of a homography at a point is that of det(H) times the point's w."""
    if not usable(homography):
        return np.nan
    h, w = shape
    homography = np.asarray(homography, dtype=float)
    depth = (homography @ np.array([w / 2, h - 1, 1.0]))[2]
    return float(np.linalg.det(homography) * depth < 0)


def keypoint_residuals(keypoints, homography: np.ndarray, pitch: SoccerPitch | None = None,
                       min_confidence: float = 0.5) -> np.ndarray:
    """`(vertex, pixels)` rows: distance of each confident keypoint from its landmark projected by `homography`
    (frame -> pitch) that lies in front of the camera; empty without keypoints."""
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 2))
    vertices = (pitch or SoccerPitch()).vertices.astype(float)
    to_frame = np.linalg.inv(np.asarray(homography, dtype=float))
    xy, front = project_points(vertices, to_frame, tuple(vertices.mean(axis=0)))
    confident = (keypoints.confidence[0] >= min_confidence) & front
    index = np.flatnonzero(confident)
    distance = np.linalg.norm(keypoints.xy[0][index] - xy[index], axis=1)
    return np.column_stack([index, distance])
