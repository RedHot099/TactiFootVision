"""Frame -> pitch homography fitted to SoccerNet pitch-line annotations, and the 32-vertex keypoint labels.

Units. Annotations come as fractions of the image; everything here works in 720p pixels (1280 x 720), so residuals and
uncertainties from 1920 x 1080 GSR frames and 960 x 540 Calibration images are comparable. Pitch units are the
package's `SoccerPitch()` metres: corner origin, x along the length, y = 0 at the far touchline ("Side line top").

The fit (`fit_lines`):
1. Straight annotated lines are point-on-line constraints (each annotated point lies on its pitch line). Lines through
   circles add linear constraints too: the image line meets the image conic fitted to the circle's points where the
   pitch line meets the circle, and the conic's tangents there are the circle's tangents (both orderings of each
   crossing pair are tried). Goal posts and crossbars are not on the ground plane and are skipped.
2. A normalised linear system over the 9 homography entries; the true matrix lies in the span of its 3 smallest
   singular vectors whenever the constraints leave at most 2 degrees of freedom (e.g. middle line, touchline and centre
   circle). That span is searched on a grid for the lowest residual among camera views (`views_pitch`), then refined.
3. Levenberg-Marquardt on pixel residuals: point-to-line distance for straight lines, Sampson distance for circles.
4. The labels' uncertainty: the residual covariance propagated to every in-frame vertex (`vertex_sd`); an
   underdetermined fit (e.g. a circle and one line) has an unbounded vertex uncertainty and is rejected by the rule.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from tactifoot_vision.pitch import SoccerPitch

PITCH = SoccerPitch()
VERTICES = PITCH.vertices.astype(float)
FRAME_W, FRAME_H = 1280, 720
BOTTOM_CENTRE = (FRAME_W / 2, FRAME_H - 1.0)   # on the ground in a broadcast view
RADIUS = float(PITCH.circle_radii[0])
_SPOT, _HW = PITCH.penalty_spot_distance, PITCH.width / 2

# SoccerNet line name -> the two package vertices it runs between (y = 0 is "top", x = 0 is "left").
STRAIGHT: dict[str, tuple[int, int]] = {
    "Side line top": (0, 24), "Side line bottom": (5, 29), "Side line left": (0, 5), "Side line right": (24, 29),
    "Middle line": (13, 16),
    "Big rect. left top": (1, 9), "Big rect. left main": (9, 12), "Big rect. left bottom": (4, 12),
    "Big rect. right top": (25, 17), "Big rect. right main": (17, 20), "Big rect. right bottom": (28, 20),
    "Small rect. left top": (2, 6), "Small rect. left main": (6, 7), "Small rect. left bottom": (3, 7),
    "Small rect. right top": (26, 22), "Small rect. right main": (22, 23), "Small rect. right bottom": (27, 23),
}  # fmt: skip
# Circle name -> centre (radius RADIUS); "Circle left/right" are the penalty arcs outside the box.
CIRCLES: dict[str, tuple[float, float]] = {
    "Circle central": (PITCH.length / 2, _HW), "Circle left": (_SPOT, _HW), "Circle right": (PITCH.length - _SPOT, _HW),
}
# Straight lines that cross a circle: (line, circle) -> the two crossing points' y on the pitch (far first).
CROSSINGS: dict[tuple[str, str], tuple[float, float]] = {}
for _line, _circle in (("Middle line", "Circle central"), ("Big rect. left main", "Circle left"),
                       ("Big rect. right main", "Circle right")):
    _x = VERTICES[STRAIGHT[_line][0], 0]
    _dy = (RADIUS**2 - (_x - CIRCLES[_circle][0]) ** 2) ** 0.5
    CROSSINGS[(_line, _circle)] = (_HW - _dy, _HW + _dy)
SKIPPED = ("Goal left crossbar", "Goal left post left", "Goal left post right",
           "Goal right crossbar", "Goal right post left", "Goal right post right")
# Mirror image of the line names under a horizontal image flip (x -> L - x on the pitch).
FLIP_NAMES = {n: n.replace("left", "@").replace("right", "left").replace("@", "right") for n in (*STRAIGHT, *CIRCLES)}
FLIP_NAMES.update({n: n.replace("left", "@").replace("right", "left").replace("@", "right") for n in SKIPPED})

SD_FLOOR_PX = 1.0           # annotation noise floor for the vertex uncertainty (720p px)
_GRID = 4000                # directions searched in the 3-dim span


def pitch_line(name: str) -> np.ndarray:
    """Homogeneous pitch line (a, b, c), a^2 + b^2 = 1, of a straight SoccerNet line."""
    a, b = STRAIGHT[name]
    line = np.cross([*VERTICES[a], 1.0], [*VERTICES[b], 1.0])
    return line / np.hypot(line[0], line[1])


def circle_conic(centre: tuple[float, float], radius: float = RADIUS) -> np.ndarray:
    """Symmetric 3x3 matrix C with q^T C q = 0 on the circle."""
    cx, cy = centre
    return np.array([[1.0, 0, -cx], [0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy - radius * radius]])


def fractions(points) -> np.ndarray:
    """One annotated line as an (N, 2) array of image fractions, from SoccerNet's [{"x", "y"}, ...] or an array."""
    if isinstance(points, np.ndarray):
        return points.astype(float).reshape(-1, 2)
    return np.array([[p["x"], p["y"]] for p in points], dtype=float).reshape(-1, 2)


def annotation_points(lines: dict, size: tuple[int, int] = (FRAME_W, FRAME_H)) -> dict[str, np.ndarray]:
    """SoccerNet `{name: [{"x": fx, "y": fy}, ...]}` or `{name: (N, 2) array}` (fractions of the image) -> {name: (N, 2)
    720p pixels}.

    Only the ground-plane lines this module models, with at least two points; duplicates are dropped.
    """
    out = {}
    for name, points in lines.items():
        if name not in STRAIGHT and name not in CIRCLES:
            continue
        xy = np.unique(fractions(points), axis=0) * size
        if len(xy) >= 2:
            out[name] = xy
    return out


# --------------------------------------------------------------------------- residuals
def _residuals(H: np.ndarray, pts: dict[str, np.ndarray]) -> np.ndarray:
    """Signed pixel residuals of every annotated point under the frame -> pitch homography(ies) `H` ((3, 3) or (M, 3, 3)).

    Straight lines: signed distance to the pitch line's image. Circles: signed Sampson distance to the circle's image
    conic. Signed, so that least squares sees the direction of each residual. Returns (n,) or (M, n).
    """
    H = np.asarray(H, dtype=float)
    single = H.ndim == 2
    H = H[None] if single else H
    out = []
    for name, xy in pts.items():
        p = np.column_stack([xy, np.ones(len(xy))])
        if name in STRAIGHT:
            img = np.einsum("mji,j->mi", H, pitch_line(name))           # H^T l
            num = img @ p.T
            out.append(num / np.maximum(np.hypot(img[:, 0], img[:, 1]), 1e-300)[:, None])
        else:
            C = np.einsum("mki,kl,mlj->mij", H, circle_conic(CIRCLES[name]), H)   # H^T C H
            Cp = np.einsum("mij,nj->mni", C, p)
            f = np.einsum("mni,ni->mn", Cp, p)
            grad = 2 * np.hypot(Cp[..., 0], Cp[..., 1])
            out.append(f / np.maximum(grad, 1e-300))
    r = np.concatenate(out, axis=1)
    return r[0] if single else r


def views_pitch(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Whether each frame -> pitch homography in `H` ((M, 3, 3)) is a camera's view at all points `xy` ((N, 2)).

    All points must lie on one side of the horizon and the map must keep the orientation (Jacobian determinant > 0,
    i.e. det(H) * w > 0): with y = 0 at the far touchline a camera sees x and y as the frame does; a mirror image
    (left and right swapped) reverses it.
    """
    w = np.einsum("mj,nj->mn", H[:, 2, :], np.column_stack([xy, np.ones(len(xy))]))
    return (np.linalg.det(H)[:, None] * w > 0).all(axis=1)


# --------------------------------------------------------------------------- linear stage
def _normaliser(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    scale = np.sqrt(2) / max(np.mean(np.linalg.norm(points - centre, axis=1)), 1e-9)
    return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1.0]])


_T_PITCH = np.array([[1 / 50, 0, -PITCH.length / 100], [0, 1 / 50, -PITCH.width / 100], [0, 0, 1.0]])


def _fit_line(xy: np.ndarray) -> np.ndarray:
    """Total-least-squares homogeneous image line through (N, 2) points."""
    centre = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centre)
    normal = vt[-1]
    return np.array([normal[0], normal[1], -normal @ centre])


def _fit_conic(xy: np.ndarray) -> np.ndarray | None:
    """Algebraic conic through >= 5 points (symmetric 3x3), or None."""
    if len(xy) < 5:
        return None
    T = _normaliser(xy)
    p = (np.column_stack([xy, np.ones(len(xy))]) @ T.T)[:, :2]
    x, y = p[:, 0], p[:, 1]
    _, s, vt = np.linalg.svd(np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)]))
    a, b, c, d, e, f = vt[-1]
    C = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    return T.T @ C @ T


def _line_conic(line: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Real crossings (k, 2) of an image line and a conic, ordered along the line's direction."""
    n = line[:2] / np.hypot(line[0], line[1])
    p0 = -line[2] / np.hypot(line[0], line[1]) * n       # closest point to the origin
    d = np.array([-n[1], n[0]])
    P0, D = np.array([*p0, 1.0]), np.array([*d, 0.0])
    a, b, c = D @ C @ D, 2 * P0 @ C @ D, P0 @ C @ P0
    disc = b * b - 4 * a * c
    if abs(a) < 1e-15 or disc < 0:
        return np.zeros((0, 2))
    t = np.sort([(-b - disc**0.5) / (2 * a), (-b + disc**0.5) / (2 * a)])
    return p0 + t[:, None] * d


def _derived(pts: dict[str, np.ndarray]) -> list[list[tuple[str, np.ndarray, np.ndarray]]]:
    """Alternative sets of derived constraints from line-circle crossings: [("point"|"line", image, pitch), ...].

    For each annotated (line, circle) pair the two crossings are assigned in both orders; the product of the choices
    is returned (at most 2^3 sets; one empty set when there is no pair).
    """
    options: list[list[list]] = []
    for (line_name, circle_name), ys in CROSSINGS.items():
        if line_name not in pts or circle_name not in pts:
            continue
        conic = _fit_conic(pts[circle_name])
        if conic is None:
            continue
        crossing = _line_conic(_fit_line(pts[line_name]), conic)
        if len(crossing) != 2:
            continue
        x = VERTICES[STRAIGHT[line_name][0], 0]
        centre = np.array(CIRCLES[circle_name])
        pair = []
        for order in ((0, 1), (1, 0)):
            items = []
            for k, y in zip(order, ys, strict=True):
                q = np.array([x, y])
                normal = (q - centre) / RADIUS
                tangent_pitch = np.array([normal[0], normal[1], -normal @ q])
                tangent_image = conic @ np.array([*crossing[k], 1.0])
                items.append(("point", crossing[k], q))
                items.append(("line", tangent_image, tangent_pitch))
            pair.append(items)
        options.append(pair)
    sets: list[list] = [[]]
    for pair in options:
        sets = [s + choice for s in sets for choice in pair]
    return sets


def _linear_rows(pts: dict[str, np.ndarray], derived: list, Ti: np.ndarray) -> np.ndarray:
    """Rows of the normalised system A g = 0, g the row-major entries of G = T_pitch H Ti^-1."""
    Tp_inv_t = np.linalg.inv(_T_PITCH).T
    rows = []
    for name, xy in pts.items():
        if name not in STRAIGHT:
            continue
        line = Tp_inv_t @ pitch_line(name)
        line /= np.hypot(line[0], line[1])
        for p in np.column_stack([xy, np.ones(len(xy))]) @ Ti.T:
            rows.append(np.outer(line, p).ravel())
    for kind, image, pitch in derived:
        if kind == "point":
            p = Ti @ np.array([*image, 1.0])
            q = _T_PITCH @ np.array([*pitch, 1.0])
            zero = np.zeros(3)
            rows.append(np.concatenate([zero, -q[2] * p, q[1] * p]))
            rows.append(np.concatenate([q[2] * p, zero, -q[0] * p]))
        else:  # image line l_i ~ H^T l_p  ->  l_p^T G p = 0 for two points p on l_i
            li = np.linalg.inv(Ti).T @ image
            lp = Tp_inv_t @ pitch
            lp /= np.hypot(lp[0], lp[1])
            n = li[:2] / np.hypot(li[0], li[1])
            foot = -li[2] / np.hypot(li[0], li[1]) * n
            d = np.array([-n[1], n[0]])
            for t in (-1.0, 1.0):
                rows.append(np.outer(lp, [*(foot + t * d), 1.0]).ravel())
    return np.array(rows)


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    z = k / n                                           # z in (0, 1]: one hemisphere (G and -G are one homography)
    phi = np.pi * (1 + 5**0.5) * k
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


_DIRECTIONS = _fibonacci_hemisphere(_GRID)


# --------------------------------------------------------------------------- the fit
@dataclass
class LineFit:
    """A frame -> pitch homography (720p pixels -> metres) fitted to one image's line annotations."""

    H: np.ndarray | None
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))   # |pixel residual| per annotated point
    n_points: int = 0
    n_lines: int = 0
    cov: np.ndarray | None = None       # (8, 8) covariance of the update parameters `params`
    reason: str = ""                    # why there is no fit
    front: tuple[float, float] = BOTTOM_CENTRE   # a pixel on the ground: the annotated points' mean
    # H = unnormalise(G0 (I + D(params))) Ti: the parametrisation the covariance refers to
    G0: np.ndarray | None = None
    params: np.ndarray | None = None
    Ti: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.H is not None

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if len(self.residuals) else float("nan")

    @property
    def max(self) -> float:
        return float(self.residuals.max()) if len(self.residuals) else float("nan")


def _compose(G0: np.ndarray, params: np.ndarray) -> np.ndarray:
    D = np.append(params, 0.0).reshape(3, 3)
    return G0 @ (np.eye(3) + D)


def fit_lines(pts: dict[str, np.ndarray], min_points: int = 9) -> LineFit:
    """Fit the frame -> pitch homography to annotation points (`annotation_points` output, 720p pixels)."""
    n_points = sum(len(v) for v in pts.values())
    if n_points < min_points or not any(n in STRAIGHT for n in pts):
        return LineFit(None, n_points=n_points, n_lines=len(pts), reason="too few points")
    all_xy = np.vstack(list(pts.values()))
    Ti = _normaliser(all_xy)
    unnorm = np.linalg.inv(_T_PITCH)
    best = None
    for derived in _derived(pts):
        A = _linear_rows(pts, derived, Ti)
        _, _, vt = np.linalg.svd(A, full_matrices=True)
        basis = vt[-3:][::-1]                                         # smallest first
        candidates = np.vstack([np.eye(3)[:1], _DIRECTIONS]) @ basis  # the smallest vector, then the grid
        G = candidates.reshape(-1, 3, 3)
        H = unnorm @ G @ Ti
        valid = views_pitch(H, all_xy)
        if not valid.any():
            continue
        r = _residuals(H[valid], pts)
        cost = np.mean(np.minimum(np.abs(r), 50.0) ** 2, axis=1)
        for idx in np.argsort(cost)[:3]:
            G0 = G[valid][idx]
            G0 = G0 / np.linalg.norm(G0)

            def fun(params, G0=G0):
                return _residuals(unnorm @ _compose(G0, params) @ Ti, pts)

            sol = least_squares(fun, np.zeros(8), method="lm", x_scale="jac", max_nfev=400)
            Hs = unnorm @ _compose(G0, sol.x) @ Ti
            if not np.isfinite(Hs).all() or np.linalg.cond(Hs) > 1e12 or not views_pitch(Hs[None], all_xy)[0]:
                continue                                              # singular (maps the frame onto a line)
            cost_s = float(np.mean(sol.fun**2))
            if best is None or cost_s < best[0]:
                best = (cost_s, Hs / Hs[2, 2] if abs(Hs[2, 2]) > 1e-12 else Hs, sol, G0)
    if best is None:
        return LineFit(None, n_points=n_points, n_lines=len(pts), reason="no camera view fits")
    _, H, sol, G0 = best
    residuals = np.abs(_residuals(H, pts))
    dof = max(len(residuals) - 8, 1)
    sigma2 = max(float(np.sum(residuals**2)) / dof, SD_FLOOR_PX**2)
    JtJ = sol.jac.T @ sol.jac
    cov = sigma2 * np.linalg.pinv(JtJ) if np.linalg.cond(JtJ) < 1e12 else np.full((8, 8), np.inf)
    front = tuple(float(v) for v in all_xy.mean(axis=0))
    return LineFit(H, residuals, n_points, len(pts), cov, front=front, G0=G0, params=sol.x, Ti=Ti)


def pitch_to_frame(H: np.ndarray, xy: np.ndarray, front=BOTTOM_CENTRE) -> tuple[np.ndarray, np.ndarray]:
    """Pitch points (N, 2) -> 720p pixels through the inverse of the frame -> pitch `H`, and whether each is in front
    of the camera (on the same side of the horizon as the frame pixel `front`, e.g. an annotated line point)."""
    inv = np.linalg.inv(H)
    p = np.column_stack([xy, np.ones(len(xy))]) @ inv.T
    side = np.sign((H @ np.array([front[0], front[1], 1.0]))[2])
    ok = np.sign(p[:, 2]) == side
    return p[:, :2] / np.where(np.abs(p[:, 2]) > 1e-12, p[:, 2], 1e-12)[:, None], ok


def vertex_sd(fit: LineFit, xy: np.ndarray = VERTICES, eps: float = 1e-6) -> np.ndarray:
    """2-D standard deviation (720p px, sqrt of the trace) of each pitch point's image under the fit's covariance."""
    if not fit.ok:
        return np.full(len(xy), np.inf)
    if not np.isfinite(fit.cov).all():
        return np.full(len(xy), np.inf)
    unnorm = np.linalg.inv(_T_PITCH)

    def image(params):
        return pitch_to_frame(unnorm @ _compose(fit.G0, params) @ fit.Ti, xy, fit.front)[0]

    base = image(fit.params)
    J = np.stack([(image(fit.params + eps * np.eye(8)[k]) - base) / eps for k in range(8)], axis=-1)   # (N, 2, 8)
    var = np.einsum("nik,kl,nil->n", J, fit.cov, J)
    return np.sqrt(np.maximum(var, 0.0))


# --------------------------------------------------------------------------- labels
@dataclass
class Labels:
    """The 32 vertices in 720p pixels, their visibility (2 in frame and in front, else 0) and the pitch's frame box."""

    xy: np.ndarray            # (32, 2)
    visible: np.ndarray       # (32,) bool
    box: tuple[float, float, float, float] | None   # x0, y0, x1, y1 of the pitch area in the frame (720p px)


def labels(H: np.ndarray, front=BOTTOM_CENTRE, step: int = 8) -> Labels:
    """Labels of a frame -> pitch homography; `front` is a frame pixel on the ground (default the bottom centre)."""
    xy, in_front = pitch_to_frame(H, VERTICES, front)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < FRAME_W) & (xy[:, 1] >= 0) & (xy[:, 1] < FRAME_H)
    visible = in_front & inside
    gx, gy = np.meshgrid(np.arange(step / 2, FRAME_W, step), np.arange(step / 2, FRAME_H, step))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    q = np.column_stack([grid, np.ones(len(grid))]) @ H.T
    side = np.sign((H @ np.array([front[0], front[1], 1.0]))[2])
    ground = np.sign(q[:, 2]) == side
    q = q[:, :2] / np.where(np.abs(q[:, 2]) > 1e-12, q[:, 2], 1e-12)[:, None]
    on_pitch = ground & (q[:, 0] >= 0) & (q[:, 0] <= PITCH.length) & (q[:, 1] >= 0) & (q[:, 1] <= PITCH.width)
    box = None
    if on_pitch.any():
        g = grid[on_pitch]
        box = (max(g[:, 0].min() - step / 2, 0.0), max(g[:, 1].min() - step / 2, 0.0),
               min(g[:, 0].max() + step / 2, FRAME_W), min(g[:, 1].max() + step / 2, FRAME_H))
    return Labels(np.where(visible[:, None], xy, 0.0), visible, box)


def image_up_direction(H: np.ndarray) -> np.ndarray:
    """Unit pitch direction of 'up the frame' at the bottom centre: (dx, dy). A main-camera view has dy < 0."""
    q = np.array([[FRAME_W / 2, FRAME_H - 1, 1.0], [FRAME_W / 2, FRAME_H - 101, 1.0]]) @ H.T
    d = q[1, :2] / q[1, 2] - q[0, :2] / q[0, 2]
    return d / max(np.linalg.norm(d), 1e-12)

