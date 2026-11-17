"""The homography fit: one robust fit of a frame -> pitch homography to line and circle points.

Evidence, in frame pixels: points on a known straight marking (a
:class:`~tactifoot_vision.enums.PitchMarking` line) and points on a known
circle (the centre circle, the penalty arcs), each with a weight (the line
network's probability).

The residuals are image distances, in pixels of the frame scaled to 960 px
wide: a line point's distance to its marking's image line (``H^T L``) and a
circle point's Sampson distance to its marking's image conic (``H^T C H``).
Coordinates are normalised (Hartley) on both sides and the matrix is
updated multiplicatively, ``H = H0 (I + D)``; Levenberg-Marquardt (IRLS on
a soft-L1 loss) uses the Jacobian in closed form.

Starts: a direct linear solve with every line point on its line (four lines,
two in each direction, determine a homography) and its mirror images when it
comes out mirrored; the caller's previous matrix; and, only when there is no
previous matrix, the ``template_top`` best distinct cameras of a bank of
1,440 main-camera views. Each start is refined, the lowest robust cost wins,
evidence the fit leaves far off is dropped and the fit redone. A result is
refused when it is singular, mirrored (a real camera above the pitch never
flips its orientation), puts evidence beyond its horizon, or when the
evidence leaves it undetermined: the fit's covariance, propagated to pitch
positions over the lower frame, is its uncertainty in metres.

Matrices are frame -> pitch in standard-pitch metres (105 x 68 m, corner
origin; :mod:`tactifoot_vision.pitch.markings`).
"""

from dataclasses import dataclass, field

import numpy as np

from tactifoot_vision.pitch.markings import GEOMETRY, NUM_MARKINGS

UNIT_WIDTH = 960.0  # residuals are pixels of the frame scaled to this width
_PITCH_CENTRE = np.array([52.5, 34.0])
# The eight update parameters: entry (k, j) of D in H = H0 (I + D) (D[2, 2] stays 0).
_PARAMS = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]])
_PITCH_SCALE = 50.0
# Pitch landmarks in normalised pitch coordinates (homogeneous).
_LINES = np.array(
    [m.line if m.kind == "line" else np.zeros(3) for m in GEOMETRY], float
)
_CONICS = np.array(
    [m.conic if m.kind == "circle" else np.zeros((3, 3)) for m in GEOMETRY], float
)
_IS_CIRCLE = np.array([m.kind == "circle" for m in GEOMETRY])


def _pitch_norm() -> np.ndarray:
    s = 1.0 / _PITCH_SCALE
    return np.array(
        [[s, 0, -_PITCH_CENTRE[0] * s], [0, s, -_PITCH_CENTRE[1] * s], [0, 0, 1.0]]
    )


T_PITCH = _pitch_norm()
T_PITCH_INV = np.linalg.inv(T_PITCH)
_LINES_N = _LINES @ T_PITCH_INV  # (T^-T L)^T = L^T T^-1
_LINES_N /= np.maximum(np.hypot(_LINES_N[:, 0], _LINES_N[:, 1]), 1e-12)[:, None]
_CONICS_N = np.einsum("ji,cjk,kl->cil", T_PITCH_INV, _CONICS, T_PITCH_INV)


def image_norm(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    s = 2.0 / max(w, h)
    return np.array([[s, 0, -w / 2 * s], [0, s, -h / 2 * s], [0, 0, 1.0]])


@dataclass
class Evidence:
    """What one frame shows: points of known markings, in frame pixels, with their weights."""

    size: tuple[int, int]
    pt_class: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    pt_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pt_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def empty(self) -> bool:
        return len(self.pt_class) == 0


@dataclass
class LinePoints:
    """A frame's points on the painted markings, as the line network finds them.

    ``marking`` holds indices into :data:`~tactifoot_vision.pitch.markings.MARKINGS`,
    ``xy`` frame pixels and ``prob`` the network's probability of each point.
    """

    marking: np.ndarray
    xy: np.ndarray
    prob: np.ndarray

    def __post_init__(self) -> None:
        self.marking = np.asarray(self.marking, dtype=np.int64).reshape(-1)
        self.xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        self.prob = np.asarray(self.prob, dtype=np.float32).reshape(-1)
        if not len(self.marking) == len(self.xy) == len(self.prob):
            raise ValueError(
                f"{len(self.marking)} markings, {len(self.xy)} points and {len(self.prob)} probabilities"
            )
        if len(self.marking) and not (
            0 <= self.marking.min() and self.marking.max() < NUM_MARKINGS
        ):
            raise ValueError(f"marking indices must be in [0, {NUM_MARKINGS})")

    def __len__(self) -> int:
        return len(self.marking)

    @classmethod
    def empty(cls) -> "LinePoints":
        return cls(
            np.zeros(0, np.int64), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        )


def select_points(
    points: LinePoints,
    size: tuple[int, int],
    *,
    min_prob: float = 0.3,
    per_class: int = 24,
) -> "Evidence":
    """The evidence of a frame of ``size`` (width, height): points of at least ``min_prob``, at most ``per_class`` per marking.

    The points kept of a marking are evenly spaced along its longer image
    axis, so a long line does not cost more than it informs; each weighs its
    probability.
    """
    keep = points.prob >= min_prob
    cls = points.marking[keep]
    xy = points.xy[keep].astype(float)
    prob = points.prob[keep].astype(float)
    take = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        if len(idx) > per_class:
            spread = np.ptp(xy[idx], axis=0)
            order = idx[np.argsort(xy[idx, int(spread[1] > spread[0])])]
            idx = order[np.linspace(0, len(order) - 1, per_class).round().astype(int)]
        take.append(idx)
    take = np.concatenate(take) if take else np.zeros(0, int)
    return Evidence(size, cls[take].astype(int), xy[take], prob[take])


@dataclass(frozen=True)
class FitConfig:
    """Tunables of the joint fit (pixels at 960 px wide; metres on the pitch)."""

    loss_scale_px: float = 2.0  # soft-L1
    max_evaluations: int = 30  # LM iterations
    max_uncertainty_m: float = (
        2.0  # reject above: the evidence leaves the matrix undetermined
    )
    sigma_px: float = 1.5  # residual noise assumed by the covariance
    inlier_px: float = 3.0
    min_front_share: float = 0.95  # evidence points on the visible side of the horizon
    template_top: int = 3  # distinct templates refined when there is no previous matrix
    gate_px: float = (
        8.0  # second pass without the evidence the first fit leaves this far off
    )
    refine_top: int = 2  # starts refined, by their cost before refinement (the previous matrix always)


@dataclass
class Fit:
    """A solved frame: ``matrix`` frame -> pitch (corner metres) or None, and how it went."""

    matrix: np.ndarray | None
    uncertainty_m: float = np.inf
    inlier_share: float = 0.0
    cost: float = np.inf
    start: str = ""
    reason: str = ""
    candidate: np.ndarray | None = None  # the best matrix before the checks
    tried_templates: bool = (
        False  # the template search ran (the caller may back it off)
    )


class _Problem:
    """The normalised evidence of one frame and its residual function."""

    def __init__(self, ev: Evidence, config: FitConfig) -> None:
        self.ev = ev
        self.config = config
        self.t_img = image_norm(ev.size)
        self.unit = UNIT_WIDTH / 2.0 * (max(ev.size) / ev.size[0])
        pts = np.column_stack([ev.pt_xy, np.ones(len(ev.pt_xy))]) @ self.t_img.T
        circle = _IS_CIRCLE[ev.pt_class] if len(ev.pt_class) else np.zeros(0, bool)
        self.ln_idx, self.ci_idx = np.flatnonzero(~circle), np.flatnonzero(circle)
        self.ln_x, self.ln_c = pts[~circle], ev.pt_class[~circle]
        self.ln_w = np.sqrt(ev.pt_weight[~circle])
        self.ci_x, self.ci_c = pts[circle], ev.pt_class[circle]
        self.ci_w = np.sqrt(ev.pt_weight[circle])
        self.ci_classes = np.unique(self.ci_c)
        self.n = len(self.ln_x) + len(self.ci_x)

    def matrix(self, h0: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """``h0 (I + D(delta))`` for one ``delta`` (8,) or a batch (B, 8)."""
        delta = np.asarray(delta, float)
        flat = delta.reshape(-1, 8)
        d = np.zeros((len(flat), 3, 3))
        d[:, [0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1]] = flat
        d += np.eye(3)
        out = h0 @ d
        return out[0] if delta.ndim == 1 else out

    def residuals_of(self, hn: np.ndarray, weighted: bool = True) -> np.ndarray:
        """Per-point image distances (unit px), lines then circles, weighted. ``hn`` (3, 3) or (B, 3, 3)."""
        single = hn.ndim == 2
        hb = hn[None] if single else hn
        out = []
        if len(self.ln_x):
            image_lines = np.einsum("pi,bij->bpj", _LINES_N[self.ln_c], hb)
            num = (image_lines * self.ln_x[None]).sum(-1)
            den = np.hypot(image_lines[..., 0], image_lines[..., 1])
            r = num / np.maximum(den, 1e-12)
            out.append(r * self.ln_w[None] if weighted else r)
        if len(self.ci_x):
            r = np.empty((len(hb), len(self.ci_x)))
            for c in self.ci_classes:
                sel = self.ci_c == c
                conic = np.einsum("bki,kl,blj->bij", hb, _CONICS_N[c], hb)
                v = np.einsum("pj,bij->bpi", self.ci_x[sel], conic)
                f = (v * self.ci_x[sel][None]).sum(-1)
                gr = 2 * np.hypot(v[..., 0], v[..., 1])
                r[:, sel] = f / np.maximum(gr, 1e-12)
            out.append(r * self.ci_w[None] if weighted else r)
        res = np.concatenate(out, axis=1) * self.unit if out else np.zeros((len(hb), 0))
        return res[0] if single else res

    def jacobian(
        self, hn0: np.ndarray, h: np.ndarray, weighted: bool = True
    ) -> np.ndarray:
        """d residuals / d delta (n x 8) at ``h = hn0 (I + D(delta))``, in closed form.

        ``dH / d delta_(k, j) = hn0 E_kj`` whatever delta is, so for a line
        point (image line ``l = H^T L``, residual ``l . x / |l_12|``) the
        derivative of ``l`` is ``(hn0^T L)_k e_j``; for a circle point
        (``y = H x``, ``f = y^T C y``, ``v = H^T C y``, Sampson residual
        ``f / 2 |v_12|``) it is ``df = 2 x_j (hn0^T C y)_k`` and
        ``dv_a = [a = j] (hn0^T C y)_k + x_j (H^T C hn0)_ak``.
        """
        k_idx, j_idx = _PARAMS[:, 0], _PARAMS[:, 1]
        out = []
        if len(self.ln_x):
            lines = _LINES_N[self.ln_c]
            m0, img = lines @ hn0, lines @ h
            x = self.ln_x
            num = (img * x).sum(1)
            den = np.maximum(np.hypot(img[:, 0], img[:, 1]), 1e-12)
            lj = np.where(j_idx[None] < 2, img[:, np.minimum(j_idx, 1)], 0.0)
            jac = m0[:, k_idx] * (
                x[:, j_idx] / den[:, None] - num[:, None] * lj / den[:, None] ** 3
            )
            out.append(jac * self.ln_w[:, None] if weighted else jac)
        if len(self.ci_x):
            jac = np.empty((len(self.ci_x), 8))
            for c in self.ci_classes:
                sel = self.ci_c == c
                conic = _CONICS_N[c]
                x = self.ci_x[sel]
                y = x @ h.T
                cy = y @ conic  # C symmetric
                v = cy @ h  # rows: (H^T C y)^T
                f = (cy * y).sum(1)
                n = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-12)
                q = cy @ hn0  # rows: (hn0^T C y)^T
                s_mat = h.T @ conic @ hn0
                xj, qk = x[:, j_idx], q[:, k_idx]
                df = 2 * xj * qk
                dv0 = (j_idx[None] == 0) * qk + xj * s_mat[0, k_idx][None]
                dv1 = (j_idx[None] == 1) * qk + xj * s_mat[1, k_idx][None]
                jac[sel] = df / (2 * n[:, None]) - f[:, None] * (
                    v[:, :1] * dv0 + v[:, 1:2] * dv1
                ) / (2 * n[:, None] ** 3)
            out.append(jac * self.ci_w[:, None] if weighted else jac)
        return np.vstack(out) * self.unit if out else np.zeros((0, 8))

    def distances(self, hn: np.ndarray) -> np.ndarray:
        """Unweighted distance (unit px) of each point, in the evidence's order."""
        d = np.abs(self.residuals_of(hn, weighted=False))
        pt = np.empty(len(self.ev.pt_class))
        pt[self.ln_idx] = d[: len(self.ln_idx)]
        pt[self.ci_idx] = d[len(self.ln_idx) :]
        return pt

    def _rho(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Soft-L1 cost per residual and its IRLS weight (derivative w.r.t. r^2 / 2)."""
        s = self.config.loss_scale_px
        z = (r / s) ** 2
        return 2 * s * s * (np.sqrt(1 + z) - 1), 1.0 / np.sqrt(1 + z)

    def cost(self, hn: np.ndarray) -> np.ndarray:
        """Robust cost of one matrix or a batch."""
        r = self.residuals_of(hn)
        return self._rho(r)[0].sum(-1)

    def refine(self, hn0: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Levenberg-Marquardt (IRLS on the robust loss) from ``hn0``: matrix, cost, weighted Jacobian (n x 8)."""
        c = self.config
        delta = np.zeros(8)
        lam = 1e-3
        h = self.matrix(hn0, delta)
        r = self.residuals_of(h)
        cost = self._rho(r)[0].sum()
        small = False
        for _ in range(c.max_evaluations):
            jac = self.jacobian(hn0, self.matrix(hn0, delta))
            w = self._rho(r)[1]
            jw = jac * w[:, None]
            a = jac.T @ jw
            g = jw.T @ r
            improved = False
            for _ in range(6):
                try:
                    step = -np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-9), g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                cand = delta + step
                r_new = self.residuals_of(self.matrix(hn0, cand))
                cost_new = self._rho(r_new)[0].sum()
                if np.isfinite(cost_new) and cost_new < cost:
                    improved = True
                    small = (
                        cost - cost_new < 1e-6 * max(cost, 1e-12)
                        or np.abs(step).max() < 1e-9
                    )
                    delta, r, cost = cand, r_new, cost_new
                    lam = max(lam / 3, 1e-9)
                    break
                lam *= 4
            if not improved or small:
                break
        hn = self.matrix(hn0, delta)
        hn = hn / np.linalg.norm(hn)
        # the Jacobian about the solution itself (delta = 0 at hn), as the covariance reads it
        jac = self.jacobian(hn, hn)
        wj = jac * np.sqrt(self._rho(r)[1])[:, None]
        return hn, float(cost), wj


def to_pixels(hn: np.ndarray, t_img: np.ndarray) -> np.ndarray:
    m = T_PITCH_INV @ hn @ t_img
    return m / np.linalg.norm(m)


def to_normalised(matrix: np.ndarray, t_img: np.ndarray) -> np.ndarray:
    m = T_PITCH @ np.asarray(matrix, float) @ np.linalg.inv(t_img)
    return m / np.linalg.norm(m)


def line_start(ev: Evidence, prob: "_Problem") -> np.ndarray | None:
    """Direct linear solve (normalised): every line point on its class's line (L^T H x = 0).

    Circle points are left out (their constraint is quadratic). Four lines,
    two in each direction, fix the eight degrees of freedom.
    """
    rows = []
    if len(prob.ln_x):
        big_l = _LINES_N[prob.ln_c]
        rows.append(
            np.einsum("pi,pj->pij", big_l, prob.ln_x).reshape(-1, 9)
            * prob.ln_w[:, None]
        )
    if not rows:
        return None
    a = np.vstack(rows)
    if len(a) < 8:
        return None
    _, s, vt = np.linalg.svd(a)  # vt is 9 x 9 even for the minimal 8 x 9 system
    if len(s) < 8 or s[7] < 1e-6 * s[0]:  # rank 8: one null vector, the matrix
        return None
    hn = vt[-1].reshape(3, 3)
    if not np.isfinite(hn).all() or abs(np.linalg.det(hn)) < 1e-12:
        return None
    return hn / np.linalg.norm(hn)


_BANKS: dict[tuple[int, int], np.ndarray] = {}


def template_bank(size: tuple[int, int]) -> np.ndarray:
    """Normalised frame -> pitch matrices of 1,440 main-camera views (behind the near touchline, panning, tilting, zooming).

    A start for evidence no linear solve can use (a circle and one or two
    lines): the templates with the lowest robust cost are refined. Every
    template is a real camera above the pitch, so none is mirrored.
    """
    key = size
    if key in _BANKS:
        return _BANKS[key]

    scale = max(size) / 1920
    out = []
    for back, height in ((25.0, 14.0), (45.0, 22.0), (70.0, 30.0)):
        for tx in np.arange(-5.0, 111.0, 5.0):
            for ty in (5.0, 20.0, 34.0, 48.0):
                for focal in (900.0, 1400.0, 2100.0, 3200.0, 4800.0):
                    m = _camera((52.5, 68.0 + back, height), (tx, ty, 0.0),
                               focal * scale, size)  # fmt: skip
                    out.append(to_normalised(m, image_norm(size)))
    _BANKS[key] = np.array(out)
    return _BANKS[key]


def _template_distances(
    prob: "_Problem", bank: np.ndarray, x, cls, kind: str, cap: float
) -> np.ndarray:
    """(templates, points) image distances (unit px) of points of ``cls``; beyond a template's horizon: ``cap``."""
    front = np.einsum("tj,pj->tp", bank[:, 2], x)
    if kind == "line":
        image_lines = np.einsum("pi,tij->tpj", _LINES_N[cls], bank)
        num = np.einsum("tpj,pj->tp", image_lines, x)
        den = np.hypot(image_lines[..., 0], image_lines[..., 1])
    else:
        conics = np.einsum("tki,ckl,tlj->tcij", bank, _CONICS_N, bank)
        v = np.einsum("pj,tpij->tpi", x, conics[:, cls])
        num = (v * x[None]).sum(-1)
        den = 2 * np.hypot(v[..., 0], v[..., 1])
    d = np.abs(num) / np.maximum(den, 1e-12) * prob.unit
    width, height = prob.ev.size
    lower = prob.t_img @ np.array([width / 2, height - 1, 1.0])
    sign = np.sign(bank[:, 2] @ lower)
    return np.where(front * sign[:, None] > 0, np.minimum(d, cap), cap)


def _template_costs(prob: "_Problem", bank: np.ndarray) -> np.ndarray:
    """Robust cost of every template on the evidence (points beyond a template's horizon count as far off)."""
    cost = np.zeros(len(bank))
    cap = 50.0  # px: a point far off or beyond the horizon
    for x, cls, w, kind in ((prob.ln_x, prob.ln_c, prob.ln_w, "line"),
                            (prob.ci_x, prob.ci_c, prob.ci_w, "circle")):  # fmt: skip
        if not len(x):
            continue
        d = _template_distances(prob, bank, x, cls, kind, cap)
        z = (d / prob.config.loss_scale_px) ** 2
        cost += (2 * (np.sqrt(1 + z) - 1) * w[None] ** 2).sum(1)
    return cost


def template_starts(prob: "_Problem") -> list[tuple[str, np.ndarray]]:
    """The ``template_top`` templates of lowest cost whose views differ (no two within 10 m over the lower frame)."""
    k = prob.config.template_top
    bank = template_bank(prob.ev.size)
    cost = _template_costs(prob, bank)
    if k <= 1:
        return [("template", bank[int(np.argmin(cost))])]
    px = np.column_stack([lower_frame(prob.ev.size), np.ones(20)]) @ prob.t_img.T
    chosen, views = [], []
    for i in np.argsort(cost):
        p = px @ bank[i].T
        if np.any(np.abs(p[:, 2]) < 1e-9):
            continue
        xy = (T_PITCH_INV @ p.T).T
        xy = xy[:, :2] / xy[:, 2:]
        if all(np.nanmedian(np.linalg.norm(xy - v, axis=1)) > 10.0 for v in views):
            chosen.append(("template", bank[i]))
            views.append(xy)
        if len(chosen) == k:
            break
    return chosen


def orientation(matrix: np.ndarray, pixels) -> float:
    """Sign of the frame -> pitch map's Jacobian, by majority over ``pixels`` (on the pitch): +1 for a real camera above it.

    The pitch frame (x along, y across towards the main camera) seen from
    above keeps the image's orientation; a mirrored matrix flips it. Beyond
    the horizon the sign flips too, hence pixels that show the pitch.
    """
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    w = np.column_stack([pixels, np.ones(len(pixels))]) @ matrix[2]
    return float(np.sign(np.sign(np.linalg.det(matrix) / w**3).sum()))


def lower_frame(size: tuple[int, int], top: float = 0.35) -> np.ndarray:
    """A 5 x 4 grid of pixels over the frame from ``top`` of its height down: what must be on the pitch side."""
    w, h = size
    xs = (np.arange(5) + 0.5) / 5 * w
    ys = h * top + (np.arange(4) + 0.5) / 4 * h * (1 - top)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _front_sign(matrix: np.ndarray, size: tuple[int, int]) -> float:
    w, h = size
    return float(np.sign(matrix[2] @ np.array([w / 2, h - 1, 1.0])))


def uncertainty_m(
    prob: _Problem, hn: np.ndarray, jac: np.ndarray, probes_px: np.ndarray
) -> float:
    """Median std (m) of the pitch positions of ``probes_px`` under the fit's covariance."""
    sigma = prob.config.sigma_px
    jtj = jac.T @ jac
    try:
        cov = sigma**2 * np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return np.inf
    if not np.isfinite(cov).all():
        return np.inf
    probes = np.column_stack([probes_px, np.ones(len(probes_px))]) @ prob.t_img.T

    def positions(delta):
        m = T_PITCH_INV @ prob.matrix(hn, delta)
        p = probes @ m.T
        return (p[:, :2] / p[:, 2:3]).ravel()

    base = positions(np.zeros(8))
    eps = 1e-6
    d = np.empty((len(base), 8))
    for k in range(8):
        step = np.zeros(8)
        step[k] = eps
        d[:, k] = (positions(step) - base) / eps
    var = np.einsum("ik,kl,il->i", d, cov, d).reshape(-1, 2).sum(1)
    if not np.isfinite(var).all() or (var < 0).any():
        return np.inf
    return float(np.median(np.sqrt(var)))


def usable(matrix: np.ndarray | None) -> bool:
    """The benchmark's test: finite and not singular after rescaling pixels by 1000 and metres by 100."""
    if matrix is None or not np.isfinite(matrix).all():
        return False
    m = np.diag([0.01, 0.01, 1.0]) @ matrix @ np.diag([1000.0, 1000.0, 1.0])
    s = np.linalg.svd(m, compute_uv=False)
    return s[0] > 0 and s[-1] / s[0] > 1e-6


def subset(ev: Evidence, keep: np.ndarray) -> Evidence:
    return Evidence(ev.size, ev.pt_class[keep], ev.pt_xy[keep], ev.pt_weight[keep])


def solve(
    ev: Evidence,
    config: FitConfig,
    previous: np.ndarray | None = None,
    *,
    templates: bool = True,
) -> Fit:
    """The joint fit of one frame's evidence; ``previous`` (frame -> pitch) is a start.

    1. Starts: the linear solve (and its mirror images when it comes out
       mirrored) and the previous matrix; each refined on all the evidence,
       the valid one with the lowest robust cost kept.
    2. Only when there is no previous matrix (and ``templates``): the
       ``template_top`` distinct templates of lowest cost, refined too.
    3. Gating: evidence the fit leaves far off is dropped and the fit redone.
    4. Checks: the evidence on the visible side of the horizon (the side of
       the frame's bottom centre) and the uncertainty at most
       ``max_uncertainty_m``.
    """
    if ev.empty():
        return Fit(None, reason="no evidence")
    prob = _Problem(ev, config)
    if prob.n < 8:
        return Fit(None, reason="too little evidence")
    starts = _line_starts(ev, prob, "lines")
    if previous is not None and usable(previous):
        starts.append(("previous", to_normalised(previous, prob.t_img)))
    best = _best_refined(prob, starts, config, top=config.refine_top)
    use_templates = templates and previous is None
    if use_templates:
        other = _best_refined(prob, template_starts(prob), config)
        if other is not None and (best is None or other[2] < best[2]):
            best = other
    fit = _checked(prob, ev, config, best)
    fit.tried_templates = use_templates
    return fit


def _checked(prob: "_Problem", ev: Evidence, config: FitConfig, best) -> Fit:
    """Gating, then the horizon and uncertainty checks, of the best refined start."""
    if best is None:
        return Fit(None, reason="no valid fit")
    name, hn, cost, jac, matrix = best
    # gating: drop what the fit leaves far off, fit again
    keep = prob.distances(hn) < config.gate_px
    if not keep.all():
        part = subset(prob.ev, keep)
        gated = _Problem(part, config)
        if gated.n >= 8:
            refined = _best_refined(gated, [(name + "+gated", hn)], config)
            if refined is not None:
                prob = gated
                name, hn, cost, jac, matrix = refined
    fit = Fit(None, cost=cost, start=name, candidate=matrix)
    dist = prob.distances(hn)
    fit.inlier_share = float(np.mean(dist < config.inlier_px)) if len(dist) else 0.0
    # every evidence pixel must be on the side of the horizon the frame's bottom centre is on
    sign = _front_sign(matrix, ev.size)
    pixels = prob.ev.pt_xy
    w = np.column_stack([pixels, np.ones(len(pixels))]) @ matrix[2]
    if len(w) and np.mean(w * sign > 0) < config.min_front_share:
        fit.reason = "evidence beyond the horizon"
        return fit
    lower = lower_frame(ev.size)
    ahead = np.column_stack([lower, np.ones(len(lower))]) @ matrix[2] * sign > 0
    probes = lower[ahead] if ahead.sum() >= 5 else pixels
    fit.uncertainty_m = uncertainty_m(prob, hn, jac, probes)
    if not fit.uncertainty_m <= config.max_uncertainty_m:
        fit.reason = "undetermined"
        return fit
    fit.matrix = matrix
    return fit


def _line_starts(ev: Evidence, prob: "_Problem", name: str) -> list:
    """The linear solve, and its two mirror images when it comes out mirrored (a symmetric set fits both)."""
    hn = line_start(ev, prob)
    if hn is None:
        return []
    out = [(name, hn)]
    m = to_pixels(hn, prob.t_img)
    if orientation(m, ev.pt_xy) < 0:
        for mirror in (_MIRROR_X, _MIRROR_Y):
            out.append((name + "-mirrored", to_normalised(mirror @ m, prob.t_img)))
    return out


def _best_refined(prob: _Problem, starts, config: FitConfig, top: int | None = None):
    """Refine the ``top`` starts of lowest cost (all when None); the valid fit (usable, not mirrored) with the lowest robust cost."""
    best = None
    if top is not None and len(starts) > top:
        # the previous frame's matrix is always refined: two mirror starts must not push it out
        kept = [s for s in starts if s[0] == "previous"]
        rest = [s for s in starts if s[0] != "previous"]
        costs = prob.cost(np.array([h for _, h in rest]))
        costs = np.where(np.isfinite(costs), costs, np.inf)
        starts = kept + [rest[i] for i in np.argsort(costs, kind="stable")[:top]]
    for name, hn0 in starts:
        try:
            hn, cost, jac = prob.refine(hn0)
        except (np.linalg.LinAlgError, ValueError):
            continue
        matrix = to_pixels(hn, prob.t_img)
        if not usable(matrix):
            continue
        if orientation(matrix, prob.ev.pt_xy) < 0:
            continue
        if best is None or cost < best[2]:
            best = (name, hn, cost, jac, matrix)
    return best


_MIRROR_X = np.array([[-1.0, 0, 105.0], [0, 1, 0], [0, 0, 1]])
_MIRROR_Y = np.array([[1.0, 0, 0], [0, -1, 68.0], [0, 0, 1]])


def _camera(
    position=(52.5, 108.0, 22.0),
    target=(52.5, 30.0, 0.0),
    focal_px=1500.0,
    size=(1920, 1080),
    roll_deg=0.0,
) -> np.ndarray:
    """Frame -> pitch homography of a pinhole camera at ``position`` (pitch metres, height up) looking at ``target``.

    The pitch frame (x along, y across towards the main camera) with the
    height up is left-handed, so the right-handed world here has z = -height.
    """
    flip = np.array([1.0, 1.0, -1.0])
    pos = np.asarray(position, float) * flip
    forward = np.asarray(target, float) * flip - pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    roll = np.deg2rad(roll_deg)
    right, down = (
        np.cos(roll) * right + np.sin(roll) * down,
        -np.sin(roll) * right + np.cos(roll) * down,
    )
    rot = np.vstack([right, down, forward])  # world -> camera
    k = np.array([[focal_px, 0, size[0] / 2], [0, focal_px, size[1] / 2], [0, 0, 1]])
    t = -rot @ pos
    pitch_to_image = k @ np.column_stack([rot[:, 0], rot[:, 1], t])
    m = np.linalg.inv(pitch_to_image)
    return m / np.linalg.norm(m)
