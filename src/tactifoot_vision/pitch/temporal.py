"""The temporal layer: a constant-velocity Kalman filter on a video's frame -> pitch homographies, and its RTS pass.

No camera model and no evidence of its own. The state is the pitch position
(standard-pitch metres) of four fixed image points, the corners of a box at
10-90 % of the frame's width and 50-95 % of its height, plus their rates.
Those four points fix the homography and are well conditioned where the
pitch is in view. Each fresh per-frame fit is a measurement of the eight
coordinates.

A measurement whose Mahalanobis distance from the prediction is above
``gate_chi2`` is not used, and the frame keeps its own matrix. ``reset_after``
such rejects in a row (another view: a cut), or more than ``max_gap`` steps
without a fresh fit, start a new segment; a reset hands every frame since
the run of rejects began (rejected frames and frames without a fit between
them) to the new segment. :meth:`HomographyFilter.step` is causal (the
filtered state). :meth:`HomographyFilter.smoothed` is the Rauch-Tung-Striebel
pass over the whole video, per segment, never across a reset.

Time: the settings are per step of ``1 / rate_hz`` seconds (25 fps, the rate
they were chosen at). ``step(matrix, dt)`` takes the time since the previous
step in those units, so a run at another frame rate or stride is filtered in
the same physical time; ``dt=1`` everywhere reproduces the 25 fps filter.
"""

from dataclasses import dataclass

import numpy as np

# image points (fractions of the frame) whose pitch positions are the state
POINTS = np.array([[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]])
_N = 8
_EYE = np.eye(_N)


@dataclass(frozen=True)
class FilterConfig:
    """Tunables of :class:`HomographyFilter`.

    Attributes:
        accel_m: white-noise acceleration of each coordinate, metres per step²
            (a step is ``1 / rate_hz`` seconds).
        meas_m: measurement noise of each coordinate, metres.
        gate_chi2: a fit further than this (squared Mahalanobis distance, 8
            degrees of freedom) from the prediction is rejected.
        reset_after: this many rejects in a row start a new segment.
        max_gap: more than this many steps without a fresh fit end the segment.
        rate_hz: the steps per second the other settings are given in.
    """

    accel_m: float = 0.15
    meas_m: float = 0.3
    gate_chi2: float = 60.0
    reset_after: int = 3
    max_gap: int = 12
    rate_hz: float = 25.0

    def __post_init__(self) -> None:
        if not self.accel_m > 0 or not self.meas_m > 0 or not self.gate_chi2 > 0:
            raise ValueError("accel_m, meas_m and gate_chi2 must be > 0")
        if self.reset_after < 1 or self.max_gap < 0:
            raise ValueError("reset_after must be >= 1 and max_gap >= 0")
        if not self.rate_hz > 0:
            raise ValueError(f"rate_hz must be > 0, got {self.rate_hz}")


def _pack(p: np.ndarray) -> np.ndarray | tuple[float, float, float, float]:
    """A covariance as its 2 x 2 block, when it is exactly that block times the identity (else as it is).

    Every coordinate has the same noise and is measured alike, so the
    covariance is ``kron([[a, b], [c, d]], I_8)``: four numbers instead of
    256. The check is exact, so :func:`_unpack` gives back the same matrix
    bit for bit, and a covariance off the pattern is kept whole.
    """
    a, b, c, d = p[0, 0], p[0, _N], p[_N, 0], p[_N, _N]
    if np.array_equal(np.kron(np.array([[a, b], [c, d]]), _EYE), p):
        return (float(a), float(b), float(c), float(d))
    return p


def _unpack(p) -> np.ndarray:
    if isinstance(p, np.ndarray):
        return p
    return np.kron(np.array(p).reshape(2, 2), _EYE)


def _four_point(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    h = np.linalg.solve(np.array(rows), dst.reshape(-1))
    return np.append(h, 1.0).reshape(3, 3)


class HomographyFilter:
    """The filter over one video of ``frame_size`` (width, height): ``step(fit or None, dt)`` per frame, then ``smoothed()``.

    Matrices are frame -> standard pitch (105 x 68 m); ``step`` takes the
    frame's fresh fit, or ``None`` when the frame has none. With
    ``keep_history=False`` nothing per frame is kept (the causal use, whose
    memory then stays the same however long the video) and
    :meth:`smoothed` is unavailable.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        config: FilterConfig | None = None,
        *,
        keep_history: bool = True,
    ) -> None:
        self.config = config or FilterConfig()
        self.pixels = POINTS * np.asarray(frame_size, float)
        self.keep_history = keep_history
        self.R = np.eye(_N) * self.config.meas_m**2
        self.H = np.hstack([np.eye(_N), np.zeros((_N, _N))])
        self._models: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.F, self.Q = self._model(1.0)
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.rejects = 0
        self.gap = 0.0
        self._reject_start = 0  # index into records of the run of rejects' first frame
        # per frame: (x_pred, p_pred, x_post, p_post, restart, F into this frame), or None outside a segment
        self.records: list = []

    def _model(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition and process noise over ``dt`` steps (white-noise acceleration)."""
        if dt not in self._models:
            F = np.block(
                [[np.eye(_N), dt * np.eye(_N)], [np.zeros((_N, _N)), np.eye(_N)]]
            )
            q = np.zeros((2 * _N, 2 * _N))
            v = self.config.accel_m**2
            for k in range(_N):
                q[k, k] = v / 4 * dt**4
                q[k, k + _N] = q[k + _N, k] = v / 2 * dt**3
                q[k + _N, k + _N] = v * dt**2
            self._models[dt] = (F, q)
        return self._models[dt]

    def measure(self, matrix: np.ndarray | None) -> np.ndarray | None:
        """The eight coordinates of the four image points under ``matrix`` (``None`` when one is beyond its horizon)."""
        if matrix is None or not np.isfinite(matrix).all():
            return None
        xyw = np.column_stack([self.pixels, np.ones(4)]) @ np.asarray(matrix).T
        bottom = np.asarray(matrix) @ np.array(
            [self.pixels[2:, 0].mean(), self.pixels[2, 1], 1.0]
        )
        if (xyw[:, 2] * bottom[2] <= 0).any():
            return None
        return (xyw[:, :2] / xyw[:, 2:]).reshape(-1)

    def matrix(self, state: np.ndarray) -> np.ndarray:
        """The frame -> pitch matrix of a state (unit Frobenius norm)."""
        out = _four_point(self.pixels, np.asarray(state[:_N]).reshape(4, 2))
        return out / np.linalg.norm(out)

    def _record(self, record) -> None:
        if self.keep_history:
            if record is not None:  # the covariances packed (_pack): memory per frame
                x_pred, p_pred, x_post, p_post, restart, F = record
                record = (x_pred, _pack(p_pred), x_post, _pack(p_post), restart, F)
            self.records.append(record)

    def _reset(self, z: np.ndarray) -> None:
        self.x = np.concatenate([z, np.zeros(_N)])
        self.p = np.diag(
            [self.config.meas_m**2] * _N + [(4 * self.config.meas_m) ** 2] * _N
        )
        self.rejects, self.gap = 0, 0.0
        self._record(
            (self.x.copy(), self.p.copy(), self.x.copy(), self.p.copy(), True, None)
        )

    def drop(self) -> None:
        """End the current segment here (the caller stopped trusting it); the next fit starts a new one."""
        self.x = None
        self.rejects, self.gap = 0, 0.0

    def step(self, matrix: np.ndarray | None, dt: float = 1.0) -> np.ndarray | None:
        """This frame's matrix: the filtered state; the fit itself when it is rejected; ``None`` without a state.

        ``dt``: steps (``1 / rate_hz`` seconds) since the previous call.
        """
        if not dt > 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        c = self.config
        z = self.measure(matrix)
        if self.x is None:
            if z is None:
                self._record(None)
                return None
            self._reset(z)
            return self.matrix(self.x)
        F, Q = self._model(dt)
        x_pred = F @ self.x
        p_pred = F @ self.p @ F.T + Q
        if z is None:
            self.gap += dt
            if self.gap > c.max_gap:
                self.x = None
                self._record(None)
                return None
            self.x, self.p = x_pred, p_pred
            self._record((x_pred, p_pred, self.x.copy(), self.p.copy(), False, F))
            return self.matrix(self.x)
        s = self.H @ p_pred @ self.H.T + self.R
        innovation = z - self.H @ x_pred
        d2 = float(innovation @ np.linalg.solve(s, innovation))
        if d2 > c.gate_chi2:
            if self.rejects == 0:
                self._reject_start = len(self.records)
            self.rejects += 1
            if self.rejects >= c.reset_after:  # another view: start again from it
                # every frame since the run of rejects began belongs to the new view:
                # out of the old segment's RTS (they keep their own matrices)
                for i in range(self._reject_start, len(self.records)):
                    self.records[i] = None
                self._reset(z)
                return self.matrix(self.x)
            self.x, self.p = x_pred, p_pred
            self._record((x_pred, p_pred, self.x.copy(), self.p.copy(), False, F))
            # until a reset confirms another view, the frame keeps its own fit
            return np.asarray(matrix, float) / np.linalg.norm(matrix)
        gain = p_pred @ self.H.T @ np.linalg.inv(s)
        self.x = x_pred + gain @ innovation
        self.p = (np.eye(2 * _N) - gain @ self.H) @ p_pred
        self.p = (self.p + self.p.T) / 2
        self.rejects, self.gap = 0, 0.0
        self._record((x_pred, p_pred, self.x.copy(), self.p.copy(), False, F))
        return self.matrix(self.x)

    def smoothed(self) -> list[np.ndarray | None]:
        """RTS per segment, one entry per step (``None`` outside a segment)."""
        if not self.keep_history:
            raise RuntimeError("smoothed() needs HomographyFilter(keep_history=True)")
        out: list[np.ndarray | None] = [None] * len(self.records)
        later = None
        for k in range(len(self.records) - 1, -1, -1):
            rec = self.records[k]
            if rec is None:
                later = None
                continue
            nxt = self.records[k + 1] if k + 1 < len(self.records) else None
            _, _, x_post, p_post, _, _ = rec
            p_post = _unpack(p_post)
            if later is None or nxt is None or nxt[4]:
                xs, ps = x_post, p_post
            else:
                p_next = _unpack(nxt[1])
                gain = p_post @ nxt[5].T @ np.linalg.inv(p_next)
                xs = x_post + gain @ (later[0] - nxt[0])
                ps = p_post + gain @ (later[1] - p_next) @ gain.T
            later = (xs, ps)
            out[k] = self.matrix(xs)
        return out
