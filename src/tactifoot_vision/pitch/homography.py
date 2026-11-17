"""Frame <-> pitch homography from the painted pitch markings."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from tactifoot_vision.pitch.fit import FitConfig, LinePoints, select_points, solve
from tactifoot_vision.pitch.markings import standard_to_pitch
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.pitch.projection import project_points
from tactifoot_vision.pitch.temporal import FilterConfig, HomographyFilter

_DEFAULT_FILTER = FilterConfig()  # the declared temporal layer (ADR 0005)


def frame_to_pitch(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` frame pixels to pitch coordinates with a frame->pitch homography."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def pitch_to_frame(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` pitch coordinates to frame pixels (inverse of :func:`frame_to_pitch`)."""
    return frame_to_pitch(points, np.linalg.inv(homography))


def visible_area(
    homography: np.ndarray | None, frame_size: tuple[int, int]
) -> np.ndarray | None:
    """The part of the pitch a frame shows: its outline through a frame->pitch homography.

    ``frame_size`` is the frame's ``(width, height)``. Returns the ``(4, 2)``
    pitch positions of the frame's corners, clockwise on screen from the
    top-left one, taken at the pixel edges (``(0, 0)`` to ``(width, height)``)
    as :func:`~tactifoot_vision.viz.warp_to_pitch` warps them. ``None`` when
    part of the frame lies beyond the horizon, where the outline has no finite
    pitch position, or without a homography.
    """
    if homography is None:
        return None
    width, height = frame_size
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    outline, in_front = project_points(corners, homography, (width / 2, height / 2))
    return outline if in_front.all() else None


@dataclass(frozen=True)
class HomographyFit:
    """How the current frame's homography came about (:attr:`HomographyEstimator.fit`).

    Attributes:
        fresh: the matrix was fitted on this frame; ``False`` when it is
            held from an earlier frame or there is none.
        age: frames since the last fresh fit (0 when fresh).
        points: line points the frame's fit was given.
        uncertainty_m: the frame's fit's position uncertainty over the lower
            frame (metres on the standard pitch, one sigma); ``inf`` without
            a fit.
        inlier_share: share of the points the fit kept within ``inlier_px``
            of their marking (``FitConfig.inlier_px``, 3 px by default; pixels
            of the frame scaled to 960 wide). After the gated
            second pass these are the points that passed the gate, not all
            ``points``.
        start: where the winning fit started (``"lines"``, ``"previous"``,
            ``"template"``, ... with ``"+gated"`` after a second pass); empty
            without one.
        reason: why the frame has no fresh fit (``"no evidence"``,
            ``"undetermined"``, ...); empty when it has one.
        matrix: the estimator's matrix before the temporal filter (frame ->
            pitch units): the frame's own fit when fresh, else the held one;
            ``None`` without either. The whole-video pass
            (:func:`~tactifoot_vision.pipeline.smooth_homographies`) reads
            these.
    """

    fresh: bool
    age: int
    points: int
    uncertainty_m: float
    inlier_share: float
    start: str
    reason: str
    matrix: np.ndarray | None = field(default=None, compare=False, repr=False)


class HomographyEstimator:
    """Fits a frame->pitch homography per frame to the line network's points, causally.

    Per frame, the points of at least ``min_prob`` (at most ``per_class``
    per marking, spread along it) go to one robust fit
    (:func:`~tactifoot_vision.pitch.fit.solve`, tuned by ``fit``: a
    :class:`~tactifoot_vision.pitch.fit.FitConfig` or its fields), started
    from the points themselves and from the previous matrix; without a
    previous matrix it also tries a bank of camera templates. A fit the
    evidence does not determine, or one that is mirrored or puts the points
    beyond its horizon, is refused.

    When the fit started with the previous matrix is refused (a cut, say),
    the frame is solved again on its own, templates included, and that fit
    is taken when its uncertainty is at most half the rejection limit
    (``fit.max_uncertainty_m / 2``, :attr:`restart_uncertainty_m`).

    A frame without a fit keeps the last matrix for up to ``max_age_s``
    seconds (``None``: forever), flagged not fresh in :attr:`fit`, and the
    last matrix stays the next fit's start. After the template search ran
    and found nothing it is skipped for 1, 2, 4, ... processed frames, up to
    ``backoff_max_s`` seconds (0: never skipped), which bounds the cost of a
    run of frames that show no usable markings. Both are counted in
    processed frames at the frame rate :meth:`reset` takes (rounded down, at
    least one frame for a positive value):
    at 25 fps, 2 s is 50 frames and 1.28 s is 32; at stride 2, 25 and 16.

    ``temporal_filter`` (a :class:`~tactifoot_vision.pitch.temporal.FilterConfig`
    or its fields; ``None``: off) runs the fresh fits through a
    constant-velocity Kalman filter, causally: the returned matrix is the
    filtered one, a fit the filter rejects is returned as it is, and a cut
    resets the filter (ADR 0005). A frame without a fit then gets the
    filter's prediction for up to its ``max_gap`` steps, then the held
    matrix; nothing outlives ``max_age_s``. The filter keeps no history, so
    memory does not grow with the video. Its settings are per 1/25 s:
    :meth:`reset` takes the processed frame rate (``fps / stride``; the
    pipeline passes it) and the filter steps in physical time. :attr:`fit`
    keeps the matrix before the filter. The whole-video pass on a saved result is the filter's RTS smoother,
    :func:`~tactifoot_vision.pipeline.smooth_homographies` (ADR 0002).

    The returned matrices map frame pixels to ``pitch`` coordinates.
    """

    def __init__(
        self,
        pitch: SoccerPitch | None = None,
        *,
        min_prob: float = 0.3,
        per_class: int = 24,
        max_age_s: float | None = 2.0,
        backoff_max_s: float = 1.28,
        fit: FitConfig | Mapping[str, Any] | None = None,
        temporal_filter: FilterConfig | Mapping[str, Any] | None = _DEFAULT_FILTER,
    ) -> None:
        if not 0 <= min_prob <= 1:
            raise ValueError(f"min_prob must be in [0, 1], got {min_prob}")
        if per_class < 2:
            raise ValueError(f"per_class must be >= 2, got {per_class}")
        if max_age_s is not None and not max_age_s >= 0:
            raise ValueError(f"max_age_s must be None or >= 0, got {max_age_s}")
        if not backoff_max_s >= 0:
            raise ValueError(f"backoff_max_s must be >= 0, got {backoff_max_s}")
        self.pitch = pitch or SoccerPitch()
        self.min_prob = min_prob
        self.per_class = per_class
        self.max_age_s = max_age_s
        self.backoff_max_s = backoff_max_s
        self.fit_config = (
            fit if isinstance(fit, FitConfig) else FitConfig(**dict(fit or {}))
        )
        self.temporal_filter = (
            None
            if temporal_filter is None
            else temporal_filter
            if isinstance(temporal_filter, FilterConfig)
            else FilterConfig(**dict(temporal_filter))
        )
        self.reset()

    @property
    def restart_uncertainty_m(self) -> float:
        """The uncertainty a fit made without the previous matrix may have to replace a refused one."""
        return self.fit_config.max_uncertainty_m / 2

    def reset(self, frame_rate: float | None = None) -> None:
        """Forget the previous frames (a new video) processed at ``frame_rate`` per second (``None``: 25)."""
        if frame_rate is not None and not frame_rate > 0:
            raise ValueError(f"frame_rate must be > 0, got {frame_rate}")
        rate = self.temporal_filter.rate_hz if self.temporal_filter else 25.0
        self._dt = 1.0 if frame_rate is None else rate / frame_rate
        frames = frame_rate or 25.0  # processed frames per second
        self._max_age = (
            None if self.max_age_s is None else _frames(self.max_age_s, frames)
        )
        self._backoff_max = _frames(self.backoff_max_s, frames)
        self.matrix: np.ndarray | None = None
        self.fit: HomographyFit | None = None
        self._last: np.ndarray | None = None  # standard-pitch metres
        self._age = 0
        self._frame = 0
        self._skip_until, self._wait = -1, 1
        self._filter: HomographyFilter | None = None
        self._filter_size: tuple[int, int] | None = None

    def update(
        self, points: LinePoints | None, frame_size: tuple[int, int]
    ) -> np.ndarray | None:
        """Fit this frame's line points (frame pixels of a ``(width, height)`` frame); returns the current matrix."""
        frame = self._frame
        self._frame += 1
        evidence = select_points(
            points if points is not None else LinePoints.empty(),
            frame_size,
            min_prob=self.min_prob,
            per_class=self.per_class,
        )
        templates = self._backoff_max == 0 or frame >= self._skip_until
        fit = solve(evidence, self.fit_config, previous=self._last, templates=templates)
        if fit.matrix is None and self._last is not None:
            # Refused with the previous matrix as a start, as on the first frame after
            # a cut: the frame on its own, taken only when it is well determined.
            alone = solve(evidence, self.fit_config, previous=None, templates=templates)
            if (
                alone.matrix is not None
                and alone.uncertainty_m <= self.restart_uncertainty_m
            ):
                fit = alone
            elif alone.tried_templates:
                fit.tried_templates = True
        if fit.matrix is not None:
            self._last, self._age, self._wait = fit.matrix, 0, 1
        else:
            if self._backoff_max and fit.tried_templates:
                self._skip_until = frame + self._wait
                self._wait = min(2 * self._wait, self._backoff_max)
            if self._last is not None:
                self._age += 1
                if self._max_age is not None and self._age > self._max_age:
                    self._last = None
        fresh = fit.matrix is not None
        self.fit = HomographyFit(
            fresh=fresh,
            age=self._age,
            points=len(evidence.pt_class),
            uncertainty_m=float(fit.uncertainty_m),
            inlier_share=float(fit.inlier_share),
            start=fit.start if fresh else "",
            reason=fit.reason,
            matrix=self._in_pitch_units(self._last),
        )
        emitted = self._last
        if self.temporal_filter is not None:
            kalman = self._temporal(frame_size)
            if self._last is None:  # no fit to hold (none yet, or older than max_age_s)
                kalman.drop()
            filtered = kalman.step(fit.matrix, self._dt)
            if filtered is not None:  # without a state: the fit, or the held matrix
                emitted = filtered
        self.matrix = self._in_pitch_units(emitted)
        return self.matrix

    def _temporal(self, frame_size: tuple[int, int]) -> HomographyFilter:
        size = (int(frame_size[0]), int(frame_size[1]))
        if self._filter is None or self._filter_size != size:
            self._filter = HomographyFilter(
                size, self.temporal_filter, keep_history=False
            )
            self._filter_size = size
        return self._filter

    def _in_pitch_units(self, matrix: np.ndarray | None) -> np.ndarray | None:
        return None if matrix is None else standard_to_pitch(self.pitch) @ matrix

    def __repr__(self) -> str:
        return (
            f"HomographyEstimator(pitch={self.pitch!r}, min_prob={self.min_prob}, "
            f"per_class={self.per_class}, max_age_s={self.max_age_s}, "
            f"backoff_max_s={self.backoff_max_s}, fit={self.fit_config!r}, "
            f"temporal_filter={self.temporal_filter!r})"
        )


def _frames(seconds: float, frame_rate: float) -> int:
    """Whole processed frames in ``seconds`` at ``frame_rate``: rounded down (exact multiples stay exact), at least one when ``seconds > 0``."""
    whole = int(np.floor(seconds * frame_rate + 1e-9))
    return max(whole, 1) if seconds > 0 else 0
