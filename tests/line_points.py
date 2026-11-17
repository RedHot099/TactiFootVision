"""Synthetic line evidence shared by the tests: the points a perfect line model would find for a known camera."""

import numpy as np

from tactifoot_vision.enums import PitchMarking
from tactifoot_vision.pitch import LinePoints, SoccerPitch
from tactifoot_vision.pitch.markings import GEOMETRY, INDEX, MARKINGS, standard_to_pitch


def marking_points(name: PitchMarking, step: float = 0.5) -> np.ndarray:
    """Points every ``step`` metres along a marking (standard pitch); the penalty arcs outside their box only."""
    m = GEOMETRY[INDEX[name]]
    if m.kind == "line":
        n = max(2, int(np.hypot(*(m.end - m.start)) / step) + 1)
        t = np.linspace(0, 1, n)[:, None]
        return m.start + t * (m.end - m.start)
    angles = np.linspace(0, 2 * np.pi, int(2 * np.pi * m.radius / step), endpoint=False)
    pts = m.centre + m.radius * np.column_stack([np.cos(angles), np.sin(angles)])
    if name == PitchMarking.CIRCLE_LEFT:
        pts = pts[pts[:, 0] > 16.5]
    elif name == PitchMarking.CIRCLE_RIGHT:
        pts = pts[pts[:, 0] < 105 - 16.5]
    return pts


def project(
    matrix: np.ndarray, pitch_xy: np.ndarray, size
) -> tuple[np.ndarray, np.ndarray]:
    """Pitch points into the frame of a frame -> pitch ``matrix``: (pixels, inside the frame and in front)."""
    g = np.linalg.inv(matrix)
    w, h = size
    front = matrix @ np.array([w / 2, h - 1, 1.0])
    ref = np.sign(g[2] @ (front / front[2]))
    p = np.column_stack([pitch_xy, np.ones(len(pitch_xy))]) @ g.T
    ok = p[:, 2] * ref > 1e-12
    xy = p[:, :2] / np.where(ok, p[:, 2], 1.0)[:, None]
    inside = ok & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return xy, inside


def line_points(
    frame_to_pitch: np.ndarray,
    size: tuple[int, int],
    pitch: SoccerPitch | None = None,
    names=None,
    per_class: int = 20,
    noise_px: float = 0.0,
    prob: float = 0.9,
    rng=None,
) -> LinePoints:
    """What a perfect line model finds in a ``size`` frame of ``frame_to_pitch`` (into ``pitch`` units)."""
    rng = rng or np.random.default_rng(0)
    standard = np.linalg.inv(standard_to_pitch(pitch or SoccerPitch())) @ frame_to_pitch
    cls, pts = [], []
    for name in names or MARKINGS:
        xy, ok = project(standard, marking_points(PitchMarking(name)), size)
        xy = xy[ok]
        if len(xy) < 2:
            continue
        take = np.linspace(0, len(xy) - 1, min(per_class, len(xy))).astype(int)
        pts.append(xy[take] + rng.normal(0, noise_px, (len(take), 2)))
        cls.append(np.full(len(take), INDEX[PitchMarking(name)]))
    if not pts:
        return LinePoints.empty()
    cls = np.concatenate(cls)
    return LinePoints(cls, np.vstack(pts), np.full(len(cls), prob))
