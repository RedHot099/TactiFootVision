"""The painted markings as lines and circles: what the line network predicts and the homography fit uses.

Each :class:`~tactifoot_vision.enums.PitchMarking` is a straight line (the
infinite line through its segment) or a circle (the penalty arcs are the
part of their circle outside the box; their points lie on the circle all
the same), on the standard 105 x 68 m pitch in corner-origin metres, the
frame :class:`SoccerPitch` uses. A pitch of other units (StatsBomb's 120 x 80)
keeps these proportions, so a homography found here is scaled to it.
"""

from dataclasses import dataclass

import numpy as np

from tactifoot_vision.enums import PitchMarking
from tactifoot_vision.pitch.pitch import SoccerPitch

STANDARD_PITCH = SoccerPitch()
_V = STANDARD_PITCH.vertices.astype(float)
CIRCLE_RADIUS = 9.15  # metres

# Straight marking -> (first, last) SoccerPitch vertex of its segment.
_SEGMENTS = {
    PitchMarking.SIDE_LINE_TOP: (0, 24),
    PitchMarking.SIDE_LINE_BOTTOM: (5, 29),
    PitchMarking.SIDE_LINE_LEFT: (0, 5),
    PitchMarking.SIDE_LINE_RIGHT: (24, 29),
    PitchMarking.MIDDLE_LINE: (13, 16),
    PitchMarking.BIG_RECT_LEFT_TOP: (1, 9),
    PitchMarking.BIG_RECT_LEFT_MAIN: (9, 12),
    PitchMarking.BIG_RECT_LEFT_BOTTOM: (4, 12),
    PitchMarking.BIG_RECT_RIGHT_TOP: (25, 17),
    PitchMarking.BIG_RECT_RIGHT_MAIN: (17, 20),
    PitchMarking.BIG_RECT_RIGHT_BOTTOM: (28, 20),
    PitchMarking.SMALL_RECT_LEFT_TOP: (2, 6),
    PitchMarking.SMALL_RECT_LEFT_MAIN: (6, 7),
    PitchMarking.SMALL_RECT_LEFT_BOTTOM: (3, 7),
    PitchMarking.SMALL_RECT_RIGHT_TOP: (26, 22),
    PitchMarking.SMALL_RECT_RIGHT_MAIN: (22, 23),
    PitchMarking.SMALL_RECT_RIGHT_BOTTOM: (27, 23),
}
_CIRCLES = {
    PitchMarking.CIRCLE_CENTRAL: (STANDARD_PITCH.length / 2, STANDARD_PITCH.width / 2),
    PitchMarking.CIRCLE_LEFT: tuple(_V[8]),  # the penalty spots
    PitchMarking.CIRCLE_RIGHT: tuple(_V[21]),
}
MARKINGS: tuple[PitchMarking, ...] = tuple(PitchMarking)
INDEX = {m: i for i, m in enumerate(MARKINGS)}
NUM_MARKINGS = len(MARKINGS)


@dataclass(frozen=True)
class Marking:
    """One marking's geometry in standard-pitch metres."""

    name: PitchMarking
    kind: str  # "line" or "circle"
    start: np.ndarray  # line: the segment's ends; circle: its centre twice
    end: np.ndarray
    centre: np.ndarray
    radius: float

    @property
    def line(self) -> np.ndarray:
        """Homogeneous pitch line (a, b, c), a x + b y + c = 0 with (a, b) a unit normal."""
        line = np.cross([*self.start, 1.0], [*self.end, 1.0])
        return line / np.hypot(line[0], line[1])

    @property
    def conic(self) -> np.ndarray:
        """The circle as a symmetric 3 x 3 conic, X^T C X = 0, scaled by 1 / radius."""
        cx, cy = self.centre
        r = self.radius
        c = np.array(
            [[1.0, 0, -cx], [0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy - r * r]]
        )
        return c / r


def _marking(name: PitchMarking) -> Marking:
    if name in _SEGMENTS:
        i, j = _SEGMENTS[name]
        return Marking(name, "line", _V[i], _V[j], np.zeros(2), 0.0)
    c = np.array(_CIRCLES[name], float)
    return Marking(name, "circle", c, c, c, CIRCLE_RADIUS)


GEOMETRY: tuple[Marking, ...] = tuple(_marking(m) for m in MARKINGS)


def _flip_along(name: str) -> str:
    if "left" in name:
        return name.replace("left", "right")
    return name.replace("right", "left")


def _flip_across(name: str) -> str:
    if name.endswith("top"):
        return name.removesuffix("top") + "bottom"
    if name.endswith("bottom"):
        return name.removesuffix("bottom") + "top"
    return name


# A horizontal image flip mirrors the pitch along its length in a main-camera
# view (left <-> right) and across it in a view along the pitch (top <-> bottom).
FLIP_ALONG: tuple[int, ...] = tuple(
    INDEX[PitchMarking(_flip_along(m.value))] for m in MARKINGS
)
FLIP_ACROSS: tuple[int, ...] = tuple(
    INDEX[PitchMarking(_flip_across(m.value))] for m in MARKINGS
)


def standard_to_pitch(pitch: SoccerPitch) -> np.ndarray:
    """The 3 x 3 map from standard-pitch metres to ``pitch``'s units (identity for the default pitch)."""
    return np.diag(
        [pitch.length / STANDARD_PITCH.length, pitch.width / STANDARD_PITCH.width, 1.0]
    )


def marking_distance(marking: np.ndarray, pitch_xy: np.ndarray) -> np.ndarray:
    """Distance (standard-pitch metres) of each pitch point to its marking's line or circle.

    ``marking`` holds indices into :data:`MARKINGS`; a straight marking counts
    as its whole line, a circle as its whole circle.
    """
    marking = np.asarray(marking, dtype=int)
    p = np.asarray(pitch_xy, dtype=float).reshape(-1, 2)
    out = np.full(len(p), np.inf)
    for k in np.unique(marking):
        m, rows = GEOMETRY[k], marking == k
        if m.kind == "line":
            out[rows] = np.abs(p[rows] @ m.line[:2] + m.line[2])
        else:
            out[rows] = np.abs(np.linalg.norm(p[rows] - m.centre, axis=1) - m.radius)
    return out
