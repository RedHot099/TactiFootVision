"""Coordinate conventions of the detector features (xg-detector spec, Extraction; tests in test_xgd_conventions.py).

Pitch metres are the package's: `SoccerPitch()` is 105 x 68 m with the origin in a corner, x along the length and y
across. On broadcast frames y grows towards the camera, so y = 0 is the far touchline. The attacked goal is the goal
nearer the shot location. "Aligned" coordinates rotate the pitch by 180 degrees when the attacked goal is at x = 0, so
the attacked goal is always at x = 105 and the attacker's left stays at small y (a rotation, never a mirror image).

StatsBomb (audit only) draws a 120 x 80 pitch with the attacking team always towards x = 120, origin top left, so in
the aligned frame its y = 0 is also the attacker's left. Its markings near the goal are in yards (posts at y = 36 and
44, box 18 yards deep and 44 wide), so the conversion keeps yards as yards: metres = goal centre + 0.9144 x the yard
offset from StatsBomb's goal centre (120, 40). Scaling the whole pitch instead (105/120 along x, 68/80 across) would put
the posts 6.8 m apart and the box edge 0.7 m too deep.
"""

import numpy as np

from tactifoot_vision.pitch import SoccerPitch

PITCH = SoccerPitch()
LENGTH, WIDTH = PITCH.length, PITCH.width
GOAL_HALF_WIDTH = 7.32 / 2
GOAL = np.array([LENGTH, WIDTH / 2])          # centre of the attacked goal in aligned coordinates
POSTS = (np.array([LENGTH, WIDTH / 2 - GOAL_HALF_WIDTH]), np.array([LENGTH, WIDTH / 2 + GOAL_HALF_WIDTH]))
OTHER_GOAL = np.array([0.0, WIDTH / 2])

YARD = 0.9144
STATSBOMB_GOAL = np.array([120.0, 40.0])


def attacks_right(shot_xy) -> bool:
    """Whether the goal nearer the shot location is the one at x = 105 (ties go to x = 105)."""
    x = float(np.asarray(shot_xy, dtype=float)[0])
    return abs(LENGTH - x) <= abs(x)


def align(xy, attack_right: bool) -> np.ndarray:
    """Pitch points (`(2,)` or `(N, 2)`) in aligned coordinates: unchanged when attacking x = 105, else rotated 180 deg."""
    xy = np.asarray(xy, dtype=float)
    return xy.copy() if attack_right else np.array([LENGTH, WIDTH]) - xy


def statsbomb_to_metres(xy) -> np.ndarray:
    """StatsBomb pitch units (120 x 80, attacking x = 120) -> aligned metres; yards kept as yards around the goal centre."""
    return GOAL + (np.asarray(xy, dtype=float) - STATSBOMB_GOAL) * YARD
