"""Pitch landmark coordinates for the positional features.

Copied from the pre-rewrite ``tactifoot_vision.geometry.pitch_definitions`` so
earlier results stay reproducible. The rewritten package's ``SoccerPitch`` puts
keypoints 10, 11, 18, 19 at the penalty-arc intersection (up to 1.85 m away),
so it is deliberately not used here. Only the vertices are kept.
"""

import numpy as np

# Markings as ratios of a standard 105 m x 68 m pitch.
_PENALTY_BOX_LENGTH = 16.5 / 105.0
_PENALTY_BOX_WIDTH = 40.32 / 68.0
_GOAL_BOX_LENGTH = 5.5 / 105.0
_GOAL_BOX_WIDTH = 18.32 / 68.0
_CENTRE_CIRCLE_RADIUS = 9.15 / 68.0
_PENALTY_SPOT_DISTANCE = 11.0 / 105.0


def pitch_vertices(length: float, width: float) -> np.ndarray:
    """``(32, 2)`` float32 landmark coordinates; index ``i`` is keypoint ``i``."""
    pb_len, pb_half = length * _PENALTY_BOX_LENGTH, width * _PENALTY_BOX_WIDTH / 2
    gb_len, gb_half = length * _GOAL_BOX_LENGTH, width * _GOAL_BOX_WIDTH / 2
    circle = width * _CENTRE_CIRCLE_RADIUS
    spot = length * _PENALTY_SPOT_DISTANCE
    hw, hl = width / 2, length / 2
    points = [
        (0, 0), (0, hw - pb_half), (0, hw - gb_half), (0, hw + gb_half), (0, hw + pb_half), (0, width),
        (gb_len, hw - gb_half), (gb_len, hw + gb_half),
        (spot, hw),
        (pb_len, hw - pb_half), (pb_len, hw - gb_half), (pb_len, hw + gb_half), (pb_len, hw + pb_half),
        (hl, 0), (hl, hw - circle), (hl, hw + circle), (hl, width),
        (length - pb_len, hw - pb_half), (length - pb_len, hw - gb_half),
        (length - pb_len, hw + gb_half), (length - pb_len, hw + pb_half),
        (length - spot, hw),
        (length - gb_len, hw - gb_half), (length - gb_len, hw + gb_half),
        (length, 0), (length, hw - pb_half), (length, hw - gb_half),
        (length, hw + gb_half), (length, hw + pb_half), (length, width),
        (hl - circle, hw), (hl + circle, hw),
    ]  # fmt: skip
    return np.asarray(points, dtype=np.float32)
