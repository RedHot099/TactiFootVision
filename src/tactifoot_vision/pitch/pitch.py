"""Pitch model: the 32 landmark vertices the keypoint models predict, and the line layout."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# Markings as fractions of a standard 105 m x 68 m pitch, so any pitch size
# (e.g. StatsBomb's 120 x 80 units) keeps the same proportions.
_STD_LENGTH, _STD_WIDTH = 105.0, 68.0
_PENALTY_BOX = (16.5 / _STD_LENGTH, 40.32 / _STD_WIDTH)
_GOAL_BOX = (5.5 / _STD_LENGTH, 18.32 / _STD_WIDTH)
_CENTRE_CIRCLE_RADIUS = 9.15 / _STD_WIDTH
_PENALTY_SPOT = 11.0 / _STD_LENGTH

# Pairs of vertex indices joined by a painted line.
EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # left goal line
    (6, 7),  # left goal box front
    (9, 10), (10, 11), (11, 12),  # left penalty box front
    (13, 14), (14, 15), (15, 16),  # halfway line
    (17, 18), (18, 19), (19, 20),  # right penalty box front
    (22, 23),  # right goal box front
    (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),  # right goal line
    (0, 13), (13, 24), (5, 16), (16, 29),  # touch lines
    (1, 9), (4, 12), (2, 6), (3, 7),  # left box sides
    (17, 25), (20, 28), (22, 26), (23, 27),  # right box sides
)  # fmt: skip


@dataclass(frozen=True)
class SoccerPitch:
    """A pitch of ``length`` x ``width`` units with the origin in a corner.

    ``x`` runs along the length (goal to goal), ``y`` across the width. The
    vertex order matches the pitch-keypoint dataset (index ``i`` = keypoint ``i``).
    """

    length: float = _STD_LENGTH
    width: float = _STD_WIDTH

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError(
                f"Pitch length and width must be > 0, got {self.length} x {self.width}"
            )

    @property
    def penalty_box_length(self) -> float:
        return self.length * _PENALTY_BOX[0]

    @property
    def penalty_box_width(self) -> float:
        return self.width * _PENALTY_BOX[1]

    @property
    def goal_box_length(self) -> float:
        return self.length * _GOAL_BOX[0]

    @property
    def goal_box_width(self) -> float:
        return self.width * _GOAL_BOX[1]

    @property
    def centre_circle_radius(self) -> float:
        return self.width * _CENTRE_CIRCLE_RADIUS

    @property
    def penalty_spot_distance(self) -> float:
        return self.length * _PENALTY_SPOT

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return EDGES

    @cached_property
    def vertices(self) -> np.ndarray:
        """``(32, 2)`` float32 landmark coordinates in pitch units."""
        L, W = self.length, self.width
        hw = W / 2
        pb, gb = self.penalty_box_width / 2, self.goal_box_width / 2
        pl, gl = self.penalty_box_length, self.goal_box_length
        spot, r = self.penalty_spot_distance, self.centre_circle_radius
        # Keypoints 10, 11, 18, 19 are labelled where the penalty arc meets the
        # penalty-box line (checked against the dataset), not at goal-box width.
        arc = (r**2 - (pl - spot) ** 2) ** 0.5
        points = [
            (0, 0), (0, hw - pb), (0, hw - gb), (0, hw + gb), (0, hw + pb), (0, W),  # 0-5
            (gl, hw - gb), (gl, hw + gb),  # 6-7
            (spot, hw),  # 8
            (pl, hw - pb), (pl, hw - arc), (pl, hw + arc), (pl, hw + pb),  # 9-12
            (L / 2, 0), (L / 2, hw - r), (L / 2, hw + r), (L / 2, W),  # 13-16
            (L - pl, hw - pb), (L - pl, hw - arc), (L - pl, hw + arc), (L - pl, hw + pb),  # 17-20
            (L - spot, hw),  # 21
            (L - gl, hw - gb), (L - gl, hw + gb),  # 22-23
            (L, 0), (L, hw - pb), (L, hw - gb), (L, hw + gb), (L, hw + pb), (L, W),  # 24-29
            (L / 2 - r, hw), (L / 2 + r, hw),  # 30-31
        ]  # fmt: skip
        return np.asarray(points, dtype=np.float32)

    @property
    def labels(self) -> list[str]:
        return [f"{i:02d}" for i in range(len(self.vertices))]

    @property
    def num_keypoints(self) -> int:
        return len(self.vertices)
