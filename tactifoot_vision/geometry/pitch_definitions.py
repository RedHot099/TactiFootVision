# tactifoot_vision/geometry/pitch_definitions.py
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    length: float
    width: float

    _PENALTY_BOX_LENGTH_RATIO: float = 16.5 / 105.0
    _PENALTY_BOX_WIDTH_RATIO: float = 40.32 / 68.0
    _GOAL_BOX_LENGTH_RATIO: float = 5.5 / 105.0
    _GOAL_BOX_WIDTH_RATIO: float = 18.32 / 68.0
    _CENTRE_CIRCLE_RADIUS_RATIO: float = 9.15 / 68.0
    _PENALTY_SPOT_DISTANCE_RATIO: float = 11.0 / 105.0

    penalty_box_length: float = field(init=False)
    penalty_box_width: float = field(init=False)
    goal_box_length: float = field(init=False)
    goal_box_width: float = field(init=False)
    centre_circle_radius: float = field(init=False)
    penalty_spot_distance: float = field(init=False)

    def __post_init__(self):
        self.penalty_box_length = self.length * self._PENALTY_BOX_LENGTH_RATIO
        self.penalty_box_width = self.width * self._PENALTY_BOX_WIDTH_RATIO
        self.goal_box_length = self.length * self._GOAL_BOX_LENGTH_RATIO
        self.goal_box_width = self.width * self._GOAL_BOX_WIDTH_RATIO
        self.centre_circle_radius = self.width * self._CENTRE_CIRCLE_RADIUS_RATIO
        self.penalty_spot_distance = self.length * self._PENALTY_SPOT_DISTANCE_RATIO

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        half_width = self.width / 2.0
        half_length = self.length / 2.0
        half_penalty_width = self.penalty_box_width / 2.0
        half_goal_width = self.goal_box_width / 2.0

        return [
            (0.0, 0.0),  # 0
            (0.0, half_width - half_penalty_width),  # 1
            (0.0, half_width - half_goal_width),  # 2
            (0.0, half_width + half_goal_width),  # 3
            (0.0, half_width + half_penalty_width),  # 4
            (0.0, self.width),  # 5
            (self.goal_box_length, half_width - half_goal_width),  # 6
            (self.goal_box_length, half_width + half_goal_width),  # 7
            (self.penalty_spot_distance, half_width),  # 8
            (self.penalty_box_length, half_width - half_penalty_width),  # 9
            (self.penalty_box_length, half_width - half_goal_width),  # 10
            (self.penalty_box_length, half_width + half_goal_width),  # 11
            (self.penalty_box_length, half_width + half_penalty_width),  # 12
            (half_length, 0.0),  # 13
            (
                half_length,
                half_width - self.centre_circle_radius,
            ),  # 14 - Not drawn as line
            (
                half_length,
                half_width + self.centre_circle_radius,
            ),  # 15 - Not drawn as line
            (half_length, self.width),  # 16
            (
                self.length - self.penalty_box_length,
                half_width - half_penalty_width,
            ),  # 17
            (self.length - self.penalty_box_length, half_width - half_goal_width),  # 18
            (self.length - self.penalty_box_length, half_width + half_goal_width),  # 19
            (
                self.length - self.penalty_box_length,
                half_width + half_penalty_width,
            ),  # 20
            (self.length - self.penalty_spot_distance, half_width),  # 21
            (self.length - self.goal_box_length, half_width - half_goal_width),  # 22
            (self.length - self.goal_box_length, half_width + half_goal_width),  # 23
            (self.length, 0.0),  # 24
            (self.length, half_width - half_penalty_width),  # 25
            (self.length, half_width - half_goal_width),  # 26
            (self.length, half_width + half_goal_width),  # 27
            (self.length, half_width + half_penalty_width),  # 28
            (self.length, self.width),  # 29
            (
                half_length - self.centre_circle_radius,
                half_width,
            ),  # 30 - Not drawn as line
            (
                half_length + self.centre_circle_radius,
                half_width,
            ),  # 31 - Not drawn as line
        ]

    # --- CORRECTED AND COMPLETED EDGES LIST ---
    # Indices refer to the order in the vertices list above
    edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            # Outer boundary
            (0, 5),  # Left sideline
            (5, 29),  # Top sideline
            (29, 24),  # Right sideline
            (24, 0),  # Bottom sideline
            # Midfield line
            (13, 16),
            # Left Penalty area
            (1, 9),
            (9, 12),
            (12, 4),
            # Left Goal area (6-yard box)
            (2, 6),
            (6, 7),
            (7, 3),
            # Right Penalty area
            (25, 17),
            (17, 20),
            (20, 28),
            # Right Goal area (6-yard box)
            (26, 22),
            (22, 23),
            (23, 27),
            # Connect penalty/goal areas to outer boundary if needed (optional visual style)
            # (1, 0), (4, 5), (25, 24), (28, 29) # If you want lines from box corners to sidelines
        ]
    )
    # -----------------------------------------

    labels: List[str] = field(
        default_factory=lambda: [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ]  # Adjusted labels to be 0-indexed for clarity
    )
