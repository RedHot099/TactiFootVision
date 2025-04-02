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

        # Vertex indices are 0-based
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
            (half_length, half_width - self.centre_circle_radius),  # 14
            (half_length, half_width + self.centre_circle_radius),  # 15
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
            (half_length - self.centre_circle_radius, half_width),  # 30
            (half_length + self.centre_circle_radius, half_width),  # 31
        ]

    edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),  # Left goal line segments (Indices 0-5)
            (6, 7),  # Left 6-yard box inner line (Indices 6-7)
            (9, 10),
            (10, 11),
            (11, 12),  # Left penalty box inner lines (Indices 9-12)
            (13, 14),
            (14, 15),
            (15, 16),  # Midfield line segments (Indices 13-16)
            (17, 18),
            (18, 19),
            (19, 20),  # Right penalty box inner lines (Indices 17-20)
            (22, 23),  # Right 6-yard box inner line (Indices 22-23)
            (24, 25),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),  # Right goal line segments (Indices 24-29)
            (0, 13),  # Bottom sideline left half
            (1, 9),  # Left penalty box side (bottom)
            (2, 6),  # Left 6-yard box side (bottom)
            (3, 7),  # Left 6-yard box side (top)
            (4, 12),  # Left penalty box side (top)
            (5, 16),  # Top sideline left half
            (13, 24),  # Bottom sideline right half
            (17, 25),  # Right penalty box side (bottom)
            (22, 26),  # Right 6-yard box side (bottom)
            (23, 27),  # Right 6-yard box side (top)
            (20, 28),  # Right penalty box side (top)
            (16, 29),  # Top sideline right half
        ]
    )

    edges_working_for_main_frame_1_based = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (7, 8),
        (10, 11),
        (11, 12),
        (12, 13),
        (14, 15),
        (15, 16),
        (16, 17),
        (18, 19),
        (19, 20),
        (20, 21),
        (23, 24),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (1, 14),
        (2, 10),
        (3, 7),
        (4, 8),
        (5, 13),
        (6, 17),
        (14, 25),
        (18, 26),
        (23, 27),
        (24, 28),
        (21, 29),
        (17, 30),
    ]

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
        ]
    )
