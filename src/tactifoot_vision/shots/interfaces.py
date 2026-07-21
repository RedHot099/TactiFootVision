from collections.abc import Sequence
from typing import Protocol

from tactifoot_vision.ball import BallTrajectory
from tactifoot_vision.domain import FrameResult
from tactifoot_vision.shots.results import ShotCandidate


class ShotDetector(Protocol):
    def detect(
        self,
        ball_trajectory: BallTrajectory,
        frame_results: Sequence[FrameResult] = (),
    ) -> tuple[ShotCandidate, ...]:
        raise NotImplementedError
