from typing import Protocol

from tactifoot_vision.ball.results import BallTrajectory
from tactifoot_vision.domain import PipelineResult


class BallTrajectoryReconstructor(Protocol):
    def reconstruct(self, result: PipelineResult) -> BallTrajectory:
        raise NotImplementedError
