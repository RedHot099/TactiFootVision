"""Match analysis: pure functions over a :class:`~tactifoot_vision.pipeline.PipelineResult`."""

from tactifoot_vision.analysis.direction import attack_direction
from tactifoot_vision.analysis.space import SpaceControl, space_control
from tactifoot_vision.analysis.velocity import velocities

__all__ = ["SpaceControl", "attack_direction", "space_control", "velocities"]
