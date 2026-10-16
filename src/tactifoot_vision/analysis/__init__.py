"""Match analysis: pure functions over a :class:`~tactifoot_vision.pipeline.PipelineResult`."""

from tactifoot_vision.analysis.direction import attack_direction
from tactifoot_vision.analysis.space import SpaceControl, space_control

__all__ = ["SpaceControl", "attack_direction", "space_control"]
