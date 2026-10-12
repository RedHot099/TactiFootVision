"""Inference pipeline: detection -> tracking -> pitch projection -> teams; single frames too."""

from tactifoot_vision.pipeline.frame import FrameAnalyzer, fill_ball
from tactifoot_vision.pipeline.pipeline import Pipeline
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    FrameResult,
    ObjectMasks,
    PipelineResult,
)

__all__ = [
    "NO_TEAM",
    "FrameAnalyzer",
    "FrameResult",
    "ObjectMasks",
    "Pipeline",
    "PipelineResult",
    "fill_ball",
]
