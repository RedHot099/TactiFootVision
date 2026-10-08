"""Inference pipeline: detection -> tracking -> pitch projection -> teams."""

from tactifoot_vision.pipeline.pipeline import Pipeline
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    FrameResult,
    ObjectMasks,
    PipelineResult,
)

__all__ = ["NO_TEAM", "FrameResult", "ObjectMasks", "Pipeline", "PipelineResult"]
