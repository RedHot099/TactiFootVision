"""Inference pipeline: detection -> tracking -> pitch projection -> teams."""

from tactifoot_vision.pipeline.pipeline import Pipeline
from tactifoot_vision.pipeline.postprocess import (
    POSTPROCESSES,
    remap_track_ids,
    reproject,
)
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    AppliedPostProcess,
    FrameResult,
    ObjectMasks,
    PipelineResult,
)

__all__ = [
    "NO_TEAM",
    "POSTPROCESSES",
    "AppliedPostProcess",
    "FrameResult",
    "ObjectMasks",
    "Pipeline",
    "PipelineResult",
    "remap_track_ids",
    "reproject",
]
