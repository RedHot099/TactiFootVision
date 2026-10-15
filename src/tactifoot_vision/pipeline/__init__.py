"""Inference pipeline: detection -> tracking -> pitch projection -> teams."""

from tactifoot_vision.pipeline.ball import track_ball
from tactifoot_vision.pipeline.pipeline import Pipeline
from tactifoot_vision.pipeline.postprocess import (
    POSTPROCESSES,
    remap_track_ids,
    reproject,
)
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    ROLES,
    AppliedPostProcess,
    FrameResult,
    ObjectMasks,
    PipelineResult,
)
from tactifoot_vision.pipeline.roles import assign_roles
from tactifoot_vision.pipeline.tracks import join_tracks

__all__ = [
    "NO_TEAM",
    "POSTPROCESSES",
    "ROLES",
    "AppliedPostProcess",
    "FrameResult",
    "ObjectMasks",
    "Pipeline",
    "PipelineResult",
    "assign_roles",
    "join_tracks",
    "remap_track_ids",
    "reproject",
    "track_ball",
]
