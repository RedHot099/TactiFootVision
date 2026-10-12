"""Team assignment from player crops: crop extraction, embedders, clustering, single-frame teams."""

from tactifoot_vision.teams.classifier import NO_TEAM, TeamClassifier
from tactifoot_vision.teams.crops import extract_crops
from tactifoot_vision.teams.embedders import (
    EMBEDDERS,
    Embedder,
    ResNetEmbedder,
    SigLIPEmbedder,
    available_embedders,
)
from tactifoot_vision.teams.frame import assign_frame_teams

__all__ = [
    "EMBEDDERS",
    "NO_TEAM",
    "Embedder",
    "ResNetEmbedder",
    "SigLIPEmbedder",
    "TeamClassifier",
    "assign_frame_teams",
    "available_embedders",
    "extract_crops",
]
