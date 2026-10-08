"""Team assignment from player crops: crop extraction, embedders, clustering."""

from tactifoot_vision.teams.classifier import TeamClassifier
from tactifoot_vision.teams.crops import extract_crops
from tactifoot_vision.teams.embedders import (
    EMBEDDERS,
    Embedder,
    ResNetEmbedder,
    SigLIPEmbedder,
    available_embedders,
)

__all__ = [
    "EMBEDDERS",
    "Embedder",
    "ResNetEmbedder",
    "SigLIPEmbedder",
    "TeamClassifier",
    "available_embedders",
    "extract_crops",
]
