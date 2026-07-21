from tactifoot_vision.team_assignment.crops import crop_bbox, crop_sample
from tactifoot_vision.team_assignment.results import CropSample, TeamAssignmentReport

__all__ = [
    "ColorHistogramEmbedder",
    "CropSample",
    "ResNetEmbedder",
    "SigLIPEmbedder",
    "TeamAssignmentReport",
    "TeamAssigner",
    "crop_bbox",
    "crop_sample",
]


def __getattr__(name: str) -> object:
    if name == "TeamAssigner":
        from tactifoot_vision.team_assignment.assignment import TeamAssigner

        return TeamAssigner
    if name in {"ColorHistogramEmbedder", "ResNetEmbedder", "SigLIPEmbedder"}:
        from tactifoot_vision.team_assignment import embeddings

        return getattr(embeddings, name)
    raise AttributeError(
        f"module 'tactifoot_vision.team_assignment' has no attribute {name!r}"
    )
