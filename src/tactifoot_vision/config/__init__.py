from tactifoot_vision.config.loaders import (
    dump_config,
    load_config,
    load_experiment_config,
    load_pipeline_config,
)
from tactifoot_vision.config.schemas import (
    BallReconstructionConfig,
    DetectionConfig,
    DetectionTrainingConfig,
    ExperimentConfig,
    ExportConfig,
    KeypointsConfig,
    PathsConfig,
    PipelineConfig,
    ProjectionConfig,
    SAM2Config,
    ShotDetectionConfig,
    TeamAssignmentConfig,
    TrackingConfig,
    VideoXgConfig,
    XgEstimatorConfig,
)

__all__ = [
    "BallReconstructionConfig",
    "DetectionConfig",
    "DetectionTrainingConfig",
    "ExperimentConfig",
    "ExportConfig",
    "KeypointsConfig",
    "PathsConfig",
    "PipelineConfig",
    "ProjectionConfig",
    "SAM2Config",
    "ShotDetectionConfig",
    "TeamAssignmentConfig",
    "TrackingConfig",
    "VideoXgConfig",
    "XgEstimatorConfig",
    "build_detector",
    "build_pipeline",
    "build_tracker",
    "dump_config",
    "load_config",
    "load_experiment_config",
    "load_pipeline_config",
]


def __getattr__(name: str) -> object:
    if name in {"build_detector", "build_pipeline", "build_tracker"}:
        from tactifoot_vision.config import factories

        return getattr(factories, name)
    raise AttributeError(f"module 'tactifoot_vision.config' has no attribute {name!r}")
