import importlib

__all__ = [
    "BoTSORTTracker",
    "ByteTrackTracker",
    "DetectionTrainingConfig",
    "InferencePipeline",
    "PipelineConfig",
    "RFDETRDetectionModel",
    "SAM2Tracker",
    "YOLODetectionModel",
]


def __getattr__(name: str) -> object:
    if name in {"DetectionTrainingConfig", "PipelineConfig"}:
        config = importlib.import_module("tactifoot_vision.config")

        return getattr(config, name)
    if name in {"RFDETRDetectionModel", "YOLODetectionModel"}:
        from tactifoot_vision import detection

        return getattr(detection, name)
    if name == "InferencePipeline":
        from tactifoot_vision import pipeline

        return getattr(pipeline, name)
    if name in {"BoTSORTTracker", "ByteTrackTracker", "SAM2Tracker"}:
        from tactifoot_vision import tracking

        return getattr(tracking, name)
    raise AttributeError(f"module 'tactifoot_vision' has no attribute {name!r}")
