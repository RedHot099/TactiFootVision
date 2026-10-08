"""Model training and inference behind one interface (see :class:`Model`).

Backends register themselves in ``MODELS`` on import; their heavy libraries
(ultralytics, rfdetr, torch) load only when a model is created.
"""

from tactifoot_vision.models.base import (
    MODELS,
    Model,
    TrainConfig,
    TrainResult,
    available_models,
    load_model,
    train,
)
from tactifoot_vision.models.rfdetr import RFDETRDetector
from tactifoot_vision.models.ultralytics import YOLODetector, YOLOPoseModel

__all__ = [
    "MODELS",
    "Model",
    "RFDETRDetector",
    "TrainConfig",
    "TrainResult",
    "YOLODetector",
    "YOLOPoseModel",
    "available_models",
    "load_model",
    "train",
]
