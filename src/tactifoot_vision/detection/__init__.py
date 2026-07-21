from tactifoot_vision.detection.adapters.fake import FakeDetector
from tactifoot_vision.detection.adapters.rfdetr import RFDETRDetectionModel
from tactifoot_vision.detection.adapters.rfdetr_seg import RFDETRSegDetectionModel
from tactifoot_vision.detection.adapters.yolo import YOLODetectionModel
from tactifoot_vision.detection.interfaces import Detector, TrainableDetectionModel
from tactifoot_vision.detection.results import DetectionModelInfo

__all__ = [
    "DetectionModelInfo",
    "Detector",
    "FakeDetector",
    "RFDETRDetectionModel",
    "RFDETRSegDetectionModel",
    "TrainableDetectionModel",
    "YOLODetectionModel",
]
