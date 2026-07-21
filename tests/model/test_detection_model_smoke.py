from pathlib import Path

import cv2
import pytest

from tactifoot_vision.config import build_detector, load_pipeline_config
from tactifoot_vision.domain import DetectionSet, Frame


@pytest.mark.model
@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/pipeline/yolo_model_smoke.yaml"),
        Path("configs/pipeline/rfdetr_model_smoke.yaml"),
        Path("configs/pipeline/rfdetr_seg_model_smoke.yaml"),
    ],
)
def test_detection_model_smoke(config_path: Path) -> None:
    config = load_pipeline_config(config_path)
    if config.detection.checkpoint is None:
        pytest.skip(f"Detection checkpoint is not configured in {config_path}")
    if not config.detection.checkpoint.is_file():
        pytest.skip(f"Detection checkpoint not found: {config.detection.checkpoint}")
    frame_path = Path("data/soccernet_dummy/img1/frame_0001.jpg")
    image = cv2.imread(str(frame_path))
    if image is None:
        pytest.skip(f"Frame could not be read: {frame_path}")

    detector = build_detector(config)
    detections = detector.predict(Frame(index=0, image=image, path=frame_path))

    assert isinstance(detections, DetectionSet)
    for detection in detections:
        assert detection.bbox.width >= 0
        assert detection.bbox.height >= 0
        assert detection.class_name
        assert detection.confidence is None or 0.0 <= detection.confidence <= 1.0
