from pathlib import Path

import cv2
import pytest

from tactifoot_vision.domain import Frame
from tactifoot_vision.keypoints import KeypointSet, YOLOPoseKeypointModel


@pytest.mark.model
def test_yolo_pose_smoke() -> None:
    checkpoint = Path("models/yolo_pose.pt")
    if not checkpoint.is_file():
        pytest.skip(f"YOLO-pose checkpoint not found: {checkpoint}")
    frame_path = Path("data/soccernet_dummy/img1/frame_0001.jpg")
    image = cv2.imread(str(frame_path))
    if image is None:
        pytest.skip(f"Frame could not be read: {frame_path}")

    result = YOLOPoseKeypointModel.from_weights(checkpoint).predict(
        Frame(index=0, image=image, path=frame_path)
    )

    assert isinstance(result, KeypointSet)
    for keypoint in result:
        assert keypoint.x >= 0
        assert keypoint.y >= 0
        assert keypoint.confidence is None or 0.0 <= keypoint.confidence <= 1.0
