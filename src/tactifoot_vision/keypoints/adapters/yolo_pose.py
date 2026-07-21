import urllib.error
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from tactifoot_vision.domain import BBox, Frame, ModelArtifactNotFound
from tactifoot_vision.keypoints.results import Keypoint, KeypointSet

KNOWN_YOLO_POSE_ARTIFACTS = {
    "yolov8n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
    ),
}


class YOLOPoseKeypointModel:
    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.5,
        auto_download: bool = True,
    ) -> None:
        self.weights = Path(weights)
        self.confidence = confidence
        self._model: Any | None = None
        if not self.weights.is_file():
            if auto_download and self.weights.name in KNOWN_YOLO_POSE_ARTIFACTS:
                _download_model_artifact(
                    KNOWN_YOLO_POSE_ARTIFACTS[self.weights.name], self.weights
                )
        if not self.weights.is_file():
            raise ModelArtifactNotFound(f"YOLO-pose weights not found: {self.weights}")

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        *,
        confidence: float = 0.5,
        auto_download: bool = True,
    ) -> "YOLOPoseKeypointModel":
        return cls(weights, confidence=confidence, auto_download=auto_download)

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights, task="pose")
        return self._model

    def predict(self, frame: Frame) -> KeypointSet:
        results = self._load().predict(frame.image, conf=self.confidence, verbose=False)
        if not results or results[0].keypoints is None:
            return KeypointSet.empty()
        keypoint_array = np.asarray(
            results[0].keypoints.data.cpu().numpy(), dtype=np.float32
        )
        if keypoint_array.size == 0:
            return KeypointSet.empty()
        if keypoint_array.ndim == 3:
            keypoint_array = keypoint_array[0]
        if keypoint_array.shape[-1] < 2:
            return KeypointSet.empty()
        keypoints = []
        for index, values in enumerate(keypoint_array):
            confidence = float(values[2]) if len(values) > 2 else None
            keypoints.append(
                Keypoint(
                    index=index,
                    x=float(values[0]),
                    y=float(values[1]),
                    confidence=confidence,
                )
            )
        pitch_bbox = _pitch_bbox(results[0])
        return KeypointSet(tuple(keypoints), pitch_bbox=pitch_bbox)


def _pitch_bbox(result: Any) -> BBox | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return None
    xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
    if xyxy.size == 0:
        return None
    return BBox.from_xyxy(xyxy[0])


def _download_model_artifact(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ModelArtifactNotFound(
            f"YOLO-pose weights not found and download failed: {destination}"
        ) from error
