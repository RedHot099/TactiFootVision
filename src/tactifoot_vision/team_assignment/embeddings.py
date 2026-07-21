import cv2
import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.enums import Device


class ColorHistogramEmbedder:
    def __init__(self, *, bins: int = 16) -> None:
        self.bins = bins

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        vectors: list[NDArray[np.float32]] = []
        for crop in crops:
            if crop.size == 0:
                vectors.append(np.zeros(self.bins * 3, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            channels = [
                cv2.calcHist([hsv], [channel], None, [self.bins], [0, 256]).ravel()
                for channel in range(3)
            ]
            vector = np.concatenate(channels).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm > 0 else vector)
        return (
            np.vstack(vectors)
            if vectors
            else np.empty((0, self.bins * 3), dtype=np.float32)
        )


class ResNetEmbedder:
    def __init__(
        self,
        *,
        device: Device = Device.AUTO,
        batch_size: int = 64,
        model_name: str = "resnet18",
    ) -> None:
        if model_name != "resnet18":
            raise ValueError("Only resnet18 embeddings are supported.")
        import torch
        from PIL import Image
        from torchvision.models import ResNet18_Weights, resnet18

        self._torch = torch
        self._image_cls = Image
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.feature_dim = 512

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        tensors = []
        for crop in crops:
            if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(self._image_cls.fromarray(rgb)))
        if not tensors:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        batches = []
        with self._torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = self._torch.stack(tensors[start : start + self.batch_size]).to(
                    self.device
                )
                batches.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)


class SigLIPEmbedder:
    def __init__(
        self,
        *,
        device: Device = Device.AUTO,
        batch_size: int = 32,
        model_name: str = "google/siglip-base-patch16-224",
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.feature_dim = int(getattr(self.model.config, "hidden_size", 0) or 768)

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        valid = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops if crop.size]
        if not valid:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        batches = []
        with self._torch.inference_mode():
            for start in range(0, len(valid), self.batch_size):
                batch_images = valid[start : start + self.batch_size]
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.model(**inputs)
                features = output.last_hidden_state.mean(dim=1)
                batches.append(features.detach().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)


def _resolve_device(device: Device) -> str:
    if device == Device.AUTO:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device.value
