from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tactifoot_vision.synloc.camera import pitch_points_to_image
from tactifoot_vision.synloc.data import SynLocSplitData

if TYPE_CHECKING:
    from config.synloc_models import SynLocPrediction


class PointOffsetRegressor(nn.Module):
    def __init__(self, input_size: int = 96) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


@dataclass(frozen=True)
class PointRegressorExample:
    image_path: Path
    bbox_xyxy: list[float]
    target_offset_xy: list[float]


class SynLocPointRegressorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples: Iterable[PointRegressorExample],
        input_size: int = 96,
        *,
        augment: bool = False,
    ):
        self.examples = list(examples)
        self.input_size = int(input_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        image = cv2.imread(str(example.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {example.image_path}")
        crop = crop_bbox(image, example.bbox_xyxy, size=self.input_size)
        if self.augment:
            crop = _augment_crop(crop)
        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
        target = torch.tensor(example.target_offset_xy, dtype=torch.float32)
        return tensor, target


def crop_bbox(
    image: np.ndarray,
    bbox_xyxy: list[float],
    *,
    size: int = 96,
    padding_ratio: float = 0.1,
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    ix1 = max(0, int(round(x1 - pad_x)))
    iy1 = max(0, int(round(y1 - pad_y)))
    ix2 = min(w, int(round(x2 + pad_x)))
    iy2 = min(h, int(round(y2 + pad_y)))
    crop = image[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def normalized_offset_for_bbox(
    bbox_xyxy: list[float],
    image_point_xy: list[float],
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    px, py = [float(v) for v in image_point_xy]
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    return [
        float(np.clip((px - x1) / width, 0.0, 1.0)),
        float(np.clip((py - y1) / height, 0.0, 1.0)),
    ]


def denormalize_offset_to_image_point(
    bbox_xyxy: list[float],
    offset_xy: list[float] | np.ndarray | torch.Tensor,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if isinstance(offset_xy, torch.Tensor):
        offset = offset_xy.detach().cpu().numpy()
    else:
        offset = np.asarray(offset_xy, dtype=np.float32)
    return [
        float(x1 + offset[0] * max(1e-6, x2 - x1)),
        float(y1 + offset[1] * max(1e-6, y2 - y1)),
    ]


def build_regressor_examples(split_data: SynLocSplitData) -> list[PointRegressorExample]:
    examples: list[PointRegressorExample] = []
    for annotation in split_data.annotations:
        if annotation.bbox_xywh is None:
            continue
        image_record = split_data.images_by_id.get(annotation.image_id)
        if image_record is None:
            continue
        x, y, w, h = [float(v) for v in annotation.bbox_xywh]
        bbox_xyxy = [x, y, x + w, y + h]
        image_point = pitch_points_to_image(
            [annotation.position_on_pitch_xyz],
            image_record.camera_matrix,
            image_record.dist_poly,
            image_shape=image_record.image_shape,
        )[0].tolist()
        examples.append(
            PointRegressorExample(
                image_path=image_record.file_path,
                bbox_xyxy=bbox_xyxy,
                target_offset_xy=normalized_offset_for_bbox(bbox_xyxy, image_point),
            )
        )
    return examples


def build_regressor_examples_from_predictions(
    split_data: SynLocSplitData,
    predictions: Iterable["SynLocPrediction"],
    *,
    min_iou: float = 0.1,
) -> list[PointRegressorExample]:
    from config.synloc_models import SynLocPrediction

    examples: list[PointRegressorExample] = []
    for prediction in predictions:
        if not isinstance(prediction, SynLocPrediction):
            continue
        image_record = split_data.images_by_id.get(prediction.image_id)
        if image_record is None:
            continue
        matches = []
        for annotation in split_data.annotations_by_image.get(prediction.image_id, []):
            if annotation.bbox_xywh is None:
                continue
            x, y, w, h = [float(v) for v in annotation.bbox_xywh]
            gt_bbox = [x, y, x + w, y + h]
            iou = _bbox_iou(prediction.bbox_xyxy, gt_bbox)
            if iou >= min_iou:
                matches.append((iou, annotation))
        if not matches:
            continue
        _, match = max(matches, key=lambda item: item[0])
        image_point = pitch_points_to_image(
            [match.position_on_pitch_xyz],
            image_record.camera_matrix,
            image_record.dist_poly,
            image_shape=image_record.image_shape,
        )[0].tolist()
        examples.append(
            PointRegressorExample(
                image_path=image_record.file_path,
                bbox_xyxy=[float(v) for v in prediction.bbox_xyxy],
                target_offset_xy=normalized_offset_for_bbox(prediction.bbox_xyxy, image_point),
            )
        )
    return examples


def predict_image_point(
    model: PointOffsetRegressor,
    image: np.ndarray,
    bbox_xyxy: list[float],
    *,
    device: str | torch.device | None = None,
) -> list[float]:
    crop = crop_bbox(image, bbox_xyxy, size=model.input_size)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(tensor.to(device))[0]
    return denormalize_offset_to_image_point(bbox_xyxy, pred)


def train_point_regressor(
    model: PointOffsetRegressor,
    train_examples: list[PointRegressorExample],
    *,
    val_examples: list[PointRegressorExample] | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_path: Path | None = None,
    augment: bool = True,
) -> dict[str, list[float]]:
    train_dataset = SynLocPointRegressorDataset(
        train_examples,
        input_size=model.input_size,
        augment=augment,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_examples:
        val_dataset = SynLocPointRegressorDataset(
            val_examples,
            input_size=model.input_size,
            augment=False,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    history = {"train_loss": [], "val_loss": []}

    for _ in range(int(epochs)):
        model.train()
        running = 0.0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(inputs)
            count += len(inputs)
        history["train_loss"].append(running / max(1, count))

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    preds = model(inputs.to(device))
                    loss = criterion(preds, targets.to(device))
                    val_running += float(loss.item()) * len(inputs)
                    val_count += len(inputs)
            history["val_loss"].append(val_running / max(1, val_count))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": model.input_size,
                "history": history,
            },
            output_path,
        )
    return history


def load_point_regressor(checkpoint_path: Path, *, map_location: str | torch.device = "cpu") -> PointOffsetRegressor:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = PointOffsetRegressor(input_size=int(checkpoint.get("input_size", 96)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    return model


def _augment_crop(crop: np.ndarray) -> np.ndarray:
    result = crop.copy()
    if np.random.rand() < 0.5:
        sigma = float(np.random.uniform(0.0, 1.2))
        result = cv2.GaussianBlur(result, (3, 3), sigmaX=sigma)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0.0, 4.0, size=result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    if np.random.rand() < 0.5:
        encode_ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if encode_ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                result = decoded
    return result


def _bbox_iou(lhs: list[float], rhs: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in lhs]
    bx1, by1, bx2, by2 = [float(v) for v in rhs]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
