from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.synloc_models import SynLocPrediction
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.point_regressor import (
    PointOffsetRegressor,
    build_regressor_examples,
    build_regressor_examples_from_predictions,
    train_point_regressor,
)
from config.synloc_models import SynLocDatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SynLoc learned-offset point regressor.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SoccerNet/SpiideoSynLoc"))
    parser.add_argument("--train-split", choices=["train", "val", "valid"], default="train")
    parser.add_argument("--val-split", choices=["train", "val", "valid"], default="val")
    parser.add_argument("--output-path", type=Path, default=Path("results/synloc/point_regressor.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-detections", type=Path, default=None, help="Optional raw detector results.json for the training split.")
    parser.add_argument("--val-detections", type=Path, default=None, help="Optional raw detector results.json for the validation split.")
    parser.add_argument("--no-augment", action="store_true", help="Disable crop augmentations during training.")
    args = parser.parse_args()

    train_split = load_synloc_split(SynLocDatasetConfig(root=args.dataset_root, split=args.train_split))
    val_split = load_synloc_split(SynLocDatasetConfig(root=args.dataset_root, split=args.val_split))
    train_examples = _load_examples(train_split, args.train_detections)
    val_examples = _load_examples(val_split, args.val_detections)

    model = PointOffsetRegressor()
    history = train_point_regressor(
        model,
        train_examples,
        val_examples=val_examples,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_path=args.output_path,
        augment=not args.no_augment,
    )
    print(history)


def _load_examples(split_data, detections_path: Path | None):
    if detections_path is None:
        return build_regressor_examples(split_data)
    predictions = [
        SynLocPrediction(
            image_id=int(item["image_id"]),
            category_id=int(item.get("category_id", 1)),
            score=float(item["score"]),
            bbox_xyxy=_xywh_to_xyxy(item.get("bbox", [0.0, 0.0, 0.0, 0.0])),
            image_point_xy=_extract_image_point(item),
            position_on_pitch_xyz=[float(v) for v in item.get("position_on_pitch", [0.0, 0.0, 0.0])],
        )
        for item in json.loads(detections_path.read_text(encoding="utf-8"))
    ]
    return build_regressor_examples_from_predictions(split_data, predictions)


def _extract_image_point(item: dict[str, object]) -> list[float]:
    if "keypoints" not in item:
        bbox = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x, y, w, h = [float(v) for v in bbox]
        return [x + w / 2.0, y + h]
    keypoints = [float(v) for v in item["keypoints"]]  # type: ignore[index]
    if len(keypoints) >= 6:
        return [keypoints[3], keypoints[4]]
    return [0.0, 0.0]


def _xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


if __name__ == "__main__":
    main()
