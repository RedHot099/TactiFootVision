from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config.models import DetectionConfig, DetectionModelType, TrainingDetectionConfig
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.synloc.data import export_synloc_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train a SynLoc person detector.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SoccerNet/SpiideoSynLoc"))
    parser.add_argument("--auxiliary-root", type=Path, default=Path("data/SoccerNetGS"))
    parser.add_argument("--auxiliary-tasks", nargs="*", choices=["gamestate-2024", "gamestate-2025"], default=[])
    parser.add_argument("--max-aux-images-per-split", type=int, default=None)
    parser.add_argument("--prepared-dataset-dir", type=Path, default=Path("data/SoccerNet/SpiideoSynLoc_detection"))
    parser.add_argument("--model-type", choices=["yolo", "rfdetr"], default="yolo")
    parser.add_argument("--base-model", type=str, default="yolo11m.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synloc/training"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        for key, value in payload.items():
            attr_name = key.replace("-", "_")
            if not hasattr(args, attr_name):
                continue
            if getattr(args, attr_name) == parser.get_default(attr_name):
                setattr(args, attr_name, value)

    prepared = export_synloc_detection_dataset(
        args.dataset_root,
        args.prepared_dataset_dir,
        symlink_images=not args.copy_images,
        auxiliary_roots=tuple((args.auxiliary_root / task).resolve() for task in args.auxiliary_tasks),
        auxiliary_tasks=tuple(args.auxiliary_tasks),
        max_aux_images_per_split=args.max_aux_images_per_split,
    )
    print(f"Prepared dataset -> COCO: {prepared['coco_root']} YOLO: {prepared['yolo_yaml']}")
    if args.prepare_only:
        return

    detection_config = DetectionConfig(
        model_type=DetectionModelType.YOLO if args.model_type == "yolo" else DetectionModelType.RFDETR,
        checkpoint_path=args.checkpoint_path,
        confidence_threshold=0.25,
        nms_threshold=0.5,
        classes={"person": 0},
        include_labels=["person"],
    )

    training_config = TrainingDetectionConfig(
        dataset_path=prepared["yolo_yaml"] if args.model_type == "yolo" else prepared["coco_root"],
        dataset_format="yolo" if args.model_type == "yolo" else "coco",
        base_model=None if args.checkpoint_path else args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        imgsz=args.train_imgsz,
        device=args.device,
        project_name=args.project_name or str(args.output_dir),
        run_name=args.run_name or f"synloc_{args.model_type}_person",
    )

    handler_cls = YOLOHandler if args.model_type == "yolo" else RFDETRHandler
    handler = handler_cls(detection_config, training_config, model_dir=args.model_dir)
    handler.train()


if __name__ == "__main__":
    main()
