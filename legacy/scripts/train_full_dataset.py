#!/usr/bin/env python3
"""
Trains RF-DETR (Base & Seg) on the full merged SoccerNet Tracking dataset.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.models import DetectionConfig, DetectionModelType, TrainingDetectionConfig
from tactifoot_vision.data.soccernet_tracking import SOCCERNET_CLASS_TO_ID
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, default=Path("data/soccernet/tracking/coco_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/project/full_training"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-type", choices=["base", "seg"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for output models
    final_model_path = args.output_dir / f"rfdetr_{args.model_type}_full.pth"
    training_work_dir = args.output_dir / f"work_{args.model_type}"
    training_work_dir.mkdir(parents=True, exist_ok=True)

    # Configs
    checkpoint = "rf-detr-base.pth" if args.model_type == "base" else "rf-detr-seg-preview.pt"
    
    det_config = DetectionConfig(
        model_type=DetectionModelType.RFDETR if args.model_type == "base" else DetectionModelType.RFDETR_SEG,
        checkpoint_path=Path(checkpoint), 
        classes=SOCCERNET_CLASS_TO_ID,
        confidence_threshold=0.3
    )
    
    train_config = TrainingDetectionConfig(
        dataset_path=str(args.coco_root),
        dataset_format="coco",
        output_dir=str(training_work_dir),
        save_checkpoint_path=str(final_model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        num_workers=4,
        optimizer="AdamW",
        grad_accum_steps=1
    )

    logger.info(f"Starting training for {args.model_type}...")
    logger.info(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    
    if args.model_type == "base":
        handler = RFDETRHandler(det_config, training_config=train_config, model_dir=project_root)
    else:
        handler = RFDETRSegHandler(det_config, training_config=train_config, model_dir=project_root)
        
    handler.train()
    
    logger.success(f"Training complete. Model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
