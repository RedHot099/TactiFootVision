#!/usr/bin/env python3
"""
Builds a monolithic COCO dataset from ALL available SoccerNet Tracking splits (train + test)
that possess Ground Truth.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tactifoot_vision.data.soccernet_tracking import export_mot_to_coco, iter_sequence_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, default=Path("data/soccernet/tracking/extracted"),
                        help="Root containing 'train', 'test', 'challenge' folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/soccernet/tracking/coco_full"),
                        help="Output directory for the COCO dataset.")
    args = parser.parse_args()

    # Define source folders
    splits = ["train", "test"]
    valid_seq_dirs = []

    logger.info(f"Scanning for sequences with GT in: {args.extracted_root}")

    for split in splits:
        split_dir = args.extracted_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        # Find sequences
        seqs = iter_sequence_dirs(split_dir)
        for seq in seqs:
            gt_file = seq / "gt" / "gt.txt"
            if gt_file.is_file() and gt_file.stat().st_size > 0:
                valid_seq_dirs.append(seq)
            else:
                logger.debug(f"Skipping {seq.name} (no GT found)")

    if not valid_seq_dirs:
        logger.error("No valid sequences with GT found!")
        sys.exit(1)

    logger.info(f"Found {len(valid_seq_dirs)} valid sequences.")

    # Create output structure
    # We want everything in 'train'. Valid/Test can be empty or duplicates.
    # For training 'RF-DETR', we usually point it to a dataset with train/valid/test.
    # We will put EVERYTHING in 'train' and symlink it to 'valid' and 'test' so we can evaluate on it too if needed.
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # We need to hack `export_mot_to_coco` slightly or just use it by passing a custom list of dirs.
    # But `export_mot_to_coco` takes a `split_root`.
    # It's easier to use the underlying logic or create a temporary "merged" root using symlinks?
    # No, creating symlinks for 100 folders is messy.
    
    # Let's inspect `tactifoot_vision/data/soccernet_tracking.py` to see if we can pass a list of dirs.
    # It seems `export_mot_to_coco` iterates `split_root`.
    
    # Alternative: We modify `export_mot_to_coco` or create a wrapper that mocks `iter_sequence_dirs`.
    # Or cleaner: We just call the processing function directly if exposed.
    
    # Actually, the simplest way is to manually build the dataset using the same logic as `export_mot_to_coco`.
    # But to avoid code duplication, I will create a temporary "virtual root" with symlinks.
    
    temp_root = args.output_dir / "temp_source_root"
    temp_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating symlinks in {temp_root}...")
    for seq_dir in valid_seq_dirs:
        link_path = temp_root / seq_dir.name
        if link_path.exists():
            link_path.unlink() # Refresh
        link_path.symlink_to(seq_dir.resolve())
        
    logger.info("Generating COCO annotations...")
    # Now we can call export_mot_to_coco on temp_root
    # We set valid_fraction=0.0 because we want everything in TRAIN.
    
    export_mot_to_coco(
        dataset_root=temp_root,
        output_root=args.output_dir,
        valid_fraction=0.0,
        seed=42,
        every_nth_frame=1,
        max_sequences=None, # All
        symlink_images=True
    )
    
    # Cleanup temp root? Maybe keep it for debugging.
    # Ensure 'valid' and 'test' folders exist in output (copies of train annotation)
    # export_mot_to_coco with valid_fraction=0.0 puts everything in train and valid/test might be empty or missing.
    # RF-DETR training script usually expects 'train' and 'valid'.
    
    train_json = args.output_dir / "train" / "_annotations.coco.json"
    valid_dir = args.output_dir / "valid"
    valid_dir.mkdir(exist_ok=True)
    valid_json = valid_dir / "_annotations.coco.json"
    
    if train_json.exists():
        import shutil
        if not valid_json.exists():
            shutil.copy(train_json, valid_json)
            logger.info("Copied train annotations to valid (for training compatibility).")
            
    logger.success(f"Full COCO dataset ready at: {args.output_dir}")

if __name__ == "__main__":
    main()
