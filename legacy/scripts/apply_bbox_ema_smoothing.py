#!/usr/bin/env python3
"""Apply EMA smoothing to MOT tracking results (post-processing).

This script applies Exponential Moving Average (EMA) smoothing to bounding boxes
in existing MOT tracking results, which helps reduce trajectory jitter and improves
the MSS (Motion Smoothness Score) metric.

Usage:
    python scripts/apply_bbox_ema_smoothing.py \
        --input-dir results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg/trackeval/data/trackers \
        --trackers rfdetr_base__sam2 rfdetr_seg__sam2 \
        --alpha 0.5 \
        --output-suffix _smoothed
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from loguru import logger


def apply_ema_to_mot_file(
    input_path: Path,
    output_path: Path,
    alpha: float = 0.5,
) -> None:
    """Apply EMA smoothing to bounding boxes in a MOT file.
    
    MOT format: frame, id, x, y, w, h, conf, class_id, visibility
    
    Args:
        input_path: Path to input MOT txt file
        output_path: Path to output smoothed MOT txt file
        alpha: EMA alpha (0 = full smoothing / no change, 1 = no smoothing / use current)
               For smoothing: new = alpha * current + (1 - alpha) * previous
    """
    # Read all lines
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    # Parse into dict: track_id -> list of (frame, x, y, w, h, conf, class_id, visibility, line_idx)
    tracks: dict[int, list[tuple]] = defaultdict(list)
    parsed_lines: list[tuple] = []
    
    for line_idx, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        class_id = int(float(parts[7])) if len(parts) > 7 else -1
        visibility = float(parts[8]) if len(parts) > 8 else 1.0
        
        tracks[track_id].append((frame, x, y, w, h, conf, class_id, visibility, line_idx))
        parsed_lines.append((frame, track_id, x, y, w, h, conf, class_id, visibility))
    
    # Apply EMA smoothing per track
    smoothed: dict[int, dict[int, tuple]] = {}  # track_id -> frame -> (x, y, w, h)
    
    for track_id, detections in tracks.items():
        # Sort by frame
        detections_sorted = sorted(detections, key=lambda d: d[0])
        
        prev_box = None
        smoothed[track_id] = {}
        
        for det in detections_sorted:
            frame, x, y, w, h = det[0], det[1], det[2], det[3], det[4]
            current_box = np.array([x, y, w, h], dtype=np.float64)
            
            if prev_box is not None:
                # EMA: new = alpha * current + (1 - alpha) * prev
                smoothed_box = alpha * current_box + (1 - alpha) * prev_box
            else:
                smoothed_box = current_box
            
            smoothed[track_id][frame] = tuple(smoothed_box)
            prev_box = smoothed_box.copy()
    
    # Write output with smoothed boxes
    output_lines = []
    for frame, track_id, x, y, w, h, conf, class_id, visibility in parsed_lines:
        if track_id in smoothed and frame in smoothed[track_id]:
            sx, sy, sw, sh = smoothed[track_id][frame]
        else:
            sx, sy, sw, sh = x, y, w, h
        
        output_lines.append(f"{frame},{track_id},{sx:.4f},{sy:.4f},{sw:.4f},{sh:.4f},{conf},{class_id},{visibility}\n")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Apply EMA smoothing to MOT results")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing tracker results (e.g., .../trackeval/data/trackers)",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["rfdetr_base__sam2", "rfdetr_seg__sam2"],
        help="Tracker names to process",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="EMA alpha (0.5 = moderate smoothing, 0.3 = stronger smoothing)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_smoothed",
        help="Suffix to add to output tracker folder names",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files instead of creating new tracker folders",
    )
    args = parser.parse_args()
    
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Find all MOT files for specified trackers
    processed = 0
    for tracker_name in args.trackers:
        # Search in all challenge types (player, goalkeeper, referee, ball)
        tracker_dirs = list(input_dir.rglob(f"**/{tracker_name}"))
        
        for tracker_dir in tracker_dirs:
            data_dir = tracker_dir / "data"
            if not data_dir.exists():
                continue
            
            for mot_file in data_dir.glob("*.txt"):
                if mot_file.name.endswith("_summary.txt"):
                    continue
                
                if args.inplace:
                    output_file = mot_file
                else:
                    # Create new tracker dir with suffix
                    new_tracker_name = f"{tracker_name}{args.output_suffix}"
                    new_tracker_dir = tracker_dir.parent / new_tracker_name / "data"
                    output_file = new_tracker_dir / mot_file.name
                
                logger.info(f"Smoothing: {mot_file} -> {output_file}")
                apply_ema_to_mot_file(mot_file, output_file, alpha=args.alpha)
                processed += 1
    
    logger.info(f"Processed {processed} MOT files with alpha={args.alpha}")
    return 0


if __name__ == "__main__":
    exit(main())
