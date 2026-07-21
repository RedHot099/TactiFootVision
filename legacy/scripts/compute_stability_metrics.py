#!/usr/bin/env python3
"""Compute trajectory stability metrics on existing MOT predictions.

This script reads MOT prediction files from TrackEval format and computes
new stability metrics (ISR, ORC, DRR, AOR, PPS, MSS, TCI) for all trackers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tactifoot_vision.metrics.trajectory_stability import compute_all_stability_metrics


def parse_mot_file(path: Path) -> list[list[float]]:
    """Parse a MOT format text file into rows."""
    rows = []
    if not path.is_file():
        return rows
    
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:10]])
        except ValueError:
            continue
    return rows


def find_trackers(trackers_root: Path) -> dict[str, dict[str, Path]]:
    """Find all tracker predictions organized by class and tracker name.
    
    Returns:
        Dict[class_name, Dict[tracker_name, data_dir]]
    """
    result = defaultdict(dict)
    
    # Look for pattern: trackers_root/SNMOT_<class>-<split>/<tracker>/data/
    for benchmark_dir in trackers_root.iterdir():
        if not benchmark_dir.is_dir():
            continue
        
        benchmark_name = benchmark_dir.name
        if not benchmark_name.startswith("SNMOT_"):
            continue
        
        # Parse class from benchmark name (e.g., "SNMOT_player-test" -> "player")
        class_name = benchmark_name.replace("SNMOT_", "").split("-")[0]
        
        for tracker_dir in benchmark_dir.iterdir():
            if not tracker_dir.is_dir():
                continue
            
            data_dir = tracker_dir / "data"
            if data_dir.is_dir():
                tracker_name = tracker_dir.name
                result[class_name][tracker_name] = data_dir
    
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description="Compute stability metrics on existing MOT predictions")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking"),
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to results_dir/stability_metrics.csv)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Image width for metric calculations",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=25,
        help="Video frame rate (fps)",
    )
    args = parser.parse_args()
    
    results_dir = args.results_dir
    trackers_root = results_dir / "trackeval" / "data" / "trackers" / "mot_challenge"
    
    if not trackers_root.is_dir():
        logger.error("Trackers directory not found: {}", trackers_root)
        sys.exit(1)
    
    output_path = args.output or (results_dir / "stability_metrics.csv")
    
    logger.info("Scanning for tracker predictions in: {}", trackers_root)
    trackers_by_class = find_trackers(trackers_root)
    
    if not trackers_by_class:
        logger.error("No tracker predictions found")
        sys.exit(1)
    
    logger.info("Found {} classes with trackers", len(trackers_by_class))
    
    all_rows = []
    
    for class_name, tracker_dict in sorted(trackers_by_class.items()):
        logger.info("Processing class: {} ({} trackers)", class_name, len(tracker_dict))
        
        for tracker_name, data_dir in sorted(tracker_dict.items()):
            # Skip tuning variants for main analysis
            if "_tune_" in tracker_name:
                continue
            
            # Collect all predictions for this tracker
            all_rows_for_tracker = []
            sequence_count = 0
            
            for mot_file in data_dir.glob("*.txt"):
                rows = parse_mot_file(mot_file)
                if rows:
                    all_rows_for_tracker.extend(rows)
                    sequence_count += 1
            
            if not all_rows_for_tracker:
                logger.warning("No predictions found for {}/{}", class_name, tracker_name)
                continue
            
            # Compute stability metrics
            metrics = compute_all_stability_metrics(
                all_rows_for_tracker,
                image_width=args.image_width,
                frame_rate=args.frame_rate,
            )
            
            # Parse tracker variant info
            parts = tracker_name.split("__")
            detector = parts[0] if len(parts) > 1 else "unknown"
            tracker_type = parts[1] if len(parts) > 1 else tracker_name
            
            row = {
                "class": class_name,
                "tracker": tracker_name,
                "detector": detector,
                "tracker_type": tracker_type,
                "sequences": sequence_count,
                "total_detections": len(all_rows_for_tracker),
                **metrics,
            }
            all_rows.append(row)
            
            logger.info(
                "  {} ({} seqs, {} dets): ISR={:.3f} DRR={:.3f} PPS={:.3f} TCI={:.3f}",
                tracker_name,
                sequence_count,
                len(all_rows_for_tracker),
                metrics["isr_mean"],
                metrics["drr"],
                metrics["pps"],
                metrics["tci"],
            )
    
    if not all_rows:
        logger.error("No metrics computed")
        sys.exit(1)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    logger.success("Wrote stability metrics to: {}", output_path)
    
    # Print summary table
    print("\n=== Stability Metrics Summary ===\n")
    summary_cols = [
        "class",
        "tracker_type",
        "isr_mean",
        "isr_ge_0.8",
        "orc@30",
        "drr",
        "aor",
        "pps",
        "mss_mean",
        "tci",
    ]
    if all(col in df.columns for col in summary_cols):
        summary = df[summary_cols].copy()
        for col in ["isr_mean", "isr_ge_0.8", "orc@30", "drr", "aor", "pps", "mss_mean", "tci"]:
            summary[col] = summary[col].apply(lambda x: f"{x:.3f}")
        print(summary.to_string(index=False))

    # Print detailed dynamics breakdown
    print("\n=== Dynamics Breakdown (DRR/AOR/PPS) ===\n")
    dynamics_cols = [
        "class",
        "tracker_type",
        "drr",
        "drr_tracks_affected",
        "aor",
        "aor_total_outliers",
        "pps",
        "pps_speed_violations",
        "pps_accel_violations",
        "pps_max_speed_observed",
        "pps_max_accel_observed",
    ]
    if all(col in df.columns for col in dynamics_cols):
        dynamics_df = df[dynamics_cols].copy()
        for col in ["drr", "drr_tracks_affected", "aor", "pps", "pps_max_speed_observed", "pps_max_accel_observed"]:
            dynamics_df[col] = dynamics_df[col].apply(lambda x: f"{x:.3f}")
        print(dynamics_df.to_string(index=False))
    
    # Also output per-tracker summary grouped
    print("\n=== Per-Tracker Aggregates (across classes) ===\n")
    if "tracker_type" in df.columns:
        agg = df.groupby("tracker_type").agg({
            "isr_mean": "mean",
            "isr_ge_0.8": "mean",
            "orc@30": "mean",
            "drr": "mean",
            "aor": "mean",
            "pps": "mean",
            "mss_mean": "mean",
            "tci": "mean",
        }).round(3)
        print(agg.to_string())


if __name__ == "__main__":
    main()
