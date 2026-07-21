#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from tactifoot_vision.data.soccernet_tracking import export_mot_to_coco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SoccerNet Tracking (MOT) to Roboflow-style COCO.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/train"),
        help="Path to extracted SoccerNet tracking train split (contains SNMOT-* dirs).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/soccernet/tracking/coco_tracking_2023"),
        help="Output directory for COCO dataset.",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.2,
        help="Fraction of sequences assigned to the valid split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split selection.")
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Subsample frames: keep only every Nth frame.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="If >0, limit number of sequences processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Copy images instead of symlinking them (uses more disk).",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional JSON file to save conversion metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    logger.info("Converting MOT to COCO")
    logger.info("Dataset root: {}", dataset_root)
    logger.info("Output root: {}", output_root)

    meta = export_mot_to_coco(
        dataset_root,
        output_root,
        valid_fraction=float(args.valid_fraction),
        seed=int(args.seed),
        every_nth_frame=int(args.every_nth_frame),
        max_sequences=int(args.max_sequences),
        symlink_images=not bool(args.no_symlinks),
    )
    logger.success("Done. Wrote COCO dataset to {}", output_root)
    logger.info("Metadata: {}", meta)
    if args.meta_out is not None:
        out_path = args.meta_out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(meta, indent=2))
        logger.info("Saved metadata to {}", out_path)


if __name__ == "__main__":
    main()

