#!/usr/bin/env python3
"""Create a small player-crop dataset from SoccerNet Tracking (MOT format).

Input is an extracted SoccerNet Tracking split (e.g. train/) containing sequence folders
like SNMOT-060 with:
  - img1/*.jpg
  - gt/gt.txt
  - gameinfo.ini (maps track_id -> semantic label incl. team left/right)

Output follows the pipeline's expected folder structure:
  {output_dir}/{match_id}/{team_no}/{frame_no}_{track_id}.jpg

We only collect player crops for the two teams (left/right) and ignore referees/ball/etc.
"""

from __future__ import annotations

import argparse
import configparser
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


TEAM_LEFT = 0
TEAM_RIGHT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract player crops from SoccerNet Tracking sequences.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/train"),
        help="Path to extracted split root containing SNMOT-* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/soccernet/tracking/player_crops_sample"),
        help="Directory where crops will be written.",
    )
    parser.add_argument(
        "--target-per-team",
        type=int,
        default=50,
        help="How many crops to save per team (left/right). Use 0 for no limit.",
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=5,
        help="Subsample frames: keep only every Nth frame.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=12,
        help="Discard boxes with width/height smaller than this many pixels.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="If >0, limit number of sequences processed.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def clamp_box(x: float, y: float, w: float, h: float, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + w)))
    y2 = min(height, int(np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_tracklet_team_map(gameinfo_path: Path) -> Dict[int, int]:
    """Parse gameinfo.ini mapping trackletID_N to team left/right for players/goalkeepers."""
    parser = configparser.ConfigParser()
    parser.read(gameinfo_path)
    if "Sequence" not in parser:
        return {}
    section = parser["Sequence"]
    mapping: Dict[int, int] = {}
    num = section.getint("num_tracklets", fallback=0)
    for idx in range(1, max(0, num) + 1):
        raw = section.get(f"trackletID_{idx}", fallback="")
        if not raw:
            continue
        raw_lower = raw.lower().strip()
        # Examples:
        #  "player team left;10"
        #  "goalkeepers team left;y"
        #  "goalkeeper team right;X"
        #  "referee;main"
        is_team_person = raw_lower.startswith("player ") or raw_lower.startswith("goalkeeper")
        if not is_team_person:
            continue
        if "team left" in raw_lower:
            mapping[idx] = TEAM_LEFT
        elif "team right" in raw_lower:
            mapping[idx] = TEAM_RIGHT
    return mapping


def load_gt(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path, header=None)
    df = df.iloc[:, :7]
    df.columns = ["frame", "track_id", "x", "y", "w", "h", "confidence"]
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def iter_sequence_dirs(root: Path, max_sequences: int) -> Iterable[Path]:
    seqs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()])
    if max_sequences and max_sequences > 0:
        seqs = seqs[: int(max_sequences)]
    return seqs


def frame_path(img_dir: Path, frame_no: int) -> Path:
    return img_dir / f"{int(frame_no):06d}.jpg"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[int, int] = defaultdict(int)
    target_total = None if int(args.target_per_team) <= 0 else int(args.target_per_team) * 2

    for seq_dir in iter_sequence_dirs(dataset_root, args.max_sequences):
        if target_total is not None and counts[TEAM_LEFT] >= args.target_per_team and counts[TEAM_RIGHT] >= args.target_per_team:
            break

        gameinfo_path = seq_dir / "gameinfo.ini"
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        if not gameinfo_path.is_file() or not gt_path.is_file() or not img_dir.is_dir():
            logging.warning("Skipping %s (missing files)", seq_dir.name)
            continue

        team_map = parse_tracklet_team_map(gameinfo_path)
        if not team_map:
            logging.warning("Skipping %s (no player team mapping in gameinfo.ini)", seq_dir.name)
            continue

        gt = load_gt(gt_path)
        if gt.empty:
            logging.warning("Skipping %s (empty gt.txt)", seq_dir.name)
            continue

        logging.info("Processing %s (gt rows=%d)", seq_dir.name, len(gt))
        match_id = seq_dir.name

        # group by frame to minimize disk IO
        grouped = gt.groupby("frame", sort=True)
        for frame_no, rows in grouped:
            if frame_no % max(1, int(args.every_nth_frame)) != 0:
                continue
            if target_total is not None and sum(counts.values()) >= target_total:
                break

            img_path = frame_path(img_dir, int(frame_no))
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]

            for _, r in rows.iterrows():
                track_id = int(r["track_id"])
                team = team_map.get(track_id)
                if team is None:
                    continue
                if int(args.target_per_team) > 0 and counts[team] >= args.target_per_team:
                    continue
                box = clamp_box(float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]), width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                out_path = args.output_dir / match_id / str(team) / f"{int(frame_no):06d}_{track_id}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(out_path), crop):
                    counts[team] += 1
                    if target_total is not None and sum(counts.values()) >= target_total:
                        break

    logging.info("Saved crops per team: %s", dict(counts))
    logging.info("Output directory: %s", args.output_dir.resolve())
    total = sum(counts.values())
    if target_total is not None and total < target_total:
        logging.warning("Only saved %d/%d crops; consider lowering --min-size or --every-nth-frame.", total, target_total)


if __name__ == "__main__":
    main()
