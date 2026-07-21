#!/usr/bin/env python3
"""Prepare a small balanced set of player crops from the local soccer_track data.

The script reads detections from the wide-view CSV annotations, aligns them with
the corresponding MP4 clips, and saves centered crops for a fixed number of
samples per team. The output folder follows the expected layout:

    {output_dir}/{match_id}/{team_id}/{frame_no}_{player_id}.jpg

By default we create 50 crops per team (100 in total for two teams), which is
enough for a quick smoke test of the downstream classification pipeline.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd


BBOX_FIELDS = ("bb_left", "bb_top", "bb_width", "bb_height")
DEFAULT_TEAMS = ("0", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a small balanced set of player crops from soccer_track clips."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/soccer_track/wide-view"),
        help="Directory with paired MP4/CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/soccer_track/player_crops_sample"),
        help="Where to save cropped players.",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=list(DEFAULT_TEAMS),
        help="Team identifiers from the CSV header to include.",
    )
    parser.add_argument(
        "--target-per-team",
        type=int,
        default=50,
        help="How many crops to save for each team.",
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Process every Nth frame to subsample dense clips.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=8,
        help="Discard boxes with width/height smaller than this many pixels.",
    )
    return parser.parse_args()


def load_annotations(csv_path: Path, teams: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the multi-index CSV and return frame numbers alongside annotations."""
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    frame_col = df.columns[0]
    frame_numbers_raw = pd.to_numeric(df[frame_col], errors="coerce")
    valid_mask = frame_numbers_raw.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    frame_numbers = frame_numbers_raw.loc[valid_mask].astype(int).reset_index(drop=True)

    wanted_columns = []
    for team in teams:
        for player_id in df.columns.get_level_values(1).unique():
            if player_id == "Unnamed: 0_level_1":
                continue
            player_id = str(player_id)
            cols_for_player = [(team, player_id, field) for field in BBOX_FIELDS]
            if all(col in df.columns for col in cols_for_player):
                wanted_columns.extend(cols_for_player)
    filtered_df = df.loc[:, wanted_columns].copy()
    return filtered_df, frame_numbers


def iter_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def save_crop(frame: np.ndarray, box: Tuple[int, int, int, int], path: Path, min_size: int) -> bool:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), crop))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.teams = [str(t) for t in args.teams]
    counts: Dict[str, int] = defaultdict(int)
    output_total = args.target_per_team * len(args.teams)

    video_paths = sorted(args.input_dir.glob("*.mp4"))
    if not video_paths:
        raise SystemExit(f"No MP4 files found in {args.input_dir}")

    for video_path in video_paths:
        csv_path = video_path.with_suffix(".csv")
        if not csv_path.is_file():
            logging.warning("Skipping %s (missing %s)", video_path.name, csv_path.name)
            continue

        if all(counts[t] >= args.target_per_team for t in args.teams):
            break

        annotations, frame_numbers = load_annotations(csv_path, args.teams)
        logging.info("Processing %s (%d frames)", video_path.name, len(frame_numbers))

        match_id = video_path.stem
        frame_iter = iter_video_frames(video_path)

        for row_idx, (frame_no, frame) in enumerate(zip(frame_numbers, frame_iter), start=0):
            if row_idx % max(1, args.every_nth_frame) != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            for team in args.teams:
                if counts[team] >= args.target_per_team:
                    continue

                team_cols = [col for col in annotations.columns if col[0] == team]
                player_ids = sorted(set(col[1] for col in team_cols))

                for player_id in player_ids:
                    cols = {(team, player_id, field): annotations[(team, player_id, field)] for field in BBOX_FIELDS if (team, player_id, field) in annotations.columns}
                    if len(cols) != 4:
                        continue
                    left = cols[(team, player_id, "bb_left")].iloc[row_idx]
                    top = cols[(team, player_id, "bb_top")].iloc[row_idx]
                    width = cols[(team, player_id, "bb_width")].iloc[row_idx]
                    height = cols[(team, player_id, "bb_height")].iloc[row_idx]

                    if any(pd.isna(v) for v in (left, top, width, height)):
                        continue

                    x1 = int(round(left))
                    y1 = int(round(top))
                    x2 = int(round(left + width))
                    y2 = int(round(top + height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    filename = f"{int(frame_no):06d}_{player_id}.jpg"
                    save_path = args.output_dir / match_id / team / filename
                    if save_crop(frame, (x1, y1, x2, y2), save_path, args.min_size):
                        counts[team] += 1
                        if counts[team] >= args.target_per_team:
                            break

                if counts[team] >= args.target_per_team:
                    continue

            if sum(counts.values()) >= output_total:
                break

    logging.info("Saved crops per team: %s", dict(counts))
    logging.info("Output directory: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
