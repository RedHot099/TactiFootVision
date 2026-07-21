#!/usr/bin/env python3
"""Evaluate detection outputs against SoccerNet MOT format ground truth.

This utility compares a CSV exported by `tests/test_soccernet_tracking.py`
with the corresponding `gt.txt` file from the SoccerNet tracking dataset.
It reports simple detection metrics (precision/recall/F1) using IoU matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

MOT_COLUMNS = ["frame", "id", "x", "y", "width", "height", "score", "x3", "y3", "z3"]


def _load_predictions(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    required = {"frame", "x", "y", "width", "height"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Prediction CSV missing required columns: {missing}")
    return df


def _load_ground_truth(gt_path: Path) -> pd.DataFrame:
    return pd.read_csv(gt_path, header=None, names=MOT_COLUMNS)


def _as_xyxy(arr: np.ndarray) -> np.ndarray:
    xyxy = arr.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _match_frame(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int, float, list[tuple[int, int, float]]]:
    """Returns (tp, fp, fn, iou_sum, matches) for a single frame."""
    if gt_boxes.size == 0 and pred_boxes.size == 0:
        return 0, 0, 0, 0.0, []
    if gt_boxes.size == 0:
        return 0, pred_boxes.shape[0], 0, 0.0, []
    if pred_boxes.size == 0:
        return 0, 0, gt_boxes.shape[0], 0.0, []

    iou_matrix = sv.box_iou_batch(gt_boxes, pred_boxes)
    if iou_matrix.size == 0:
        return 0, pred_boxes.shape[0], gt_boxes.shape[0], 0.0, []

    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = 0
    iou_sum = 0.0
    matches: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        val = float(iou_matrix[r, c])
        if val >= iou_threshold:
            tp += 1
            iou_sum += val
            matches.append((int(r), int(c), val))
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp
    return tp, fp, fn, iou_sum, matches


def evaluate(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> Dict[str, float]:
    if frame_start is not None:
        pred_df = pred_df[pred_df["frame"] >= frame_start]
        gt_df = gt_df[gt_df["frame"] >= frame_start]
    if frame_end is not None:
        pred_df = pred_df[pred_df["frame"] <= frame_end]
        gt_df = gt_df[gt_df["frame"] <= frame_end]

    frames = sorted(set(pred_df["frame"]).union(gt_df["frame"]))
    tp_total = fp_total = fn_total = 0
    iou_sum_total = 0.0
    matches = 0
    id_switches = 0
    prev_assignments: dict[int, int] = {}

    for frame in frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_boxes = _as_xyxy(gt_frame[["x", "y", "width", "height"]].to_numpy(float))
        pred_boxes = _as_xyxy(pred_frame[["x", "y", "width", "height"]].to_numpy(float))
        tp, fp, fn, iou_sum, matched_pairs = _match_frame(
            gt_boxes, pred_boxes, iou_threshold
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn
        iou_sum_total += iou_sum
        matches += tp

        current_assignments: dict[int, int] = {}
        if "track_id" in pred_frame.columns and not pred_frame.empty:
            pred_ids = pred_frame["track_id"].to_numpy()
            gt_ids = gt_frame["id"].to_numpy()
            for gt_idx, pred_idx, _iou in matched_pairs:
                if pred_idx >= len(pred_ids) or gt_idx >= len(gt_ids):
                    continue
                try:
                    pred_tid = int(pred_ids[pred_idx])
                    gt_tid = int(gt_ids[gt_idx])
                except (TypeError, ValueError):
                    continue
                if pred_tid < 0:
                    continue
                if pred_tid in prev_assignments and prev_assignments[pred_tid] != gt_tid:
                    id_switches += 1
                current_assignments[pred_tid] = gt_tid
        prev_assignments = current_assignments

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = iou_sum_total / matches if matches else 0.0

    valid_tracks = pred_df[pred_df.get("track_id", -1) >= 0]
    track_lengths = (
        valid_tracks.groupby("track_id")["frame"].nunique().astype(float)
        if not valid_tracks.empty and "track_id" in valid_tracks
        else pd.Series(dtype=float)
    )

    frames_evaluated = len(frames)

    return {
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "matches": int(matches),
        "frames_evaluated": frames_evaluated,
        "iou_threshold": iou_threshold,
        "id_switches": int(id_switches),
        "id_switch_rate": (id_switches / matches) if matches else 0.0,
        "avg_track_length": float(track_lengths.mean()) if not track_lengths.empty else 0.0,
        "median_track_length": float(track_lengths.median()) if not track_lengths.empty else 0.0,
        "max_track_length": float(track_lengths.max()) if not track_lengths.empty else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", required=True, type=Path, help="Path to CSV produced by the pipeline export.")
    parser.add_argument("--gt", required=True, type=Path, help="Path to SoccerNet gt.txt file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to store metrics. When omitted, prints to stdout.",
    )
    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Optional inclusive frame range to evaluate.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for true positives (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_df = _load_predictions(args.pred_csv)
    gt_df = _load_ground_truth(args.gt)

    frame_start, frame_end = (args.frame_range or (None, None))

    metrics = evaluate(
        pred_df,
        gt_df,
        iou_threshold=args.iou_threshold,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    metrics["predictions"] = int(len(pred_df))
    metrics["ground_truth"] = int(len(gt_df))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
