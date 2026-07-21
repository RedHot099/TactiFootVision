from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

MOT_COLUMNS = ["frame", "id", "x", "y", "width", "height", "score", "x3", "y3", "z3"]
PIPELINE_CSV_MARKER_COLUMNS = {
    "timestamp_seconds",
    "class_id",
    "class_name",
}


def load_prediction_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {"frame", "x", "y", "width", "height"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV missing required columns: {', '.join(sorted(missing))}"
        )
    return dataframe


def load_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        return load_mot_ground_truth(path).rename(columns={"id": "track_id"})
    return load_prediction_csv(path)


def load_mot_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=MOT_COLUMNS)


def evaluate_tracking_files(
    pred_path: Path,
    gt_path: Path,
    *,
    iou_threshold: float = 0.5,
    frame_start: int | None = None,
    frame_end: int | None = None,
    prediction_frame_offset: int | None = None,
) -> dict[str, float]:
    predictions = load_prediction_file(pred_path)
    ground_truth = load_mot_ground_truth(gt_path)
    offset = (
        infer_prediction_frame_offset(predictions, ground_truth, pred_path=pred_path)
        if prediction_frame_offset is None
        else int(prediction_frame_offset)
    )
    metrics = evaluate_mot_tracking(
        predictions,
        ground_truth,
        iou_threshold=iou_threshold,
        frame_start=frame_start,
        frame_end=frame_end,
        prediction_frame_offset=offset,
    )
    metrics["prediction_frame_offset"] = float(offset)
    return metrics


def evaluate_mot_tracking(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    frame_start: int | None = None,
    frame_end: int | None = None,
    prediction_frame_offset: int = 0,
) -> dict[str, float]:
    if prediction_frame_offset:
        predictions = predictions.copy()
        predictions["frame"] = predictions["frame"] + int(prediction_frame_offset)
    predictions = _filter_frame_range(predictions, frame_start, frame_end)
    ground_truth = _filter_frame_range(ground_truth, frame_start, frame_end)
    frames = sorted(set(predictions["frame"]).union(ground_truth["frame"]))
    tp_total = fp_total = fn_total = 0
    iou_sum_total = 0.0
    matches_total = 0
    id_switches = 0
    last_assignments: dict[int, int] = {}
    for frame in frames:
        pred_frame = predictions[predictions["frame"] == frame]
        gt_frame = ground_truth[ground_truth["frame"] == frame]
        tp, fp, fn, iou_sum, matches = match_frame(
            _as_xyxy(gt_frame[["x", "y", "width", "height"]].to_numpy(float)),
            _as_xyxy(pred_frame[["x", "y", "width", "height"]].to_numpy(float)),
            iou_threshold=iou_threshold,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        iou_sum_total += iou_sum
        matches_total += tp
        if "track_id" in pred_frame.columns:
            pred_ids = pred_frame["track_id"].to_numpy()
            gt_ids = gt_frame["id"].to_numpy()
            for gt_index, pred_index, _iou in matches:
                if pred_index >= len(pred_ids) or gt_index >= len(gt_ids):
                    continue
                pred_id = int(pred_ids[pred_index])
                gt_id = int(gt_ids[gt_index])
                previous_pred_id = last_assignments.get(gt_id)
                if previous_pred_id is not None and previous_pred_id != pred_id:
                    id_switches += 1
                last_assignments[gt_id] = pred_id
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    valid_tracks = predictions[predictions.get("track_id", -1) >= 0]
    track_lengths = (
        valid_tracks.groupby("track_id")["frame"].nunique().astype(float)
        if not valid_tracks.empty and "track_id" in valid_tracks
        else pd.Series(dtype=float)
    )
    return {
        "tp": float(tp_total),
        "fp": float(fp_total),
        "fn": float(fn_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": iou_sum_total / matches_total if matches_total else 0.0,
        "matches": float(matches_total),
        "frames_evaluated": float(len(frames)),
        "iou_threshold": float(iou_threshold),
        "id_switches": float(id_switches),
        "id_switch_rate": id_switches / matches_total if matches_total else 0.0,
        "avg_track_length": float(track_lengths.mean())
        if not track_lengths.empty
        else 0.0,
        "median_track_length": float(track_lengths.median())
        if not track_lengths.empty
        else 0.0,
        "max_track_length": float(track_lengths.max())
        if not track_lengths.empty
        else 0.0,
    }


def match_frame(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, *, iou_threshold: float
) -> tuple[int, int, int, float, list[tuple[int, int, float]]]:
    if gt_boxes.size == 0 and pred_boxes.size == 0:
        return 0, 0, 0, 0.0, []
    if gt_boxes.size == 0:
        return 0, pred_boxes.shape[0], 0, 0.0, []
    if pred_boxes.size == 0:
        return 0, 0, gt_boxes.shape[0], 0.0, []
    iou_matrix = sv.box_iou_batch(gt_boxes, pred_boxes)
    row_indices, column_indices = linear_sum_assignment(1.0 - iou_matrix)
    matches: list[tuple[int, int, float]] = []
    iou_sum = 0.0
    for row, column in zip(row_indices, column_indices, strict=False):
        value = float(iou_matrix[row, column])
        if value >= iou_threshold:
            matches.append((int(row), int(column), value))
            iou_sum += value
    tp = len(matches)
    return tp, pred_boxes.shape[0] - tp, gt_boxes.shape[0] - tp, iou_sum, matches


def infer_prediction_frame_offset(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    pred_path: Path | None = None,
) -> int:
    if predictions.empty or ground_truth.empty:
        return 0
    if pred_path is not None and pred_path.suffix.lower() == ".txt":
        return 0
    if not PIPELINE_CSV_MARKER_COLUMNS.issubset(predictions.columns):
        return 0
    pred_min = int(predictions["frame"].min())
    gt_min = int(ground_truth["frame"].min())
    if pred_min == 0 and gt_min == 1:
        return 1
    return 0


def _as_xyxy(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.reshape(0, 4)
    xyxy = values.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _filter_frame_range(
    dataframe: pd.DataFrame, frame_start: int | None, frame_end: int | None
) -> pd.DataFrame:
    if frame_start is not None:
        dataframe = dataframe[dataframe["frame"] >= frame_start]
    if frame_end is not None:
        dataframe = dataframe[dataframe["frame"] <= frame_end]
    return dataframe
