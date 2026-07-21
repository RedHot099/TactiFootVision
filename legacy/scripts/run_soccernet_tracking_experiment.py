#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from loguru import logger
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path when invoked as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.models import (
    DetectionConfig,
    DetectionModelType,
    SAM2Config,
    TrackingConfig,
    TrainingDetectionConfig,
)
from tactifoot_vision.data.soccernet_tracking import (
    SOCCERNET_CLASS_TO_ID,
    count_unique_gt_tracks_by_class,
    export_mot_to_coco,
    iter_sequence_dirs,
    load_mot_gt,
    parse_tracklet_class_map,
    read_seqinfo,
)
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler
from tactifoot_vision.metrics.trajectory_stability import compute_all_stability_metrics
from tactifoot_vision.tracking.botsort_tracker import BoTSORTArgs, BoTSORTTracker
from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker
from tactifoot_vision.tracking.tracker import Tracker


CLASSES_ORDERED = ["player", "goalkeeper", "referee", "ball"]


@dataclass(frozen=True)
class Variant:
    name: str
    detector: str
    tracker: str


@dataclass
class TimeStats:
    seconds: float = 0.0
    frames: int = 0

    def add(self, seconds: float, frames: int) -> None:
        self.seconds += float(seconds)
        self.frames += int(frames)

    @property
    def fps(self) -> float:
        return (self.frames / self.seconds) if self.seconds > 0 else 0.0


def _cuda_sync_if_available() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _iter_frames(seq_dir: Path, *, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    seqinfo = read_seqinfo(seq_dir)
    img_dir = seq_dir / "img1"
    limit = max(1, seqinfo.seq_length)
    if max_frames is not None and int(max_frames) > 0:
        limit = min(limit, int(max_frames))
    for frame_idx in range(1, limit + 1):
        img_path = img_dir / f"{frame_idx:06d}{seqinfo.image_ext}"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        yield frame_idx, frame


def _filter_by_class(detections: sv.Detections, class_id: int) -> sv.Detections:
    if len(detections) == 0:
        return detections
    if detections.class_id is None:
        return sv.Detections.empty()
    mask = detections.class_id.astype(int) == int(class_id)
    try:
        return detections[mask]
    except Exception:
        return sv.Detections.empty()


def _detections_to_mot_rows(
    detections: sv.Detections,
    frame_idx: int,
    *,
    default_confidence: float = 1.0,
) -> list[list[float]]:
    if len(detections) == 0:
        return []
    tracker_ids = (
        detections.tracker_id.astype(int) if detections.tracker_id is not None else None
    )
    rows: list[list[float]] = []
    for det_idx in range(len(detections)):
        if tracker_ids is None or det_idx >= len(tracker_ids):
            continue
        tid = int(tracker_ids[det_idx])
        x1, y1, x2, y2 = [float(v) for v in detections.xyxy[det_idx]]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        conf = default_confidence
        if detections.confidence is not None and det_idx < len(detections.confidence):
            candidate = detections.confidence[det_idx]
            if candidate is not None:
                try:
                    conf = float(candidate)
                except (TypeError, ValueError):
                    conf = default_confidence
        rows.append([frame_idx, tid, x1, y1, w, h, conf, -1, -1, -1])
    return rows


def _write_mot(rows: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "track_id",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "x3",
            "y3",
            "z3",
        ],
    )
    df.to_csv(path, header=False, index=False, float_format="%.3f")


def _match_by_iou(
    tracked: sv.Detections,
    dets: sv.Detections,
    *,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    if len(tracked) == 0 or len(dets) == 0:
        return []
    tracked_boxes = tracked.xyxy.astype(np.float32)
    det_boxes = dets.xyxy.astype(np.float32)
    if tracked_boxes.size == 0 or det_boxes.size == 0:
        return []
    iou_matrix = sv.box_iou_batch(tracked_boxes, det_boxes)
    if tracked.class_id is not None and dets.class_id is not None:
        tracked_cls = tracked.class_id.astype(int)[:, None]
        det_cls = dets.class_id.astype(int)[None, :]
        iou_matrix = iou_matrix * (tracked_cls == det_cls)
    if iou_matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    matches: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if float(iou_matrix[r, c]) >= float(iou_threshold):
            matches.append((int(r), int(c)))
    return matches


def _reseed_sam2(
    *,
    frame: np.ndarray,
    sam2_tracker: SAM2Tracker,
    tracked: sv.Detections,
    dets: sv.Detections,
    iou_threshold: float,
    reseed_mode: str,
    drop_after: int,
    stale_counts: dict[int, int],
) -> sv.Detections:
    matches = _match_by_iou(tracked, dets, iou_threshold=iou_threshold)
    matched_det = {d for _, d in matches}

    if reseed_mode != "reanchor":
        new_indices = [j for j in range(len(dets)) if j not in matched_det]
        if not new_indices:
            return tracked
        new_boxes = dets.xyxy[new_indices]
        new_classes = (
            dets.class_id[new_indices].astype(int)
            if dets.class_id is not None
            else np.full(len(new_boxes), -1, dtype=int)
        )
        new_ids = sam2_tracker.allocate_ids(len(new_boxes))
        tracked_boxes = (
            tracked.xyxy.astype(np.float32)
            if len(tracked) > 0
            else np.empty((0, 4), dtype=np.float32)
        )
        tracked_ids = (
            tracked.tracker_id.astype(int)
            if tracked.tracker_id is not None and len(tracked) > 0
            else np.empty((0,), dtype=int)
        )
        tracked_classes = (
            tracked.class_id.astype(int)
            if tracked.class_id is not None and len(tracked) > 0
            else np.full(len(tracked_boxes), -1, dtype=int)
        )
        combined_boxes = (
            np.vstack([tracked_boxes, new_boxes]) if tracked_boxes.size else new_boxes
        )
        combined_classes = (
            np.concatenate([tracked_classes, new_classes]) if tracked_classes.size else new_classes
        )
        combined_ids = np.concatenate([tracked_ids, new_ids]) if tracked_ids.size else new_ids
        return sam2_tracker.refresh_prompts(frame, combined_boxes, combined_classes, combined_ids)

    tracked_boxes = (
        tracked.xyxy.astype(np.float32)
        if len(tracked) > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    tracked_ids = (
        tracked.tracker_id.astype(int)
        if tracked.tracker_id is not None and len(tracked) > 0
        else np.empty((0,), dtype=int)
    )
    tracked_classes = (
        tracked.class_id.astype(int)
        if tracked.class_id is not None and len(tracked) > 0
        else np.full(len(tracked_boxes), -1, dtype=int)
    )
    det_boxes = (
        dets.xyxy.astype(np.float32)
        if len(dets) > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    det_classes = (
        dets.class_id.astype(int)
        if dets.class_id is not None and len(dets) > 0
        else np.full(len(det_boxes), -1, dtype=int)
    )

    combined_boxes: list[np.ndarray] = []
    combined_classes: list[int] = []
    combined_ids: list[int] = []

    matched_track_ids: set[int] = set()
    for track_idx, det_idx in matches:
        if track_idx >= len(tracked_ids) or det_idx >= len(det_boxes):
            continue
        tid = int(tracked_ids[track_idx])
        matched_track_ids.add(tid)
        combined_boxes.append(det_boxes[det_idx])
        combined_ids.append(tid)
        combined_classes.append(int(det_classes[det_idx]))
        stale_counts[tid] = 0

    for track_idx, tid in enumerate(tracked_ids):
        if int(tid) in matched_track_ids:
            continue
        if int(drop_after) > 0:
            stale_counts[tid] = stale_counts.get(tid, 0) + 1
            if stale_counts[tid] >= int(drop_after):
                continue
        combined_boxes.append(tracked_boxes[track_idx])
        combined_ids.append(int(tid))
        combined_classes.append(int(tracked_classes[track_idx]))

    new_indices = [j for j in range(len(det_boxes)) if j not in matched_det]
    if new_indices:
        new_boxes = det_boxes[new_indices]
        new_classes = det_classes[new_indices]
        new_ids = sam2_tracker.allocate_ids(len(new_boxes))
        for box, cls, tid in zip(new_boxes, new_classes, new_ids):
            combined_boxes.append(box)
            combined_ids.append(int(tid))
            combined_classes.append(int(cls))

    if not combined_boxes:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_classes = np.empty((0,), dtype=int)
        empty_ids = np.empty((0,), dtype=int)
        return sam2_tracker.refresh_prompts(frame, empty_boxes, empty_classes, empty_ids)

    combined_boxes_arr = np.stack(combined_boxes, axis=0).astype(np.float32)
    combined_classes_arr = np.array(combined_classes, dtype=int)
    combined_ids_arr = np.array(combined_ids, dtype=int)
    return sam2_tracker.refresh_prompts(
        frame, combined_boxes_arr, combined_classes_arr, combined_ids_arr
    )


def _snap_boxes_to_detections(
    tracked: sv.Detections,
    dets: sv.Detections,
    *,
    iou_threshold: float,
    drop_unmatched: bool = False,
    blend_alpha: float | None = None,
) -> sv.Detections:
    if len(tracked) == 0 or len(dets) == 0:
        return sv.Detections.empty() if drop_unmatched else tracked
    matches = _match_by_iou(tracked, dets, iou_threshold=iou_threshold)
    if not matches:
        return sv.Detections.empty() if drop_unmatched else tracked
    new_xyxy = tracked.xyxy.copy()
    new_conf = (
        tracked.confidence.copy()
        if tracked.confidence is not None
        else None
    )
    keep = np.zeros(len(tracked), dtype=bool)
    for track_idx, det_idx in matches:
        if track_idx >= len(new_xyxy) or det_idx >= len(dets.xyxy):
            continue
        keep[track_idx] = True
        det_box = dets.xyxy[det_idx]
        if blend_alpha is not None:
            track_box = new_xyxy[track_idx]
            alpha = float(blend_alpha)
            new_xyxy[track_idx] = (alpha * det_box) + ((1.0 - alpha) * track_box)
        else:
            new_xyxy[track_idx] = det_box
        if new_conf is not None and dets.confidence is not None:
            try:
                new_conf[track_idx] = float(dets.confidence[det_idx])
            except (TypeError, ValueError):
                pass
    updated = sv.Detections(
        xyxy=new_xyxy,
        mask=tracked.mask,
        confidence=new_conf if new_conf is not None else tracked.confidence,
        class_id=tracked.class_id,
        tracker_id=tracked.tracker_id,
    )
    if drop_unmatched:
        updated = updated[keep]
    return updated


def _track_length_stats(rows: list[list[float]], image_width: int = 1920) -> dict[str, float]:
    """Compute track length statistics and trajectory stability metrics.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        image_width: Image width for stability metric calculations
        
    Returns:
        Dict with track length stats and stability metrics (ISR, ORC, DRR, AOR, PPS, MSS, TCI)
    """
    # Default empty result with all metrics
    empty_result = {
        "pred_dets": 0.0,
        "pred_tracks": 0.0,
        "mean_track_len": 0.0,
        "median_track_len": 0.0,
        "p90_track_len": 0.0,
        "max_track_len": 0.0,
        "pct_tracks_lt_5": 0.0,
        "pct_tracks_lt_10": 0.0,
        # Stability metrics
        "isr_mean": 0.0,
        "isr_median": 0.0,
        "isr_ge_0.8": 0.0,
        "isr_ge_0.9": 0.0,
        "orc@15": 1.0,
        "orc@30": 1.0,
        "orc@60": 1.0,
        "drr": 0.0,
        "drr_tracks_affected": 0.0,
        "drr_total_reversals": 0,
        "aor": 0.0,
        "aor_median": 0.0,
        "aor_total_outliers": 0,
        "pps": 1.0,
        "pps_speed_violations": 0,
        "pps_accel_violations": 0,
        "pps_max_speed_observed": 0.0,
        "pps_max_accel_observed": 0.0,
        "mss_mean": 0.0,
        "mss_median": 0.0,
        "tci": 0.0,
    }
    
    if not rows:
        return empty_result

    frames_by_tid: dict[int, set[int]] = {}
    for row in rows:
        try:
            frame = int(row[0])
            tid = int(row[1])
        except Exception:
            continue
        frames_by_tid.setdefault(tid, set()).add(frame)

    lengths = np.array([len(frames) for frames in frames_by_tid.values()], dtype=float)
    if lengths.size == 0:
        result = empty_result.copy()
        result["pred_dets"] = float(len(rows))
        return result

    # Basic track length stats
    basic_stats = {
        "pred_dets": float(len(rows)),
        "pred_tracks": float(lengths.size),
        "mean_track_len": float(lengths.mean()),
        "median_track_len": float(np.median(lengths)),
        "p90_track_len": float(np.percentile(lengths, 90)),
        "max_track_len": float(lengths.max()),
        "pct_tracks_lt_5": float((lengths < 5).mean()),
        "pct_tracks_lt_10": float((lengths < 10).mean()),
    }
    
    # Compute stability metrics
    stability_metrics = compute_all_stability_metrics(rows, image_width=image_width)
    
    # Combine results
    return {**basic_stats, **stability_metrics}


def _prepare_trackeval_gt_by_class(
    *,
    extracted_root: Path,
    sequences: Iterable[str],
    output_gt_root: Path,
    benchmark_prefix: str,
    split_name: str,
    max_frames: int | None = None,
) -> dict[str, dict]:
    """Create TrackEval-compatible GT folders filtered per class."""
    output_gt_root.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for class_name in CLASSES_ORDERED:
        benchmark = f"{benchmark_prefix}_{class_name}"
        gt_set_dir = output_gt_root / f"{benchmark}-{split_name}"
        gt_set_dir.mkdir(parents=True, exist_ok=True)
        meta[class_name] = {"benchmark": benchmark, "gt_set_dir": str(gt_set_dir)}

        for seq in sequences:
            seq_src = extracted_root / seq
            class_map = parse_tracklet_class_map(seq_src / "gameinfo.ini")
            gt_path = seq_src / "gt" / "gt.txt"
            gt_df = pd.read_csv(gt_path, header=None)
            # SoccerNet provides MOT-style 10-column txt (frame,id,x,y,w,h,conf,-1,-1,-1).
            # TrackEval expects >=8 columns for GT; we keep the original shape and only filter rows.
            if gt_df.shape[1] < 8:
                raise ValueError(f"Unexpected SoccerNet gt.txt format (need >=8 columns): {gt_path}")
            gt_df = gt_df.iloc[:, :10]
            gt_df.columns = ["frame", "track_id", "x", "y", "w", "h", "conf", "x3", "y3", "z3"]
            if max_frames is not None and int(max_frames) > 0:
                gt_df = gt_df[gt_df["frame"].astype(int) <= int(max_frames)]
            keep_ids = {tid for tid, cls in class_map.items() if cls == class_name}
            if keep_ids:
                filtered = gt_df[gt_df["track_id"].astype(int).isin(sorted(keep_ids))]
            else:
                filtered = gt_df.iloc[:0]

            seq_dst = gt_set_dir / seq
            (seq_dst / "gt").mkdir(parents=True, exist_ok=True)
            (seq_dst / "gt" / "gt.txt").write_text(filtered.to_csv(header=False, index=False), encoding="utf-8")
            # TrackEval expects seqinfo.ini
            seqinfo_src = seq_src / "seqinfo.ini"
            if seqinfo_src.is_file():
                target = seq_dst / "seqinfo.ini"
                if not target.exists():
                    if max_frames is None or int(max_frames) <= 0:
                        target.symlink_to(os.path.relpath(seqinfo_src, start=seq_dst))
                    else:
                        content = seqinfo_src.read_text(encoding="utf-8").splitlines()
                        updated: list[str] = []
                        for line in content:
                            if line.startswith("seqLength="):
                                try:
                                    seq_len = int(line.split("=", 1)[1].strip())
                                except (IndexError, ValueError):
                                    seq_len = int(max_frames)
                                line = f"seqLength={min(seq_len, int(max_frames))}"
                            updated.append(line)
                        target.write_text("\n".join(updated) + "\n", encoding="utf-8")

    return meta


def _run_trackeval(
    *,
    trackeval_root: Path,
    gt_root: Path,
    trackers_root: Path,
    benchmark: str,
    split_name: str,
    seqmap_file: Path,
    tracker_names: list[str],
    iou_threshold: float = 0.5,
) -> dict:
    import numpy as np

    # TrackEval upstream still uses deprecated NumPy aliases (np.float/np.int) which are removed in NumPy>=2.0.
    # Patch them for compatibility.
    if not hasattr(np, "float"):
        setattr(np, "float", float)  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        setattr(np, "int", int)  # type: ignore[attr-defined]

    sys.path.insert(0, str(trackeval_root.resolve()))
    import trackeval  # type: ignore

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["PRINT_RESULTS"] = False
    default_eval_config["PRINT_ONLY_COMBINED"] = True
    default_eval_config["TIME_PROGRESS"] = True
    default_eval_config["OUTPUT_SUMMARY"] = True
    default_eval_config["OUTPUT_DETAILED"] = False
    default_eval_config["PLOT_CURVES"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_root.resolve())
    dataset_config["TRACKERS_FOLDER"] = str(trackers_root.resolve())
    dataset_config["BENCHMARK"] = benchmark
    dataset_config["SPLIT_TO_EVAL"] = split_name
    dataset_config["SEQMAP_FILE"] = str(seqmap_file.resolve())
    dataset_config["TRACKERS_TO_EVAL"] = tracker_names
    # SoccerNet tracking GT uses dummy class values (-1). TrackEval preprocessing would reject this,
    # so we disable preprocessing (as in the official SoccerNet tracking evaluation scripts).
    dataset_config["DO_PREPROC"] = False

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": float(iou_threshold)}

    evaluator = trackeval.Evaluator(default_eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    output_res, _output_msg = evaluator.evaluate(dataset_list, metrics_list)
    return output_res


def _extract_combined_metrics(
    trackeval_output: dict,
    tracker_name: str,
    *,
    debug_zero_fp: bool | None = None,
) -> dict[str, float]:
    def _scalar(v) -> float:
        if v is None:
            return 0.0
        try:
            import numpy as _np

            if isinstance(v, _np.ndarray):
                return float(_np.mean(v)) if v.size else 0.0
            if isinstance(v, (_np.floating, _np.integer)):
                return float(v)
        except Exception:
            pass
        if isinstance(v, (list, tuple)):
            return float(sum(float(x) for x in v) / len(v)) if v else 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    dataset_key = next(iter(trackeval_output.keys()))
    combined = trackeval_output[dataset_key][tracker_name]["COMBINED_SEQ"]
    # TrackEval nests results under class names (MotChallenge2DBox defaults to 'pedestrian').
    # We evaluate one class at a time via separate benchmarks (SNMOT_player, ...),
    # so pull the single class entry when present.
    if isinstance(combined, dict) and "HOTA" not in combined and "CLEAR" not in combined and combined:
        first = next(iter(combined.values()))
        if isinstance(first, dict) and (
            "HOTA" in first or "CLEAR" in first or "Identity" in first or "Count" in first
        ):
            combined = first
    hota = combined.get("HOTA", {})
    clear = combined.get("CLEAR", {})
    identity = combined.get("Identity", {})
    if debug_zero_fp is None:
        debug_zero_fp = os.environ.get("TACTIFOOT_DEBUG_ZERO_FP_METRICS", "") == "1"

    hota_v = _scalar(hota.get("HOTA", 0.0))
    idf1_v = _scalar(identity.get("IDF1", 0.0))
    mota_v = _scalar(clear.get("MOTA", 0.0))
    fp_v = _scalar(clear.get("CLR_FP", 0.0))
    fn_v = _scalar(clear.get("CLR_FN", 0.0))
    idsw_v = _scalar(clear.get("IDSW", 0.0))
    frag_v = _scalar(clear.get("Frag", 0.0))

    if debug_zero_fp:
        # TrackEval CLEAR MOTA satisfies: MOTA = 1 - (FP + FN + IDSW) / GT.
        # Recover GT and recompute MOTA with FP forced to 0.
        tp_v = _scalar(clear.get("CLR_TP", 0.0))
        gt = tp_v + fn_v if tp_v > 0.0 else 0.0
        if gt <= 0.0:
            denom = 1.0 - mota_v
            if denom != 0.0:
                gt = (fp_v + fn_v + idsw_v) / denom
        if gt > 0.0:
            mota_fp0 = 1.0 - ((fn_v + idsw_v) / gt)
            mota_v = float(mota_fp0)
        fp_v = 0.0

    return {
        "HOTA": hota_v,
        "IDF1": idf1_v,
        "MOTA": mota_v,
        "FP": fp_v,
        "FN": fn_v,
        "ID-switch": idsw_v,
        "Frag": frag_v,
    }


def _make_plots(
    results_dir: Path,
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    per_sequence_df: pd.DataFrame,
) -> None:
    plots_dir = Path("results/detection_tracking/plots") / results_dir.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    def pretty_variant(name: str) -> str:
        return (
            name.replace("rfdetr_", "RF-DETR ")
            .replace("__", " + ")
            .replace("botsort_reid", "BoT-SORT(ReID)")
            .replace("bytetrack", "ByteTrack")
            .replace("sam2", "SAM2")
            .replace("base", "Base")
            .replace("seg", "Seg")
        )

    summary_plot = summary_df.copy()
    summary_plot["variant_label"] = summary_plot["variant"].map(pretty_variant)
    variant_labels = summary_plot["variant_label"].dropna().tolist()
    unique_variants = list(dict.fromkeys(variant_labels))
    variant_colors = sns.color_palette("tab10", len(unique_variants))
    variant_palette = dict(zip(unique_variants, variant_colors))

    # 1) Weighted core metrics
    metrics_cols = ["weighted_HOTA", "weighted_IDF1", "weighted_MOTA"]
    melt = summary_plot.melt(
        id_vars=["variant_label"],
        value_vars=metrics_cols,
        var_name="metric",
        value_name="value",
    )
    melt["metric"] = melt["metric"].str.replace("weighted_", "", regex=False)
    plt.figure(figsize=(10, 5), dpi=200)
    sns.barplot(
        data=melt,
        x="metric",
        y="value",
        hue="variant_label",
        palette=variant_palette,
        hue_order=unique_variants,
    )
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("SoccerNet Tracking: Weighted Metrics by Variant (higher is better)")
    plt.legend(title="Variant", loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "weighted_metrics.png")
    plt.close()

    # 2) FPS
    plt.figure(figsize=(12, 4.5), dpi=200)
    sns.barplot(
        data=summary_plot,
        x="variant_label",
        y="fps",
        hue="variant_label",
        palette=variant_palette,
        hue_order=unique_variants,
        dodge=False,
    )
    plt.legend([], [], frameon=False)
    plt.ylabel("FPS (end-to-end, higher is better)")
    plt.xlabel("")
    plt.title("Inference Throughput (higher is better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "fps.png")
    plt.close()

    # 3) Per-class HOTA heatmap
    per_class_plot = per_class_df.copy()
    per_class_plot["variant_label"] = per_class_plot["variant"].map(pretty_variant)
    pivot = per_class_plot.pivot_table(index="class", columns="variant_label", values="HOTA", aggfunc="mean")
    plt.figure(figsize=(14, 3.5), dpi=200)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("HOTA per Class (higher is better)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(plots_dir / "hota_per_class_heatmap.png")
    plt.close()

    # 4) id_ratio distribution for players (per sequence)
    seq_plot = per_sequence_df.copy()
    seq_plot["variant_label"] = seq_plot["variant"].map(pretty_variant)
    players = seq_plot[seq_plot["class"] == "player"].copy()
    if not players.empty:
        plt.figure(figsize=(12, 4.5), dpi=200)
        sns.boxplot(
            data=players,
            x="variant_label",
            y="id_ratio",
            hue="variant_label",
            palette=variant_palette,
            hue_order=unique_variants,
            dodge=False,
        )
        plt.legend([], [], frameon=False)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.ylabel("id_ratio (#pred IDs / #GT tracks, closer to 1 is better)")
        plt.xlabel("")
        plt.title("ID Inflation (Players) – per Match (closer to 1 is better)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "id_ratio_players_boxplot.png")
        plt.close()

        # 4b) Absolute predicted IDs (tracks) with GT baseline distribution.
        # GT baseline is the number of ground-truth tracks for the same sequences.
        ids_plot = players[["variant_label", "sequence", "pred_tracks", "gt_tracks"]].copy()
        ids_plot = ids_plot.rename(columns={"pred_tracks": "pred_ids", "gt_tracks": "gt_ids"})
        long = pd.concat(
            [
                ids_plot[["variant_label", "sequence", "pred_ids"]].rename(
                    columns={"pred_ids": "ids"}
                ).assign(kind="Pred"),
                ids_plot[["variant_label", "sequence", "gt_ids"]].rename(
                    columns={"gt_ids": "ids"}
                ).assign(kind="GT (baseline)"),
            ],
            ignore_index=True,
        )
        plt.figure(figsize=(12, 4.5), dpi=200)
        sns.boxplot(
            data=long,
            x="variant_label",
            y="ids",
            hue="kind",
            palette=dict(zip(["Pred", "GT (baseline)"], sns.color_palette("deep", 2))),
        )
        plt.ylabel("# unique IDs (tracks)")
        plt.xlabel("")
        plt.title("Unique IDs (Players) – Pred vs GT Baseline (closer overlap is better)")
        plt.xticks(rotation=25, ha="right")
        plt.legend(title="")
        plt.tight_layout()
        plt.savefig(plots_dir / "ids_players_pred_vs_gt_boxplot.png")
        plt.close()

        plt.figure(figsize=(12, 4.5), dpi=200)
        sns.boxplot(
            data=players,
            x="variant_label",
            y="mean_track_len",
            hue="variant_label",
            palette=variant_palette,
            hue_order=unique_variants,
            dodge=False,
        )
        plt.legend([], [], frameon=False)
        plt.ylabel("Mean predicted track length (frames)")
        plt.xlabel("")
        plt.title("Predicted Track Length (Players) – per Match (higher is better)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "track_length_players_boxplot.png")
        plt.close()

    # 5) ID switches comparison between trackers (from TrackEval summary metrics).
    # Prefer weighted ID switches if available; fall back to macro.
    idsw_col = "weighted_ID-switch" if "weighted_ID-switch" in summary_plot.columns else None
    if idsw_col is None and "macro_ID-switch" in summary_plot.columns:
        idsw_col = "macro_ID-switch"
    if idsw_col is not None:
        idsw_plot = summary_plot.copy()
        idsw_plot["detector_label"] = idsw_plot["detector"].map(
            lambda x: str(x).replace("rfdetr_", "RF-DETR ").replace("base", "Base").replace("seg", "Seg")
        )
        idsw_plot["tracker_label"] = idsw_plot["tracker"].map(
            lambda x: str(x).replace("botsort", "BoT-SORT(ReID)").replace("bytetrack", "ByteTrack").replace("sam2", "SAM2")
        )
        detector_labels = idsw_plot["detector_label"].dropna().tolist()
        unique_detectors = list(dict.fromkeys(detector_labels))
        detector_colors = sns.color_palette("tab10", len(unique_detectors))
        detector_palette = dict(zip(unique_detectors, detector_colors))
        plt.figure(figsize=(10, 4.5), dpi=200)
        sns.barplot(
            data=idsw_plot,
            x="tracker_label",
            y=idsw_col,
            hue="detector_label",
            palette=detector_palette,
            hue_order=unique_detectors,
        )
        plt.ylabel("ID switches (lower is better)")
        plt.xlabel("")
        plt.title(f"ID Switches by Tracker ({idsw_col}, lower is better)")
        plt.tight_layout()
        plt.savefig(plots_dir / "id_switches_by_tracker.png")
        plt.close()


def _tune_trackers_for_detector(
    *,
    detector_name: str,
    detector,
    tune_seq_dirs: list[Path],
    gt_root: Path,
    trackers_root: Path,
    seqmap_tune: Path,
    max_frames_per_seq: int,
    base_bytetrack_cfg: TrackingConfig,
    base_botsort_args: BoTSORTArgs,
) -> dict:
    """Tune ByteTrack and BoT-SORT params on the tune split using player class only."""
    player_class_id = SOCCERNET_CLASS_TO_ID["player"]

    bytetrack_candidates: list[tuple[str, TrackingConfig]] = []
    for idx, params in enumerate(
        [
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.8, lost_track_buffer=30),
            dict(track_activation_threshold=0.35, minimum_matching_threshold=0.8, lost_track_buffer=30),
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.7, lost_track_buffer=30),
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.8, lost_track_buffer=60),
        ]
    ):
        cfg = base_bytetrack_cfg.model_copy(
            update={
                "track_activation_threshold": params["track_activation_threshold"],
                "minimum_matching_threshold": params["minimum_matching_threshold"],
                "lost_track_buffer": params["lost_track_buffer"],
            }
        )
        bytetrack_candidates.append((f"{detector_name}__bytetrack_tune_{idx}", cfg))

    botsort_candidates: list[tuple[str, BoTSORTArgs]] = []
    for idx, params in enumerate(
        [
            dict(match_thresh=0.8, appearance_thresh=0.8, track_buffer=30),
            dict(match_thresh=0.7, appearance_thresh=0.8, track_buffer=30),
            dict(match_thresh=0.8, appearance_thresh=0.7, track_buffer=30),
        ]
    ):
        args = BoTSORTArgs(
            track_high_thresh=base_botsort_args.track_high_thresh,
            track_low_thresh=base_botsort_args.track_low_thresh,
            new_track_thresh=base_botsort_args.new_track_thresh,
            track_buffer=int(params["track_buffer"]),
            match_thresh=float(params["match_thresh"]),
            fuse_score=base_botsort_args.fuse_score,
            gmc_method=base_botsort_args.gmc_method,
            proximity_thresh=base_botsort_args.proximity_thresh,
            appearance_thresh=float(params["appearance_thresh"]),
            with_reid=base_botsort_args.with_reid,
            model=base_botsort_args.model,
        )
        botsort_candidates.append((f"{detector_name}__botsort_tune_{idx}", args))

    bytetrack_trackers = {name: Tracker(cfg) for name, cfg in bytetrack_candidates}
    botsort_trackers = {
        name: BoTSORTTracker(args, frame_rate=int(base_bytetrack_cfg.frame_rate or 25))
        for name, args in botsort_candidates
    }

    all_tracker_names = list(bytetrack_trackers.keys()) + list(botsort_trackers.keys())
    logger.info(
        "Tuning on {} sequences for {} (candidates: bytetrack={}, botsort={})",
        len(tune_seq_dirs),
        detector_name,
        len(bytetrack_candidates),
        len(botsort_candidates),
    )

    # Run inference once per detector and update all candidates.
    for seq_dir in tune_seq_dirs:
        for tracker in bytetrack_trackers.values():
            tracker.reset()
        for tracker in botsort_trackers.values():
            tracker.reset()

        rows_by_tracker: dict[str, list[list[float]]] = {name: [] for name in all_tracker_names}

        for frame_no, frame in _iter_frames(seq_dir, max_frames=max_frames_per_seq):
            dets = detector.detect(frame)
            dets_player = _filter_by_class(dets, player_class_id)

            for name, tracker in bytetrack_trackers.items():
                tracked = tracker.update(dets_player)
                rows_by_tracker[name].extend(_detections_to_mot_rows(tracked, frame_no, default_confidence=1.0))

            for name, tracker in botsort_trackers.items():
                tracked = tracker.update(dets_player, frame)
                rows_by_tracker[name].extend(_detections_to_mot_rows(tracked, frame_no, default_confidence=1.0))

        # Write MOT files for TrackEval.
        for tracker_name, rows in rows_by_tracker.items():
            out_dir = trackers_root / "SNMOT_player-tune" / tracker_name / "data"
            out_path = out_dir / f"{seq_dir.name}.txt"
            _write_mot(rows, out_path)

    # Evaluate all candidates at once.
    output = _run_trackeval(
        trackeval_root=Path("external/TrackEval"),
        gt_root=gt_root,
        trackers_root=trackers_root,
        benchmark="SNMOT_player",
        split_name="tune",
        seqmap_file=seqmap_tune,
        tracker_names=all_tracker_names,
        iou_threshold=0.5,
    )

    candidate_rows: list[dict] = []
    for name in all_tracker_names:
        m = _extract_combined_metrics(output, name)
        candidate_rows.append({"detector": detector_name, "candidate": name, **m})
    candidates_df = pd.DataFrame(candidate_rows)

    def pick_best(prefix: str) -> str:
        df = candidates_df[candidates_df["candidate"].str.contains(prefix, regex=False)].copy()
        df = df.sort_values(["HOTA", "IDF1"], ascending=[False, False])
        return str(df["candidate"].iloc[0])

    best_bytetrack_name = pick_best("bytetrack_tune_")
    best_botsort_name = pick_best("botsort_tune_")

    best_bytetrack_cfg = next(cfg for name, cfg in bytetrack_candidates if name == best_bytetrack_name)
    best_botsort_args = next(args for name, args in botsort_candidates if name == best_botsort_name)

    return {
        "detector": detector_name,
        "candidates_df": candidates_df,
        "best_bytetrack_cfg": best_bytetrack_cfg,
        "best_botsort_args": best_botsort_args,
        "best": {
            "bytetrack": {"name": best_bytetrack_name, "config": best_bytetrack_cfg.model_dump()},
            "botsort_reid": {"name": best_botsort_name, "args": best_botsort_args.__dict__},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SoccerNet tracking-2023 detection+tracking experiment.")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/train"),
        help="Path to extracted tracking-2023 train split (SNMOT-* dirs).",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/test"),
        help="Path to extracted tracking-2023 test split (SNMOT-* dirs).",
    )
    parser.add_argument(
        "--seqmap-test",
        type=Path,
        default=Path("external/sn-tracking/tools/SNMOT-test.txt"),
        help="Seqmap file for TrackEval.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("data/soccernet/tracking/coco_tracking_2023"),
        help="COCO dataset output root for RF-DETR training.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/detection_tracking/raw/soccernet_tracking_2023_detection_tracking"),
        help="Output directory for experiment results.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip RF-DETR training if checkpoints exist.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip tracker hyperparameter tuning on the tune split.",
    )
    parser.add_argument(
        "--tune-max-sequences",
        type=int,
        default=0,
        help="If >0, limit number of tune sequences (for quicker tuning).",
    )
    parser.add_argument(
        "--tune-max-frames",
        type=int,
        default=200,
        help="Max frames per tune sequence used during tuning (default: 200).",
    )
    parser.add_argument(
        "--max-test-sequences",
        type=int,
        default=0,
        help="If >0, limit number of test sequences (for debugging).",
    )
    parser.add_argument(
        "--max-frames-per-seq",
        type=int,
        default=0,
        help="If >0, limit frames processed per sequence (debug only; affects metrics).",
    )
    parser.add_argument(
        "--train-max-sequences",
        type=int,
        default=0,
        help="Max number of sequences to use for training (default: 0 = all).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50 for base, 10 for seg).",
    )
    parser.add_argument(
        "--train-every-nth-frame",
        type=int,
        default=1,
        help="Subsampling for training data generation (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    test_root = args.test_root.resolve()
    coco_root = args.coco_root.resolve()
    results_dir = args.results_dir.resolve()
    seqmap_test = args.seqmap_test.resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "run_manifest.json"

    # Prepare COCO dataset for training if missing.
    coco_train_ann = coco_root / "train" / "_annotations.coco.json"
    coco_valid_ann = coco_root / "valid" / "_annotations.coco.json"
    coco_test_ann = coco_root / "test" / "_annotations.coco.json"
    if not coco_train_ann.is_file() or not coco_valid_ann.is_file() or not coco_test_ann.is_file():
        logger.info("COCO dataset missing, generating at {}", coco_root)
        export_mot_to_coco(
            train_root,
            coco_root,
            valid_fraction=0.2,
            seed=42,
            every_nth_frame=args.train_every_nth_frame,
            max_sequences=args.train_max_sequences,
        )

    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = models_dir / "rfdetr_base_soccernet2023.pth"
    seg_ckpt = models_dir / "rfdetr_seg_soccernet2023.pth"

    training_times: dict[str, float] = {}

    if not args.skip_training or not base_ckpt.is_file():
        logger.info("Training RF-DETR Base on COCO: {}", coco_root)
        det_cfg = DetectionConfig(
            model_type=DetectionModelType.RFDETR,
            checkpoint_path=Path("rf-detr-base.pth"),
            confidence_threshold=0.3,
            nms_threshold=0.5,
            classes=SOCCERNET_CLASS_TO_ID,
        )
        train_cfg = TrainingDetectionConfig(
            dataset_path=coco_root,
            dataset_format="coco",
            output_dir=results_dir / "training" / "rfdetr_base",
            save_checkpoint_path=base_ckpt,
            epochs=args.train_epochs,
            batch_size=8,
            grad_accum_steps=2,
            num_workers=2,
            learning_rate=1e-4,
            optimizer="AdamW",
        )
        t0 = time.perf_counter()
        handler = RFDETRHandler(det_cfg, training_config=train_cfg, model_dir=project_root)
        handler.train()
        training_times["rfdetr_base_s"] = time.perf_counter() - t0
    else:
        logger.info("Skipping RF-DETR Base training (checkpoint exists): {}", base_ckpt)

    if not args.skip_training or not seg_ckpt.is_file():
        logger.info("Training RF-DETR Seg on COCO (pseudo masks): {}", coco_root)
        det_cfg = DetectionConfig(
            model_type=DetectionModelType.RFDETR_SEG,
            checkpoint_path=Path("rf-detr-seg-preview.pt"),
            confidence_threshold=0.3,
            nms_threshold=0.5,
            classes=SOCCERNET_CLASS_TO_ID,
        )
        train_cfg = TrainingDetectionConfig(
            dataset_path=coco_root,
            dataset_format="coco",
            output_dir=results_dir / "training" / "rfdetr_seg",
            save_checkpoint_path=seg_ckpt,
            epochs=args.train_epochs,
            batch_size=2,
            grad_accum_steps=1,
            num_workers=2,
            learning_rate=1e-4,
            optimizer="AdamW",
        )
        t0 = time.perf_counter()
        handler = RFDETRSegHandler(det_cfg, training_config=train_cfg, model_dir=project_root)
        handler.train()
        training_times["rfdetr_seg_s"] = time.perf_counter() - t0
    else:
        logger.info("Skipping RF-DETR Seg training (checkpoint exists): {}", seg_ckpt)

    # Instantiate detection handlers for inference.
    logger.info("Loading trained detectors...")
    base_handler = RFDETRHandler(
        DetectionConfig(
            model_type=DetectionModelType.RFDETR,
            checkpoint_path=base_ckpt,
            confidence_threshold=0.3,
            nms_threshold=0.5,
            classes=SOCCERNET_CLASS_TO_ID,
        ),
        model_dir=project_root,
    )
    seg_handler = RFDETRSegHandler(
        DetectionConfig(
            model_type=DetectionModelType.RFDETR_SEG,
            checkpoint_path=seg_ckpt,
            confidence_threshold=0.3,
            nms_threshold=0.5,
            classes=SOCCERNET_CLASS_TO_ID,
        ),
        model_dir=project_root,
    )

    # TrackEval folder layout.
    trackeval_data_dir = results_dir / "trackeval" / "data"
    gt_root = trackeval_data_dir / "gt" / "mot_challenge"
    trackers_root = trackeval_data_dir / "trackers" / "mot_challenge"
    gt_root.mkdir(parents=True, exist_ok=True)
    trackers_root.mkdir(parents=True, exist_ok=True)

    # Prepare GT for all classes (test split).
    test_seq_dirs = iter_sequence_dirs(test_root)
    if args.max_test_sequences and args.max_test_sequences > 0:
        test_seq_dirs = test_seq_dirs[: int(args.max_test_sequences)]
    test_seq_names = [p.name for p in test_seq_dirs]

    # Use a seqmap file matching the selected sequence subset (helps debugging via --max-test-sequences).
    seqmap_effective = seqmap_test
    if args.max_test_sequences and args.max_test_sequences > 0:
        seqmap_effective = results_dir / "SNMOT-test.subset.txt"
        seqmap_effective.write_text("name\n" + "\n".join(test_seq_names) + "\n", encoding="utf-8")
    gt_meta = _prepare_trackeval_gt_by_class(
        extracted_root=test_root,
        sequences=test_seq_names,
        output_gt_root=gt_root,
        benchmark_prefix="SNMOT",
        split_name="test",
    )

    # Tracker parameters (placeholder defaults; tuning step is implemented later).
    bytetrack_cfg = TrackingConfig(
        enabled=True,
        backend="bytetrack",
        frame_rate=25,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    )
    botsort_args = BoTSORTArgs(with_reid=True, model=str((project_root / "yolo11n.pt").resolve()))
    sam2_cfg = TrackingConfig(
        enabled=True,
        backend="sam2",
        frame_rate=25,
        sam2=SAM2Config(
            checkpoint_path=Path("external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"),
            config_path=Path("external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"),
            mask_filter_distance=300.0,
            max_side=1024,
            max_objects=40,
            reseed_interval=30,
            reseed_iou_threshold=0.3,
            reseed_mode="add_new",
            drop_after=0,
            mask_threshold=0.0,
            mask_open=0,
            mask_close=0,
            output_box_mode="mask",
            output_box_iou_threshold=0.3,
            bbox_ema_alpha=0.5,  # EMA smoothing for trajectory stability
        ),
    )

    variants = [
        Variant(name="rfdetr_base__bytetrack", detector="rfdetr_base", tracker="bytetrack"),
        Variant(name="rfdetr_base__botsort_reid", detector="rfdetr_base", tracker="botsort"),
        Variant(name="rfdetr_base__sam2", detector="rfdetr_base", tracker="sam2"),
        Variant(name="rfdetr_seg__bytetrack", detector="rfdetr_seg", tracker="bytetrack"),
        Variant(name="rfdetr_seg__botsort_reid", detector="rfdetr_seg", tracker="botsort"),
        Variant(name="rfdetr_seg__sam2", detector="rfdetr_seg", tracker="sam2"),
    ]

    # Tracker tuning on a held-out subset of train (tune split).
    bytetrack_cfg_by_detector: dict[str, TrackingConfig] = {
        "rfdetr_base": bytetrack_cfg,
        "rfdetr_seg": bytetrack_cfg,
    }
    botsort_args_by_detector: dict[str, BoTSORTArgs] = {
        "rfdetr_base": botsort_args,
        "rfdetr_seg": botsort_args,
    }
    tuning_best: dict[str, dict] = {}

    def _reuse_tuning_from_csv(det_name: str) -> bool:
        candidates_path = results_dir / f"tuning_candidates_{det_name}.csv"
        if not candidates_path.is_file():
            return False
        try:
            df = pd.read_csv(candidates_path)
        except Exception:
            return False
        if df.empty or "candidate" not in df.columns:
            return False

        def pick(prefix: str) -> tuple[str, int] | None:
            sub = df[df["candidate"].astype(str).str.contains(prefix, regex=False)].copy()
            if sub.empty:
                return None
            sub = sub.sort_values(["HOTA", "IDF1"], ascending=[False, False])
            name = str(sub["candidate"].iloc[0])
            try:
                idx = int(name.split("_")[-1])
            except Exception:
                return None
            return name, idx

        bt = pick("bytetrack_tune_")
        bs = pick("botsort_tune_")
        if bt is None or bs is None:
            return False

        bt_name, bt_idx = bt
        bs_name, bs_idx = bs

        bytetrack_param_grid = [
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.8, lost_track_buffer=30),
            dict(track_activation_threshold=0.35, minimum_matching_threshold=0.8, lost_track_buffer=30),
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.7, lost_track_buffer=30),
            dict(track_activation_threshold=0.25, minimum_matching_threshold=0.8, lost_track_buffer=60),
        ]
        botsort_param_grid = [
            dict(match_thresh=0.8, appearance_thresh=0.8, track_buffer=30),
            dict(match_thresh=0.7, appearance_thresh=0.8, track_buffer=30),
            dict(match_thresh=0.8, appearance_thresh=0.7, track_buffer=30),
        ]
        if bt_idx < 0 or bt_idx >= len(bytetrack_param_grid) or bs_idx < 0 or bs_idx >= len(botsort_param_grid):
            return False

        bt_params = bytetrack_param_grid[bt_idx]
        bs_params = botsort_param_grid[bs_idx]

        bytetrack_cfg_det = bytetrack_cfg.model_copy(update=bt_params)
        botsort_args_det = BoTSORTArgs(
            track_high_thresh=botsort_args.track_high_thresh,
            track_low_thresh=botsort_args.track_low_thresh,
            new_track_thresh=botsort_args.new_track_thresh,
            track_buffer=int(bs_params["track_buffer"]),
            match_thresh=float(bs_params["match_thresh"]),
            fuse_score=botsort_args.fuse_score,
            gmc_method=botsort_args.gmc_method,
            proximity_thresh=botsort_args.proximity_thresh,
            appearance_thresh=float(bs_params["appearance_thresh"]),
            with_reid=botsort_args.with_reid,
            model=botsort_args.model,
        )

        bytetrack_cfg_by_detector[det_name] = bytetrack_cfg_det
        botsort_args_by_detector[det_name] = botsort_args_det
        tuning_best[det_name] = {
            "candidates_csv": str(candidates_path),
            "bytetrack": {"name": bt_name, "config": bytetrack_cfg_det.model_dump()},
            "botsort_reid": {"name": bs_name, "args": botsort_args_det.__dict__},
        }
        logger.info("Reused tuning for {} from {}", det_name, candidates_path)
        return True

    if args.skip_tuning:
        for det_name in ("rfdetr_base", "rfdetr_seg"):
            _reuse_tuning_from_csv(det_name)
    else:
        splits_path = coco_root / "sequence_splits.json"
        split_map = json.loads(splits_path.read_text()) if splits_path.is_file() else {}
        tune_seq_names = [k for k, v in split_map.items() if v == "valid"]
        if not tune_seq_names:
            # Fallback: take 20% sequences from train_root
            all_train_seqs = [p.name for p in iter_sequence_dirs(train_root)]
            tune_seq_names = all_train_seqs[: max(1, int(round(len(all_train_seqs) * 0.2)))]
        tune_seq_names = sorted(tune_seq_names)
        if args.tune_max_sequences and args.tune_max_sequences > 0:
            tune_seq_names = tune_seq_names[: int(args.tune_max_sequences)]

        tune_seq_dirs = [train_root / name for name in tune_seq_names if (train_root / name).is_dir()]
        seqmap_tune = results_dir / "SNMOT-tune.txt"
        seqmap_tune.write_text("name\n" + "\n".join([p.name for p in tune_seq_dirs]) + "\n", encoding="utf-8")

        _prepare_trackeval_gt_by_class(
            extracted_root=train_root,
            sequences=[p.name for p in tune_seq_dirs],
            output_gt_root=gt_root,
            benchmark_prefix="SNMOT",
            split_name="tune",
        )

        for det_name, det_handler in {"rfdetr_base": base_handler, "rfdetr_seg": seg_handler}.items():
            tune_result = _tune_trackers_for_detector(
                detector_name=det_name,
                detector=det_handler,
                tune_seq_dirs=tune_seq_dirs,
                gt_root=gt_root,
                trackers_root=trackers_root,
                seqmap_tune=seqmap_tune,
                max_frames_per_seq=int(args.tune_max_frames),
                base_bytetrack_cfg=bytetrack_cfg,
                base_botsort_args=botsort_args,
            )
            candidates_df: pd.DataFrame = tune_result["candidates_df"]
            candidates_path = results_dir / f"tuning_candidates_{det_name}.csv"
            candidates_df.to_csv(candidates_path, index=False)

            bytetrack_cfg_by_detector[det_name] = tune_result["best_bytetrack_cfg"]
            botsort_args_by_detector[det_name] = tune_result["best_botsort_args"]
            tuning_best[det_name] = {
                "candidates_csv": str(candidates_path),
                **tune_result["best"],
            }
            logger.info("Tuning best for {}: {}", det_name, tune_result["best"])

    # Inference + tracking (write MOT outputs per class into TrackEval structure).
    timing: dict[str, TimeStats] = {v.name: TimeStats() for v in variants}
    per_sequence_rows: list[dict] = []

    logger.info("Running inference+tracking on test split (sequences={}): {}", len(test_seq_dirs), test_root)
    warmup_frames = 5

    # Optimise detectors for inference when available.
    for handler in (base_handler, seg_handler):
        model = getattr(handler, "model", None)
        if model is not None and hasattr(model, "optimize_for_inference"):
            try:
                model.optimize_for_inference()
            except Exception:
                pass

    detector_handlers = {"rfdetr_base": base_handler, "rfdetr_seg": seg_handler}
    tracker_suites: dict[str, dict] = {}
    for detector_name in detector_handlers.keys():
        bytetrack_cfg_det = bytetrack_cfg_by_detector.get(detector_name, bytetrack_cfg)
        botsort_args_det = botsort_args_by_detector.get(detector_name, botsort_args)
        tracker_suites[detector_name] = {
            "bytetrack": {c: Tracker(bytetrack_cfg_det) for c in CLASSES_ORDERED},
            "botsort_reid": {
                c: BoTSORTTracker(botsort_args_det, frame_rate=int(bytetrack_cfg_det.frame_rate or 25))
                for c in CLASSES_ORDERED
            },
            "sam2": None,
        }
        try:
            tracker_suites[detector_name]["sam2"] = SAM2Tracker(sam2_cfg)
        except Exception as e:
            logger.error("SAM2Tracker init failed ({}): {}", detector_name, e)
            tracker_suites[detector_name]["sam2"] = None

    reseed_interval = None
    reseed_iou_threshold = 0.3
    reseed_mode = "add_new"
    drop_after = 0
    output_box_mode = "mask"
    output_box_iou = 0.3
    output_box_blend_alpha = 0.7
    if sam2_cfg.sam2 is not None:
        reseed_interval = sam2_cfg.sam2.reseed_interval
        reseed_iou_threshold = float(sam2_cfg.sam2.reseed_iou_threshold)
        reseed_mode = str(sam2_cfg.sam2.reseed_mode)
        drop_after = int(sam2_cfg.sam2.drop_after)
        output_box_mode = str(sam2_cfg.sam2.output_box_mode)
        output_box_iou = float(sam2_cfg.sam2.output_box_iou_threshold)
        output_box_blend_alpha = float(sam2_cfg.sam2.output_box_blend_alpha)
    snap_to_detector = output_box_mode in {"detector", "detector_strict", "detector_blend"}
    drop_unmatched = output_box_mode == "detector_strict"
    blend_alpha = output_box_blend_alpha if output_box_mode == "detector_blend" else None

    for seq_dir in test_seq_dirs:
        logger.info("Sequence {}", seq_dir.name)
        gt_counts_seq = count_unique_gt_tracks_by_class(seq_dir)

        for detector_name, detector in detector_handlers.items():
            bytetrack_trackers: dict[str, Tracker] = tracker_suites[detector_name]["bytetrack"]
            botsort_trackers: dict[str, BoTSORTTracker] = tracker_suites[detector_name]["botsort_reid"]
            sam2_tracker: SAM2Tracker | None = tracker_suites[detector_name]["sam2"]

            # Reset per-sequence tracker state.
            for tracker in bytetrack_trackers.values():
                tracker.reset()
            for tracker in botsort_trackers.values():
                tracker.reset()

            sam2_initialized = False
            sam2_reset_done = False
            sam2_stale_counts: dict[int, int] = {}

            rows_by_variant_and_class: dict[str, dict[str, list[list[float]]]] = {
                f"{detector_name}__bytetrack": {c: [] for c in CLASSES_ORDERED},
                f"{detector_name}__botsort_reid": {c: [] for c in CLASSES_ORDERED},
                f"{detector_name}__sam2": {c: [] for c in CLASSES_ORDERED},
            }

            max_frames = int(args.max_frames_per_seq) if args.max_frames_per_seq else None
            frames_processed = 0
            for idx, (frame_idx, frame) in enumerate(_iter_frames(seq_dir, max_frames=max_frames)):
                frames_processed += 1
                do_time = idx >= warmup_frames

                _cuda_sync_if_available()
                t0 = time.perf_counter()
                dets = detector.detect(frame)
                _cuda_sync_if_available()
                det_time = time.perf_counter() - t0

                # ByteTrack (per class).
                _cuda_sync_if_available()
                t0 = time.perf_counter()
                for cname in CLASSES_ORDERED:
                    class_id = SOCCERNET_CLASS_TO_ID[cname]
                    dets_c = _filter_by_class(dets, class_id)
                    tracked = bytetrack_trackers[cname].update(dets_c)
                    rows_by_variant_and_class[f"{detector_name}__bytetrack"][cname].extend(
                        _detections_to_mot_rows(tracked, frame_idx, default_confidence=1.0)
                    )
                _cuda_sync_if_available()
                bytetrack_time = time.perf_counter() - t0

                # BoT-SORT (per class, with ReID).
                _cuda_sync_if_available()
                t0 = time.perf_counter()
                for cname in CLASSES_ORDERED:
                    class_id = SOCCERNET_CLASS_TO_ID[cname]
                    dets_c = _filter_by_class(dets, class_id)
                    tracked = botsort_trackers[cname].update(dets_c, frame)
                    rows_by_variant_and_class[f"{detector_name}__botsort_reid"][cname].extend(
                        _detections_to_mot_rows(tracked, frame_idx, default_confidence=1.0)
                    )
                _cuda_sync_if_available()
                botsort_time = time.perf_counter() - t0

                # SAM2 (all classes together, with optional reseeding).
                _cuda_sync_if_available()
                t0 = time.perf_counter()
                tracked_sam2 = sv.Detections.empty()
                if sam2_tracker is not None:
                    if not sam2_reset_done:
                        sam2_tracker.initialize(frame, np.empty((0, 4), dtype=np.float32), None)
                        sam2_reset_done = True

                    if not sam2_initialized:
                        if len(dets) > 0:
                            tracked_sam2 = sam2_tracker.initialize(frame, dets.xyxy, dets.class_id)
                            sam2_initialized = True
                    else:
                        tracked_sam2 = sam2_tracker.track(frame)

                    if snap_to_detector and len(tracked_sam2) > 0:
                        tracked_sam2 = _snap_boxes_to_detections(
                            tracked_sam2,
                            dets,
                            iou_threshold=output_box_iou,
                            drop_unmatched=drop_unmatched,
                            blend_alpha=blend_alpha,
                        )

                    if (
                        sam2_initialized
                        and reseed_interval is not None
                        and reseed_interval > 0
                        and (idx % int(reseed_interval) == 0)
                        and len(dets) > 0
                    ):
                        tracked_sam2 = _reseed_sam2(
                            frame=frame,
                            sam2_tracker=sam2_tracker,
                            tracked=tracked_sam2,
                            dets=dets,
                            iou_threshold=reseed_iou_threshold,
                            reseed_mode=reseed_mode,
                            drop_after=drop_after,
                            stale_counts=sam2_stale_counts,
                        )
                        if snap_to_detector and len(tracked_sam2) > 0:
                            tracked_sam2 = _snap_boxes_to_detections(
                                tracked_sam2,
                                dets,
                                iou_threshold=output_box_iou,
                                drop_unmatched=drop_unmatched,
                                blend_alpha=blend_alpha,
                            )

                    for cname in CLASSES_ORDERED:
                        class_id = SOCCERNET_CLASS_TO_ID[cname]
                        tracked_c = _filter_by_class(tracked_sam2, class_id)
                        rows_by_variant_and_class[f"{detector_name}__sam2"][cname].extend(
                            _detections_to_mot_rows(tracked_c, frame_idx, default_confidence=1.0)
                        )
                _cuda_sync_if_available()
                sam2_time = time.perf_counter() - t0

                if do_time:
                    timing[f"{detector_name}__bytetrack"].add(det_time + bytetrack_time, 1)
                    timing[f"{detector_name}__botsort_reid"].add(det_time + botsort_time, 1)
                    timing[f"{detector_name}__sam2"].add(det_time + sam2_time, 1)

            # Write results for this sequence to TrackEval tracker folders.
            for tracker_key in ("bytetrack", "botsort_reid", "sam2"):
                variant_name = f"{detector_name}__{tracker_key}"
                for class_name in CLASSES_ORDERED:
                    benchmark = gt_meta[class_name]["benchmark"]
                    tracker_dir = trackers_root / f"{benchmark}-test" / variant_name / "data"
                    out_path = tracker_dir / f"{seq_dir.name}.txt"
                    class_rows = rows_by_variant_and_class[variant_name][class_name]
                    _write_mot(class_rows, out_path)
                    stats = _track_length_stats(class_rows)
                    gt_tracks = int(gt_counts_seq.get(class_name, 0))
                    pred_tracks = int(stats["pred_tracks"])
                    id_ratio = (pred_tracks / gt_tracks) if gt_tracks > 0 else float("nan")
                    per_sequence_rows.append(
                        {
                            "sequence": seq_dir.name,
                            "variant": variant_name,
                            "detector": detector_name,
                            "tracker": tracker_key,
                            "class": class_name,
                            "frames_processed": int(frames_processed),
                            "gt_tracks": gt_tracks,
                            "id_ratio": id_ratio,
                            **stats,
                        }
                    )

    # Run TrackEval per class, then aggregate macro/weighted.
    logger.info("Running TrackEval (HOTA/CLEAR/Identity)...")
    tracker_names = [v.name for v in variants]
    per_class_rows: list[dict] = []
    for class_name in CLASSES_ORDERED:
        benchmark = gt_meta[class_name]["benchmark"]
        output = _run_trackeval(
            trackeval_root=Path("external/TrackEval"),
            gt_root=gt_root,
            trackers_root=trackers_root,
            benchmark=benchmark,
            split_name="test",
            seqmap_file=seqmap_effective,
            tracker_names=tracker_names,
            iou_threshold=0.5,
        )
        for v in variants:
            metrics = _extract_combined_metrics(output, v.name)
            row = {
                "variant": v.name,
                "detector": v.detector,
                "tracker": v.tracker,
                "class": class_name,
                **metrics,
            }
            per_class_rows.append(row)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = results_dir / "metrics_per_class.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # Compute weights by GT track counts.
    weights_total: Dict[str, int] = {c: 0 for c in CLASSES_ORDERED}
    for seq_dir in test_seq_dirs:
        counts = count_unique_gt_tracks_by_class(seq_dir)
        for cls, val in counts.items():
            if cls in weights_total:
                weights_total[cls] += int(val)

    summary_rows: list[dict] = []
    for v in variants:
        df_v = per_class_df[per_class_df["variant"] == v.name]
        macro = {m: float(df_v[m].mean()) for m in ["HOTA", "IDF1", "MOTA", "FP", "FN", "ID-switch", "Frag"]}
        weighted = {}
        for m in ["HOTA", "IDF1", "MOTA", "FP", "FN", "ID-switch", "Frag"]:
            num = 0.0
            den = 0.0
            for cls in CLASSES_ORDERED:
                w = float(weights_total.get(cls, 0))
                val = float(df_v[df_v["class"] == cls][m].iloc[0])
                num += w * val
                den += w
            weighted[m] = (num / den) if den > 0 else 0.0
        t = timing.get(v.name, TimeStats())
        summary_rows.append(
            {
                "variant": v.name,
                "detector": v.detector,
                "tracker": v.tracker,
                "fps": t.fps,
                "inference_time_s": t.seconds,
                "frames_timed": t.frames,
                **{f"macro_{k}": v_ for k, v_ in macro.items()},
                **{f"weighted_{k}": v_ for k, v_ in weighted.items()},
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    per_sequence_df = pd.DataFrame(per_sequence_rows)
    per_sequence_csv = results_dir / "per_sequence_stats.csv"
    per_sequence_df.to_csv(per_sequence_csv, index=False)

    track_stats_summary = (
        per_sequence_df.groupby(["variant", "class"], as_index=False)
        .agg(
            id_ratio_mean=("id_ratio", "mean"),
            mean_track_len_mean=("mean_track_len", "mean"),
            median_track_len_mean=("median_track_len", "mean"),
            p90_track_len_mean=("p90_track_len", "mean"),
            pct_tracks_lt_5_mean=("pct_tracks_lt_5", "mean"),
            pct_tracks_lt_10_mean=("pct_tracks_lt_10", "mean"),
            # Stability metrics aggregates
            isr_mean_avg=("isr_mean", "mean"),
            isr_median_avg=("isr_median", "mean"),
            isr_ge_0_8_avg=("isr_ge_0.8", "mean"),
            isr_ge_0_9_avg=("isr_ge_0.9", "mean"),
            orc_15_avg=("orc@15", "mean"),
            orc_30_avg=("orc@30", "mean"),
            orc_60_avg=("orc@60", "mean"),
            drr_avg=("drr", "mean"),
            aor_avg=("aor", "mean"),
            pps_avg=("pps", "mean"),
            mss_mean_avg=("mss_mean", "mean"),
            mss_median_avg=("mss_median", "mean"),
            tci_avg=("tci", "mean"),
        )
    )
    track_stats_csv = results_dir / "track_stats_summary.csv"
    track_stats_summary.to_csv(track_stats_csv, index=False)

    try:
        _make_plots(results_dir, summary_df, per_class_df, per_sequence_df)
    except Exception as plot_err:  # pragma: no cover
        logger.error("Plot generation failed: {}", plot_err)

    manifest = {
        "train_root": str(train_root),
        "test_root": str(test_root),
        "coco_root": str(coco_root),
        "results_dir": str(results_dir),
        "seqmap_test": str(seqmap_test),
        "models": {"base": str(base_ckpt), "seg": str(seg_ckpt)},
        "training_times_s": training_times,
        "tuning_best": tuning_best,
        "gt_weights_tracks": weights_total,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.success("Experiment complete.")
    logger.info("Per-class metrics: {}", per_class_csv)
    logger.info("Summary: {}", summary_csv)
    logger.info("Per-sequence stats: {}", per_sequence_csv)
    logger.info("Track-stats summary: {}", track_stats_csv)
    logger.info("Manifest: {}", manifest_path)


if __name__ == "__main__":
    main()
