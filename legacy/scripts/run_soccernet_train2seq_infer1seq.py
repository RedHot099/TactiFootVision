#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from loguru import logger
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when invoked as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.models import (  # noqa: E402
    DetectionConfig,
    DetectionModelType,
    SAM2Config,
    TrackingConfig,
    TrainingDetectionConfig,
)
from tactifoot_vision.data.soccernet_tracking import (  # noqa: E402
    SOCCERNET_CLASS_TO_ID,
    export_mot_to_coco,
    iter_sequence_dirs,
    read_seqinfo,
)
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler  # noqa: E402
from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler  # noqa: E402
from tactifoot_vision.tracking.botsort_tracker import BoTSORTArgs, BoTSORTTracker  # noqa: E402
from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker  # noqa: E402
from tactifoot_vision.tracking.tracker import Tracker  # noqa: E402


CLASSES_ORDERED = ["player", "goalkeeper", "referee", "ball"]


@dataclass(frozen=True)
class Variant:
    name: str
    detector: str
    tracker: str


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
    limit = max(1, int(seqinfo.seq_length))
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


def _filter_by_geometry(
    detections: sv.Detections,
    *,
    frame_shape: tuple[int, int],
    min_area_ratio_by_class: dict[int, float] | None = None,
    max_area_ratio_by_class: dict[int, float] | None = None,
    max_aspect_by_class: dict[int, float] | None = None,
) -> sv.Detections:
    if len(detections) == 0 or detections.class_id is None:
        return detections
    xyxy = detections.xyxy.astype(np.float32)
    if xyxy.size == 0:
        return detections
    widths = np.clip(xyxy[:, 2] - xyxy[:, 0], 0.0, None)
    heights = np.clip(xyxy[:, 3] - xyxy[:, 1], 0.0, None)
    frame_h, frame_w = frame_shape
    frame_area = float(max(1, frame_h * frame_w))
    area_ratio = (widths * heights) / frame_area
    aspect = np.maximum(
        widths / np.maximum(heights, 1e-6),
        heights / np.maximum(widths, 1e-6),
    )
    class_ids = detections.class_id.astype(int)
    keep = np.ones(len(detections), dtype=bool)
    for idx, class_id in enumerate(class_ids):
        if widths[idx] <= 0.0 or heights[idx] <= 0.0:
            keep[idx] = False
            continue
        if min_area_ratio_by_class and class_id in min_area_ratio_by_class:
            if area_ratio[idx] < float(min_area_ratio_by_class[class_id]):
                keep[idx] = False
                continue
        if max_area_ratio_by_class and class_id in max_area_ratio_by_class:
            if area_ratio[idx] > float(max_area_ratio_by_class[class_id]):
                keep[idx] = False
                continue
        if max_aspect_by_class and class_id in max_aspect_by_class:
            if aspect[idx] > float(max_aspect_by_class[class_id]):
                keep[idx] = False
                continue
    try:
        return detections[keep]
    except Exception:
        return detections


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


def _parse_kv_floats(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object for key-value floats.")
        return {str(k): float(v) for k, v in data.items()}
    pairs = [item.strip() for item in text.split(",") if item.strip()]
    if not pairs:
        return None
    parsed: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid key=value pair: {pair}")
        key, value = pair.split("=", 1)
        parsed[key.strip()] = float(value)
    return parsed or None


def _map_class_values(
    raw: dict[str, float] | None, class_map: dict[str, int]
) -> dict[int, float] | None:
    if not raw:
        return None
    mapped = {
        int(class_map[name]): float(value)
        for name, value in raw.items()
        if name in class_map
    }
    return mapped or None


def _match_by_iou(
    tracked: sv.Detections,
    dets: sv.Detections,
    *,
    iou_threshold: float,
    return_iou: bool = False,
) -> list[tuple[int, int]] | list[tuple[int, int, float]]:
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
    matches: list[tuple[int, int]] | list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        iou = float(iou_matrix[r, c])
        if iou >= float(iou_threshold):
            if return_iou:
                matches.append((int(r), int(c), iou))
            else:
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
    blend_alpha_by_class: dict[int, float] | None = None,
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
            if blend_alpha_by_class is not None and tracked.class_id is not None:
                try:
                    cls_id = int(tracked.class_id[track_idx])
                    alpha = float(blend_alpha_by_class.get(cls_id, alpha))
                except Exception:
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


def _drop_unmatched_tracks(
    *,
    frame: np.ndarray,
    sam2_tracker: SAM2Tracker,
    tracked: sv.Detections,
    dets: sv.Detections,
    iou_threshold: float,
    unmatched_counts: dict[int, int],
    drop_after: int,
) -> sv.Detections:
    if len(tracked) == 0:
        return tracked
        
    # Match to find who is matched
    if len(dets) > 0:
        matches = _match_by_iou(tracked, dets, iou_threshold=iou_threshold)
        matched_track_indices = {track_idx for track_idx, _ in matches}
    else:
        matched_track_indices = set()

    current_ids = tracked.tracker_id.astype(int)
    
    # Identify IDs to remove
    ids_to_remove = []
    
    # Clean up counts for IDs that no longer exist
    current_id_set = set(current_ids)
    for tid in list(unmatched_counts):
        if tid not in current_id_set:
            unmatched_counts.pop(tid, None)

    for i, tid in enumerate(current_ids):
        if i in matched_track_indices:
            unmatched_counts.pop(tid, None) # Reset count if matched
        else:
            # Unmatched: increment count
            count = unmatched_counts.get(tid, 0) + 1
            unmatched_counts[tid] = count
            
            # Dynamic Lifespan Logic
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            
            limit = drop_after # Default
            
            # If very confident, allow longer life (fill gaps / occlusion)
            if conf > 0.85:
                limit = 30 
            # If uncertain, kill quickly (ghosts)
            elif conf < 0.60:
                limit = 1
                
            if count >= limit:
                ids_to_remove.append(tid)

    if not ids_to_remove:
        return tracked

    for tid in ids_to_remove:
        unmatched_counts.pop(tid, None)
    return sam2_tracker.remove_ids(frame, ids_to_remove)


def _track_length_stats(rows: list[list[float]]) -> dict[str, float]:
    if not rows:
        return {
            "pred_dets": 0.0,
            "pred_tracks": 0.0,
            "mean_track_len": 0.0,
            "median_track_len": 0.0,
            "p90_track_len": 0.0,
            "max_track_len": 0.0,
        }

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
        return {
            "pred_dets": float(len(rows)),
            "pred_tracks": 0.0,
            "mean_track_len": 0.0,
            "median_track_len": 0.0,
            "p90_track_len": 0.0,
            "max_track_len": 0.0,
        }

    return {
        "pred_dets": float(len(rows)),
        "pred_tracks": float(lengths.size),
        "mean_track_len": float(lengths.mean()),
        "median_track_len": float(np.median(lengths)),
        "p90_track_len": float(np.percentile(lengths, 90)),
        "max_track_len": float(lengths.max()),
    }


def _draw_tracked(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    *,
    class_name: str,
    color: tuple[int, int, int],
) -> None:
    if len(detections) == 0:
        return
    ids = detections.tracker_id.astype(int) if detections.tracker_id is not None else None
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int).tolist()
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = class_name
        if ids is not None and i < len(ids):
            label = f"{class_name}:{int(ids[i])}"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _variant_sanity_summary(
    *,
    rows_by_class: dict[str, list[list[float]]],
    frames_processed: int,
) -> dict:
    per_class: dict[str, dict] = {}
    total_rows = 0
    total_tracks = 0
    for cname, rows in rows_by_class.items():
        stats = _track_length_stats(rows)
        unique_tracks = int(stats["pred_tracks"])
        per_class[cname] = {
            "rows": int(len(rows)),
            "tracks": unique_tracks,
            "rows_per_frame": float(len(rows) / frames_processed) if frames_processed > 0 else 0.0,
            **stats,
        }
        total_rows += int(len(rows))
        total_tracks += unique_tracks
    return {
        "frames_processed": int(frames_processed),
        "total_rows": int(total_rows),
        "total_tracks": int(total_tracks),
        "per_class": per_class,
    }


def _make_sanity_plots(*, results_dir: Path, infer_out_dir: Path) -> None:
    plots_dir = Path("results/detection_tracking/plots") / results_dir.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    sanity_csv = results_dir / "sanity_summary.csv"
    if not sanity_csv.is_file():
        return
    sanity_df = pd.read_csv(sanity_csv)
    if sanity_df.empty:
        return

    sanity_df = sanity_df.sort_values("variant")

    def _bar(metric: str, title: str, ylabel: str, out_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        ax.bar(sanity_df["variant"].astype(str).tolist(), sanity_df[metric].astype(float).tolist())
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(sanity_df["variant"].astype(str).tolist(), rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(plots_dir / out_name, bbox_inches="tight")
        plt.close(fig)

    _bar("total_rows", "Sanity comparison: total MOT rows", "rows", "sanity_total_rows.png")
    _bar("total_tracks", "Sanity comparison: total unique tracks", "tracks", "sanity_total_tracks.png")

    # Per-class heatmap-like plot (matplotlib only).
    records: list[dict] = []
    for variant in sanity_df["variant"].astype(str).tolist():
        sanity_path = infer_out_dir / variant / "sanity.json"
        if not sanity_path.is_file():
            continue
        try:
            data = json.loads(sanity_path.read_text())
        except Exception:
            continue
        per_class = data.get("per_class") or {}
        frames = int(data.get("frames_processed") or 0)
        if frames <= 0:
            continue
        for cname, stats in per_class.items():
            try:
                rows = int(stats.get("rows") or 0)
                tracks = int(stats.get("tracks") or 0)
            except Exception:
                continue
            records.append(
                {
                    "variant": variant,
                    "class": str(cname),
                    "rows_per_frame": float(rows / frames) if frames else 0.0,
                    "tracks": tracks,
                }
            )

    if records:
        per_class_df = pd.DataFrame(records)
        pivot = per_class_df.pivot_table(
            index="class", columns="variant", values="rows_per_frame", aggfunc="mean"
        ).fillna(0.0)
        fig, ax = plt.subplots(figsize=(10, 2.6), dpi=200)
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title("Sanity comparison: rows/frame (per class)")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rows/frame")
        fig.tight_layout()
        fig.savefig(plots_dir / "sanity_rows_per_frame_heatmap.png", bbox_inches="tight")
        plt.close(fig)


def _pick_threshold_for_detector(
    detector,
    seq_dir: Path,
    *,
    thresholds: list[float],
    sample_frames: int = 60,
    min_total_per_frame: float = 1.0,
    min_player_per_frame: float = 1.0,
) -> float:
    player_id = SOCCERNET_CLASS_TO_ID["player"]
    best = thresholds[-1]

    for thr in thresholds:
        try:
            detector.detection_config.confidence_threshold = float(thr)
        except Exception:
            pass

        total = 0
        total_player = 0
        frames = 0
        for frame_idx, frame in _iter_frames(seq_dir, max_frames=sample_frames):
            _ = frame_idx
            dets = detector.detect(frame)
            total += int(len(dets))
            dets_player = _filter_by_class(dets, player_id)
            total_player += int(len(dets_player))
            frames += 1
        if frames <= 0:
            continue
        mean_total = total / frames
        mean_player = total_player / frames
        logger.info(
            "Threshold {:.2f}: mean_total={:.2f} mean_player={:.2f} (frames={})",
            thr,
            mean_total,
            mean_player,
            frames,
        )
        if mean_total >= float(min_total_per_frame) and mean_player >= float(min_player_per_frame):
            return float(thr)
        best = float(thr)
    return float(best)


def _maybe_optimize_detector(detector) -> None:
    model = getattr(detector, "model", None)
    if model is not None and hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            return


def _best_checkpoint_in_dir(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "checkpoint_best_total.pth",
        output_dir / "checkpoint_best_ema.pth",
        output_dir / "checkpoint_best_regular.pth",
        output_dir / "checkpoint.pth",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_existing_sanity(out_variant: Path) -> dict | None:
    sanity_path = out_variant / "sanity.json"
    if not sanity_path.is_file():
        return None
    try:
        data = json.loads(sanity_path.read_text())
    except Exception:
        return None
    try:
        frames = int(data.get("frames_processed", 0))
        total_rows = int(data.get("total_rows", 0))
        total_tracks = int(data.get("total_tracks", 0))
    except Exception:
        return None
    if frames <= 0 or total_rows <= 0:
        return None
    return {"frames": frames, "total_rows": total_rows, "total_tracks": total_tracks}


def _ensure_exported_checkpoint(
    *,
    name: str,
    final_ckpt: Path,
    training_output_dir: Path,
    force_train: bool,
    train_fn,
) -> tuple[Path, float, dict]:
    """Return (checkpoint_path, training_seconds, meta)."""
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "name": name,
        "final_ckpt": str(final_ckpt),
        "training_output_dir": str(training_output_dir),
        "force_train": bool(force_train),
    }

    if final_ckpt.is_file() and not force_train:
        meta.update({"used": "existing_final_ckpt"})
        return final_ckpt, 0.0, meta

    best = _best_checkpoint_in_dir(training_output_dir)
    if best is not None and not force_train:
        shutil.copy2(best, final_ckpt)
        meta.update({"used": "copied_from_training_output", "source_ckpt": str(best)})
        return final_ckpt, 0.0, meta

    t0 = time.perf_counter()
    train_fn()
    train_s = time.perf_counter() - t0

    if final_ckpt.is_file():
        meta.update({"used": "trained_and_saved", "training_seconds": float(train_s)})
        return final_ckpt, float(train_s), meta

    best_after = _best_checkpoint_in_dir(training_output_dir)
    if best_after is not None:
        shutil.copy2(best_after, final_ckpt)
        meta.update(
            {
                "used": "trained_then_copied_from_training_output",
                "source_ckpt": str(best_after),
                "training_seconds": float(train_s),
            }
        )
        return final_ckpt, float(train_s), meta

    best = float(best)
    return best


def _interpolate_tracks(rows: list[list[float]], max_gap: int = 5) -> list[list[float]]:
    """
    Linearly interpolates missing frames in tracks.
    rows: list of [frame, id, x, y, w, h, conf, ...]
    """
    if not rows:
        return []
    
    # Organize by track_id
    tracks = {}
    for r in rows:
        tid = int(r[1])
        # frame = int(r[0])
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append(r)
        
    interpolated_rows = []
    
    for tid, track_rows in tracks.items():
        # Sort by frame
        track_rows.sort(key=lambda x: int(x[0]))
        
        new_track_rows = []
        for i in range(len(track_rows)):
            curr_row = track_rows[i]
            new_track_rows.append(curr_row)
            
            if i < len(track_rows) - 1:
                next_row = track_rows[i+1]
                frame_diff = int(next_row[0]) - int(curr_row[0])
                
                if 1 < frame_diff <= max_gap + 1:
                    # Interpolate
                    start_frame = int(curr_row[0])
                    end_frame = int(next_row[0])
                    
                    start_box = np.array(curr_row[2:6]) # x, y, w, h
                    end_box = np.array(next_row[2:6])
                    
                    for f in range(start_frame + 1, end_frame):
                        alpha = (f - start_frame) / (end_frame - start_frame)
                        interp_box = start_box + alpha * (end_box - start_box)
                        
                        # Copy row structure (conf, class, etc) from start
                        # Set conf to mean or start? Let's use start conf but maybe lower?
                        # Using start conf is standard.
                        interp_row = list(curr_row)
                        interp_row[0] = float(f)
                        interp_row[2:6] = interp_box.tolist()
                        interp_row[6] = curr_row[6] * 0.9 # Penalize confidence slightly
                        
                        new_track_rows.append(interp_row)
                        
        interpolated_rows.extend(new_track_rows)
        
    # Sort all by frame again (optional but good for writing)
    interpolated_rows.sort(key=lambda x: int(x[0]))
    return interpolated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR Base+Seg on first 2 SoccerNet train sequences (100 epochs) and run inference+tracking on the 1st sequence."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/train"),
        help="Path to extracted tracking-2023 train split (SNMOT-* dirs).",
    )
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=None,
        help="Path to extracted split used for inference (default: same as --train-root).",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("data/soccernet/tracking/coco_tracking_2023_first2seq"),
        help="COCO dataset output root (will be created if missing).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/detection_tracking/raw/soccernet_tracking_train2seq_100ep_infer1seq"),
        help="Output directory for training+inference outputs.",
    )
    parser.add_argument(
        "--train-sequences",
        type=int,
        default=2,
        help="Number of first sequences from train used to build the COCO dataset (default: 2).",
    )
    parser.add_argument(
        "--infer-sequence-index",
        type=int,
        default=0,
        help="Index (0-based) of the sequence under --infer-root to run inference on (default: 0 -> first).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for each detector (default: 100).",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining even if checkpoints already exist under results_dir.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip any training and only run inference+tracking using existing checkpoints.",
    )
    parser.add_argument(
        "--detectors",
        choices=["base", "seg", "both"],
        default="both",
        help="Which detector(s) to run during inference (default: both).",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint for RF-DETR Base inference (overrides results_dir discovery).",
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint for RF-DETR Seg inference (overrides results_dir discovery).",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=8,
        help="How many preview frames to save per variant (default: 8).",
    )
    parser.add_argument(
        "--resume-inference",
        action="store_true",
        help="If variant outputs already exist, reuse them and only (re)write summaries/manifests.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional max frames for inference (0 = full sequence).",
    )
    parser.add_argument(
        "--detector-confidence-threshold",
        type=float,
        default=None,
        help="Force detector confidence threshold (skip auto-picking when set).",
    )
    parser.add_argument(
        "--detector-per-class-thresholds",
        type=str,
        default="",
        help="Per-class thresholds, e.g. 'player=0.25,ball=0.1' or JSON dict.",
    )
    parser.add_argument(
        "--detector-nms-threshold",
        type=float,
        default=None,
        help="Override detector NMS threshold (applied after per-class filtering).",
    )
    parser.add_argument(
        "--base-pretrain",
        type=Path,
        default=Path("rf-detr-base.pth"),
        help="RF-DETR Base pretrain weights path.",
    )
    parser.add_argument(
        "--seg-pretrain",
        type=Path,
        default=Path("rf-detr-seg-preview.pt"),
        help="RF-DETR Seg (preview) pretrain weights path.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"),
        help="SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--sam2-config",
        type=Path,
        default=Path("external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"),
        help="SAM2 config yaml path.",
    )
    parser.add_argument(
        "--sam2-max-side",
        type=int,
        default=768,
        help="Max image side for SAM2 (downscale for VRAM, default: 768).",
    )
    parser.add_argument(
        "--sam2-max-objects",
        type=int,
        default=32,
        help="Max objects to prompt for SAM2 (default: 32).",
    )
    parser.add_argument(
        "--sam2-reseed-interval",
        type=int,
        default=45,
        help="Reseed interval for SAM2 (default: 45).",
    )
    parser.add_argument(
        "--sam2-reseed-iou",
        type=float,
        default=0.3,
        help="IoU threshold for SAM2 reseeding matches (default: 0.3).",
    )
    parser.add_argument(
        "--sam2-reseed-skip-iou",
        type=float,
        default=0.0,
        help="Skip reseed if all matches have IoU >= threshold (0 disables).",
    )
    parser.add_argument(
        "--sam2-reseed-mode",
        choices=["add_new", "reanchor"],
        default="add_new",
        help="Reseeding strategy for SAM2 (default: add_new).",
    )
    parser.add_argument(
        "--sam2-drop-after",
        type=int,
        default=0,
        help="Drop tracks after N reseeds without a detection match (0 disables).",
    )
    parser.add_argument(
        "--sam2-unmatched-drop-after",
        type=int,
        default=0,
        help="Drop tracks after N consecutive frames without a detection match (0 disables).",
    )
    parser.add_argument(
        "--sam2-mask-threshold",
        type=float,
        default=0.0,
        help="Mask logit threshold for SAM2 (default: 0.0).",
    )
    parser.add_argument(
        "--sam2-mask-filter-distance",
        type=float,
        default=300.0,
        help="Connected-components distance filter for SAM2 masks (default: 300).",
    )
    parser.add_argument(
        "--sam2-mask-open",
        type=int,
        default=0,
        help="Morphological open kernel size for SAM2 masks (0 disables).",
    )
    parser.add_argument(
        "--sam2-mask-close",
        type=int,
        default=0,
        help="Morphological close kernel size for SAM2 masks (0 disables).",
    )
    parser.add_argument(
        "--sam2-output-mode",
        choices=["mask", "detector", "detector_strict", "detector_blend"],
        default="mask",
        help=(
            "SAM2 output box source: mask, detector (snap), "
            "detector_strict (snap + drop unmatched), "
            "or detector_blend (snap + blend)."
        ),
    )
    parser.add_argument(
        "--sam2-output-iou",
        type=float,
        default=0.3,
        help="IoU threshold for snapping SAM2 boxes to detector boxes.",
    )
    parser.add_argument(
        "--sam2-output-blend-alpha",
        type=float,
        default=0.7,
        help="Blend factor for detector_blend mode (1.0 = detector, 0.0 = mask).",
    )
    parser.add_argument(
        "--sam2-output-blend-alpha-by-class",
        type=str,
        default="",
        help="Per-class blend alpha, e.g. 'player=0.5,ball=0.8' or JSON dict.",
    )
    parser.add_argument(
        "--sam2-geom-min-area-ratio",
        type=str,
        default="",
        help="Per-class min area ratio (bbox_area / frame_area), e.g. 'player=0.0004'.",
    )
    parser.add_argument(
        "--sam2-geom-max-area-ratio",
        type=str,
        default="",
        help="Per-class max area ratio (bbox_area / frame_area), e.g. 'player=0.2'.",
    )
    parser.add_argument(
        "--sam2-geom-max-aspect",
        type=str,
        default="",
        help="Per-class max aspect ratio (max(w/h, h/w)), e.g. 'player=8'.",
    )
    parser.add_argument(
        "--sam2-allowed-classes",
        type=str,
        default="",
        help="Comma-separated class names allowed for SAM2 (empty=all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    infer_root = (args.infer_root or args.train_root).resolve()
    coco_root = args.coco_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs_infer = iter_sequence_dirs(infer_root)
    if not seq_dirs_infer:
        raise RuntimeError(f"Expected sequences under {infer_root}, got 0")
    infer_seq_dir = seq_dirs_infer[int(args.infer_sequence_index)]
    infer_seqinfo = read_seqinfo(infer_seq_dir)

    if not bool(args.skip_training):
        seq_dirs_train = iter_sequence_dirs(train_root)
        if len(seq_dirs_train) < max(1, int(args.train_sequences)):
            raise RuntimeError(
                f"Expected at least {int(args.train_sequences)} sequences under {train_root}, got {len(seq_dirs_train)}"
            )
        train_seq_dirs = seq_dirs_train[: int(args.train_sequences)]
        logger.info("Train sequences: {}", [p.name for p in train_seq_dirs])

        # Build COCO dataset (train/valid/test) from the first N train sequences.
        coco_train_ann = coco_root / "train" / "_annotations.coco.json"
        coco_valid_ann = coco_root / "valid" / "_annotations.coco.json"
        coco_test_ann = coco_root / "test" / "_annotations.coco.json"
        if not coco_train_ann.is_file() or not coco_valid_ann.is_file() or not coco_test_ann.is_file():
            logger.info("COCO dataset missing, generating at {}", coco_root)
            export_mot_to_coco(
                train_root,
                coco_root,
                valid_fraction=0.0,
                seed=42,
                every_nth_frame=1,
                max_sequences=int(args.train_sequences),
                symlink_images=True,
            )
    else:
        train_seq_dirs = []

    logger.info("Inference root: {}", infer_root)
    logger.info("Inference sequence: {}", infer_seq_dir.name)

    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = models_dir / "rfdetr_base_train2seq_100ep.pth"
    seg_ckpt = models_dir / "rfdetr_seg_train2seq_100ep.pth"

    want_base = args.detectors in ("base", "both")
    want_seg = args.detectors in ("seg", "both")
    per_class_thresholds = _parse_kv_floats(args.detector_per_class_thresholds)
    if per_class_thresholds:
        per_class_thresholds = {
            name: value
            for name, value in per_class_thresholds.items()
            if name in SOCCERNET_CLASS_TO_ID
        }
    detector_nms_threshold = (
        float(args.detector_nms_threshold)
        if args.detector_nms_threshold is not None
        else 0.5
    )

    # Prepare training configs (training may be skipped if checkpoints already exist).
    base_det_cfg = DetectionConfig(
        model_type=DetectionModelType.RFDETR,
        checkpoint_path=args.base_pretrain,
        confidence_threshold=0.3,
        nms_threshold=detector_nms_threshold,
        per_class_confidence_thresholds=per_class_thresholds,
        classes=SOCCERNET_CLASS_TO_ID,
    )
    base_train_cfg = TrainingDetectionConfig(
        dataset_path=coco_root,
        dataset_format="coco",
        output_dir=results_dir / "training" / "rfdetr_base",
        save_checkpoint_path=base_ckpt,
        epochs=int(args.epochs),
        batch_size=8,
        grad_accum_steps=2,
        num_workers=2,
        learning_rate=1e-4,
        optimizer="AdamW",
        early_stopping=False,
    )
    seg_det_cfg = None
    seg_train_cfg = None
    if want_seg:
        seg_det_cfg = DetectionConfig(
            model_type=DetectionModelType.RFDETR_SEG,
            checkpoint_path=args.seg_pretrain,
            confidence_threshold=0.3,
            nms_threshold=detector_nms_threshold,
            per_class_confidence_thresholds=per_class_thresholds,
            classes=SOCCERNET_CLASS_TO_ID,
        )
        seg_train_cfg = TrainingDetectionConfig(
            dataset_path=coco_root,
            dataset_format="coco",
            output_dir=results_dir / "training" / "rfdetr_seg",
            save_checkpoint_path=seg_ckpt,
            epochs=int(args.epochs),
            batch_size=2,
            grad_accum_steps=1,
            num_workers=2,
            learning_rate=1e-4,
            optimizer="AdamW",
            early_stopping=False,
        )

    base_train_s = 0.0
    seg_train_s = 0.0
    base_meta: dict = {"used": "n/a"}
    seg_meta: dict = {"used": "n/a"}

    if want_base:
        if args.base_checkpoint is not None:
            base_ckpt = args.base_checkpoint.resolve()
            if not base_ckpt.is_file():
                raise FileNotFoundError(f"--base-checkpoint not found: {base_ckpt}")
            base_meta = {"used": "explicit_arg", "path": str(base_ckpt)}
        elif bool(args.skip_training):
            # Discover an existing checkpoint from prior runs.
            if base_ckpt.is_file():
                base_meta = {"used": "existing_final_ckpt", "path": str(base_ckpt)}
            else:
                candidate = _best_checkpoint_in_dir(results_dir / "training" / "rfdetr_base")
                if candidate is None:
                    raise FileNotFoundError(
                        "skip-training enabled but no RF-DETR Base checkpoint found under "
                        f"{results_dir / 'models'} or {results_dir / 'training' / 'rfdetr_base'}"
                    )
                base_ckpt = candidate
                base_meta = {"used": "discovered_training_ckpt", "path": str(base_ckpt)}
        else:
            logger.info(
                "Ensuring RF-DETR Base checkpoint (epochs={} force_train={})...",
                int(args.epochs),
                bool(args.force_train),
            )
            base_ckpt, base_train_s, base_meta = _ensure_exported_checkpoint(
                name="rfdetr_base",
                final_ckpt=base_ckpt,
                training_output_dir=Path(base_train_cfg.output_dir),
                force_train=bool(args.force_train),
                train_fn=lambda: RFDETRHandler(
                    base_det_cfg, training_config=base_train_cfg, model_dir=project_root
                ).train(),
            )
        logger.success(
            "RF-DETR Base ready -> {} (train_time={:.1f}s, mode={})",
            base_ckpt,
            float(base_train_s),
            base_meta.get("used"),
        )

    if want_seg:
        if args.seg_checkpoint is not None:
            seg_ckpt = args.seg_checkpoint.resolve()
            if not seg_ckpt.is_file():
                raise FileNotFoundError(f"--seg-checkpoint not found: {seg_ckpt}")
            seg_meta = {"used": "explicit_arg", "path": str(seg_ckpt)}
        elif bool(args.skip_training):
            if seg_ckpt.is_file():
                seg_meta = {"used": "existing_final_ckpt", "path": str(seg_ckpt)}
            else:
                candidate = _best_checkpoint_in_dir(results_dir / "training" / "rfdetr_seg")
                if candidate is None:
                    raise FileNotFoundError(
                        "skip-training enabled but no RF-DETR Seg checkpoint found under "
                        f"{results_dir / 'models'} or {results_dir / 'training' / 'rfdetr_seg'}"
                    )
                seg_ckpt = candidate
                seg_meta = {"used": "discovered_training_ckpt", "path": str(seg_ckpt)}
        else:
            assert seg_det_cfg is not None and seg_train_cfg is not None
            logger.info(
                "Ensuring RF-DETR Seg checkpoint (epochs={} force_train={})...",
                int(args.epochs),
                bool(args.force_train),
            )
            seg_ckpt, seg_train_s, seg_meta = _ensure_exported_checkpoint(
                name="rfdetr_seg",
                final_ckpt=seg_ckpt,
                training_output_dir=Path(seg_train_cfg.output_dir),
                force_train=bool(args.force_train),
                train_fn=lambda: RFDETRSegHandler(
                    seg_det_cfg, training_config=seg_train_cfg, model_dir=project_root
                ).train(),
            )
        logger.success(
            "RF-DETR Seg ready -> {} (train_time={:.1f}s, mode={})",
            seg_ckpt,
            float(seg_train_s),
            seg_meta.get("used"),
        )

    # Prepare inference output structure.
    infer_out_dir = results_dir / "inference" / infer_seq_dir.name
    infer_out_dir.mkdir(parents=True, exist_ok=True)

    # Load detectors for inference (one at a time to keep VRAM stable).
    detector_specs = []
    if want_base:
        detector_specs.append(("rfdetr_base", DetectionModelType.RFDETR, base_ckpt))
    if want_seg:
        detector_specs.append(("rfdetr_seg", DetectionModelType.RFDETR_SEG, seg_ckpt))

    # Preview frame indices.
    preview_n = max(0, int(args.preview_frames))
    preview_frames: set[int] = set()
    if preview_n > 0:
        choices = np.linspace(1, max(1, int(infer_seqinfo.seq_length)), preview_n, dtype=int).tolist()
        preview_frames = {int(x) for x in choices if int(x) > 0}

    # Tracker configs.
    bytetrack_cfg = TrackingConfig(
        enabled=True,
        backend="bytetrack",
        frame_rate=int(infer_seqinfo.frame_rate),
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    )
    botsort_args = BoTSORTArgs(with_reid=True, model=str((project_root / "yolo11n.pt").resolve()))
    sam2_cfg = TrackingConfig(
        enabled=True,
        backend="sam2",
        frame_rate=int(infer_seqinfo.frame_rate),
        sam2=SAM2Config(
            checkpoint_path=args.sam2_checkpoint,
            config_path=args.sam2_config,
            max_side=int(args.sam2_max_side),
            max_objects=int(args.sam2_max_objects),
            reseed_interval=int(args.sam2_reseed_interval),
            reseed_iou_threshold=float(args.sam2_reseed_iou),
            reseed_skip_iou_threshold=float(args.sam2_reseed_skip_iou),
            reseed_mode=str(args.sam2_reseed_mode),
            drop_after=int(args.sam2_drop_after),
            unmatched_drop_after=int(args.sam2_unmatched_drop_after),
            mask_filter_distance=float(args.sam2_mask_filter_distance),
            mask_threshold=float(args.sam2_mask_threshold),
            mask_open=int(args.sam2_mask_open),
            mask_close=int(args.sam2_mask_close),
            output_box_mode=str(args.sam2_output_mode),
            output_box_iou_threshold=float(args.sam2_output_iou),
            output_box_blend_alpha=float(args.sam2_output_blend_alpha),
            output_box_blend_alpha_by_class=_parse_kv_floats(args.sam2_output_blend_alpha_by_class),
            output_box_geom_min_area_ratio_by_class=_parse_kv_floats(
                args.sam2_geom_min_area_ratio
            ),
            output_box_geom_max_area_ratio_by_class=_parse_kv_floats(
                args.sam2_geom_max_area_ratio
            ),
            output_box_geom_max_aspect_ratio_by_class=_parse_kv_floats(
                args.sam2_geom_max_aspect
            ),
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

    sanity_rows: list[dict] = []
    manifest: dict = {
        "train_root": str(train_root),
        "infer_root": str(infer_root),
        "coco_root": str(coco_root),
        "results_dir": str(results_dir),
        "train_sequences": [p.name for p in train_seq_dirs],
        "infer_sequence": infer_seq_dir.name,
        "models": {"base": str(base_ckpt), "seg": str(seg_ckpt)},
        "training_time_s": {"rfdetr_base": float(base_train_s), "rfdetr_seg": float(seg_train_s)},
        "training_resume": {"rfdetr_base": base_meta, "rfdetr_seg": seg_meta},
        "variants": [v.__dict__ for v in variants],
        "tracker_configs": {
            "bytetrack": bytetrack_cfg.model_dump(),
            "botsort_reid": botsort_args.__dict__,
            "sam2": sam2_cfg.model_dump(),
        },
        "detector_settings": {
            "nms_threshold": float(detector_nms_threshold),
            "per_class_thresholds": per_class_thresholds,
        },
    }

    for detector_name, model_type, ckpt in detector_specs:
        logger.info("Loading detector: {} from {}", detector_name, ckpt)
        det_handler = (
            RFDETRHandler(
                DetectionConfig(
                    model_type=DetectionModelType.RFDETR,
                    checkpoint_path=ckpt,
                    confidence_threshold=0.3,
                    nms_threshold=detector_nms_threshold,
                    per_class_confidence_thresholds=per_class_thresholds,
                    classes=SOCCERNET_CLASS_TO_ID,
                ),
                model_dir=project_root,
            )
            if model_type == DetectionModelType.RFDETR
            else RFDETRSegHandler(
                DetectionConfig(
                    model_type=DetectionModelType.RFDETR_SEG,
                    checkpoint_path=ckpt,
                    confidence_threshold=0.3,
                    nms_threshold=detector_nms_threshold,
                    per_class_confidence_thresholds=per_class_thresholds,
                    classes=SOCCERNET_CLASS_TO_ID,
                ),
                model_dir=project_root,
            )
        )
        _maybe_optimize_detector(det_handler)

        forced_thr = args.detector_confidence_threshold
        if forced_thr is not None:
            det_handler.detection_config.confidence_threshold = float(forced_thr)
            manifest.setdefault("chosen_thresholds", {})[detector_name] = float(forced_thr)
            logger.success(
                "Detector {} using forced confidence_threshold={:.2f}",
                detector_name,
                float(forced_thr),
            )
        else:
            # Auto-pick a threshold that yields non-empty predictions on this sequence.
            chosen_thr = _pick_threshold_for_detector(
                det_handler,
                infer_seq_dir,
                thresholds=[0.3, 0.2, 0.1, 0.05],
                sample_frames=min(60, int(infer_seqinfo.seq_length)),
                min_total_per_frame=1.0,
                min_player_per_frame=1.0,
            )
            det_handler.detection_config.confidence_threshold = float(chosen_thr)
            manifest.setdefault("chosen_thresholds", {})[detector_name] = float(chosen_thr)
            logger.success("Detector {} using confidence_threshold={:.2f}", detector_name, chosen_thr)

        max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

        # Pass 1: ByteTrack + BoT-SORT (share detections).
        bt_variant_name = f"{detector_name}__bytetrack"
        bs_variant_name = f"{detector_name}__botsort_reid"
        bt_out_variant = infer_out_dir / bt_variant_name
        bs_out_variant = infer_out_dir / bs_variant_name

        resumed_bt_bs = False
        if bool(args.resume_inference):
            bt_existing = _load_existing_sanity(bt_out_variant)
            bs_existing = _load_existing_sanity(bs_out_variant)
            if bt_existing is not None and bs_existing is not None:
                sanity_rows.append(
                    {
                        "variant": bt_variant_name,
                        "detector": detector_name,
                        "tracker": "bytetrack",
                        "frames": int(bt_existing["frames"]),
                        "total_rows": int(bt_existing["total_rows"]),
                        "total_tracks": int(bt_existing["total_tracks"]),
                    }
                )
                sanity_rows.append(
                    {
                        "variant": bs_variant_name,
                        "detector": detector_name,
                        "tracker": "botsort_reid",
                        "frames": int(bs_existing["frames"]),
                        "total_rows": int(bs_existing["total_rows"]),
                        "total_tracks": int(bs_existing["total_tracks"]),
                    }
                )
                logger.success(
                    "Resumed existing outputs for {}: ByteTrack(rows={},tracks={}) BoT-SORT(rows={},tracks={})",
                    detector_name,
                    bt_existing["total_rows"],
                    bt_existing["total_tracks"],
                    bs_existing["total_rows"],
                    bs_existing["total_tracks"],
                )
                resumed_bt_bs = True

        if not resumed_bt_bs:
            logger.info("Running {} + ByteTrack/BoT-SORT on {}", detector_name, infer_seq_dir.name)
            bytetrack_trackers = {c: Tracker(bytetrack_cfg) for c in CLASSES_ORDERED}
            botsort_trackers = {
                c: BoTSORTTracker(botsort_args, frame_rate=int(infer_seqinfo.frame_rate))
                for c in CLASSES_ORDERED
            }

            rows_bt: dict[str, list[list[float]]] = {c: [] for c in CLASSES_ORDERED}
            rows_bs: dict[str, list[list[float]]] = {c: [] for c in CLASSES_ORDERED}

            frames_processed = 0
            for frame_idx, frame in _iter_frames(infer_seq_dir, max_frames=max_frames):
                frames_processed += 1
                dets = det_handler.detect(frame)
                tracked_bt_by_class: dict[str, sv.Detections] = {}
                tracked_bs_by_class: dict[str, sv.Detections] = {}
                for cname in CLASSES_ORDERED:
                    class_id = SOCCERNET_CLASS_TO_ID[cname]
                    dets_c = _filter_by_class(dets, class_id)
                    tracked_bt = bytetrack_trackers[cname].update(dets_c)
                    tracked_bs = botsort_trackers[cname].update(dets_c, frame)
                    tracked_bt_by_class[cname] = tracked_bt
                    tracked_bs_by_class[cname] = tracked_bs
                    rows_bt[cname].extend(
                        _detections_to_mot_rows(tracked_bt, frame_idx, default_confidence=1.0)
                    )
                    rows_bs[cname].extend(
                        _detections_to_mot_rows(tracked_bs, frame_idx, default_confidence=1.0)
                    )

                if frame_idx in preview_frames:
                    bt_img = frame.copy()
                    bs_img = frame.copy()
                    for cname in CLASSES_ORDERED:
                        class_id = SOCCERNET_CLASS_TO_ID[cname]
                        color = (
                            (0, 255, 0)
                            if class_id == SOCCERNET_CLASS_TO_ID["player"]
                            else (255, 0, 0)
                        )
                        _draw_tracked(
                            bt_img,
                            tracked_bt_by_class.get(cname, sv.Detections.empty()),
                            class_name=cname,
                            color=color,
                        )
                        _draw_tracked(
                            bs_img,
                            tracked_bs_by_class.get(cname, sv.Detections.empty()),
                            class_name=cname,
                            color=color,
                        )
                    (bt_out_variant / "preview").mkdir(parents=True, exist_ok=True)
                    (bs_out_variant / "preview").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(bt_out_variant / "preview" / f"{frame_idx:06d}.jpg"), bt_img)
                    cv2.imwrite(str(bs_out_variant / "preview" / f"{frame_idx:06d}.jpg"), bs_img)

            for tracker_key, rows_by_class in [("bytetrack", rows_bt), ("botsort_reid", rows_bs)]:
                variant_name = f"{detector_name}__{tracker_key}"
                out_variant = infer_out_dir / variant_name
                mot_dir = out_variant / "mot"
                mot_dir.mkdir(parents=True, exist_ok=True)
                for cname, rows in rows_by_class.items():
                    _write_mot(rows, mot_dir / f"{cname}.txt")
                summary = _variant_sanity_summary(
                    rows_by_class=rows_by_class, frames_processed=frames_processed
                )
                (out_variant / "sanity.json").write_text(json.dumps(summary, indent=2))
                sanity_rows.append(
                    {
                        "variant": variant_name,
                        "detector": detector_name,
                        "tracker": tracker_key,
                        "frames": int(frames_processed),
                        "total_rows": int(summary["total_rows"]),
                        "total_tracks": int(summary["total_tracks"]),
                    }
                )
                if int(summary["total_rows"]) == 0:
                    raise RuntimeError(
                        f"Variant {variant_name} produced zero rows; check training/thresholds."
                    )
                logger.success(
                    "Sanity {}: frames={} rows={} tracks={}",
                    variant_name,
                    summary["frames_processed"],
                    summary["total_rows"],
                    summary["total_tracks"],
                )

        # Pass 2: SAM2 (separate run to keep VRAM stable).
        sam2_variant_name = f"{detector_name}__sam2"
        out_variant = infer_out_dir / sam2_variant_name
        if bool(args.resume_inference):
            existing = _load_existing_sanity(out_variant)
            if existing is not None:
                sanity_rows.append(
                    {
                        "variant": sam2_variant_name,
                        "detector": detector_name,
                        "tracker": "sam2",
                        "frames": int(existing["frames"]),
                        "total_rows": int(existing["total_rows"]),
                        "total_tracks": int(existing["total_tracks"]),
                    }
                )
                logger.success(
                    "Resumed existing outputs for {}: SAM2(rows={},tracks={})",
                    detector_name,
                    existing["total_rows"],
                    existing["total_tracks"],
                )
                import torch  # type: ignore

                try:
                    del det_handler
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue

        logger.info("Running {} + SAM2 on {}", detector_name, infer_seq_dir.name)
        (out_variant / "mot").mkdir(parents=True, exist_ok=True)
        (out_variant / "preview").mkdir(parents=True, exist_ok=True)

        import torch  # type: ignore

        torch.cuda.empty_cache()
        sam2_tracker = SAM2Tracker(sam2_cfg)
        sam2_initialized = False
        sam2_reset_done = False
        reseed_interval = sam2_cfg.sam2.reseed_interval if sam2_cfg.sam2 is not None else None
        reseed_iou_threshold = float(sam2_cfg.sam2.reseed_iou_threshold) if sam2_cfg.sam2 is not None else 0.3
        reseed_skip_iou_threshold = (
            float(sam2_cfg.sam2.reseed_skip_iou_threshold) if sam2_cfg.sam2 is not None else 0.0
        )
        reseed_mode = str(sam2_cfg.sam2.reseed_mode) if sam2_cfg.sam2 is not None else "add_new"
        drop_after = int(sam2_cfg.sam2.drop_after) if sam2_cfg.sam2 is not None else 0
        unmatched_drop_after = (
            int(sam2_cfg.sam2.unmatched_drop_after) if sam2_cfg.sam2 is not None else 0
        )
        output_box_mode = str(sam2_cfg.sam2.output_box_mode) if sam2_cfg.sam2 is not None else "mask"
        output_box_iou = float(sam2_cfg.sam2.output_box_iou_threshold) if sam2_cfg.sam2 is not None else 0.3
        output_box_blend_alpha = (
            float(sam2_cfg.sam2.output_box_blend_alpha) if sam2_cfg.sam2 is not None else 0.7
        )
        snap_to_detector = output_box_mode in {"detector", "detector_strict", "detector_blend"}
        drop_unmatched = output_box_mode == "detector_strict"
        blend_alpha = output_box_blend_alpha if output_box_mode == "detector_blend" else None
        blend_alpha_by_class = None
        if (
            output_box_mode == "detector_blend"
            and sam2_cfg.sam2 is not None
            and sam2_cfg.sam2.output_box_blend_alpha_by_class
        ):
            blend_alpha_by_class = _map_class_values(
                sam2_cfg.sam2.output_box_blend_alpha_by_class,
                SOCCERNET_CLASS_TO_ID,
            )
        geom_min_area_by_class = None
        geom_max_area_by_class = None
        geom_max_aspect_by_class = None
        if sam2_cfg.sam2 is not None:
            geom_min_area_by_class = _map_class_values(
                sam2_cfg.sam2.output_box_geom_min_area_ratio_by_class,
                SOCCERNET_CLASS_TO_ID,
            )
            geom_max_area_by_class = _map_class_values(
                sam2_cfg.sam2.output_box_geom_max_area_ratio_by_class,
                SOCCERNET_CLASS_TO_ID,
            )
            geom_max_aspect_by_class = _map_class_values(
                sam2_cfg.sam2.output_box_geom_max_aspect_ratio_by_class,
                SOCCERNET_CLASS_TO_ID,
            )
        sam2_stale_counts: dict[int, int] = {}
        sam2_unmatched_counts: dict[int, int] = {}
        allowed_classes_raw = str(args.sam2_allowed_classes).strip()
        sam2_allowed_class_ids = None
        if allowed_classes_raw:
            names = [c.strip() for c in allowed_classes_raw.split(",") if c.strip()]
            sam2_allowed_class_ids = [
                SOCCERNET_CLASS_TO_ID[name]
                for name in names
                if name in SOCCERNET_CLASS_TO_ID
            ]
            if not sam2_allowed_class_ids:
                sam2_allowed_class_ids = None

        rows_sam2: dict[str, list[list[float]]] = {c: [] for c in CLASSES_ORDERED}
        frames_processed = 0

        for idx, (frame_idx, frame) in enumerate(_iter_frames(infer_seq_dir, max_frames=max_frames)):
            frames_processed += 1
            dets = det_handler.detect(frame)
            dets_for_sam2 = dets
            if sam2_allowed_class_ids is not None:
                if dets.class_id is not None and len(dets) > 0:
                    mask = np.isin(dets.class_id, sam2_allowed_class_ids)
                    dets_for_sam2 = dets[mask]
                else:
                    dets_for_sam2 = sv.Detections.empty()

            tracked = sv.Detections.empty()
            if not sam2_reset_done:
                sam2_tracker.initialize(frame, np.empty((0, 4), dtype=np.float32), None)
                sam2_reset_done = True

            if not sam2_initialized:
                if len(dets_for_sam2) > 0:
                    sam2_tracker.initialize(frame, dets_for_sam2.xyxy, dets_for_sam2.class_id)
                    sam2_initialized = True
                    tracked = sam2_tracker.track(frame)
            else:
                tracked = sam2_tracker.track(frame)

            if snap_to_detector and len(tracked) > 0:
                tracked = _snap_boxes_to_detections(
                    tracked,
                    dets_for_sam2,
                    iou_threshold=output_box_iou,
                    drop_unmatched=drop_unmatched,
                    blend_alpha=blend_alpha,
                    blend_alpha_by_class=blend_alpha_by_class,
                )

            if (
                sam2_initialized
                and reseed_interval is not None
                and int(reseed_interval) > 0
                and (idx % int(reseed_interval) == 0)
                and len(dets_for_sam2) > 0
            ):
                do_reseed = True
                if (
                    reseed_skip_iou_threshold > 0.0
                    and len(tracked) > 0
                    and dets_for_sam2.class_id is not None
                ):
                    matches = _match_by_iou(
                        tracked,
                        dets_for_sam2,
                        iou_threshold=reseed_iou_threshold,
                        return_iou=True,
                    )
                    if matches:
                        min_iou = min(match[2] for match in matches)
                        if len(matches) == len(tracked) and min_iou >= reseed_skip_iou_threshold:
                            do_reseed = False

                if do_reseed:
                    tracked = _reseed_sam2(
                        frame=frame,
                        sam2_tracker=sam2_tracker,
                        tracked=tracked,
                        dets=dets_for_sam2,
                        iou_threshold=reseed_iou_threshold,
                        reseed_mode=reseed_mode,
                        drop_after=drop_after,
                        stale_counts=sam2_stale_counts,
                    )

                    if snap_to_detector and len(tracked) > 0:
                        tracked = _snap_boxes_to_detections(
                            tracked,
                            dets_for_sam2,
                            iou_threshold=output_box_iou,
                            drop_unmatched=drop_unmatched,
                            blend_alpha=blend_alpha,
                            blend_alpha_by_class=blend_alpha_by_class,
                        )

            if unmatched_drop_after > 0 and len(tracked) > 0:
                tracked = _drop_unmatched_tracks(
                    frame=frame,
                    sam2_tracker=sam2_tracker,
                    tracked=tracked,
                    dets=dets_for_sam2,
                    iou_threshold=output_box_iou,
                    unmatched_counts=sam2_unmatched_counts,
                    drop_after=unmatched_drop_after,
                )
                if snap_to_detector and len(tracked) > 0:
                    tracked = _snap_boxes_to_detections(
                        tracked,
                        dets_for_sam2,
                        iou_threshold=output_box_iou,
                        drop_unmatched=drop_unmatched,
                        blend_alpha=blend_alpha,
                        blend_alpha_by_class=blend_alpha_by_class,
                    )
                if tracked.tracker_id is not None:
                    current_ids = {int(tid) for tid in tracked.tracker_id if tid is not None}
                    for tid in list(sam2_stale_counts):
                        if tid not in current_ids:
                            sam2_stale_counts.pop(tid, None)

            if (
                len(tracked) > 0
                and (geom_min_area_by_class or geom_max_area_by_class or geom_max_aspect_by_class)
            ):
                tracked = _filter_by_geometry(
                    tracked,
                    frame_shape=frame.shape[:2],
                    min_area_ratio_by_class=geom_min_area_by_class,
                    max_area_ratio_by_class=geom_max_area_by_class,
                    max_aspect_by_class=geom_max_aspect_by_class,
                )

            for cname in CLASSES_ORDERED:
                class_id = SOCCERNET_CLASS_TO_ID[cname]
                tracked_c = _filter_by_class(tracked, class_id)
                rows_sam2[cname].extend(_detections_to_mot_rows(tracked_c, frame_idx, default_confidence=1.0))

            if frame_idx in preview_frames and len(tracked) > 0:
                img = frame.copy()
                for cname in CLASSES_ORDERED:
                    class_id = SOCCERNET_CLASS_TO_ID[cname]
                    color = (0, 255, 0) if class_id == SOCCERNET_CLASS_TO_ID["player"] else (255, 0, 0)
                    _draw_tracked(img, _filter_by_class(tracked, class_id), class_name=cname, color=color)
                cv2.imwrite(str(out_variant / "preview" / f"{frame_idx:06d}.jpg"), img)

        for cname, rows in rows_sam2.items():
            rows = _interpolate_tracks(rows, max_gap=5)
            _write_mot(rows, out_variant / "mot" / f"{cname}.txt")
        summary = _variant_sanity_summary(rows_by_class=rows_sam2, frames_processed=frames_processed)
        (out_variant / "sanity.json").write_text(json.dumps(summary, indent=2))
        sanity_rows.append(
            {
                "variant": sam2_variant_name,
                "detector": detector_name,
                "tracker": "sam2",
                "frames": int(frames_processed),
                "total_rows": int(summary["total_rows"]),
                "total_tracks": int(summary["total_tracks"]),
            }
        )
        if int(summary["total_rows"]) == 0:
            raise RuntimeError(f"Variant {sam2_variant_name} produced zero rows; check training/thresholds.")
        logger.success(
            "Sanity {}: frames={} rows={} tracks={}",
            sam2_variant_name,
            summary["frames_processed"],
            summary["total_rows"],
            summary["total_tracks"],
        )

        # Clean up SAM2 VRAM before loading the next detector.
        del sam2_tracker
        torch.cuda.empty_cache()

        # Best-effort cleanup for detector VRAM.
        try:
            del det_handler
        except Exception:
            pass
        torch.cuda.empty_cache()

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_csv = results_dir / "sanity_summary.csv"
    sanity_df.to_csv(sanity_csv, index=False)

    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    try:
        _make_sanity_plots(results_dir=results_dir, infer_out_dir=infer_out_dir)
    except Exception as plot_err:  # pragma: no cover
        logger.error("Sanity plot generation failed: {}", plot_err)
    logger.success("Done. Sanity summary: {}", sanity_csv)


if __name__ == "__main__":
    main()
