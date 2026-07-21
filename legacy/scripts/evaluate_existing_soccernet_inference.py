#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on sys.path when invoked as a script
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.models import (  # noqa: E402
    DetectionConfig,
    DetectionModelType,
    TrackingConfig,
)
from scripts.run_soccernet_tracking_experiment import (  # noqa: E402
    CLASSES_ORDERED,
    TimeStats,
    _make_plots,
    _prepare_trackeval_gt_by_class,
    _run_trackeval,
    _extract_combined_metrics,
)
from tactifoot_vision.data.soccernet_tracking import (  # noqa: E402
    SOCCERNET_CLASS_TO_ID,
    count_unique_gt_tracks_by_class,
    iter_sequence_dirs,
    read_seqinfo,
)
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler  # noqa: E402
from tactifoot_vision.detection.rfdetr_seg_handler import RFDETRSegHandler  # noqa: E402
from tactifoot_vision.metrics.trajectory_stability import compute_all_stability_metrics  # noqa: E402
from tactifoot_vision.tracking.botsort_tracker import (  # noqa: E402
    BoTSORTArgs,
    BoTSORTTracker,
)
from tactifoot_vision.tracking.sam2_tracker import SAM2Tracker  # noqa: E402
from tactifoot_vision.tracking.tracker import Tracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate existing inference results (MOT per-class outputs) with TrackEval and "
            "generate the same plots as `results/detection_tracking/plots/soccernet_tracking_2023_detection_tracking`."
        )
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/detection_tracking/raw/soccernet_tracking_train2seq_100ep_infer1seq"),
        help="Results dir containing `inference/<SEQ>/<variant>/mot/*.txt`.",
    )
    p.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/soccernet/tracking/extracted/test"),
        help="Extracted SoccerNet root for the split matching the inference results.",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence name (e.g. SNMOT-116). If omitted, evaluates multiple sequences under results_dir/inference.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="If >0 and --sequence is not provided, limit how many sequences to evaluate (alphabetical order).",
    )
    p.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Split name passed to TrackEval folder naming (default: test).",
    )
    p.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip (re)timing FPS and set fps=0 in summary.csv.",
    )
    p.add_argument(
        "--time-frames",
        type=int,
        default=25,
        help="How many initial frames to time per variant (default: 25).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, truncate GT to the first N frames for evaluation.",
    )
    return p.parse_args()


def _read_mot_rows(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:10]])
        except Exception:
            continue
    return rows


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


def _iter_frames(seq_dir: Path, *, max_frames: int) -> list[tuple[int, np.ndarray]]:
    import cv2

    seqinfo = read_seqinfo(seq_dir)
    img_dir = seq_dir / "img1"
    limit = min(max(1, int(seqinfo.seq_length)), max(1, int(max_frames)))
    frames: list[tuple[int, np.ndarray]] = []
    for frame_idx in range(1, limit + 1):
        img_path = img_dir / f"{frame_idx:06d}{seqinfo.image_ext}"
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frames.append((frame_idx, frame))
    return frames


def _maybe_optimize_detector(det_handler) -> None:
    model = getattr(det_handler, "model", None)
    if model is not None and hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            pass


def _time_variant(
    *,
    seq_dir: Path,
    det_handler,
    variant: str,
    tracker_key: str,
    bytetrack_cfg: TrackingConfig,
    botsort_args: BoTSORTArgs,
    sam2_cfg: TrackingConfig,
    frames_to_time: int,
) -> TimeStats:
    import supervision as sv

    frames = _iter_frames(seq_dir, max_frames=frames_to_time)
    if not frames:
        return TimeStats(0.0, 0)

    if tracker_key == "bytetrack":
        trackers = {c: Tracker(bytetrack_cfg) for c in CLASSES_ORDERED}
    elif tracker_key == "botsort_reid":
        trackers = {c: BoTSORTTracker(botsort_args, frame_rate=int(bytetrack_cfg.frame_rate or 25)) for c in CLASSES_ORDERED}
    elif tracker_key == "sam2":
        sam2_tracker = SAM2Tracker(sam2_cfg)
        sam2_initialized = False
        sam2_reset_done = False
    else:
        raise ValueError(f"Unknown tracker_key: {tracker_key}")

    t = TimeStats()
    for _frame_idx, frame in frames:
        t0 = time.perf_counter()
        dets = det_handler.detect(frame)
        if tracker_key == "sam2":
            tracked = sv.Detections.empty()
            if not sam2_reset_done:
                sam2_tracker.initialize(frame, np.empty((0, 4), dtype=np.float32), None)
                sam2_reset_done = True
            if not sam2_initialized:
                if len(dets) > 0:
                    sam2_tracker.initialize(frame, dets.xyxy, dets.class_id)
                    sam2_initialized = True
                    tracked = sam2_tracker.track(frame)
            else:
                tracked = sam2_tracker.track(frame)
            _ = tracked
        else:
            for cname in CLASSES_ORDERED:
                class_id = SOCCERNET_CLASS_TO_ID[cname]
                dets_c = dets[dets.class_id.astype(int) == int(class_id)] if dets.class_id is not None else sv.Detections.empty()
                if tracker_key == "bytetrack":
                    _ = trackers[cname].update(dets_c)
                else:
                    _ = trackers[cname].update(dets_c, frame)
        dt = time.perf_counter() - t0
        t.add(dt, 1)

    logger.info("Timing {} ({}): fps={:.2f}", variant, tracker_key, t.fps)
    return t


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    extracted_root = args.extracted_root.resolve()
    split_name = str(args.split_name)

    inference_root = results_dir / "inference"
    if not inference_root.is_dir():
        raise FileNotFoundError(f"Missing inference dir: {inference_root}")

    candidates = sorted([p.name for p in inference_root.iterdir() if p.is_dir()])
    if not candidates:
        raise RuntimeError(f"No sequences found under {inference_root}")

    if args.sequence is not None:
        seq_names = [str(args.sequence)]
    else:
        limit = int(args.max_sequences or 0)
        seq_names = candidates[:limit] if limit > 0 else candidates

    missing = [name for name in seq_names if not (extracted_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Sequence(s) not found under extracted_root={extracted_root}: {', '.join(missing)}"
        )

    infer_first_dir = inference_root / seq_names[0]
    variants = sorted([p.name for p in infer_first_dir.iterdir() if p.is_dir()])
    if not variants:
        raise RuntimeError(f"No variants found under {infer_first_dir}")
    for seq_name in seq_names[1:]:
        infer_seq_dir = inference_root / seq_name
        for variant in variants:
            if not (infer_seq_dir / variant).is_dir():
                raise FileNotFoundError(f"Missing variant dir: {infer_seq_dir / variant}")

    # Use the first sequence as a reference for optional FPS timing.
    seq_name = seq_names[0]
    seq_dir = extracted_root / seq_name
    timing_seq_dir = seq_dir
    infer_seq_dir = inference_root / seq_name

    # Load manifest for configs/checkpoints when available.
    manifest_path = results_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    tracker_cfgs = manifest.get("tracker_configs") or {}
    bytetrack_cfg = TrackingConfig.model_validate(tracker_cfgs.get("bytetrack") or {})
    sam2_cfg = TrackingConfig.model_validate(tracker_cfgs.get("sam2") or {})
    botsort_args_raw = tracker_cfgs.get("botsort_reid") or {}
    botsort_args = BoTSORTArgs(**botsort_args_raw) if botsort_args_raw else BoTSORTArgs(with_reid=True, model=str((project_root / "yolo11n.pt").resolve()))

    chosen_thresholds = manifest.get("chosen_thresholds") or {}
    models = manifest.get("models") or {}
    base_ckpt = Path(models.get("base", results_dir / "models" / "rfdetr_base_train2seq_100ep.pth")).resolve()
    seg_ckpt = Path(models.get("seg", results_dir / "models" / "rfdetr_seg_train2seq_100ep.pth")).resolve()

    # TrackEval folder layout (same as full experiment).
    trackeval_data_dir = results_dir / "trackeval" / "data"
    gt_root = trackeval_data_dir / "gt" / "mot_challenge"
    trackers_root = trackeval_data_dir / "trackers" / "mot_challenge"
    gt_root.mkdir(parents=True, exist_ok=True)
    trackers_root.mkdir(parents=True, exist_ok=True)

    seqmap = results_dir / f"SNMOT-{split_name}.subset.txt"
    seqmap.write_text("name\n" + "\n".join(seq_names) + "\n", encoding="utf-8")

    gt_meta = _prepare_trackeval_gt_by_class(
        extracted_root=extracted_root,
        sequences=seq_names,
        output_gt_root=gt_root,
        benchmark_prefix="SNMOT",
        split_name=split_name,
        max_frames=int(args.max_frames) if int(args.max_frames) > 0 else None,
    )

    # Copy existing MOT outputs into TrackEval tracker folders.
    for seq_name in seq_names:
        infer_seq_dir = inference_root / seq_name
        for variant in variants:
            mot_dir = (infer_seq_dir / variant / "mot")
            for class_name in CLASSES_ORDERED:
                src = mot_dir / f"{class_name}.txt"
                benchmark = gt_meta[class_name]["benchmark"]
                dst_dir = trackers_root / f"{benchmark}-{split_name}" / variant / "data"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{seq_name}.txt"
                if src.is_file():
                    dst.write_text(src.read_text(), encoding="utf-8")
                else:
                    dst.write_text("", encoding="utf-8")

    # Per-sequence stats from existing MOT files.
    per_sequence_rows: list[dict] = []
    weights_total: dict[str, int] = {c: 0 for c in CLASSES_ORDERED}
    for seq_name in seq_names:
        seq_dir_local = extracted_root / seq_name
        seqinfo = read_seqinfo(seq_dir_local)
        gt_counts_seq = count_unique_gt_tracks_by_class(seq_dir_local)
        for class_name in CLASSES_ORDERED:
            weights_total[class_name] = int(weights_total.get(class_name, 0)) + int(gt_counts_seq.get(class_name, 0))

        infer_seq_dir = inference_root / seq_name
        for variant in variants:
            tracker_key = variant.split("__", 1)[1] if "__" in variant else "unknown"
            detector_name = variant.split("__", 1)[0] if "__" in variant else "unknown"
            for class_name in CLASSES_ORDERED:
                rows = _read_mot_rows(infer_seq_dir / variant / "mot" / f"{class_name}.txt")
                stats = _track_length_stats(rows)
                gt_tracks = int(gt_counts_seq.get(class_name, 0))
                pred_tracks = int(stats["pred_tracks"])
                id_ratio = (pred_tracks / gt_tracks) if gt_tracks > 0 else float("nan")
                per_sequence_rows.append(
                    {
                        "sequence": seq_name,
                        "variant": variant,
                        "detector": detector_name,
                        "tracker": tracker_key,
                        "class": class_name,
                        "frames_processed": int(seqinfo.seq_length),
                        "gt_tracks": gt_tracks,
                        "id_ratio": id_ratio,
                        **stats,
                    }
                )

    per_sequence_df = pd.DataFrame(per_sequence_rows)
    per_sequence_csv = results_dir / "per_sequence_stats.csv"
    per_sequence_df.to_csv(per_sequence_csv, index=False)

    # Run TrackEval per class, then aggregate macro/weighted.
    per_class_rows: list[dict] = []
    for class_name in CLASSES_ORDERED:
        benchmark = gt_meta[class_name]["benchmark"]
        output = _run_trackeval(
            trackeval_root=Path("external/TrackEval"),
            gt_root=gt_root,
            trackers_root=trackers_root,
            benchmark=benchmark,
            split_name=split_name,
            seqmap_file=seqmap,
            tracker_names=variants,
            iou_threshold=0.5,
        )
        for variant in variants:
            tracker_key = variant.split("__", 1)[1] if "__" in variant else "unknown"
            detector_name = variant.split("__", 1)[0] if "__" in variant else "unknown"
            metrics = _extract_combined_metrics(output, variant)
            per_class_rows.append(
                {
                    "variant": variant,
                    "detector": detector_name,
                    "tracker": tracker_key,
                    "class": class_name,
                    **metrics,
                }
            )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = results_dir / "metrics_per_class.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    timing: dict[str, TimeStats] = {v: TimeStats(0.0, 0) for v in variants}
    if not args.skip_timing:
        frames_to_time = int(args.time_frames)
        if frames_to_time > 0:
            for detector_key in sorted({v.split("__", 1)[0] for v in variants if "__" in v}):
                if detector_key == "rfdetr_base":
                    ckpt = base_ckpt
                    model_type = DetectionModelType.RFDETR
                elif detector_key == "rfdetr_seg":
                    ckpt = seg_ckpt
                    model_type = DetectionModelType.RFDETR_SEG
                else:
                    continue
                if not ckpt.is_file():
                    logger.warning("Skipping timing for {} (missing ckpt: {})", detector_key, ckpt)
                    continue

                thr = float(chosen_thresholds.get(detector_key, 0.3))
                det_cfg = DetectionConfig(
                    model_type=model_type,
                    checkpoint_path=ckpt,
                    confidence_threshold=thr,
                    nms_threshold=0.5,
                    classes=SOCCERNET_CLASS_TO_ID,
                )
                det_handler = (
                    RFDETRHandler(det_cfg, model_dir=project_root)
                    if model_type == DetectionModelType.RFDETR
                    else RFDETRSegHandler(det_cfg, model_dir=project_root)
                )
                _maybe_optimize_detector(det_handler)

                for variant in [v for v in variants if v.startswith(f"{detector_key}__")]:
                    tracker_key = variant.split("__", 1)[1]
                    try:
                        timing[variant] = _time_variant(
                            seq_dir=timing_seq_dir,
                            det_handler=det_handler,
                            variant=variant,
                            tracker_key=tracker_key,
                            bytetrack_cfg=bytetrack_cfg,
                            botsort_args=botsort_args,
                            sam2_cfg=sam2_cfg,
                            frames_to_time=frames_to_time,
                        )
                    except Exception as e:
                        logger.warning("Timing failed for {}: {}", variant, e)

    summary_rows: list[dict] = []
    for variant in variants:
        df_v = per_class_df[per_class_df["variant"] == variant]
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

        t = timing.get(variant, TimeStats(0.0, 0))
        tracker_key = variant.split("__", 1)[1] if "__" in variant else "unknown"
        detector_name = variant.split("__", 1)[0] if "__" in variant else "unknown"
        summary_rows.append(
            {
                "variant": variant,
                "detector": detector_name,
                "tracker": tracker_key,
                "fps": float(t.fps) if t.frames > 0 else 0.0,
                "inference_time_s": float(t.seconds),
                "frames_timed": int(t.frames),
                **{f"macro_{k}": v_ for k, v_ in macro.items()},
                **{f"weighted_{k}": v_ for k, v_ in weighted.items()},
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

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

    _make_plots(results_dir, summary_df, per_class_df, per_sequence_df)
    logger.success("Done. Wrote: {}, {}, {} and plots/", summary_csv, per_class_csv, per_sequence_csv)


if __name__ == "__main__":
    main()
