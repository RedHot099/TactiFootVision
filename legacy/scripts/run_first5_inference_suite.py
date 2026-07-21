from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "project"
RAW_RESULTS_ROOT = RESULTS_ROOT / "raw"
NUMERIC_RESULTS_ROOT = RESULTS_ROOT / "numeric"

MATCHES_ROOT = Path("/home/kuba/projects/ball-vision/data/20232024")
MATCH_VIDEO_NAME = "first_5.mp4"
MAX_MATCHES = None

MODELS: list[dict[str, str]] = [
    {"name": "yolov8m", "type": "yolo"},
    {"name": "yolo11m", "type": "yolo"},
    {"name": "yolo12m", "type": "yolo"},
    {"name": "rfdetr_base", "type": "rfdetr"},
]


def _load_default_keypoints_checkpoint() -> Path:
    default_cfg_path = PROJECT_ROOT / "config" / "default_config.yaml"
    raw = yaml.safe_load(default_cfg_path.read_text(encoding="utf-8"))
    kp = (raw or {}).get("keypoints") or {}
    checkpoint = kp.get("checkpoint_path")
    if not checkpoint:
        raise RuntimeError(
            f"Missing keypoints.checkpoint_path in {default_cfg_path}"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (default_cfg_path.parent / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Keypoints checkpoint not found: {checkpoint_path}"
        )
    return checkpoint_path


def _find_yolo_best_checkpoint(model_name: str) -> Optional[Path]:
    candidates = sorted(
        (RAW_RESULTS_ROOT / model_name).glob("training/**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _get_model_checkpoint(model_name: str) -> Optional[Path]:
    if model_name in {"yolov8m", "yolo11m", "yolo12m"}:
        return _find_yolo_best_checkpoint(model_name)
    if model_name == "rfdetr_base":
        ckpt = RAW_RESULTS_ROOT / "rfdetr_base" / "checkpoints" / "checkpoint_best_total.pth"
        return ckpt.resolve() if ckpt.is_file() else None
    return None


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_detection(config_path: Path) -> None:
    run_detection_script = PROJECT_ROOT / "scripts" / "run_detection.py"
    subprocess.run(
        [sys.executable, str(run_detection_script), "--config", str(config_path)],
        check=True,
    )


def _read_timing_json(timing_path: Path) -> Optional[Dict[str, Any]]:
    if not timing_path.is_file():
        return None
    try:
        return json.loads(timing_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    if not MATCHES_ROOT.is_dir():
        raise FileNotFoundError(f"MATCHES_ROOT not found: {MATCHES_ROOT}")

    keypoints_ckpt = _load_default_keypoints_checkpoint()
    model_dir = (PROJECT_ROOT / "models").resolve()

    matches = sorted([p for p in MATCHES_ROOT.iterdir() if p.is_dir()])
    if MAX_MATCHES is not None:
        matches = matches[: int(MAX_MATCHES)]
    if not matches:
        raise RuntimeError(f"No match directories found under: {MATCHES_ROOT}")

    all_rows: List[Dict[str, Any]] = []

    for model in MODELS:
        model_name = model["name"]
        model_type = model["type"]

        checkpoint = _get_model_checkpoint(model_name)
        if checkpoint is None:
            print(
                f"[SKIP] Missing checkpoint for {model_name}. Train first, then rerun."
            )
            continue

        for match_dir in matches:
            video_path = match_dir / MATCH_VIDEO_NAME
            if not video_path.is_file():
                print(f"[SKIP] Missing video: {video_path}")
                continue

            cfg_path = RAW_RESULTS_ROOT / model_name / "configs" / f"{match_dir.name}.yaml"

            output_rel = (
                Path("..")
                / "inference"
                / match_dir.name
                / f"{model_name}_first_5.mp4"
            )
            expected_timing_path = (
                (cfg_path.parent / output_rel).resolve().with_suffix("")
            )
            expected_timing_path = expected_timing_path.parent / (
                expected_timing_path.name + "_timing.json"
            )

            payload: Dict[str, Any] = {
                "project_name": f"infer_{model_name}_{match_dir.name}",
                "logging_level": "INFO",
                "paths": {
                    "input_video": str(video_path.resolve()),
                    "output_video": str(output_rel),
                    "model_dir": str(model_dir),
                    "statsbomb_input_csv": "unused.csv",
                    "pipeline_input_csv": "unused.csv",
                    "merged_output_csv": "unused.csv",
                },
                "detection": {
                    "model_type": model_type,
                    "checkpoint_path": str(checkpoint),
                    "confidence_threshold": 0.3,
                    "nms_threshold": 0.5,
                    "classes": {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3},
                },
                "keypoints": {
                    "enabled": True,
                    "model_type": "yolo_pose",
                    "checkpoint_path": str(keypoints_ckpt),
                    "confidence_threshold": 0.8,
                },
                "geometry": {
                    "min_keypoint_confidence_for_homography": 0.2,
                    "homography_smoothing_window": 15,
                    "ball_outlier_threshold_percent": 5,
                    "target_pitch_length": 120,
                    "target_pitch_width": 80,
                },
                "visualization": {
                    "enabled": False,
                    "overlay": False,
                    "draw_projected_pitch": False,
                    "draw_bounding_boxes": False,
                    "draw_keypoints": False,
                    "draw_pitch_detection": False,
                },
                "processing": {
                    "period": 1,
                    "period_start_time_seconds": 0.0,
                    "save_output_video": False,
                },
            }

            _write_yaml(cfg_path, payload)
            print(f"[RUN] {model_name} -> {match_dir.name}")
            _run_detection(cfg_path)

            timing = _read_timing_json(expected_timing_path) or {}
            row: Dict[str, Any] = {
                "model": model_name,
                "match": match_dir.name,
                "video": str(video_path),
                "checkpoint": str(checkpoint),
                "timing_json": str(expected_timing_path),
                "detection_time_avg_ms": timing.get("detection_time_avg_ms"),
                "detection_fps": timing.get("detection_fps"),
                "timed_frames": timing.get("timed_frames"),
                "frames_total": timing.get("frames_total"),
                "wall_time_total_s": timing.get("wall_time_total_s"),
                "output_pipeline_csv": timing.get("output_pipeline_csv"),
            }
            all_rows.append(row)

        model_csv_path = RAW_RESULTS_ROOT / model_name / "inference_timings.csv"
        model_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if all_rows:
            with model_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                for r in [x for x in all_rows if x.get("model") == model_name]:
                    writer.writerow(r)

    if all_rows:
        all_csv_path = NUMERIC_RESULTS_ROOT / "inference_timings_all_models.csv"
        all_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with all_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[OK] Wrote: {all_csv_path}")


if __name__ == "__main__":
    main()
