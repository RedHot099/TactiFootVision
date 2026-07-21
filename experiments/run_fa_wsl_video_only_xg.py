import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from tactifoot_vision.video_xg import run_video_only_xg_experiment


@dataclass(frozen=True, slots=True)
class DetectionRow:
    shot_id: str
    period: int
    video_file: str
    frame_index: int
    timestamp_seconds: float
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class FeatureRow:
    shot_id: str
    frame_index: int
    shot_x: float
    shot_y: float
    goal_x: float
    goal_y: float
    nearest_player_distance: float | None
    goalkeeper_distance: float | None
    defender_count_in_cone: int
    ball_speed: float | None
    shot_confidence: float
    source: str
    video_file: str
    video_second: float
    selected_frame: int
    ball_confidence: float | None
    player_count: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-dir",
        type=Path,
        default=Path(
            "/home/kuba/projects/ball-vision/data/FA_WSL_2020_2021/"
            "3775567_Chelsea_FCW_vs_Manchester_United"
        ),
    )
    parser.add_argument("--weights", type=Path, default=Path("models/yolo11m.pt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--window-before", type=float, default=2.0)
    parser.add_argument("--window-after", type=float, default=2.0)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.01)
    parser.add_argument("--save-selected-frames", action="store_true")
    args = parser.parse_args()

    match_dir = args.match_dir
    output_dir = args.output_dir or (
        match_dir / "experiments" / "video_only_xg_full_match_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_frames").mkdir(exist_ok=True)

    run_config = {
        "match_dir": str(match_dir),
        "weights": str(args.weights),
        "window_before": args.window_before,
        "window_after": args.window_after,
        "person_confidence": args.person_confidence,
        "ball_confidence": args.ball_confidence,
        "method_note": (
            "StatsBomb timestamps are used only to define shot windows and "
            "StatsBomb xG is used only as reference. xG inputs are derived from "
            "YOLO detections in video frames with image-to-pitch normalization."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    shots = _read_shots(match_dir)
    shots.to_csv(output_dir / "shot_manifest.csv", index=False)
    _write_reference(shots, output_dir / "reference_statsbomb_xg.csv")

    model = YOLO(args.weights)
    detections: list[DetectionRow] = []
    features: list[FeatureRow] = []
    quality_rows = []
    for _, shot in shots.iterrows():
        print(
            f"processing shot {int(shot['shot_number'])}/{len(shots)} "
            f"{shot['video_file']} {shot['timestamp']}",
            flush=True,
        )
        video_path = match_dir / str(shot["video_file"])
        window = _process_shot_window(
            model=model,
            video_path=video_path,
            shot=shot,
            window_before=args.window_before,
            window_after=args.window_after,
            person_confidence=args.person_confidence,
            ball_confidence=args.ball_confidence,
            save_frame_dir=output_dir / "selected_frames"
            if args.save_selected_frames
            else None,
        )
        detections.extend(window["detections"])
        features.append(window["feature"])
        quality_rows.append(window["quality"])
        _write_rows(
            [asdict(row) for row in window["detections"]],
            output_dir / "detections_by_shot" / f"{shot['shot_id']}.csv",
        )

    _write_rows([asdict(row) for row in detections], output_dir / "detections.csv")
    _write_rows(
        [asdict(row) for row in features], output_dir / "video_features_with_audit.csv"
    )
    _write_video_features(features, output_dir / "video_features.csv")
    _write_rows(quality_rows, output_dir / "feature_quality.csv")

    summary, _ = run_video_only_xg_experiment(
        features_path=output_dir / "video_features.csv",
        output_dir=output_dir / "method_comparison",
        reference_path=output_dir / "reference_statsbomb_xg.csv",
        group_id=match_dir.name,
    )
    _write_final_report(output_dir, summary, pd.DataFrame(quality_rows))
    print(json.dumps(summary, indent=2))
    print(f"output_dir={output_dir}")
    return 0


def _read_shots(match_dir: Path) -> pd.DataFrame:
    events = pd.read_parquet(match_dir / "events.parquet")
    shots = (
        events[events["type"].eq("Shot")].copy().sort_values(["period", "timestamp"])
    )
    rows = []
    for shot_number, (_, row) in enumerate(shots.iterrows(), start=1):
        video_second = _timestamp_seconds(str(row["timestamp"]))
        rows.append(
            {
                "shot_number": shot_number,
                "shot_id": str(row["id"]),
                "period": int(row["period"]),
                "timestamp": str(row["timestamp"]),
                "video_file": "part1.mp4" if int(row["period"]) == 1 else "part2.mp4",
                "video_second": video_second,
                "team": row["team"],
                "player": row["player"],
                "shot_outcome": row["shot_outcome"],
                "statsbomb_xg": float(row["shot_statsbomb_xg"]),
            }
        )
    return pd.DataFrame(rows)


def _write_reference(shots: pd.DataFrame, path: Path) -> None:
    reference = pd.DataFrame(
        {
            "shot_id": shots["shot_id"],
            "reference_xg": shots["statsbomb_xg"],
            "is_goal": shots["shot_outcome"].eq("Goal").astype(int),
        }
    )
    reference.to_csv(path, index=False)


def _process_shot_window(
    *,
    model: YOLO,
    video_path: Path,
    shot: pd.Series,
    window_before: float,
    window_after: float,
    person_confidence: float,
    ball_confidence: float,
    save_frame_dir: Path | None,
) -> dict[str, object]:
    fps, width, height, frame_count = _video_info(video_path)
    shot_frame = int(round(float(shot["video_second"]) * fps))
    start_frame = max(
        0, int(round((float(shot["video_second"]) - window_before) * fps))
    )
    end_frame = min(
        frame_count - 1,
        int(round((float(shot["video_second"]) + window_after) * fps)),
    )
    capture = cv2.VideoCapture(str(video_path))
    detections: list[DetectionRow] = []
    frames: dict[int, object] = {}
    try:
        for frame_index in range(start_frame, end_frame + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            frames[frame_index] = image
            result = model.predict(
                image,
                conf=min(ball_confidence, person_confidence),
                iou=0.5,
                classes=[0, 32],
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for class_id, confidence, xyxy in zip(
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
                boxes.xyxy.cpu().numpy(),
                strict=True,
            ):
                class_name = "player" if int(class_id) == 0 else "ball"
                confidence_value = float(confidence)
                if class_name == "player" and confidence_value < person_confidence:
                    continue
                if class_name == "ball" and confidence_value < ball_confidence:
                    continue
                if class_name == "ball" and not _valid_ball_box(xyxy):
                    continue
                detections.append(
                    DetectionRow(
                        shot_id=str(shot["shot_id"]),
                        period=int(shot["period"]),
                        video_file=video_path.name,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        class_name=class_name,
                        confidence=confidence_value,
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    )
                )
    finally:
        capture.release()

    feature = _feature_from_window(
        shot=shot,
        detections=detections,
        shot_frame=shot_frame,
        fps=fps,
        image_width=width,
        image_height=height,
    )
    if save_frame_dir is not None and feature.selected_frame in frames:
        frame = frames[feature.selected_frame].copy()
        _draw_feature(frame, feature)
        cv2.imwrite(
            str(
                save_frame_dir / f"{shot['shot_number']:02d}_{shot['shot_id'][:8]}.jpg"
            ),
            frame,
        )
    quality = {
        "shot_id": str(shot["shot_id"]),
        "period": int(shot["period"]),
        "video_file": video_path.name,
        "video_second": float(shot["video_second"]),
        "window_start_frame": start_frame,
        "window_end_frame": end_frame,
        "frames_processed": end_frame - start_frame + 1,
        "detections": len(detections),
        "ball_detections": sum(
            1 for detection in detections if detection.class_name == "ball"
        ),
        "player_detections": sum(
            1 for detection in detections if detection.class_name == "player"
        ),
        "feature_source": feature.source,
        "selected_frame": feature.selected_frame,
        "ball_confidence": feature.ball_confidence,
        "player_count_selected_frame": feature.player_count,
    }
    return {"detections": detections, "feature": feature, "quality": quality}


def _feature_from_window(
    *,
    shot: pd.Series,
    detections: list[DetectionRow],
    shot_frame: int,
    fps: float,
    image_width: int,
    image_height: int,
) -> FeatureRow:
    ball_detections = [
        detection for detection in detections if detection.class_name == "ball"
    ]
    selected_ball, source = _select_or_interpolate_ball(ball_detections, shot_frame)
    if selected_ball is None:
        # Last-resort video-only fallback: use center of the frame, with low confidence.
        ball_x, ball_y = image_width / 2.0, image_height / 2.0
        selected_frame = shot_frame
        ball_confidence = None
        source = "missing_center_fallback"
    else:
        ball_x, ball_y = selected_ball["x"], selected_ball["y"]
        selected_frame = int(selected_ball["frame_index"])
        ball_confidence = selected_ball.get("confidence")

    shot_x = min(max(ball_x / image_width, 0.0), 1.0) * 105.0
    shot_y = min(max(ball_y / image_height, 0.0), 1.0) * 68.0
    goal_x = 105.0 if shot_x >= 52.5 else 0.0
    goal_y = 34.0

    players = [
        _pitch_center(detection, image_width, image_height)
        for detection in detections
        if detection.class_name == "player" and detection.frame_index == selected_frame
    ]
    nearest_player = _nearest_distance(shot_x, shot_y, players)
    goalkeeper = _nearest_distance(goal_x, goal_y, players)
    defenders_in_cone = _count_in_goal_cone(shot_x, shot_y, goal_x, goal_y, players)
    ball_speed = _ball_speed(ball_detections, fps, image_width, image_height)
    return FeatureRow(
        shot_id=str(shot["shot_id"]),
        frame_index=int(selected_frame),
        shot_x=shot_x,
        shot_y=shot_y,
        goal_x=goal_x,
        goal_y=goal_y,
        nearest_player_distance=nearest_player,
        goalkeeper_distance=goalkeeper,
        defender_count_in_cone=defenders_in_cone,
        ball_speed=ball_speed,
        shot_confidence=1.0 if source != "missing_center_fallback" else 0.2,
        source=source,
        video_file=str(shot["video_file"]),
        video_second=float(shot["video_second"]),
        selected_frame=int(selected_frame),
        ball_confidence=None if ball_confidence is None else float(ball_confidence),
        player_count=len(players),
    )


def _select_or_interpolate_ball(
    detections: list[DetectionRow], target_frame: int
) -> tuple[dict[str, float] | None, str]:
    if not detections:
        return None, "missing"
    centers = sorted(
        {
            detection.frame_index: {
                "frame_index": float(detection.frame_index),
                "x": (detection.x1 + detection.x2) / 2.0,
                "y": (detection.y1 + detection.y2) / 2.0,
                "confidence": detection.confidence,
            }
            for detection in sorted(detections, key=lambda item: item.confidence)
        }.values(),
        key=lambda item: item["frame_index"],
    )
    nearest = min(centers, key=lambda item: abs(item["frame_index"] - target_frame))
    if abs(nearest["frame_index"] - target_frame) <= 3:
        return nearest, "observed"
    previous = [item for item in centers if item["frame_index"] < target_frame]
    following = [item for item in centers if item["frame_index"] > target_frame]
    if previous and following:
        before = previous[-1]
        after = following[0]
        span = after["frame_index"] - before["frame_index"]
        if 0 < span <= 90:
            ratio = (target_frame - before["frame_index"]) / span
            return (
                {
                    "frame_index": float(target_frame),
                    "x": before["x"] + ratio * (after["x"] - before["x"]),
                    "y": before["y"] + ratio * (after["y"] - before["y"]),
                    "confidence": min(before["confidence"], after["confidence"]) * 0.75,
                },
                "interpolated",
            )
    return nearest, "nearest_observed"


def _ball_speed(
    detections: list[DetectionRow], fps: float, image_width: int, image_height: int
) -> float | None:
    if len(detections) < 2:
        return None
    centers = sorted(detections, key=lambda item: item.frame_index)
    first = centers[0]
    last = centers[-1]
    frame_gap = last.frame_index - first.frame_index
    if frame_gap <= 0:
        return None
    x1, y1 = _pitch_center(first, image_width, image_height)
    x2, y2 = _pitch_center(last, image_width, image_height)
    return math.hypot(x2 - x1, y2 - y1) / (frame_gap / fps)


def _pitch_center(
    detection: DetectionRow, image_width: int, image_height: int
) -> tuple[float, float]:
    return (
        min(max(((detection.x1 + detection.x2) / 2.0) / image_width, 0.0), 1.0) * 105.0,
        min(max(((detection.y1 + detection.y2) / 2.0) / image_height, 0.0), 1.0) * 68.0,
    )


def _nearest_distance(
    x: float, y: float, points: list[tuple[float, float]]
) -> float | None:
    if not points:
        return None
    return min(math.hypot(px - x, py - y) for px, py in points)


def _count_in_goal_cone(
    shot_x: float,
    shot_y: float,
    goal_x: float,
    goal_y: float,
    points: list[tuple[float, float]],
) -> int:
    dx = goal_x - shot_x
    dy = goal_y - shot_y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0
    count = 0
    for px, py in points:
        t = ((px - shot_x) * dx + (py - shot_y) * dy) / length_sq
        if not 0.0 < t < 1.0:
            continue
        projected_x = shot_x + t * dx
        projected_y = shot_y + t * dy
        if math.hypot(px - projected_x, py - projected_y) <= 2.5:
            count += 1
    return count


def _valid_ball_box(xyxy: object) -> bool:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    if area < 8.0 or area > 2500.0:
        return False
    ratio = width / max(height, 1e-9)
    return 0.35 <= ratio <= 2.8


def _draw_feature(frame: object, feature: FeatureRow) -> None:
    height, width = frame.shape[:2]
    x = int(feature.shot_x / 105.0 * width)
    y = int(feature.shot_y / 68.0 * height)
    cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"{feature.source} x={feature.shot_x:.1f} y={feature.shot_y:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def _write_video_features(features: list[FeatureRow], path: Path) -> None:
    rows = []
    for feature in features:
        row = asdict(feature)
        for audit_column in [
            "source",
            "video_file",
            "video_second",
            "selected_frame",
            "ball_confidence",
            "player_count",
        ]:
            row.pop(audit_column)
        rows.append(row)
    _write_rows(rows, path)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_final_report(
    output_dir: Path, summary: dict[str, object], quality: pd.DataFrame
) -> None:
    methods = pd.DataFrame(summary["methods"])
    source_counts = quality["feature_source"].value_counts().to_dict()
    lines = [
        "# FA WSL video-only xG full-match window experiment",
        "",
        "## Inputs",
        "",
        "- Videos: `part1.mp4`, `part2.mp4`.",
        "- Shot windows: StatsBomb timestamps are used only to select evaluation windows.",
        "- Model inputs: YOLO-derived ball/player detections and image-normalized pitch coordinates.",
        "- Reference: StatsBomb xG, used only after prediction.",
        "",
        "## Method comparison",
        "",
        methods.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Feature source coverage",
        "",
        json.dumps(source_counts, indent=2),
        "",
        "## Important caveats",
        "",
        "- This run does not use calibrated homography; pitch coordinates are image-normalized approximations.",
        "- COCO YOLO distinguishes `person` and `sports ball`, but not goalkeeper/team identity.",
        "- Goalkeeper distance is approximated as the person closest to the inferred goal center.",
        "- Rows with missing ball detections use an explicit low-confidence center fallback and are counted in `feature_quality.csv`.",
    ]
    (output_dir / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, frame_count
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
