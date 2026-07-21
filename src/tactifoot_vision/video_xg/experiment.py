import csv
import json
from collections.abc import Sequence
from pathlib import Path

from tactifoot_vision.ball import BallTrajectoryReconstructor
from tactifoot_vision.domain import ExportArtifact, PipelineResult
from tactifoot_vision.enums import XgModelKind
from tactifoot_vision.shots import ShotDetector
from tactifoot_vision.video_xg.features import build_video_shot_features
from tactifoot_vision.video_xg.results import VideoShotFeatures
from tactifoot_vision.video_xg.runner import (
    VideoOnlyXgRunner,
    build_video_only_estimator,
)

MethodMetric = dict[str, float | str]


def extract_video_shot_features(
    result: PipelineResult,
    *,
    ball_reconstructor: BallTrajectoryReconstructor,
    shot_detector: ShotDetector,
    image_width: int = 1920,
    image_height: int = 1080,
    attacking_goal_x: float | None = None,
) -> tuple[VideoShotFeatures, ...]:
    trajectory = ball_reconstructor.reconstruct(result)
    candidates = shot_detector.detect(trajectory, result.frames)
    frames_by_index = {frame.frame_index: frame for frame in result.frames}
    features = []
    for index, candidate in enumerate(candidates, start=1):
        features.append(
            build_video_shot_features(
                shot_id=f"video-shot-{index:04d}",
                candidate=candidate,
                trajectory=trajectory,
                frame_result=frames_by_index.get(candidate.frame_index),
                image_width=image_width,
                image_height=image_height,
                attacking_goal_x=attacking_goal_x,
            )
        )
    return tuple(features)


def write_video_features_csv(features: Sequence[VideoShotFeatures], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_feature_fieldnames())
        writer.writeheader()
        for feature in features:
            writer.writerow(_feature_row(feature))
    return path


def run_video_only_xg_experiment(
    *,
    features_path: Path,
    output_dir: Path,
    reference_path: Path | None = None,
    group_id: str | None = None,
    model_kinds: Sequence[XgModelKind] = (
        XgModelKind.VIDEO_GEOMETRY,
        XgModelKind.VIDEO_FREEZE_CONTEXT,
        XgModelKind.VIDEO_KINEMATIC_CONTEXT,
    ),
) -> tuple[dict[str, object], tuple[ExportArtifact, ...]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ExportArtifact] = []
    methods: list[MethodMetric] = []
    for model_kind in model_kinds:
        model_output = output_dir / model_kind.value
        runner = VideoOnlyXgRunner(estimator=build_video_only_estimator(model_kind))
        model_summary, metrics, model_artifacts = runner.run(
            features_path,
            output_dir=model_output,
            reference_path=reference_path,
            group_id=group_id,
        )
        artifacts.extend(model_artifacts)
        method_row = {
            "method": model_kind.value,
            "shots": float(model_summary.shot_count),
            "total_xg": model_summary.total_xg,
            **metrics,
        }
        methods.append(method_row)
    method_metrics_path = _write_method_metrics(
        methods, output_dir / "method_metrics.csv"
    )
    report_path = _write_report(methods, output_dir / "comparison_report.md")
    summary_path = output_dir / "experiment_summary.json"
    experiment_summary: dict[str, object] = {
        "group_id": group_id,
        "features_path": str(features_path),
        "reference_path": str(reference_path) if reference_path is not None else None,
        "methods": methods,
    }
    summary_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    artifacts.extend(
        (
            ExportArtifact(
                path=method_metrics_path,
                format="video_only_xg_method_metrics_csv",
                rows=len(methods),
            ),
            ExportArtifact(
                path=report_path,
                format="video_only_xg_report_markdown",
                rows=len(methods),
            ),
            ExportArtifact(
                path=summary_path,
                format="video_only_xg_experiment_summary_json",
                rows=len(methods),
            ),
        )
    )
    return experiment_summary, tuple(artifacts)


def _write_method_metrics(methods: Sequence[MethodMetric], path: Path) -> Path:
    if not methods:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for method in methods for key in method})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(methods)
    return path


def _write_report(methods: Sequence[MethodMetric], path: Path) -> Path:
    lines = [
        "# Video-Only xG Experiment",
        "",
        "| Method | Shots | Total xG | MAE vs reference | RMSE vs reference | Total xG error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| {method} | {shots:.0f} | {total_xg:.4f} | {mae:.4f} | {rmse:.4f} | {error:.4f} |".format(
                method=method["method"],
                shots=_metric_float(method, "shots"),
                total_xg=_metric_float(method, "total_xg"),
                mae=_metric_float(method, "mae_vs_reference_xg"),
                rmse=_metric_float(method, "rmse_vs_reference_xg"),
                error=_metric_float(method, "total_xg_error"),
            )
        )
    lines.extend(
        [
            "",
            "Reference xG is used only for evaluation. The feature table is validated by the video-only protocol before any model runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _metric_float(method: MethodMetric, key: str) -> float:
    value = method.get(key, 0.0)
    if isinstance(value, str):
        return float(value)
    return value


def _feature_fieldnames() -> list[str]:
    return [
        "shot_id",
        "frame_index",
        "shot_x",
        "shot_y",
        "goal_x",
        "goal_y",
        "nearest_player_distance",
        "goalkeeper_distance",
        "defender_count_in_cone",
        "ball_speed",
        "ball_direction_to_goal",
        "shot_confidence",
    ]


def _feature_row(feature: VideoShotFeatures) -> dict[str, object]:
    return {
        "shot_id": feature.shot_id,
        "frame_index": feature.frame_index,
        "shot_x": feature.shot_x,
        "shot_y": feature.shot_y,
        "goal_x": feature.goal_x,
        "goal_y": feature.goal_y,
        "nearest_player_distance": feature.nearest_player_distance,
        "goalkeeper_distance": feature.goalkeeper_distance,
        "defender_count_in_cone": feature.defender_count_in_cone,
        "ball_speed": feature.ball_speed,
        "ball_direction_to_goal": feature.ball_direction_to_goal,
        "shot_confidence": feature.shot_confidence,
    }
