import csv
import json
from collections.abc import Mapping
from pathlib import Path

from tactifoot_vision.domain import ExportArtifact
from tactifoot_vision.enums import XgModelKind
from tactifoot_vision.video_xg.estimators import (
    VideoFreezeContextXgEstimator,
    VideoGeometryXgEstimator,
    VideoKinematicContextXgEstimator,
    VideoOnlyXgEstimator,
)
from tactifoot_vision.video_xg.evaluation import evaluate_against_reference
from tactifoot_vision.video_xg.protocol import assert_video_only_columns
from tactifoot_vision.video_xg.results import (
    VideoOnlyShotPrediction,
    VideoOnlyXgSummary,
    VideoShotFeatures,
)


class VideoOnlyXgRunner:
    def __init__(self, estimator: VideoOnlyXgEstimator | None = None) -> None:
        self.estimator = estimator or VideoFreezeContextXgEstimator()

    def run(
        self,
        features_path: Path,
        *,
        output_dir: Path,
        reference_path: Path | None = None,
        group_id: str | None = None,
    ) -> tuple[VideoOnlyXgSummary, dict[str, float], tuple[ExportArtifact, ...]]:
        features = read_video_features(features_path)
        predictions = tuple(self.estimator.predict(feature) for feature in features)
        summary = VideoOnlyXgSummary(predictions=predictions, group_id=group_id)
        reference = read_reference(reference_path) if reference_path is not None else {}
        metrics = evaluate_against_reference(predictions, reference)
        artifacts = write_outputs(summary, metrics, output_dir)
        return summary, metrics, artifacts


def build_video_only_estimator(model_kind: XgModelKind) -> VideoOnlyXgEstimator:
    if model_kind == XgModelKind.VIDEO_GEOMETRY:
        return VideoGeometryXgEstimator()
    if model_kind == XgModelKind.VIDEO_FREEZE_CONTEXT:
        return VideoFreezeContextXgEstimator()
    if model_kind == XgModelKind.VIDEO_KINEMATIC_CONTEXT:
        return VideoKinematicContextXgEstimator()
    raise ValueError(f"Unsupported video-only xG model: {model_kind}")


def read_video_features(path: Path) -> tuple[VideoShotFeatures, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert_video_only_columns(reader.fieldnames or ())
        return tuple(_feature_from_row(row) for row in reader)


def read_reference(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            shot_id = row["shot_id"]
            values = {"reference_xg": float(row["reference_xg"])}
            if row.get("is_goal") not in (None, ""):
                values["is_goal"] = float(row["is_goal"])
            rows[shot_id] = values
        return rows


def write_outputs(
    summary: VideoOnlyXgSummary, metrics: Mapping[str, float], output_dir: Path
) -> tuple[ExportArtifact, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shots_path = output_dir / "video_only_shots.csv"
    with shots_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_prediction_fieldnames())
        writer.writeheader()
        for prediction in summary.predictions:
            writer.writerow(_prediction_row(prediction))
    summary_path = output_dir / "video_only_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "group_id": summary.group_id,
                "shots": summary.shot_count,
                "total_xg": summary.total_xg,
                "metrics": dict(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        ExportArtifact(
            path=shots_path, format="video_only_xg_shots_csv", rows=summary.shot_count
        ),
        ExportArtifact(path=summary_path, format="video_only_xg_summary_json", rows=1),
    )


def _feature_from_row(row: Mapping[str, str]) -> VideoShotFeatures:
    return VideoShotFeatures(
        shot_id=row["shot_id"],
        frame_index=int(row["frame_index"]),
        shot_x=float(row["shot_x"]),
        shot_y=float(row["shot_y"]),
        goal_x=_float_or_default(row, "goal_x", 105.0),
        goal_y=_float_or_default(row, "goal_y", 34.0),
        nearest_player_distance=_optional_float(row, "nearest_player_distance"),
        goalkeeper_distance=_optional_float(row, "goalkeeper_distance"),
        defender_count_in_cone=int(row.get("defender_count_in_cone") or 0),
        ball_speed=_optional_float(row, "ball_speed"),
        ball_direction_to_goal=_optional_float(row, "ball_direction_to_goal"),
        shot_confidence=_float_or_default(row, "shot_confidence", 1.0),
    )


def _float_or_default(row: Mapping[str, str], key: str, default: float) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _prediction_fieldnames() -> list[str]:
    return [
        "shot_id",
        "frame_index",
        "model_kind",
        "xg",
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


def _prediction_row(prediction: VideoOnlyShotPrediction) -> dict[str, object]:
    features = prediction.features
    return {
        "shot_id": prediction.shot_id,
        "frame_index": prediction.frame_index,
        "model_kind": prediction.model_kind.value,
        "xg": prediction.xg,
        "shot_x": features.shot_x,
        "shot_y": features.shot_y,
        "goal_x": features.goal_x,
        "goal_y": features.goal_y,
        "nearest_player_distance": features.nearest_player_distance,
        "goalkeeper_distance": features.goalkeeper_distance,
        "defender_count_in_cone": features.defender_count_in_cone,
        "ball_speed": features.ball_speed,
        "ball_direction_to_goal": features.ball_direction_to_goal,
        "shot_confidence": features.shot_confidence,
    }
