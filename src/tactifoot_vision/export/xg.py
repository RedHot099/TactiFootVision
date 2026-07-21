import csv
import json
from pathlib import Path

from tactifoot_vision.domain import ExportArtifact
from tactifoot_vision.xg import VideoXgSummary, XgPrediction


def write_xg_shots_csv(summary: VideoXgSummary, path: Path) -> ExportArtifact:
    rows = [
        _prediction_row(summary.group_id, prediction)
        for prediction in summary.predictions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_empty_row().keys()))
        writer.writeheader()
        writer.writerows(rows)
    return ExportArtifact(path=path, format="xg_shots_csv", rows=len(rows))


def write_xg_summary_json(summary: VideoXgSummary, path: Path) -> ExportArtifact:
    payload = {
        "group_id": summary.group_id,
        "shot_count": summary.shot_count,
        "total_xg": summary.total_xg,
        "shots": [
            _prediction_row(summary.group_id, prediction)
            for prediction in summary.predictions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ExportArtifact(path=path, format="xg_summary_json", rows=summary.shot_count)


def _prediction_row(
    group_id: str | None, prediction: XgPrediction
) -> dict[str, object]:
    features = prediction.features
    candidate = prediction.candidate
    return {
        "group_id": group_id or "",
        "frame": candidate.frame_index,
        "window_start": candidate.window.start_frame,
        "window_end": candidate.window.end_frame,
        "detector_kind": candidate.detector_kind.value,
        "outcome": candidate.outcome.value,
        "confidence": candidate.confidence,
        "xg": prediction.xg,
        "model_kind": prediction.model_kind.value,
        "shot_x": features.shot_x,
        "shot_y": features.shot_y,
        "distance_to_goal": features.distance_to_goal,
        "angle_to_goal": features.angle_to_goal,
        "centrality": features.centrality,
        "ball_speed": features.ball_speed,
        "nearest_player_distance": features.nearest_player_distance,
        "goalkeeper_distance": features.goalkeeper_distance,
        "defender_count_in_cone": features.defender_count_in_cone,
        "is_penalty": features.is_penalty,
    }


def _empty_row() -> dict[str, object]:
    return {
        "group_id": "",
        "frame": "",
        "window_start": "",
        "window_end": "",
        "detector_kind": "",
        "outcome": "",
        "confidence": "",
        "xg": "",
        "model_kind": "",
        "shot_x": "",
        "shot_y": "",
        "distance_to_goal": "",
        "angle_to_goal": "",
        "centrality": "",
        "ball_speed": "",
        "nearest_player_distance": "",
        "goalkeeper_distance": "",
        "defender_count_in_cone": "",
        "is_penalty": "",
    }
