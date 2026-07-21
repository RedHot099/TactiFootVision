import json
from pathlib import Path

import pandas as pd

from tactifoot_vision.domain import ExportArtifact
from tactifoot_vision.evaluation.homography import (
    HomographyRecord,
    ProjectionRecord,
    homography_from_mapping,
    homography_to_dict,
    projection_to_dict,
)


def read_homographies(
    path: str | Path, *, allowed_frames: dict[str, set[int]] | None = None
) -> tuple[HomographyRecord, ...]:
    artifact_path = Path(path)
    rows = _read_rows(artifact_path)
    return tuple(
        homography_from_mapping(row, allowed_frames=allowed_frames) for row in rows
    )


def write_homographies_parquet(
    records: list[HomographyRecord] | tuple[HomographyRecord, ...], path: str | Path
) -> ExportArtifact:
    artifact_path = Path(path)
    rows = [homography_to_dict(record) for record in records]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_homography_columns()).to_parquet(
        artifact_path, index=False
    )
    return ExportArtifact(
        path=artifact_path, format="homographies_parquet", rows=len(rows)
    )


def write_projections_parquet(
    records: list[ProjectionRecord] | tuple[ProjectionRecord, ...], path: str | Path
) -> ExportArtifact:
    artifact_path = Path(path)
    rows = [projection_to_dict(record) for record in records]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_projection_columns()).to_parquet(
        artifact_path, index=False
    )
    return ExportArtifact(
        path=artifact_path, format="homography_projections_parquet", rows=len(rows)
    )


def write_metrics_json(metrics: dict[str, object], path: str | Path) -> ExportArtifact:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )
    return ExportArtifact(path=artifact_path, format="homography_metrics_json")


def _read_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(row)
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Expected JSON object rows in {path}")
        return payload
    raise ValueError(f"Unsupported homography artifact format: {path.suffix}")


def _homography_columns() -> list[str]:
    return [
        "sequence",
        "frame",
        "method",
        "status",
        "homography_3x3",
        "runtime_ms",
        "inliers",
        "source_artifact",
        "failure_reason",
    ]


def _projection_columns() -> list[str]:
    return [
        "sequence",
        "frame",
        "track_id",
        "role",
        "method",
        "image_x",
        "image_y",
        "pitch_x_pred",
        "pitch_y_pred",
        "pitch_x_gt",
        "pitch_y_gt",
        "error_m",
    ]


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not pd.notna(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
