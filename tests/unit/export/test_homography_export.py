import json
from pathlib import Path

import pandas as pd
import pytest

from tactifoot_vision.evaluation.homography import HomographyRecord, ProjectionRecord
from tactifoot_vision.export.homography import (
    read_homographies,
    write_homographies_parquet,
    write_metrics_json,
    write_projections_parquet,
)


def test_homography_parquet_round_trip(tmp_path: Path) -> None:
    record = HomographyRecord.available(
        sequence="SNGS-001",
        frame=1,
        method="tvcalib",
        homography_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        runtime_ms=12.5,
        inliers=8,
        source_artifact="tvcalib.jsonl",
    )

    artifact = write_homographies_parquet([record], tmp_path / "homographies.parquet")
    loaded = read_homographies(artifact.path, allowed_frames={"SNGS-001": {1}})

    assert artifact.rows == 1
    assert loaded == (record,)


def test_projection_parquet_has_common_schema(tmp_path: Path) -> None:
    record = ProjectionRecord(
        sequence="SNGS-001",
        frame=1,
        track_id=7,
        role="player",
        method="tvcalib",
        image_x=25.0,
        image_y=60.0,
        pitch_x_pred=10.0,
        pitch_y_pred=20.0,
        pitch_x_gt=10.0,
        pitch_y_gt=20.0,
        error_m=0.0,
    )

    artifact = write_projections_parquet([record], tmp_path / "projections.parquet")
    dataframe = pd.read_parquet(artifact.path)

    assert artifact.rows == 1
    assert dataframe.columns.tolist() == [
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


def test_read_homographies_rejects_invalid_matrix_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "sequence": "SNGS-001",
                "frame": 1,
                "method": "tvcalib",
                "status": "available",
                "homography_3x3": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="shape"):
        read_homographies(path, allowed_frames={"SNGS-001": {1}})


def test_read_homographies_rejects_nan_matrix(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "sequence": "SNGS-001",
                "frame": 1,
                "method": "tvcalib",
                "status": "available",
                "homography_3x3": [
                    [1.0, 0.0, 0.0],
                    [0.0, float("nan"), 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        read_homographies(path, allowed_frames={"SNGS-001": {1}})


def test_read_homographies_rejects_frame_outside_allowed_set(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "sequence": "SNGS-001",
                "frame": 99,
                "method": "tvcalib",
                "status": "available",
                "homography_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allowed frame"):
        read_homographies(path, allowed_frames={"SNGS-001": {1}})


def test_write_metrics_json_replaces_nan_with_null(tmp_path: Path) -> None:
    artifact = write_metrics_json(
        {"per_method": {"tvcalib": {"median_error_m": float("nan")}}},
        tmp_path / "metrics.json",
    )

    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert payload["per_method"]["tvcalib"]["median_error_m"] is None
