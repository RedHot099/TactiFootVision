import numpy as np
import pandas as pd

from tactifoot_vision.domain import PipelineResult


def projected_points_from_result(result: PipelineResult) -> pd.DataFrame:
    rows = []
    for frame in result.frames:
        projection = frame.projection
        if projection is None or projection.status != "available":
            continue
        for track_id, point in projection.points_by_track_id.items():
            rows.append(
                {
                    "frame": frame.frame_index,
                    "track_id": track_id,
                    "pitch_x": point.x,
                    "pitch_y": point.y,
                }
            )
    return pd.DataFrame(rows)


def evaluate_statsbomb360_projection(
    projected: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    frame_column: str = "frame",
    track_column: str = "track_id",
) -> dict[str, float]:
    """Evaluate already-normalized projection tables.

    Both inputs must use shared frame and track identifiers plus pitch_x/pitch_y
    columns. This helper does not parse native StatsBomb360 freeze-frame data.
    """
    required = {frame_column, track_column, "pitch_x", "pitch_y"}
    if projected.empty or reference.empty:
        return _empty_metrics()
    if not required.issubset(projected.columns) or not required.issubset(
        reference.columns
    ):
        missing = required - set(projected.columns) | (
            required - set(reference.columns)
        )
        raise ValueError(
            f"StatsBomb360 evaluation missing columns: {', '.join(sorted(missing))}"
        )
    merged = projected.merge(
        reference,
        on=[frame_column, track_column],
        suffixes=("_pred", "_ref"),
    )
    if merged.empty:
        metrics = _empty_metrics()
        metrics["coverage"] = 0.0
        return metrics
    distances = np.sqrt(
        (merged["pitch_x_pred"] - merged["pitch_x_ref"]) ** 2
        + (merged["pitch_y_pred"] - merged["pitch_y_ref"]) ** 2
    )
    return {
        "mean_distance": float(distances.mean()),
        "median_distance": float(distances.median()),
        "p90_distance": float(distances.quantile(0.9)),
        "coverage": float(len(merged) / len(reference)) if len(reference) else 0.0,
        "matched_points": float(len(merged)),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "mean_distance": 0.0,
        "median_distance": 0.0,
        "p90_distance": 0.0,
        "coverage": 0.0,
        "matched_points": 0.0,
    }
