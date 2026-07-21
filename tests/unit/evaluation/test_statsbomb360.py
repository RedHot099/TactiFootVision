import pandas as pd
import pytest

from tactifoot_vision.evaluation.statsbomb360 import evaluate_statsbomb360_projection


def test_statsbomb360_projection_metrics() -> None:
    projected = pd.DataFrame(
        [{"frame": 1, "track_id": 10, "pitch_x": 3.0, "pitch_y": 4.0}]
    )
    reference = pd.DataFrame(
        [{"frame": 1, "track_id": 10, "pitch_x": 0.0, "pitch_y": 0.0}]
    )

    metrics = evaluate_statsbomb360_projection(projected, reference)

    assert metrics["mean_distance"] == 5.0
    assert metrics["coverage"] == 1.0
    assert metrics["matched_points"] == 1.0


def test_statsbomb360_missing_projection_returns_zero_coverage() -> None:
    metrics = evaluate_statsbomb360_projection(
        pd.DataFrame(),
        pd.DataFrame([{"frame": 1, "track_id": 10, "pitch_x": 0.0, "pitch_y": 0.0}]),
    )

    assert metrics["coverage"] == 0.0


def test_statsbomb360_missing_columns_raise() -> None:
    projected = pd.DataFrame([{"frame": 1, "track_id": 10, "pitch_x": 3.0}])
    reference = pd.DataFrame(
        [{"frame": 1, "track_id": 10, "pitch_x": 0.0, "pitch_y": 0.0}]
    )

    with pytest.raises(ValueError, match="pitch_y"):
        evaluate_statsbomb360_projection(projected, reference)
