import math
from collections.abc import Mapping, Sequence

from tactifoot_vision.evaluation.xg import (
    brier_score,
    expected_calibration_error,
    log_loss,
)
from tactifoot_vision.video_xg.results import VideoOnlyShotPrediction


def evaluate_against_reference(
    predictions: Sequence[VideoOnlyShotPrediction],
    reference_by_shot_id: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    matched = [
        (prediction, reference_by_shot_id[prediction.shot_id])
        for prediction in predictions
        if prediction.shot_id in reference_by_shot_id
    ]
    if not matched:
        return {
            "matched_shots": 0.0,
            "mae_vs_reference_xg": 0.0,
            "rmse_vs_reference_xg": 0.0,
            "total_reference_xg": 0.0,
            "total_predicted_xg": 0.0,
            "total_xg_error": 0.0,
            "brier_vs_goal_outcome": 0.0,
            "log_loss_vs_goal_outcome": 0.0,
            "ece_vs_goal_outcome": 0.0,
        }
    predicted = [prediction.xg for prediction, _ in matched]
    reference = [float(row["reference_xg"]) for _, row in matched]
    errors = [
        prediction - actual
        for prediction, actual in zip(predicted, reference, strict=True)
    ]
    metrics = {
        "matched_shots": float(len(matched)),
        "mae_vs_reference_xg": float(sum(abs(error) for error in errors) / len(errors)),
        "rmse_vs_reference_xg": float(
            math.sqrt(sum(error * error for error in errors) / len(errors))
        ),
        "total_reference_xg": float(sum(reference)),
        "total_predicted_xg": float(sum(predicted)),
        "total_xg_error": float(sum(predicted) - sum(reference)),
    }
    outcomes = [
        float(row["is_goal"])
        for _, row in matched
        if "is_goal" in row and row["is_goal"] is not None
    ]
    if len(outcomes) == len(predicted):
        metrics.update(
            {
                "brier_vs_goal_outcome": brier_score(outcomes, predicted),
                "log_loss_vs_goal_outcome": log_loss(outcomes, predicted),
                "ece_vs_goal_outcome": expected_calibration_error(
                    outcomes, predicted, bins=5
                ),
            }
        )
    return metrics
