from pathlib import Path

import pandas as pd

from tactifoot_vision.evaluation.mot import (
    evaluate_mot_tracking,
    evaluate_tracking_files,
)


def test_mot_metrics_match_hand_computed_fixture() -> None:
    predictions = pd.DataFrame(
        [
            {"frame": 1, "track_id": 10, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 2, "track_id": 10, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
    )
    ground_truth = pd.DataFrame(
        [
            {"frame": 1, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 2, "id": 1, "x": 30, "y": 30, "width": 10, "height": 10},
        ]
    )

    metrics = evaluate_mot_tracking(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_empty_predictions_and_gt_return_stable_zeroes() -> None:
    predictions = pd.DataFrame(
        columns=["frame", "track_id", "x", "y", "width", "height"]
    )
    ground_truth = pd.DataFrame(columns=["frame", "id", "x", "y", "width", "height"])

    metrics = evaluate_mot_tracking(predictions, ground_truth)

    assert metrics["precision"] == 0.0
    assert metrics["frames_evaluated"] == 0.0


def test_id_switch_counts_gt_identity_changing_predicted_track() -> None:
    predictions = pd.DataFrame(
        [
            {"frame": 1, "track_id": 10, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 2, "track_id": 11, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
    )
    ground_truth = pd.DataFrame(
        [
            {"frame": 1, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 2, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
    )

    metrics = evaluate_mot_tracking(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["id_switches"] == 1.0


def test_prediction_frame_offset_aligns_pipeline_csv_to_mot_gt() -> None:
    predictions = pd.DataFrame(
        [
            {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "track_id": 10,
                "class_id": 2,
                "class_name": "player",
                "x": 0,
                "y": 0,
                "width": 10,
                "height": 10,
            }
        ]
    )
    ground_truth = pd.DataFrame(
        [{"frame": 1, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10}]
    )

    metrics = evaluate_mot_tracking(
        predictions, ground_truth, prediction_frame_offset=1
    )

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 0.0


def test_plain_csv_does_not_auto_shift_frames(tmp_path: Path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.txt"
    pd.DataFrame(
        [{"frame": 0, "track_id": 5, "x": 0, "y": 0, "width": 10, "height": 10}]
    ).to_csv(pred, index=False)
    gt.write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")

    metrics = evaluate_tracking_files(pred, gt)

    assert metrics["prediction_frame_offset"] == 0.0
    assert metrics["tp"] == 0.0


def test_mot_prediction_file_does_not_auto_shift(tmp_path: Path) -> None:
    pred = tmp_path / "pred.txt"
    gt = tmp_path / "gt.txt"
    pred.write_text("1,5,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")
    gt.write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")

    metrics = evaluate_tracking_files(pred, gt)

    assert metrics["prediction_frame_offset"] == 0.0
    assert metrics["tp"] == 1.0


def test_id_switch_counts_after_unmatched_gap() -> None:
    predictions = pd.DataFrame(
        [
            {"frame": 1, "track_id": 10, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 3, "track_id": 11, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
    )
    ground_truth = pd.DataFrame(
        [
            {"frame": 1, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10},
            {"frame": 2, "id": 1, "x": 100, "y": 100, "width": 10, "height": 10},
            {"frame": 3, "id": 1, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
    )

    metrics = evaluate_mot_tracking(predictions, ground_truth)

    assert metrics["id_switches"] == 1.0
