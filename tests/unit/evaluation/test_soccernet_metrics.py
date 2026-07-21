from pathlib import Path

import pandas as pd

from tactifoot_vision.evaluation.soccernet import evaluate_soccernet_tracking


def test_soccernet_tracking_evaluates_sequence_gt(tmp_path: Path) -> None:
    sequence = tmp_path / "SNMOT-001"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8"
    )
    pred = tmp_path / "pred.csv"
    pd.DataFrame(
        [{"frame": 1, "track_id": 5, "x": 0, "y": 0, "width": 10, "height": 10}]
    ).to_csv(pred, index=False)

    metrics = evaluate_soccernet_tracking(pred, sequence)

    assert metrics["tp"] == 1.0


def test_soccernet_tracking_auto_offsets_pipeline_csv(tmp_path: Path) -> None:
    sequence = tmp_path / "SNMOT-001"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8"
    )
    pred = tmp_path / "pipeline.csv"
    pd.DataFrame(
        [
            {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "track_id": 5,
                "class_id": 2,
                "class_name": "player",
                "x": 0,
                "y": 0,
                "width": 10,
                "height": 10,
            }
        ]
    ).to_csv(pred, index=False)

    metrics = evaluate_soccernet_tracking(pred, sequence)

    assert metrics["prediction_frame_offset"] == 1.0
    assert metrics["tp"] == 1.0
