from pathlib import Path

from tactifoot_vision.evaluation.mot import evaluate_tracking_files


def evaluate_soccernet_tracking(
    prediction_csv: Path,
    sequence_dir: Path,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    return evaluate_tracking_files(
        prediction_csv,
        sequence_dir / "gt" / "gt.txt",
        iou_threshold=iou_threshold,
    )
