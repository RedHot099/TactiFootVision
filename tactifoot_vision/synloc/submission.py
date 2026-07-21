from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

from config.synloc_models import SynLocPrediction, SynLocSubmissionConfig


def serialize_predictions(
    predictions: Iterable[SynLocPrediction],
    *,
    position_from_keypoint_index: int | None = None,
) -> list[dict[str, object]]:
    return [
        prediction.to_result_dict(
            idx + 1,
            position_from_keypoint_index=position_from_keypoint_index,
        )
        for idx, prediction in enumerate(predictions)
    ]


def write_submission_files(
    predictions: Iterable[SynLocPrediction],
    config: SynLocSubmissionConfig,
) -> tuple[Path, Path]:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    metadata_path = output_dir / "metadata.json"

    payload = serialize_predictions(
        _limit_predictions_per_image(predictions, topk=config.topk_per_image),
        position_from_keypoint_index=config.position_from_keypoint_index,
    )
    metadata: dict[str, object] = {"score_threshold": float(config.score_threshold)}
    if config.position_from_keypoint_index is not None:
        metadata["position_from_keypoint_index"] = int(config.position_from_keypoint_index)

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results_path, metadata_path


def build_submission_archive(
    predictions: Iterable[SynLocPrediction],
    config: SynLocSubmissionConfig,
) -> Path:
    results_path, metadata_path = write_submission_files(predictions, config)
    archive_name = config.zip_name or config.archive_name or f"{config.split}_submission.zip"
    archive_path = config.output_dir.resolve() / archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, "results.json")
        archive.write(metadata_path, "metadata.json")
    return archive_path


def _limit_predictions_per_image(
    predictions: Iterable[SynLocPrediction],
    *,
    topk: int | None,
) -> list[SynLocPrediction]:
    items = list(predictions)
    if topk is None:
        return items
    by_image: dict[int, list[SynLocPrediction]] = {}
    for prediction in items:
        by_image.setdefault(prediction.image_id, []).append(prediction)

    limited: list[SynLocPrediction] = []
    for image_id in sorted(by_image):
        limited.extend(sorted(by_image[image_id], key=lambda item: item.score, reverse=True)[:topk])
    return limited
