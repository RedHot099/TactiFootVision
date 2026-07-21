from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.synloc_models import SynLocPrediction, SynLocSubmissionConfig
from tactifoot_vision.synloc.submission import build_submission_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SynLoc results into a Codabench submission zip.")
    parser.add_argument("--results", type=Path, required=True, help="results.json or raw predictions file.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synloc/submissions"))
    parser.add_argument("--split", choices=["val", "valid", "test", "challenge"], default="challenge")
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--position-from-keypoint-index", type=int, default=None)
    parser.add_argument("--archive-name", type=str, default=None)
    parser.add_argument("--zip-name", type=str, default=None)
    parser.add_argument("--topk-per-image", type=int, default=None)
    args = parser.parse_args()

    raw_results = json.loads(args.results.read_text(encoding="utf-8"))
    predictions = [
        SynLocPrediction(
            image_id=int(item["image_id"]),
            category_id=int(item.get("category_id", 1)),
            score=float(item["score"]),
            bbox_xyxy=_xywh_to_xyxy(item.get("bbox", [0.0, 0.0, 0.0, 0.0])),
            image_point_xy=_extract_image_point(item),
            position_on_pitch_xyz=[float(v) for v in item.get("position_on_pitch", [0.0, 0.0, 0.0])],
        )
        for item in raw_results
    ]
    config = SynLocSubmissionConfig(
        split=args.split,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        position_from_keypoint_index=args.position_from_keypoint_index,
        archive_name=args.archive_name,
        zip_name=args.zip_name,
        topk_per_image=args.topk_per_image,
    )
    archive_path = build_submission_archive(predictions, config)
    print(archive_path)


def _extract_image_point(item: dict[str, object]) -> list[float]:
    if "keypoints" not in item:
        bbox = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x, y, w, h = [float(v) for v in bbox]
        return [x + w / 2.0, y + h]
    keypoints = [float(v) for v in item["keypoints"]]  # type: ignore[index]
    if len(keypoints) >= 6:
        return [keypoints[3], keypoints[4]]
    return [0.0, 0.0]


def _xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


if __name__ == "__main__":
    main()
