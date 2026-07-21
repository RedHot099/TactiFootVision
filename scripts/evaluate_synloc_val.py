from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.synloc_models import SynLocPrediction
from tactifoot_vision.synloc.eval import compare_evaluation_backends, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SynLoc results.json file against annotations.")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotations path.")
    parser.add_argument("--results", type=Path, required=True, help="results.json path.")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--position-from-keypoint-index", type=int, default=None)
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Compare the local pycocotools evaluator against the official sskit backend when available.",
    )
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
    if args.compare_reference:
        metrics = compare_evaluation_backends(
            annotation_path=args.annotations,
            predictions=predictions,
            score_threshold=args.score_threshold,
            position_from_keypoint_index=args.position_from_keypoint_index,
        )
    else:
        metrics = evaluate_predictions(
            annotation_path=args.annotations,
            predictions=predictions,
            score_threshold=args.score_threshold,
            position_from_keypoint_index=args.position_from_keypoint_index,
        )
    print(json.dumps(metrics, indent=2))


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
