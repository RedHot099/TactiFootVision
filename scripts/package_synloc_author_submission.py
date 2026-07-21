from __future__ import annotations

import argparse
from pathlib import Path

from config.synloc_models import SynLocDatasetConfig, SynLocSubmissionConfig
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.prediction_io import load_predictions_from_results_json
from tactifoot_vision.synloc.submission import build_submission_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package official author-baseline keypoint results as a SynLoc submission zip.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "valid", "test", "challenge"], default="challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synloc/author_pose_submission"))
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--position-from-keypoint-index", type=int, default=1)
    parser.add_argument("--archive-name", type=str, default=None)
    parser.add_argument("--zip-name", type=str, default=None)
    parser.add_argument("--topk-per-image", type=int, default=None)
    args = parser.parse_args()

    split_data = load_synloc_split(SynLocDatasetConfig(root=args.dataset_root.resolve(), split=args.split))
    predictions = load_predictions_from_results_json(
        args.results,
        split_data=split_data,
        position_from_keypoint_index=args.position_from_keypoint_index,
    )
    archive_path = build_submission_archive(
        predictions,
        SynLocSubmissionConfig(
            split=args.split,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            position_from_keypoint_index=args.position_from_keypoint_index,
            archive_name=args.archive_name,
            zip_name=args.zip_name,
            topk_per_image=args.topk_per_image,
        ),
    )
    print(archive_path.resolve())


if __name__ == "__main__":
    main()
