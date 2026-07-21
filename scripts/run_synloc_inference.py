from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.synloc_loaders import load_synloc_config
from tactifoot_vision.synloc.data import SynLocSplitData, load_synloc_split
from tactifoot_vision.synloc.eval import compare_evaluation_backends, evaluate_predictions
from tactifoot_vision.synloc.inference import run_inference_on_split_with_diagnostics
from tactifoot_vision.synloc.point_regressor import load_point_regressor
from tactifoot_vision.synloc.submission import build_submission_archive, serialize_predictions, write_submission_files


def _subset_split(split_data: SynLocSplitData, limit: int) -> SynLocSplitData:
    if limit <= 0 or limit >= len(split_data.images):
        return split_data
    selected_images = split_data.images[:limit]
    selected_ids = {image.image_id for image in selected_images}
    selected_annotations = [ann for ann in split_data.annotations if ann.image_id in selected_ids]
    selected_annotations_by_image = {
        image_id: annotations
        for image_id, annotations in split_data.annotations_by_image.items()
        if image_id in selected_ids
    }
    return SynLocSplitData(
        dataset_root=split_data.dataset_root,
        split=split_data.split,
        annotation_path=split_data.annotation_path,
        images=selected_images,
        images_by_id={image.image_id: image for image in selected_images},
        annotations=selected_annotations,
        annotations_by_image=selected_annotations_by_image,
        categories=split_data.categories,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SynLoc inference on a dataset split.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a SynLoc YAML config.")
    parser.add_argument("--archive", action="store_true", help="Build a submission zip after inference.")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Compare local validation metrics against the official sskit backend when available.",
    )
    parser.add_argument("--debug-summary-out", type=Path, default=None, help="Optional path for per-stage inference diagnostics JSON.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional limit for the number of images processed from the split.")
    args = parser.parse_args()

    config = load_synloc_config(args.config)
    split_data = load_synloc_split(config.dataset)
    if args.sample_limit is not None:
        split_data = _subset_split(split_data, args.sample_limit)
    point_regressor = None
    if (
        config.point_regressor_checkpoint is not None
        and config.projection.point_strategy == "learned_offset"
    ):
        point_regressor = load_point_regressor(config.point_regressor_checkpoint)

    visualize_dir = config.visuals_dir.resolve() if config.visuals_dir is not None else None
    run_result = run_inference_on_split_with_diagnostics(
        split_data,
        dataset_config=config.dataset,
        detector_config=config.detector,
        projection_config=config.projection,
        model_dir=config.model_dir,
        point_regressor=point_regressor,
        visualize_dir=visualize_dir,
    )
    predictions = run_result.predictions

    results_path, metadata_path = write_submission_files(predictions, config.submission)
    raw_results_path = config.results_path or (config.submission.output_dir / f"{config.dataset.split}_results_raw.json")
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_results_path.write_text(json.dumps(serialize_predictions(predictions), indent=2), encoding="utf-8")
    print(f"results={results_path}")
    print(f"metadata={metadata_path}")
    print(f"raw_results={raw_results_path}")
    if args.debug_summary_out is not None:
        args.debug_summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.debug_summary_out.write_text(json.dumps(run_result.summary.to_dict(), indent=2), encoding="utf-8")
        print(f"debug_summary={args.debug_summary_out}")

    if config.dataset.split in {"val", "valid"}:
        metrics = (
            compare_evaluation_backends(
                annotation_path=split_data.annotation_path,
                predictions=predictions,
            )
            if args.compare_reference
            else evaluate_predictions(
                annotation_path=split_data.annotation_path,
                predictions=predictions,
            )
        )
        print(json.dumps(metrics, indent=2))

    if args.archive:
        archive_path = build_submission_archive(predictions, config.submission)
        print(f"archive={archive_path}")


if __name__ == "__main__":
    main()
