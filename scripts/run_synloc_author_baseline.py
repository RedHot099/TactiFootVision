from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config.synloc_models import SynLocDatasetConfig, SynLocSubmissionConfig
from tactifoot_vision.synloc.author_baseline import AUTHOR_BASELINE_CONFIG, prepare_author_baseline_workspace
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.eval import compare_evaluation_backends, evaluate_predictions
from tactifoot_vision.synloc.prediction_io import load_predictions_from_results_json
from tactifoot_vision.synloc.submission import build_submission_archive, write_submission_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official Spiideo SynLoc pose baseline and import its outputs locally.")
    parser.add_argument("--repo-root", type=Path, required=True, help="Path to Spiideo/mmpose checked out on branch spiideo_scenes.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SoccerNet/SpiideoSynLoc"))
    parser.add_argument("--config", type=str, default=AUTHOR_BASELINE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test", "challenge"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synloc/author_pose_baseline"))
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_author_baseline_workspace(
        dataset_root=args.dataset_root.resolve(),
        output_dir=output_dir / "workspace",
        official_repo_root=repo_root,
        split=args.split,
        official_config=args.config,
    )

    metrics_dump_path = output_dir / "official_metrics.json"
    command = [
        sys.executable,
        "tools/test.py",
        str(prepared["override_config_path"]),
        str(args.checkpoint.resolve()),
        "--out",
        str(metrics_dump_path),
    ]
    if args.split == "challenge":
        command.append("--challenge")
    subprocess.run(command, cwd=repo_root, check=True)

    official_results = repo_root / "results.json"
    official_metadata = repo_root / "metadata.json"
    official_zip = repo_root / ("challenge_submission.zip" if args.split == "challenge" else "test_submission.zip")

    copied_results = output_dir / "official_results.json"
    copied_metadata = output_dir / "official_metadata.json"
    shutil.copyfile(official_results, copied_results)
    shutil.copyfile(official_metadata, copied_metadata)
    if official_zip.exists():
        shutil.copyfile(official_zip, output_dir / official_zip.name)

    split_name = "challenge" if args.split == "challenge" else args.split
    split_data = load_synloc_split(SynLocDatasetConfig(root=args.dataset_root.resolve(), split=split_name))
    predictions = load_predictions_from_results_json(
        copied_results,
        split_data=split_data,
        position_from_keypoint_index=1,
    )
    metadata = json.loads(copied_metadata.read_text(encoding="utf-8"))
    results_path, metadata_path = write_submission_files(
        predictions,
        SynLocSubmissionConfig(
            split=split_name,
            output_dir=output_dir,
            score_threshold=float(metadata["score_threshold"]),
            position_from_keypoint_index=1,
        ),
    )
    archive_path = build_submission_archive(
        predictions,
        SynLocSubmissionConfig(
            split=split_name,
            output_dir=output_dir,
            score_threshold=float(metadata["score_threshold"]),
            position_from_keypoint_index=1,
        ),
    )

    payload: dict[str, object] = {
        "override_config_path": str(prepared["override_config_path"]),
        "official_results": str(copied_results),
        "official_metadata": str(copied_metadata),
        "results": str(results_path),
        "metadata": str(metadata_path),
        "archive": str(archive_path),
    }
    if split_name in {"val", "test"}:
        metrics = (
            compare_evaluation_backends(
                annotation_path=split_data.annotation_path,
                predictions=predictions,
                score_threshold=float(metadata["score_threshold"]),
                position_from_keypoint_index=1,
            )
            if args.compare_reference
            else evaluate_predictions(
                annotation_path=split_data.annotation_path,
                predictions=predictions,
                score_threshold=float(metadata["score_threshold"]),
                position_from_keypoint_index=1,
            )
        )
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        payload["metrics"] = str(metrics_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
