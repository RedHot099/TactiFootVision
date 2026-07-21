import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.config import (
    DetectionTrainingConfig,
    load_experiment_config,
    load_pipeline_config,
)
from tactifoot_vision.config.factories import build_detector, build_pipeline
from tactifoot_vision.datasets import SoccerNetTrackingDataset
from tactifoot_vision.detection import RFDETRDetectionModel, YOLODetectionModel
from tactifoot_vision.detection.interfaces import TrainableDetectionModel
from tactifoot_vision.domain import Frame
from tactifoot_vision.enums import (
    DatasetFormat,
    DatasetSource,
    DetectionBackend,
    XgModelKind,
)
from tactifoot_vision.evaluation.mot import evaluate_tracking_files
from tactifoot_vision.experiments.detection_tracking import (
    DetectionTrackingExperimentRunner,
)
from tactifoot_vision.experiments.homography_comparison import (
    HomographyComparisonRunner,
)
from tactifoot_vision.experiments.team_classification import (
    TeamClassificationExperimentRunner,
)
from tactifoot_vision.experiments.video_xg import VideoXgExperimentRunner
from tactifoot_vision.video_xg import (
    DetectionBenchmarkRunner,
    VideoOnlyXgEndToEndRunner,
    VideoOnlyXgRunner,
    VideoXgWeaknessAblationRunner,
    load_video_only_xg_end_to_end_config,
)
from tactifoot_vision.video_xg.artifacts import read_dataframe_artifact
from tactifoot_vision.video_xg.experiment import run_video_only_xg_experiment
from tactifoot_vision.video_xg.runner import build_video_only_estimator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tactifoot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", required=True)
    infer_parser.add_argument("--max-frames", type=int)

    detect_parser = subparsers.add_parser("detect")
    detect_sub = detect_parser.add_subparsers(dest="target", required=True)
    detect_image = detect_sub.add_parser("image")
    detect_image.add_argument("--config", required=True)
    detect_image.add_argument("--input", required=True)

    track_parser = subparsers.add_parser("track")
    track_sub = track_parser.add_subparsers(dest="target", required=True)
    track_images = track_sub.add_parser("images")
    track_images.add_argument("--config", required=True)
    track_images.add_argument("--max-frames", type=int)

    dataset_parser = subparsers.add_parser("dataset")
    dataset_sub = dataset_parser.add_subparsers(dest="action", required=True)
    dataset_convert = dataset_sub.add_parser("convert")
    dataset_convert_sub = dataset_convert.add_subparsers(dest="dataset", required=True)
    soccernet_convert = dataset_convert_sub.add_parser("soccernet-tracking")
    soccernet_convert.add_argument("--input", required=True)
    soccernet_convert.add_argument("--output", required=True)
    soccernet_convert.add_argument("--valid-fraction", type=float, default=0.2)
    soccernet_convert.add_argument("--seed", type=int, default=42)
    soccernet_convert.add_argument("--every-nth-frame", type=int, default=1)
    soccernet_convert.add_argument("--max-sequences", type=int)
    soccernet_convert.add_argument("--copy-images", action="store_true")

    train_parser = subparsers.add_parser("train")
    train_sub = train_parser.add_subparsers(dest="target", required=True)
    train_detector = train_sub.add_parser("detector")
    train_detector.add_argument(
        "--backend",
        choices=[DetectionBackend.YOLO.value, DetectionBackend.RFDETR.value],
        required=True,
    )
    train_detector.add_argument("--weights", required=True)
    train_detector.add_argument("--data", required=True)
    train_detector.add_argument("--epochs", type=int, default=50)
    train_detector.add_argument(
        "--dataset-format",
        choices=[value.value for value in DatasetFormat],
        default=DatasetFormat.YOLO.value,
    )
    train_detector.add_argument(
        "--dataset-source",
        choices=[value.value for value in DatasetSource],
        default=DatasetSource.FILESYSTEM.value,
    )
    train_detector.add_argument("--converted-dataset-dir")
    train_detector.add_argument("--valid-fraction", type=float, default=0.2)
    train_detector.add_argument("--every-nth-frame", type=int, default=1)
    train_detector.add_argument("--max-sequences", type=int)
    train_detector.add_argument("--copy-images", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_sub = validate_parser.add_subparsers(dest="target", required=True)
    validate_detector = validate_sub.add_parser("detector")
    validate_detector.add_argument(
        "--backend",
        choices=[DetectionBackend.YOLO.value, DetectionBackend.RFDETR.value],
        required=True,
    )
    validate_detector.add_argument("--weights", required=True)
    validate_detector.add_argument("--data", required=True)

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument(
        "kind",
        choices=[
            "detection-tracking",
            "team-classification",
            "video-xg",
            "homography-comparison",
        ],
    )
    experiment_parser.add_argument("--config", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("target", choices=["tracking"])
    evaluate_parser.add_argument("--config")
    evaluate_parser.add_argument("--pred")
    evaluate_parser.add_argument("--gt")
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--pred-frame-offset", type=int)

    video_xg_parser = subparsers.add_parser("video-xg")
    video_xg_sub = video_xg_parser.add_subparsers(dest="action", required=True)
    video_xg_features = video_xg_sub.add_parser("from-features")
    video_xg_features.add_argument("--features", required=True)
    video_xg_features.add_argument("--output-dir", required=True)
    video_xg_features.add_argument("--reference")
    video_xg_features.add_argument("--group-id")
    video_xg_features.add_argument(
        "--model",
        choices=[
            XgModelKind.VIDEO_GEOMETRY.value,
            XgModelKind.VIDEO_FREEZE_CONTEXT.value,
            XgModelKind.VIDEO_KINEMATIC_CONTEXT.value,
        ],
        default=XgModelKind.VIDEO_FREEZE_CONTEXT.value,
    )
    video_xg_compare = video_xg_sub.add_parser("compare-methods")
    video_xg_compare.add_argument("--features", required=True)
    video_xg_compare.add_argument("--output-dir", required=True)
    video_xg_compare.add_argument("--reference")
    video_xg_compare.add_argument("--group-id")
    video_xg_e2e = video_xg_sub.add_parser("end-to-end")
    video_xg_e2e.add_argument("--config", required=True)
    video_xg_e2e.add_argument("--resume-from")
    video_xg_e2e.add_argument("--stop-after")
    video_xg_e2e.add_argument("--force-stage")
    video_xg_ablate = video_xg_sub.add_parser("ablate")
    video_xg_ablate.add_argument("--config", required=True)
    video_xg_benchmark = video_xg_sub.add_parser("benchmark-detection")
    video_xg_benchmark.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "infer":
        config = load_pipeline_config(args.config)
        result = build_pipeline(config).run_video(
            _require_input(config.paths.input),
            max_frames=args.max_frames,
        )
        if config.export.pipeline_csv:
            result.to_csv(config.export.pipeline_csv)
        if config.export.mot:
            result.to_mot(config.export.mot)
        return 0
    if args.command == "detect" and args.target == "image":
        config = load_pipeline_config(args.config)
        image_path = Path(args.input)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Input image could not be read: {image_path}")
        detections = build_detector(config).predict(
            Frame(index=0, image=np.asarray(image, dtype=np.uint8), path=image_path)
        )
        print(
            json.dumps(
                {
                    "backend": config.detection.backend.value,
                    "input": str(image_path),
                    "detections": len(detections),
                    "classes": sorted(
                        {detection.class_name for detection in detections}
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "track" and args.target == "images":
        config = load_pipeline_config(args.config)
        input_path = _require_input(config.paths.input)
        result = build_pipeline(config).run_video(
            input_path,
            max_frames=args.max_frames,
        )
        tracks = [
            track for frame_result in result.frames for track in frame_result.tracks
        ]
        print(
            json.dumps(
                {
                    "backend": config.tracking.backend.value,
                    "classes": sorted({track.class_name for track in tracks}),
                    "frames": len(result.frames),
                    "input": str(input_path),
                    "track_ids": sorted({track.track_id for track in tracks}),
                    "tracks": len(tracks),
                },
                sort_keys=True,
            )
        )
        return 0
    if (
        args.command == "dataset"
        and args.action == "convert"
        and args.dataset == "soccernet-tracking"
    ):
        report = SoccerNetTrackingDataset(args.input).to_coco(
            args.output,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
            every_nth_frame=args.every_nth_frame,
            max_sequences=args.max_sequences,
            symlink_images=not args.copy_images,
        )
        print(json.dumps(report.to_dict(), sort_keys=True))
        return 0
    if args.command == "train" and args.target == "detector":
        model = _detection_model(args.backend, args.weights)
        model.train(
            DetectionTrainingConfig(
                data=Path(args.data),
                epochs=args.epochs,
                dataset_format=DatasetFormat(args.dataset_format),
                dataset_source=DatasetSource(args.dataset_source),
                converted_dataset_dir=(
                    None
                    if args.converted_dataset_dir is None
                    else Path(args.converted_dataset_dir)
                ),
                valid_fraction=args.valid_fraction,
                every_nth_frame=args.every_nth_frame,
                max_sequences=args.max_sequences,
                symlink_images=not args.copy_images,
            )
        )
        return 0
    if args.command == "validate" and args.target == "detector":
        _detection_model(args.backend, args.weights).validate(args.data)
        return 0
    if args.command == "experiment":
        experiment_config = load_experiment_config(args.config)
        if args.kind == "detection-tracking":
            DetectionTrackingExperimentRunner().run(experiment_config)
        elif args.kind == "team-classification":
            TeamClassificationExperimentRunner().run(experiment_config)
        elif args.kind == "video-xg":
            VideoXgExperimentRunner().run(experiment_config)
        else:
            HomographyComparisonRunner().run(experiment_config)
        return 0
    if args.command == "evaluate" and args.target == "tracking":
        if args.pred and args.gt:
            metrics = evaluate_tracking_files(
                Path(args.pred),
                Path(args.gt),
                iou_threshold=args.iou_threshold,
                prediction_frame_offset=args.pred_frame_offset,
            )
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            else:
                print(json.dumps(metrics, sort_keys=True))
            return 0
        _ = load_pipeline_config(args.config)
        return 0
    if args.command == "video-xg" and args.action == "from-features":
        estimator = build_video_only_estimator(XgModelKind(args.model))
        _, metrics, _ = VideoOnlyXgRunner(estimator=estimator).run(
            Path(args.features),
            output_dir=Path(args.output_dir),
            reference_path=Path(args.reference) if args.reference else None,
            group_id=args.group_id,
        )
        print(json.dumps(metrics, sort_keys=True))
        return 0
    if args.command == "video-xg" and args.action == "compare-methods":
        summary, _ = run_video_only_xg_experiment(
            features_path=Path(args.features),
            output_dir=Path(args.output_dir),
            reference_path=Path(args.reference) if args.reference else None,
            group_id=args.group_id,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "video-xg" and args.action == "end-to-end":
        e2e_config = load_video_only_xg_end_to_end_config(args.config)
        e2e_result = VideoOnlyXgEndToEndRunner().run(
            e2e_config,
            resume_from=args.resume_from,
            stop_after=args.stop_after,
            force_stage=args.force_stage,
        )
        print(
            json.dumps(
                {
                    "artifacts": len(e2e_result.artifacts),
                    "metrics": e2e_result.metrics,
                    "output_dir": str(e2e_result.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "video-xg" and args.action == "ablate":
        ablation_config = load_video_only_xg_end_to_end_config(args.config)
        ablation_result = VideoXgWeaknessAblationRunner().run(ablation_config)
        print(
            json.dumps(
                {
                    "artifacts": len(ablation_result.artifacts),
                    "metrics": ablation_result.metrics,
                    "output_dir": str(ablation_result.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "video-xg" and args.action == "benchmark-detection":
        benchmark_config = load_video_only_xg_end_to_end_config(args.config)
        baseline_dir = (
            benchmark_config.ablation.baseline_run_dir or benchmark_config.output_dir
        )
        sampled = read_dataframe_artifact(baseline_dir / "01_sampled_frames.parquet")
        output_dir = benchmark_config.ablation.output_dir or benchmark_config.output_dir
        benchmark_result = DetectionBenchmarkRunner().run(
            benchmark_config,
            sampled,
            output_dir,
        )
        print(json.dumps(benchmark_result.to_dict("records"), sort_keys=True))
        return 0
    return 1


def _detection_model(backend: str, weights: str) -> TrainableDetectionModel:
    if DetectionBackend(backend) == DetectionBackend.YOLO:
        return YOLODetectionModel.from_weights(weights)
    return RFDETRDetectionModel.from_weights(weights)


def _require_input(path: Path | None) -> Path:
    if path is None:
        raise ValueError("paths.input is required")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
