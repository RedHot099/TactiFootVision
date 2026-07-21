from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config.synloc_loaders import load_synloc_config
from config.synloc_models import SynLocConfig
from tactifoot_vision.synloc.data import SynLocSplitData, load_synloc_split
from tactifoot_vision.synloc.eval import evaluate_predictions
from tactifoot_vision.synloc.inference import InferenceRunResult, run_inference_on_split_with_diagnostics
from tactifoot_vision.synloc.plan24h import (
    ExperimentRecord,
    FinalistSummary,
    PhaseRunSelection,
    finalize_24h_run,
)
from tactifoot_vision.synloc.point_regressor import load_point_regressor
from tactifoot_vision.synloc.submission import build_submission_archive, serialize_predictions, write_submission_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONEDRIVE_DIR = Path("/home/kuba/OneDrive/Uczelnia/Konferencje/synloc_24h_run/")


@dataclass(frozen=True)
class ExecutedRun:
    selection: PhaseRunSelection
    experiment: ExperimentRecord
    diagnostics: dict[str, object]
    non_empty_images: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the SynLoc 24h ranking-first sweep and package the winning submission.")
    parser.add_argument("--val-config", type=Path, default=PROJECT_ROOT / "run_config/synloc_inference_val_rfdetr_fullhd_pretrained.yaml")
    parser.add_argument("--test-config", type=Path, default=PROJECT_ROOT / "run_config/synloc_inference_test_rfdetr_fullhd_pretrained.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "run_config/synloc_detector_rfdetr_fullhd_person.yaml")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "results/synloc/24h_run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ONEDRIVE_DIR)
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    started_at = _utc_timestamp()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    val_config = load_synloc_config(args.val_config)
    test_config = load_synloc_config(args.test_config)
    split_data = load_synloc_split(val_config.dataset)
    sample_split = _subset_split(split_data, args.sample_limit)

    experiments: list[ExperimentRecord] = []
    finalists: list[FinalistSummary] = []

    phase1_runs: list[ExecutedRun] = []
    for idx, variant in enumerate(_phase1_variants(val_config), start=1):
        phase1_runs.append(
            _execute_inference_run(
                config=variant,
                split_data=sample_split,
                phase="phase1",
                run_name=f"phase1_rescue_{idx:03d}",
                run_dir=work_dir / "phase1" / f"rescue_{idx:03d}",
                archive=False,
            )
        )
    experiments.extend(run.experiment for run in phase1_runs)

    top_phase1 = sorted(
        phase1_runs,
        key=lambda item: (
            item.selection.metrics.get("map_locsim", 0.0),
            item.selection.metrics.get("recall", 0.0),
            item.non_empty_images,
        ),
        reverse=True,
    )[:3]
    if not top_phase1:
        raise RuntimeError("Phase 1 produced no runs.")

    full_val_runs: list[ExecutedRun] = []
    for idx, run in enumerate(top_phase1, start=1):
        full_val_runs.append(
            _execute_inference_run(
                config=_clone_config(run.selection.config_snapshot),
                split_data=split_data,
                phase="phase2",
                run_name=f"phase2_full_val_{idx:02d}",
                run_dir=work_dir / "phase2" / f"full_val_{idx:02d}",
                archive=False,
            )
        )
    experiments.extend(run.experiment for run in full_val_runs)

    pretrain_baseline = max(full_val_runs, key=lambda item: item.selection.metrics.get("map_locsim", 0.0))
    upgrade_runs: list[ExecutedRun] = []
    for idx, variant in enumerate(_phase2_upgrade_variants(_clone_config(pretrain_baseline.selection.config_snapshot)), start=1):
        upgrade = _execute_inference_run(
            config=variant,
            split_data=split_data,
            phase="phase2",
            run_name=f"phase2_upgrade_{idx:02d}",
            run_dir=work_dir / "phase2" / f"upgrade_{idx:02d}",
            archive=False,
        )
        baseline_map = pretrain_baseline.selection.metrics.get("map_locsim", 0.0)
        if (
            upgrade.selection.metrics.get("map_locsim", 0.0) >= baseline_map + 0.005
            or upgrade.selection.metrics.get("recall", 0.0) > pretrain_baseline.selection.metrics.get("recall", 0.0)
        ):
            upgrade_runs.append(upgrade)
            experiments.append(upgrade.experiment)

    finalists_candidates = [pretrain_baseline, *upgrade_runs]

    if not args.skip_training:
        training_runs = _run_training_qualification(
            training_config_path=args.training_config,
            base_config=_clone_config(pretrain_baseline.selection.config_snapshot),
            split_data=split_data,
            work_dir=work_dir / "phase3",
        )
        experiments.extend(run.experiment for run in training_runs)
        if training_runs:
            best_training = max(training_runs, key=lambda item: item.selection.metrics.get("map_locsim", 0.0))
            if best_training.selection.metrics.get("map_locsim", 0.0) > pretrain_baseline.selection.metrics.get("map_locsim", 0.0):
                finalists_candidates.append(best_training)

    finalists_candidates = sorted(
        finalists_candidates,
        key=lambda item: item.selection.metrics.get("map_locsim", 0.0),
        reverse=True,
    )
    winner = finalists_candidates[0]
    for run in finalists_candidates:
        metrics = run.selection.metrics
        finalists.append(
            FinalistSummary(
                run_name=run.selection.run_name,
                phase=run.experiment.phase,
                map_locsim=metrics.get("map_locsim", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1=metrics.get("f1", 0.0),
                score_threshold=metrics.get("score_threshold", 0.5),
                final_predictions=run.non_empty_images,
                archive_path=run.selection.archive_path if run.selection.archive_path.exists() else run.selection.run_dir / "placeholder.zip",
            )
        )

    winning_test_config = _apply_val_variant_to_test(
        test_config=test_config,
        val_config_snapshot=winner.selection.config_snapshot,
    )
    test_run = _execute_inference_run(
        config=winning_test_config,
        split_data=load_synloc_split(winning_test_config.dataset),
        phase="phase5",
        run_name="phase5_test_submission",
        run_dir=work_dir / "phase5" / "test_submission",
        archive=True,
    )

    best_selection = PhaseRunSelection(
        run_name=winner.selection.run_name,
        run_dir=test_run.selection.run_dir,
        archive_path=test_run.selection.archive_path,
        results_path=test_run.selection.results_path,
        metadata_path=test_run.selection.metadata_path,
        raw_results_path=test_run.selection.raw_results_path,
        metrics=winner.selection.metrics,
        config_snapshot=winner.selection.config_snapshot,
        notes=winner.selection.notes,
    )
    artifacts = finalize_24h_run(
        output_dir=args.output_dir,
        best_run=best_selection,
        finalists=finalists,
        experiments=experiments,
        run_started_at=started_at,
        run_finished_at=_utc_timestamp(),
    )

    print(f"best_submission={artifacts.best_submission_zip}")
    print(f"report={artifacts.run_report_md}")


def _phase1_variants(base_config: SynLocConfig) -> Iterable[SynLocConfig]:
    point_strategies = ["bottom_center"]
    if base_config.projection.point_regressor_checkpoint is not None:
        point_strategies.append("learned_offset")

    for confidence_threshold in (0.05, 0.10, 0.15, 0.20):
        for world_nms_radius_m in (0.0, 0.25, 0.5, 0.75):
            for image_nms_iou in (0.5, 0.6, 0.7):
                for behind_camera_policy in ("drop", "clip"):
                    for clip_margin_m in (0.0, 0.5, 1.0):
                        for point_strategy in point_strategies:
                            for max_detections in (250, 500):
                                config = _clone_config(base_config)
                                config.detector.confidence_threshold = confidence_threshold
                                config.detector.max_detections = max_detections
                                config.projection.world_nms_radius_m = world_nms_radius_m
                                config.projection.image_nms_iou = image_nms_iou
                                config.projection.behind_camera_policy = behind_camera_policy
                                config.projection.clip_margin_m = clip_margin_m
                                config.projection.point_strategy = point_strategy
                                config.submission.topk_per_image = max_detections
                                yield config


def _phase2_upgrade_variants(base_config: SynLocConfig) -> Iterable[SynLocConfig]:
    variants: list[tuple[bool, list[int], int, int]] = [
        (False, [], base_config.dataset.tile_overlap, 250),
        (True, [base_config.detector.inference_imgsz, max(base_config.detector.inference_imgsz, 1280)], base_config.dataset.tile_overlap, 250),
        (True, [], max(base_config.dataset.tile_overlap, 384), 400),
        (True, [], max(base_config.dataset.tile_overlap, 512), 600),
    ]
    for use_tiles, tta_scales, overlap, topk in variants:
        config = _clone_config(base_config)
        config.dataset.use_tiles = use_tiles
        config.dataset.tile_overlap = overlap
        config.detector.tile_overlap = overlap
        config.detector.tta_scales = tta_scales
        config.detector.max_detections = topk
        config.submission.topk_per_image = topk
        yield config


def _run_training_qualification(
    *,
    training_config_path: Path,
    base_config: SynLocConfig,
    split_data: SynLocSplitData,
    work_dir: Path,
) -> list[ExecutedRun]:
    work_dir.mkdir(parents=True, exist_ok=True)
    runs: list[ExecutedRun] = []
    for epochs in (1, 3):
        run_name = f"qualification_epochs_{epochs}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_synloc_detector.py"),
            "--config",
            str(training_config_path),
            "--epochs",
            str(epochs),
            "--run-name",
            run_name,
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        checkpoint_path = PROJECT_ROOT / "results" / "synloc" / "training" / run_name / "checkpoint_best_total.pth"
        if not checkpoint_path.is_file():
            continue
        config = _clone_config(base_config)
        config.detector.checkpoint_path = checkpoint_path
        config.detector.base_model = None
        runs.append(
            _execute_inference_run(
                config=config,
                split_data=split_data,
                phase="phase3",
                run_name=run_name,
                run_dir=work_dir / run_name,
                archive=False,
                notes=f"qualification training {epochs} epochs",
            )
        )
    return runs


def _execute_inference_run(
    *,
    config: SynLocConfig,
    split_data: SynLocSplitData,
    phase: str,
    run_name: str,
    run_dir: Path,
    archive: bool,
    notes: str = "",
) -> ExecutedRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _clone_config(config)
    config.submission.output_dir = run_dir
    config.results_path = run_dir / "raw_results.json"
    point_regressor = None
    if config.point_regressor_checkpoint is not None and config.projection.point_strategy == "learned_offset":
        point_regressor = load_point_regressor(config.point_regressor_checkpoint)
    run_result: InferenceRunResult = run_inference_on_split_with_diagnostics(
        split_data,
        dataset_config=config.dataset,
        detector_config=config.detector,
        projection_config=config.projection,
        model_dir=config.model_dir,
        point_regressor=point_regressor,
    )
    results_path, metadata_path = write_submission_files(run_result.predictions, config.submission)
    raw_results_path = config.results_path or (run_dir / "raw_results.json")
    raw_results_path.write_text(json.dumps(serialize_predictions(run_result.predictions), indent=2), encoding="utf-8")
    diagnostics_path = run_dir / "debug_summary.json"
    diagnostics_path.write_text(json.dumps(run_result.summary.to_dict(), indent=2), encoding="utf-8")

    metrics = {"map_locsim": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "score_threshold": config.submission.score_threshold}
    if split_data.split in {"val", "valid"}:
        metrics = evaluate_predictions(
            annotation_path=split_data.annotation_path,
            predictions=run_result.predictions,
        )
        (run_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    archive_path = run_dir / (config.submission.zip_name or f"{run_name}.zip")
    if archive:
        archive_path = build_submission_archive(run_result.predictions, config.submission)

    selection = PhaseRunSelection(
        run_name=run_name,
        run_dir=run_dir,
        archive_path=archive_path,
        results_path=results_path,
        metadata_path=metadata_path,
        raw_results_path=raw_results_path,
        metrics=metrics,
        config_snapshot=_config_to_payload(config),
        notes=notes,
    )
    experiment = ExperimentRecord(
        timestamp=_utc_timestamp(),
        phase=phase,
        run_name=run_name,
        detector_checkpoint=str(config.detector.checkpoint_path or config.detector.base_model or ""),
        point_strategy=config.projection.point_strategy,
        confidence_threshold=float(config.detector.confidence_threshold),
        image_nms_iou=float(config.projection.image_nms_iou),
        world_nms_radius_m=float(config.projection.world_nms_radius_m),
        behind_camera_policy=config.projection.behind_camera_policy,
        clip_margin_m=float(config.projection.clip_margin_m),
        tile_size=int(config.dataset.tile_size),
        tile_overlap=int(config.dataset.tile_overlap),
        topk_per_image=int(config.submission.topk_per_image or config.detector.max_detections),
        aux_data_used=",".join(config.dataset.auxiliary_tasks),
        val_map_locsim=float(metrics.get("map_locsim", 0.0)),
        val_recall=float(metrics.get("recall", 0.0)),
        val_precision=float(metrics.get("precision", 0.0)),
        val_f1=float(metrics.get("f1", 0.0)),
        notes=notes,
    )
    return ExecutedRun(
        selection=selection,
        experiment=experiment,
        diagnostics=run_result.summary.to_dict(),
        non_empty_images=int(run_result.summary.aggregate.non_empty_images),
    )


def _apply_val_variant_to_test(*, test_config: SynLocConfig, val_config_snapshot: dict[str, Any]) -> SynLocConfig:
    config = _clone_config(test_config)
    for section_name in ("detector", "projection", "submission", "model_dir"):
        if section_name in val_config_snapshot:
            value = val_config_snapshot[section_name]
            if section_name == "model_dir":
                config.model_dir = Path(value)
            else:
                for key, section_value in value.items():
                    setattr(getattr(config, section_name), key, section_value)
    config.dataset.use_tiles = bool(val_config_snapshot["dataset"]["use_tiles"])
    config.dataset.tile_size = int(val_config_snapshot["dataset"]["tile_size"])
    config.dataset.tile_overlap = int(val_config_snapshot["dataset"]["tile_overlap"])
    return config


def _subset_split(split_data: SynLocSplitData, limit: int) -> SynLocSplitData:
    if limit <= 0 or limit >= len(split_data.images):
        return split_data
    selected_images = split_data.images[:limit]
    selected_ids = {image.image_id for image in selected_images}
    return SynLocSplitData(
        dataset_root=split_data.dataset_root,
        split=split_data.split,
        annotation_path=split_data.annotation_path,
        images=selected_images,
        images_by_id={image.image_id: image for image in selected_images},
        annotations=[annotation for annotation in split_data.annotations if annotation.image_id in selected_ids],
        annotations_by_image={
            image_id: annotations
            for image_id, annotations in split_data.annotations_by_image.items()
            if image_id in selected_ids
        },
        categories=split_data.categories,
    )


def _clone_config(config: SynLocConfig | dict[str, Any]) -> SynLocConfig:
    if isinstance(config, SynLocConfig):
        payload = _config_to_payload(config)
    else:
        payload = json.loads(json.dumps(config))
    return SynLocConfig.model_validate(payload)


def _config_to_payload(config: SynLocConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
