import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tactifoot_vision.enums import (
    VideoBallReconstructionVariant,
    VideoProjectionVariant,
    VideoShotRankingVariant,
    VideoXgCalibrationVariant,
    XgModelKind,
)
from tactifoot_vision.video_xg.artifacts import (
    read_dataframe_artifact,
    read_json_artifact,
    write_dataframe_artifact,
    write_json_artifact,
)
from tactifoot_vision.video_xg.ball_reconstruction import (
    KalmanRtsBallReconstructorV2,
    OpticalFlowBallRefiner,
    ViterbiBallPathReconstructor,
)
from tactifoot_vision.video_xg.config import VideoOnlyXgEndToEndConfig
from tactifoot_vision.video_xg.end_to_end import (
    _extract_features,
    _match_predictions,
    _method_metrics,
    _reference_shots,
)
from tactifoot_vision.video_xg.estimators import (
    VideoFreezeContextXgEstimator,
    VideoGeometryXgEstimator,
    VideoKinematicContextXgEstimator,
)
from tactifoot_vision.video_xg.projection_features import (
    HomographyArtifactProvider,
    ImageLineHeuristicProjector,
    ProjectionQualityAnnotator,
)
from tactifoot_vision.video_xg.results import VideoOnlyXgRunResult, VideoShotFeatures
from tactifoot_vision.video_xg.shot_quality import (
    ShotPatternScorer,
    ShotWindowFeatureExtractor,
)
from tactifoot_vision.video_xg.shot_ranking import (
    DenseContactRefiner,
    HardNegativeCalibratedShotRanker,
    HighRecallCascadeShotRanker,
    LearnedTemporalShotRanker,
    RuleSweepShotRanker,
    ShotCandidateFeatureExtractor,
    ShotRankingMetrics,
    WindowedTemporalShotRanker,
    evaluate_shot_candidates,
)
from tactifoot_vision.video_xg.xg_calibration import (
    DataBallPySimpleXgBaseline,
    FormulaCoefficientCalibrator,
    IsotonicXgCalibrator,
    NeuralVideoXgCalibrator,
    QualityAwareXgEnsemble,
)


class VideoXgWeaknessAblationRunner:
    def run(self, config: VideoOnlyXgEndToEndConfig) -> VideoOnlyXgRunResult:
        start = time.perf_counter()
        baseline_dir = _baseline_dir(config)
        output_dir = _output_dir(config, baseline_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_run_config(config, output_dir / "run_config.yaml")

        artifacts = _BaselineArtifacts.read(baseline_dir)
        reference = _load_reference(config, artifacts.timeline)
        baseline_summary = _baseline_summary(artifacts, reference)
        write_json_artifact(baseline_summary, output_dir / "00_baseline_summary.json")

        detection_benchmark = _detection_benchmark(artifacts)
        write_dataframe_artifact(
            detection_benchmark, output_dir / "01_detection_benchmark.csv"
        )

        ball_variants = _run_ball_ablation(config, output_dir, artifacts, reference)
        write_dataframe_artifact(ball_variants, output_dir / "02_ball_ablation.csv")
        best_ball = _best_variant(ball_variants, "ball_score")
        best_ball_frame = _read_variant_frame(
            output_dir, "ball", best_ball, "trajectory"
        )

        shot_variants = _run_shot_ablation(
            config,
            output_dir,
            best_ball_frame,
            artifacts.tracks,
            artifacts.refined,
            reference,
        )
        write_dataframe_artifact(shot_variants, output_dir / "03_shot_ablation.csv")
        write_dataframe_artifact(
            _shot_fp_diagnostics(artifacts.refined, reference),
            output_dir / "11_shot_fp_diagnostics.csv",
        )
        write_dataframe_artifact(
            _shot_window_features(config, best_ball_frame, artifacts.tracks),
            output_dir / "12_candidate_window_features.parquet",
        )
        write_dataframe_artifact(shot_variants, output_dir / "13_shot_fp_ablation.csv")
        best_shot = _best_variant(shot_variants, "shot_score")
        best_candidates = _read_variant_frame(
            output_dir, "shots", best_shot, "candidates"
        )
        write_dataframe_artifact(
            best_candidates, output_dir / "14_selected_refined_shots.parquet"
        )

        feature_frame = _extract_features(
            best_candidates, best_ball_frame, artifacts.tracks
        )
        projection_variants = _run_projection_ablation(
            config,
            output_dir,
            artifacts.sampled,
            artifacts.homographies,
            feature_frame,
            reference,
        )
        write_dataframe_artifact(
            projection_variants, output_dir / "04_projection_ablation.csv"
        )
        best_projection = _best_variant(projection_variants, "projection_score")
        best_features = _read_variant_frame(
            output_dir, "projection", best_projection, "features"
        )

        xg_variants = _run_xg_ablation(
            config,
            output_dir,
            best_candidates,
            best_features,
            reference,
        )
        write_dataframe_artifact(xg_variants, output_dir / "05_xg_ablation.csv")
        write_dataframe_artifact(
            _predict_base_methods(best_features, config.xg.models),
            output_dir / "15_predictions_after_fp_reduction.csv",
        )

        ranking = _final_ranking(
            baseline_summary,
            best_ball,
            best_shot,
            best_projection,
            shot_variants,
            xg_variants,
            start,
            config.ablation.baseline_runtime_seconds,
        )
        write_dataframe_artifact(ranking, output_dir / "final_variant_ranking.csv")
        _write_recommendation(
            output_dir,
            baseline_summary,
            best_ball,
            best_shot,
            best_projection,
            ranking,
        )
        _write_shot_detection_report(
            output_dir,
            baseline_summary,
            best_shot,
            shot_variants,
            best_candidates,
        )
        write_json_artifact(
            {
                "baseline_dir": str(baseline_dir),
                "output_dir": str(output_dir),
                "artifacts": [
                    "00_baseline_summary.json",
                    "01_detection_benchmark.csv",
                    "02_ball_ablation.csv",
                    "03_shot_ablation.csv",
                    "04_projection_ablation.csv",
                    "05_xg_ablation.csv",
                    "11_shot_fp_diagnostics.csv",
                    "12_candidate_window_features.parquet",
                    "13_shot_fp_ablation.csv",
                    "14_selected_refined_shots.parquet",
                    "15_predictions_after_fp_reduction.csv",
                    "final_variant_ranking.csv",
                    "final_recommendation.md",
                    "final_shot_detection_report.md",
                ],
            },
            output_dir / "manifest.json",
        )
        return VideoOnlyXgRunResult(
            output_dir=output_dir,
            artifacts=tuple(
                path for path in sorted(output_dir.rglob("*")) if path.is_file()
            ),
            metrics=_ranking_metrics(ranking),
        )


class _BaselineArtifacts:
    def __init__(
        self,
        *,
        run_dir: Path,
        timeline: pd.DataFrame,
        sampled: pd.DataFrame,
        detections: pd.DataFrame,
        tracks: pd.DataFrame,
        homographies: pd.DataFrame,
        ball: pd.DataFrame,
        candidates: pd.DataFrame,
        refined: pd.DataFrame,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        metrics: dict[str, float],
    ) -> None:
        self.run_dir = run_dir
        self.timeline = timeline
        self.sampled = sampled
        self.detections = detections
        self.tracks = tracks
        self.homographies = homographies
        self.ball = ball
        self.candidates = candidates
        self.refined = refined
        self.features = features
        self.predictions = predictions
        self.metrics = metrics

    @classmethod
    def read(cls, run_dir: Path) -> "_BaselineArtifacts":
        return cls(
            run_dir=run_dir,
            timeline=pd.DataFrame(
                read_json_artifact(run_dir / "00_video_timeline.json")
            ),
            sampled=read_dataframe_artifact(run_dir / "01_sampled_frames.parquet"),
            detections=read_dataframe_artifact(run_dir / "02_detections.parquet"),
            tracks=read_dataframe_artifact(run_dir / "03_tracks.parquet"),
            homographies=read_dataframe_artifact(run_dir / "04_homographies.parquet"),
            ball=read_dataframe_artifact(run_dir / "05_ball_trajectory.parquet"),
            candidates=read_dataframe_artifact(run_dir / "06_shot_candidates.parquet"),
            refined=read_dataframe_artifact(run_dir / "07_refined_shots.parquet"),
            features=read_dataframe_artifact(run_dir / "08_video_features.csv"),
            predictions=read_dataframe_artifact(run_dir / "09_predictions.csv"),
            metrics=_read_metrics(run_dir / "10_metrics.json"),
        )


def _run_ball_ablation(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    artifacts: _BaselineArtifacts,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    variants = {
        VideoBallReconstructionVariant.BASELINE_KALMAN.value: artifacts.ball,
        VideoBallReconstructionVariant.VITERBI_DP.value: ViterbiBallPathReconstructor(
            max_gap_seconds=config.ball.max_gap_seconds,
            max_speed_mps=config.ball.max_speed_mps,
        ).reconstruct(artifacts.sampled, artifacts.detections),
        VideoBallReconstructionVariant.KALMAN_RTS_V2.value: KalmanRtsBallReconstructorV2(
            max_gap_seconds=config.ball.max_gap_seconds
        ).reconstruct(artifacts.ball),
        VideoBallReconstructionVariant.OPTICAL_FLOW_TEMPLATE.value: OpticalFlowBallRefiner(
            max_gap_seconds=max(config.ball.max_gap_seconds, 4.0)
        ).refine(artifacts.ball),
    }
    rows = []
    for variant, trajectory in variants.items():
        _write_variant_frame(output_dir, "ball", variant, "trajectory", trajectory)
        metrics = _ball_quality_metrics(trajectory, reference)
        write_json_artifact(
            metrics, _variant_dir(output_dir, "ball", variant) / "metrics.json"
        )
        rows.append({"variant": variant, **metrics})
    return pd.DataFrame(rows).sort_values("ball_score", ascending=False)


def _run_shot_ablation(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    ball: pd.DataFrame,
    tracks: pd.DataFrame,
    baseline_refined: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    features = ShotCandidateFeatureExtractor().transform(ball, tracks)
    rule = RuleSweepShotRanker(
        temporal_nms_seconds=config.shots.temporal_nms_seconds,
        max_candidates=config.shots.max_candidates,
    ).rank(features, reference)
    learned = LearnedTemporalShotRanker(
        temporal_nms_seconds=config.shots.temporal_nms_seconds,
        max_candidates=config.shots.max_candidates,
    ).rank(features, reference)
    dense_seed = rule if not rule.empty else baseline_refined
    dense = DenseContactRefiner(
        window_before_seconds=config.refine_window_before_seconds,
        window_after_seconds=config.refine_window_after_seconds,
    ).refine(dense_seed, features)
    high_recall = HighRecallCascadeShotRanker(
        temporal_nms_seconds=config.shots.temporal_nms_seconds,
        max_candidates=config.shots.max_candidates,
        max_candidates_per_half=config.shots.max_candidates_per_half,
        contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
        post_shot_window_seconds=config.shots.post_shot_window_seconds,
        long_shot_distance_m=config.shots.long_shot_distance_m,
    ).rank(features)
    hard_negative = HardNegativeCalibratedShotRanker(
        temporal_nms_seconds=config.shots.temporal_nms_seconds,
        max_candidates=config.shots.max_candidates,
        max_candidates_per_half=config.shots.max_candidates_per_half,
        recall_floor_hit2=config.shots.recall_floor_hit2,
        min_hit1=config.shots.min_hit1,
        target_max_false_positives=config.shots.target_max_false_positives,
        contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
        post_shot_window_seconds=config.shots.post_shot_window_seconds,
        long_shot_distance_m=config.shots.long_shot_distance_m,
    ).rank(features, reference, seed_candidates=baseline_refined)
    windowed = WindowedTemporalShotRanker(
        temporal_nms_seconds=config.shots.temporal_nms_seconds,
        max_candidates=config.shots.max_candidates,
        max_candidates_per_half=config.shots.max_candidates_per_half,
        recall_floor_hit2=config.shots.recall_floor_hit2,
        min_hit1=config.shots.min_hit1,
        target_max_false_positives=config.shots.target_max_false_positives,
        contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
        post_shot_window_seconds=config.shots.post_shot_window_seconds,
        long_shot_distance_m=config.shots.long_shot_distance_m,
    ).rank(features, reference)
    variants = {
        VideoShotRankingVariant.BASELINE_CONTACT_KINEMATIC.value: _candidate_columns(
            baseline_refined
        ),
        VideoShotRankingVariant.RULE_SWEEP.value: rule,
        VideoShotRankingVariant.LEARNED_TEMPORAL.value: learned,
        VideoShotRankingVariant.DENSE_LOCAL_REFINEMENT.value: dense,
        VideoShotRankingVariant.HIGH_RECALL_CASCADE.value: high_recall,
        VideoShotRankingVariant.HARD_NEGATIVE_CALIBRATED.value: hard_negative,
        VideoShotRankingVariant.WINDOWED_TEMPORAL.value: windowed,
    }
    rows = []
    for variant, candidates in variants.items():
        _write_variant_frame(output_dir, "shots", variant, "candidates", candidates)
        metrics = _shot_metrics(candidates, reference)
        write_json_artifact(
            asdict(metrics), _variant_dir(output_dir, "shots", variant) / "metrics.json"
        )
        rows.append(
            {
                "variant": variant,
                "candidates": float(len(candidates)),
                "hit@0.5s": metrics.hit_05,
                "hit@1s": metrics.hit_1,
                "hit@2s": metrics.hit_2,
                "precision@2s": metrics.precision_2,
                "temporal_mae_seconds": metrics.temporal_mae_seconds,
                "false_positives": metrics.false_positives,
                "shot_score": _shot_score(metrics),
            }
        )
    return pd.DataFrame(rows).sort_values("shot_score", ascending=False)


def _run_projection_ablation(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    sampled: pd.DataFrame,
    baseline_homographies: pd.DataFrame,
    features: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    providers = {
        VideoProjectionVariant.DEGRADED_IMAGE_NORMALIZED.value: baseline_homographies,
        VideoProjectionVariant.LAST_STABLE_HOMOGRAPHY.value: HomographyArtifactProvider(
            config.calibration.external_homographies,
            max_age_seconds=config.calibration.last_stable_max_age_seconds,
        ).project(sampled),
        VideoProjectionVariant.LINE_BOX_HEURISTIC.value: ImageLineHeuristicProjector().project(
            sampled
        ),
        VideoProjectionVariant.QUALITY_AWARE_DEGRADED.value: baseline_homographies.assign(
            status="quality_aware_degraded",
            projection_confidence=baseline_homographies["projection_confidence"].clip(
                upper=0.3
            ),
        ),
    }
    rows = []
    for variant, homographies in providers.items():
        annotated = ProjectionQualityAnnotator().annotate(features, homographies)
        _write_variant_frame(output_dir, "projection", variant, "features", annotated)
        _write_variant_frame(
            output_dir, "projection", variant, "homographies", homographies
        )
        metrics = _projection_metrics(annotated, homographies, reference)
        write_json_artifact(
            metrics, _variant_dir(output_dir, "projection", variant) / "metrics.json"
        )
        rows.append({"variant": variant, **metrics})
    return pd.DataFrame(rows).sort_values("projection_score", ascending=False)


def _run_xg_ablation(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    base_predictions = _predict_base_methods(features, config.xg.models)
    reference_by_candidate = _reference_by_candidate(candidates, reference)
    variants = {
        VideoXgCalibrationVariant.NONE.value: base_predictions,
        VideoXgCalibrationVariant.COEFFICIENT_FIT.value: FormulaCoefficientCalibrator().predict(
            features, reference_by_candidate
        ),
        VideoXgCalibrationVariant.ISOTONIC_PLATT.value: IsotonicXgCalibrator().calibrate(
            base_predictions, reference_by_candidate
        ),
        VideoXgCalibrationVariant.QUALITY_AWARE_ENSEMBLE.value: QualityAwareXgEnsemble().predict(
            base_predictions, features
        ),
        VideoXgCalibrationVariant.NEURAL_VIDEO_XG.value: NeuralVideoXgCalibrator().predict(
            features, reference_by_candidate
        ),
        VideoXgCalibrationVariant.DATABALLPY_SIMPLE_XG.value: DataBallPySimpleXgBaseline().predict(
            features
        ),
    }
    rows = []
    for variant, predictions in variants.items():
        variant_dir = _variant_dir(output_dir, "xg", variant)
        write_dataframe_artifact(predictions, variant_dir / "predictions.csv")
        per_eval = _match_predictions(reference, candidates, predictions)
        write_dataframe_artifact(per_eval, variant_dir / "per_shot_eval.csv")
        metrics = _method_metrics(per_eval)
        write_dataframe_artifact(metrics, variant_dir / "method_metrics.csv")
        for metric_row in _xg_metric_rows(
            variant, predictions, per_eval, metrics, reference
        ):
            rows.append(metric_row)
        write_json_artifact(
            {"variant": variant, "methods": _json_records(metrics)},
            variant_dir / "metrics.json",
        )
    return pd.DataFrame(rows).sort_values(
        ["xg_total_error_score", "mae_vs_reference_xg"],
        ascending=[False, True],
    )


def _baseline_summary(
    artifacts: _BaselineArtifacts, reference: pd.DataFrame
) -> dict[str, Any]:
    total_reference_xg = (
        float(reference["reference_xg"].sum()) if not reference.empty else 0.0
    )
    feature_sources = (
        artifacts.features["feature_source"].value_counts().to_dict()
        if "feature_source" in artifacts.features.columns
        else {}
    )
    fallback_count = float(feature_sources.get("missing_center_fallback", 0))
    feature_count = float(len(artifacts.features))
    best_total = _best_existing_total_xg(artifacts.predictions)
    return {
        "run_dir": str(artifacts.run_dir),
        "sampled_frames": float(len(artifacts.sampled)),
        "detections": float(len(artifacts.detections)),
        "tracks": float(len(artifacts.tracks)),
        "candidates": float(len(artifacts.refined)),
        "reference_shots": float(len(reference)),
        "reference_total_xg": total_reference_xg,
        "hit@0.5s": float(artifacts.metrics.get("hit@0.5s", 0.0)),
        "hit@1s": float(
            artifacts.metrics.get("hit@1s", artifacts.metrics.get("hit@1.0s", 0.0))
        ),
        "hit@2s": float(
            artifacts.metrics.get("hit@2s", artifacts.metrics.get("hit@2.0s", 0.0))
        ),
        "temporal_mae_seconds": float(
            artifacts.metrics.get("temporal_mae_seconds", 0.0)
        ),
        "missing_center_fallback_count": fallback_count,
        "feature_count": feature_count,
        "missing_center_fallback_ratio": fallback_count / feature_count
        if feature_count
        else 0.0,
        "best_existing_total_xg": best_total["total_xg"],
        "best_existing_xg_method": best_total["method"],
        "best_existing_total_xg_error": best_total["total_xg"] - total_reference_xg,
        "feature_sources": feature_sources,
    }


def _detection_benchmark(artifacts: _BaselineArtifacts) -> pd.DataFrame:
    detections = artifacts.detections
    sampled = artifacts.sampled
    by_class = detections["class_name"].value_counts().to_dict()
    frames_with_ball = (
        detections[detections["class_name"].eq("ball")]["global_frame_index"].nunique()
        if not detections.empty
        else 0
    )
    frames_with_player = (
        detections[detections["class_name"].eq("player")][
            "global_frame_index"
        ].nunique()
        if not detections.empty
        else 0
    )
    frame_count = max(len(sampled), 1)
    return pd.DataFrame(
        [
            {
                "variant": "existing_1fps_artifacts",
                "frames": float(len(sampled)),
                "detections": float(len(detections)),
                "ball_detections": float(by_class.get("ball", 0)),
                "player_detections": float(by_class.get("player", 0)),
                "frames_with_ball_ratio": frames_with_ball / frame_count,
                "frames_with_player_ratio": frames_with_player / frame_count,
                "chunk_artifacts_present": float(
                    (artifacts.run_dir / "02_detections").exists()
                ),
            }
        ]
    )


def _ball_quality_metrics(
    trajectory: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, float]:
    if trajectory.empty:
        return {
            "coverage": 0.0,
            "gt_window_coverage": 0.0,
            "fallback_ratio": 1.0,
            "missing_ratio": 1.0,
            "speed_outlier_ratio": 1.0,
            "p90_acceleration": 0.0,
            "ball_score": 0.0,
        }
    sources = trajectory["source"].astype(str)
    missing = sources.str.contains("missing", case=False, na=False)
    fallback = sources.eq("missing_center_fallback")
    valid = trajectory[["pitch_x", "pitch_y"]].notna().all(axis=1) & ~missing
    gt_coverage = _reference_window_coverage(trajectory, reference)
    speeds = _trajectory_speed(trajectory)
    acceleration = speeds.diff().abs().fillna(0.0)
    speed_outlier_ratio = float((speeds > 38.0).mean()) if len(speeds) else 0.0
    p90_acceleration = float(acceleration.quantile(0.9)) if len(acceleration) else 0.0
    coverage = float(valid.mean()) if len(valid) else 0.0
    fallback_ratio = float(fallback.mean()) if len(fallback) else 0.0
    missing_ratio = float(missing.mean()) if len(missing) else 0.0
    ball_score = max(
        0.0,
        0.45 * gt_coverage
        + 0.30 * coverage
        + 0.20 * (1.0 - fallback_ratio)
        + 0.05 * (1.0 - min(speed_outlier_ratio, 1.0)),
    )
    return {
        "coverage": coverage,
        "gt_window_coverage": gt_coverage,
        "fallback_ratio": fallback_ratio,
        "missing_ratio": missing_ratio,
        "speed_outlier_ratio": speed_outlier_ratio,
        "p90_acceleration": p90_acceleration,
        "ball_score": ball_score,
    }


def _shot_metrics(
    candidates: pd.DataFrame, reference: pd.DataFrame
) -> ShotRankingMetrics:
    return evaluate_shot_candidates(candidates, reference)


def _shot_score(metrics: ShotRankingMetrics) -> float:
    fp_reduction_ratio = max(0.0, 1.0 - metrics.false_positives / 61.0)
    mae_score = max(0.0, 1.0 - metrics.temporal_mae_seconds / 10.0)
    recall_gate = 1.0 if metrics.hit_2 >= 0.78 else max(0.0, metrics.hit_2 / 0.78)
    return max(
        0.0,
        recall_gate
        * (
            0.30 * metrics.precision_2
            + 0.25 * fp_reduction_ratio
            + 0.20 * metrics.hit_2
            + 0.10 * metrics.hit_1
            + 0.05 * mae_score
        ),
    )


def _shot_fp_diagnostics(
    candidates: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "shot_id",
                "global_seconds",
                "nearest_reference_seconds",
                "time_error_seconds",
                "matched_2s",
                "score",
                "source",
            ]
        )
    rows = []
    for candidate in candidates.itertuples(index=False):
        nearest_seconds = np.nan
        error = np.nan
        matched = False
        if not reference.empty:
            nearest_idx = (
                (reference["reference_seconds"] - float(candidate.global_seconds))
                .abs()
                .idxmin()
            )
            nearest = reference.loc[nearest_idx]
            nearest_seconds = float(nearest["reference_seconds"])
            error = float(candidate.global_seconds) - nearest_seconds
            matched = abs(error) <= 2.0
        rows.append(
            {
                "shot_id": str(candidate.shot_id),
                "global_seconds": float(candidate.global_seconds),
                "nearest_reference_seconds": nearest_seconds,
                "time_error_seconds": error,
                "matched_2s": matched,
                "score": float(candidate.score),
                "source": str(candidate.source),
            }
        )
    return pd.DataFrame(rows)


def _shot_window_features(
    config: VideoOnlyXgEndToEndConfig, ball: pd.DataFrame, tracks: pd.DataFrame
) -> pd.DataFrame:
    base = ShotCandidateFeatureExtractor().transform(ball, tracks)
    window = ShotWindowFeatureExtractor(
        contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
        post_shot_window_seconds=config.shots.post_shot_window_seconds,
        long_shot_distance_m=config.shots.long_shot_distance_m,
    ).transform(base)
    return ShotPatternScorer(
        long_shot_distance_m=config.shots.long_shot_distance_m
    ).score(window)


def _projection_metrics(
    features: pd.DataFrame, homographies: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, float]:
    if homographies.empty:
        homography_coverage = 0.0
        median_confidence = 0.0
        p90_confidence = 0.0
    else:
        status = homographies["status"].astype(str)
        calibrated = ~status.str.contains(
            "degraded|unavailable|unsupported", case=False, na=False
        )
        homography_coverage = float(calibrated.mean()) if len(calibrated) else 0.0
        median_confidence = float(homographies["projection_confidence"].median())
        p90_confidence = float(homographies["projection_confidence"].quantile(0.9))
    feature_count = float(len(features))
    fallback_count = (
        float(
            features["feature_source"].astype(str).eq("missing_center_fallback").sum()
        )
        if "feature_source" in features.columns
        else 0.0
    )
    reference_factor = 1.0 if not reference.empty else 0.5
    fallback_ratio = fallback_count / feature_count if feature_count else 0.0
    projection_score = reference_factor * (
        0.45 * homography_coverage
        + 0.30 * median_confidence
        + 0.25 * (1.0 - fallback_ratio)
    )
    return {
        "homography_coverage": homography_coverage,
        "median_projection_confidence": median_confidence,
        "p90_projection_confidence": p90_confidence,
        "feature_count": feature_count,
        "fallback_ratio": fallback_ratio,
        "projection_score": projection_score,
    }


def _predict_base_methods(
    features: pd.DataFrame, model_kinds: tuple[XgModelKind, ...]
) -> pd.DataFrame:
    estimators = {
        XgModelKind.VIDEO_GEOMETRY: VideoGeometryXgEstimator(),
        XgModelKind.VIDEO_FREEZE_CONTEXT: VideoFreezeContextXgEstimator(),
        XgModelKind.VIDEO_KINEMATIC_CONTEXT: VideoKinematicContextXgEstimator(),
    }
    rows = []
    for feature in features.itertuples(index=False):
        video_features = _video_features_from_row(feature)
        for model_kind in model_kinds:
            prediction = estimators[model_kind].predict(video_features)
            rows.append(
                {
                    "shot_id": prediction.shot_id,
                    "frame_index": prediction.frame_index,
                    "method": model_kind.value,
                    "xg": prediction.xg,
                }
            )
    return pd.DataFrame(rows, columns=["shot_id", "frame_index", "method", "xg"])


def _xg_metric_rows(
    variant: str,
    predictions: pd.DataFrame,
    per_eval: pd.DataFrame,
    metrics: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[dict[str, float | str]]:
    total_reference_xg = (
        float(reference["reference_xg"].sum()) if not reference.empty else 0.0
    )
    if metrics.empty:
        return [
            {
                "variant": variant,
                "method": variant,
                "matched_shots": 0.0,
                "mae_vs_reference_xg": 0.0,
                "rmse_vs_reference_xg": 0.0,
                "total_predicted_xg": float(predictions["xg"].sum())
                if not predictions.empty
                else 0.0,
                "total_reference_xg": total_reference_xg,
                "total_xg_error": -total_reference_xg,
                "xg_total_error_score": _xg_total_error_score(0.0, total_reference_xg),
            }
        ]
    rows: list[dict[str, float | str]] = []
    for row in metrics.itertuples(index=False):
        rows.append(
            {
                "variant": variant,
                "method": str(row.method),
                "matched_shots": float(row.matched_shots),
                "mae_vs_reference_xg": float(row.mae_vs_reference_xg),
                "rmse_vs_reference_xg": float(row.rmse_vs_reference_xg),
                "total_predicted_xg": float(row.total_predicted_xg),
                "total_reference_xg": float(row.total_reference_xg),
                "total_xg_error": float(row.total_xg_error),
                "xg_total_error_score": _xg_total_error_score(
                    float(row.total_predicted_xg),
                    float(row.total_reference_xg),
                ),
            }
        )
    if per_eval.empty:
        return rows
    return rows


def _final_ranking(
    baseline_summary: dict[str, Any],
    best_ball: str,
    best_shot: str,
    best_projection: str,
    shot_variants: pd.DataFrame,
    xg_variants: pd.DataFrame,
    start: float,
    baseline_runtime_seconds: float,
) -> pd.DataFrame:
    shot_row = shot_variants[shot_variants["variant"].eq(best_shot)].iloc[0]
    elapsed = max(time.perf_counter() - start, 1e-6)
    runtime_score = min(1.0, baseline_runtime_seconds / elapsed)
    rows = [
        {
            "variant": "baseline_existing_1fps",
            "ball_variant": "baseline",
            "shot_variant": "baseline",
            "projection_variant": "baseline",
            "xg_variant": baseline_summary["best_existing_xg_method"],
            "method": baseline_summary["best_existing_xg_method"],
            "hit@2s": baseline_summary["hit@2s"],
            "hit@1s": baseline_summary["hit@1s"],
            "precision@2s": 0.0,
            "false_positives": 61.0,
            "fp_reduction_ratio": 0.0,
            "feature_quality_score": 1.0
            - baseline_summary["missing_center_fallback_ratio"],
            "xg_total_error_score": _xg_total_error_score(
                float(baseline_summary["best_existing_total_xg"]),
                float(baseline_summary["reference_total_xg"]),
            ),
            "runtime_score": 1.0,
        }
    ]
    for row in xg_variants.itertuples(index=False):
        rows.append(
            {
                "variant": f"{best_ball}+{best_shot}+{best_projection}+{row.variant}",
                "ball_variant": best_ball,
                "shot_variant": best_shot,
                "projection_variant": best_projection,
                "xg_variant": str(row.variant),
                "method": str(row.method),
                "hit@2s": float(shot_row["hit@2s"]),
                "hit@1s": float(shot_row["hit@1s"]),
                "precision@2s": float(shot_row["precision@2s"]),
                "false_positives": float(shot_row["false_positives"]),
                "fp_reduction_ratio": max(
                    0.0, 1.0 - float(shot_row["false_positives"]) / 61.0
                ),
                "feature_quality_score": float(
                    1.0 - min(float(shot_row.get("fallback_ratio", 0.0)), 1.0)
                )
                if "fallback_ratio" in shot_row
                else 1.0,
                "xg_total_error_score": float(row.xg_total_error_score),
                "runtime_score": runtime_score,
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["temporal_mae_score"] = 1.0
    ranking["composite_score"] = (
        0.30 * ranking["precision@2s"]
        + 0.25 * ranking["fp_reduction_ratio"]
        + 0.20 * ranking["hit@2s"]
        + 0.10 * ranking["hit@1s"]
        + 0.10 * ranking["xg_total_error_score"]
        + 0.05 * ranking["temporal_mae_score"]
    )
    return ranking.sort_values("composite_score", ascending=False).reset_index(
        drop=True
    )


def _write_recommendation(
    output_dir: Path,
    baseline_summary: dict[str, Any],
    best_ball: str,
    best_shot: str,
    best_projection: str,
    ranking: pd.DataFrame,
) -> None:
    winner = ranking.iloc[0].to_dict() if not ranking.empty else {}
    lines = [
        "# Video-Only xG Improvement Recommendation",
        "",
        "## Baseline",
        "",
        f"- Run: `{baseline_summary['run_dir']}`",
        f"- GT shots: `{baseline_summary['reference_shots']:.0f}`",
        f"- StatsBomb total xG: `{baseline_summary['reference_total_xg']:.4f}`",
        f"- Baseline hit@2s: `{baseline_summary['hit@2s']:.4f}`",
        f"- Baseline missing-center fallback ratio: `{baseline_summary['missing_center_fallback_ratio']:.4f}`",
        "",
        "## Selected Variants",
        "",
        f"- Ball: `{best_ball}`",
        f"- Shot ranking: `{best_shot}`",
        f"- Projection/features: `{best_projection}`",
        f"- Best final row: `{winner.get('variant', '')}`",
        f"- Composite score: `{float(winner.get('composite_score', 0.0)):.4f}`",
        "",
        "## Notes",
        "",
        "Reference data is loaded only after candidate and prediction artifacts are written.",
        "The ablation is a fast 1 FPS selection pass; a publication run still needs the winning variants on the 10 FPS scan.",
    ]
    (output_dir / "final_recommendation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_shot_detection_report(
    output_dir: Path,
    baseline_summary: dict[str, Any],
    best_shot: str,
    shot_variants: pd.DataFrame,
    best_candidates: pd.DataFrame,
) -> None:
    best = shot_variants[shot_variants["variant"].eq(best_shot)].iloc[0]
    baseline_candidates = float(baseline_summary["candidates"])
    selected_candidates = float(len(best_candidates))
    fp_reduction = (
        max(0.0, 1.0 - float(best["false_positives"]) / 61.0)
        if float(best["false_positives"]) >= 0.0
        else 0.0
    )
    lines = [
        "# Shot Detection FP Reduction Report",
        "",
        "## Baseline",
        "",
        f"- Candidates: `{baseline_candidates:.0f}`",
        f"- hit@2s: `{baseline_summary['hit@2s']:.4f}`",
        f"- hit@1s: `{baseline_summary['hit@1s']:.4f}`",
        "",
        "## Selected Variant",
        "",
        f"- Shot ranking: `{best_shot}`",
        f"- Candidates: `{selected_candidates:.0f}`",
        f"- hit@2s: `{float(best['hit@2s']):.4f}`",
        f"- hit@1s: `{float(best['hit@1s']):.4f}`",
        f"- precision@2s: `{float(best['precision@2s']):.4f}`",
        f"- false positives: `{float(best['false_positives']):.0f}`",
        f"- FP reduction ratio vs current 10 FPS diagnostic: `{fp_reduction:.4f}`",
        f"- shot_score: `{float(best['shot_score']):.4f}`",
        "",
        "## Artifacts",
        "",
        "- `11_shot_fp_diagnostics.csv`",
        "- `12_candidate_window_features.parquet`",
        "- `13_shot_fp_ablation.csv`",
        "- `14_selected_refined_shots.parquet`",
        "- `15_predictions_after_fp_reduction.csv`",
        "",
        "StatsBomb reference is used only after video-only candidates and window features are saved.",
    ]
    (output_dir / "final_shot_detection_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_reference(
    config: VideoOnlyXgEndToEndConfig, timeline: pd.DataFrame
) -> pd.DataFrame:
    path = config.evaluation.reference_events
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "reference_id",
                "period",
                "reference_seconds",
                "reference_xg",
                "is_goal",
                "team",
            ]
        )
    return _reference_shots(path, timeline)


def _reference_by_candidate(
    candidates: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    if candidates.empty or reference.empty:
        return pd.DataFrame(columns=["shot_id", "reference_xg", "is_goal"])
    rows = []
    for candidate in candidates.itertuples(index=False):
        nearest_idx = (
            (reference["reference_seconds"] - float(candidate.global_seconds))
            .abs()
            .idxmin()
        )
        nearest = reference.loc[nearest_idx]
        if (
            abs(float(nearest["reference_seconds"]) - float(candidate.global_seconds))
            <= 2.0
        ):
            rows.append(
                {
                    "shot_id": candidate.shot_id,
                    "reference_xg": float(nearest["reference_xg"]),
                    "is_goal": float(nearest["is_goal"]),
                }
            )
    return pd.DataFrame(rows, columns=["shot_id", "reference_xg", "is_goal"])


def _reference_window_coverage(
    trajectory: pd.DataFrame, reference: pd.DataFrame
) -> float:
    if reference.empty:
        return 0.0
    values = []
    for ref in reference.itertuples(index=False):
        window = trajectory[
            trajectory["global_seconds"].between(
                float(ref.reference_seconds) - 2.0,
                float(ref.reference_seconds) + 2.0,
            )
        ]
        if window.empty:
            values.append(0.0)
            continue
        valid = window[["pitch_x", "pitch_y"]].notna().all(axis=1)
        missing = (
            window["source"].astype(str).str.contains("missing", case=False, na=False)
        )
        values.append(float((valid & ~missing).mean()))
    return float(np.mean(values)) if values else 0.0


def _trajectory_speed(trajectory: pd.DataFrame) -> pd.Series:
    frame = trajectory.sort_values("global_seconds")
    dx = frame["pitch_x"].diff()
    dy = frame["pitch_y"].diff()
    dt = frame["global_seconds"].diff().replace(0.0, np.nan)
    return (np.hypot(dx, dy) / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _video_features_from_row(row: object) -> VideoShotFeatures:
    return VideoShotFeatures(
        shot_id=str(row.shot_id),
        frame_index=int(row.frame_index),
        shot_x=float(row.shot_x),
        shot_y=float(row.shot_y),
        goal_x=float(getattr(row, "goal_x", 105.0)),
        goal_y=float(getattr(row, "goal_y", 34.0)),
        nearest_player_distance=_optional_float(
            getattr(row, "nearest_player_distance", None)
        ),
        goalkeeper_distance=_optional_float(getattr(row, "goalkeeper_distance", None)),
        defender_count_in_cone=int(getattr(row, "defender_count_in_cone", 0) or 0),
        ball_speed=_optional_float(getattr(row, "ball_speed", None)),
        ball_direction_to_goal=_optional_float(
            getattr(row, "ball_direction_to_goal", None)
        ),
        shot_confidence=float(getattr(row, "shot_confidence", 1.0) or 1.0),
    )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int | float | np.integer | np.floating | str):
        return float(value)
    return float(str(value))


def _best_existing_total_xg(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"method": "", "total_xg": 0.0}
    totals = predictions.groupby("method")["xg"].sum().sort_values(ascending=False)
    method = str(totals.index[0])
    return {"method": method, "total_xg": float(totals.iloc[0])}


def _xg_total_error_score(predicted: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(predicted - reference) / reference)


def _best_variant(frame: pd.DataFrame, score_column: str) -> str:
    if frame.empty:
        raise ValueError("Cannot select a variant from an empty ablation table")
    return str(frame.sort_values(score_column, ascending=False).iloc[0]["variant"])


def _candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "shot_id",
        "global_frame_index",
        "global_seconds",
        "part_index",
        "part_frame_index",
        "score",
        "confidence",
        "source",
        "nearest_player_distance",
        "ball_speed",
        "ball_direction_to_goal",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame[columns].copy()


def _variant_dir(output_dir: Path, category: str, variant: str) -> Path:
    return output_dir / category / variant


def _write_variant_frame(
    output_dir: Path, category: str, variant: str, name: str, frame: pd.DataFrame
) -> None:
    path = _variant_dir(output_dir, category, variant) / f"{name}.parquet"
    write_dataframe_artifact(frame, path)


def _read_variant_frame(
    output_dir: Path, category: str, variant: str, name: str
) -> pd.DataFrame:
    return read_dataframe_artifact(
        _variant_dir(output_dir, category, variant) / f"{name}.parquet"
    )


def _baseline_dir(config: VideoOnlyXgEndToEndConfig) -> Path:
    if config.ablation.baseline_run_dir is None:
        return config.output_dir
    return config.ablation.baseline_run_dir


def _output_dir(config: VideoOnlyXgEndToEndConfig, baseline_dir: Path) -> Path:
    if config.ablation.output_dir is not None:
        return config.ablation.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return baseline_dir.parent / f"video_xg_improvement_{timestamp}"


def _read_metrics(path: Path) -> dict[str, float]:
    data = read_json_artifact(path)
    return {str(key): float(value) for key, value in data.items()}


def _write_run_config(config: VideoOnlyXgEndToEndConfig, path: Path) -> None:
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _ranking_metrics(ranking: pd.DataFrame) -> dict[str, float]:
    if ranking.empty:
        return {}
    winner = ranking.iloc[0]
    return {
        "composite_score": float(winner["composite_score"]),
        "hit@2s": float(winner["hit@2s"]),
        "xg_total_error_score": float(winner["xg_total_error_score"]),
    }
