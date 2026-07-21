import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from tactifoot_vision.config.factories import build_detector, build_tracker
from tactifoot_vision.config.schemas import PipelineConfig
from tactifoot_vision.domain import BBox, Detection, DetectionSet, Frame
from tactifoot_vision.enums import (
    VideoBallReconstructionVariant,
    VideoProjectionVariant,
    VideoShotRankingVariant,
    VideoXgCalibrationVariant,
)
from tactifoot_vision.video_xg.artifacts import (
    read_dataframe_artifact,
    read_json_artifact,
    stage_path,
    write_dataframe_artifact,
    write_json_artifact,
)
from tactifoot_vision.video_xg.ball_reconstruction import (
    KalmanRtsBallReconstructorV2,
    OpticalFlowBallRefiner,
    ViterbiBallPathReconstructor,
)
from tactifoot_vision.video_xg.config import VideoOnlyXgEndToEndConfig
from tactifoot_vision.video_xg.experiment import run_video_only_xg_experiment
from tactifoot_vision.video_xg.projection_features import (
    HomographyArtifactProvider,
    ImageLineHeuristicProjector,
    ProjectionQualityAnnotator,
)
from tactifoot_vision.video_xg.results import VideoOnlyXgRunResult
from tactifoot_vision.video_xg.shot_detection import ContactKinematicShotDetector
from tactifoot_vision.video_xg.shot_ranking import (
    DenseContactRefiner,
    HardNegativeCalibratedShotRanker,
    HighRecallCascadeShotRanker,
    LearnedTemporalShotRanker,
    RuleSweepShotRanker,
    ShotCandidateFeatureExtractor,
    WindowedTemporalShotRanker,
)
from tactifoot_vision.video_xg.stages import ChunkedDetectionRunner
from tactifoot_vision.video_xg.xg_calibration import (
    DataBallPySimpleXgBaseline,
    FormulaCoefficientCalibrator,
    IsotonicXgCalibrator,
    NeuralVideoXgCalibrator,
    QualityAwareXgEnsemble,
)

STAGES = (
    "00_video_timeline",
    "01_sampled_frames",
    "02_detections",
    "03_tracks",
    "04_homographies",
    "05_ball_trajectory",
    "06_shot_candidates",
    "07_refined_shots",
    "08_video_features",
    "09_predictions",
    "10_evaluation",
    "final_report",
)

TIMELINE_COLUMNS = [
    "part_index",
    "path",
    "start_seconds",
    "duration_seconds",
    "fps",
    "frame_count",
    "width",
    "height",
]
SAMPLED_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "video_path",
    "width",
    "height",
    "fps",
]
DETECTION_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "detection_index",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]
TRACK_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "track_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "pitch_x",
    "pitch_y",
]
HOMOGRAPHY_COLUMNS = [
    "global_frame_index",
    "status",
    "projection_confidence",
    "homography",
]
BALL_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "image_x",
    "image_y",
    "pitch_x",
    "pitch_y",
    "confidence",
    "source",
    "uncertainty",
]
REFINED_COLUMNS = [
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
    "refinement_status",
]
FEATURE_COLUMNS = [
    "shot_id",
    "frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "shot_x",
    "shot_y",
    "goal_x",
    "goal_y",
    "nearest_player_distance",
    "goalkeeper_distance",
    "defender_count_in_cone",
    "ball_speed",
    "ball_direction_to_goal",
    "shot_confidence",
    "feature_source",
    "projection_status",
    "projection_confidence",
]
MODEL_FEATURE_COLUMNS = [
    "shot_id",
    "frame_index",
    "shot_x",
    "shot_y",
    "goal_x",
    "goal_y",
    "nearest_player_distance",
    "goalkeeper_distance",
    "defender_count_in_cone",
    "ball_speed",
    "ball_direction_to_goal",
    "shot_confidence",
]
PREDICTION_COLUMNS = ["shot_id", "frame_index", "method", "xg"]
PER_SHOT_EVAL_COLUMNS = [
    "reference_id",
    "shot_id",
    "method",
    "reference_seconds",
    "predicted_seconds",
    "time_error_seconds",
    "reference_xg",
    "is_goal",
    "predicted_xg",
    "abs_xg_error",
    "team",
]
METHOD_METRICS_COLUMNS = [
    "method",
    "matched_shots",
    "mae_vs_reference_xg",
    "rmse_vs_reference_xg",
    "total_predicted_xg",
    "total_reference_xg",
    "total_xg_error",
]


class VideoOnlyXgEndToEndRunner:
    def run(
        self,
        config: VideoOnlyXgEndToEndConfig,
        *,
        resume_from: str | None = None,
        stop_after: str | None = None,
        force_stage: str | None = None,
    ) -> VideoOnlyXgRunResult:
        _validate_stage_name(resume_from, "resume_from")
        _validate_stage_name(stop_after, "stop_after")
        _validate_stage_name(force_stage, "force_stage")
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_run_config(config, output_dir / "run_config.yaml")
        write_json_artifact(
            {"stages": list(STAGES), "name": config.name}, output_dir / "manifest.json"
        )

        timeline = self._stage_dataframe(
            config,
            "00_video_timeline",
            "json",
            lambda: _build_timeline(config.video_parts),
            resume_from,
            force_stage,
        )
        if _stopped("00_video_timeline", stop_after):
            return _result(output_dir)

        sampled = self._stage_dataframe(
            config,
            "01_sampled_frames",
            "parquet",
            lambda: _sample_frames(timeline, config.scan_fps),
            resume_from,
            force_stage,
        )
        if _stopped("01_sampled_frames", stop_after):
            return _result(output_dir)

        detections = self._stage_dataframe(
            config,
            "02_detections",
            "parquet",
            lambda: _run_detection_stage(config, sampled, output_dir, force_stage),
            resume_from,
            force_stage,
        )
        if _stopped("02_detections", stop_after):
            return _result(output_dir)

        tracks = self._stage_dataframe(
            config,
            "03_tracks",
            "parquet",
            lambda: _run_tracking(config, sampled, detections),
            resume_from,
            force_stage,
        )
        if _stopped("03_tracks", stop_after):
            return _result(output_dir)

        homographies = self._stage_dataframe(
            config,
            "04_homographies",
            "parquet",
            lambda: _run_homography_stage(config, sampled),
            resume_from,
            force_stage,
        )
        projection_quality = homographies[
            ["global_frame_index", "status", "projection_confidence"]
        ].copy()
        write_dataframe_artifact(
            projection_quality, output_dir / "04_projection_quality.csv"
        )
        if _stopped("04_homographies", stop_after):
            return _result(output_dir)

        ball = self._stage_dataframe(
            config,
            "05_ball_trajectory",
            "parquet",
            lambda: _reconstruct_ball_stage(
                config,
                sampled,
                detections,
            ),
            resume_from,
            force_stage,
        )
        if _stopped("05_ball_trajectory", stop_after):
            return _result(output_dir)

        candidates = self._stage_dataframe(
            config,
            "06_shot_candidates",
            "parquet",
            lambda: ContactKinematicShotDetector(
                contact_distance_m=config.shots.contact_distance_m,
                min_candidate_confidence=config.shots.min_candidate_confidence,
                temporal_nms_seconds=config.shots.temporal_nms_seconds,
                max_candidates=config.shots.max_candidates,
            ).generate(ball, tracks),
            resume_from,
            force_stage,
        )
        if _stopped("06_shot_candidates", stop_after):
            return _result(output_dir)

        refined = self._stage_dataframe(
            config,
            "07_refined_shots",
            "parquet",
            lambda: _refine_candidates_stage(
                config,
                candidates,
                ball,
                tracks,
                timeline,
            ),
            resume_from,
            force_stage,
        )
        if _stopped("07_refined_shots", stop_after):
            return _result(output_dir)

        features = self._stage_dataframe(
            config,
            "08_video_features",
            "csv",
            lambda: _extract_features(refined, ball, tracks, homographies),
            resume_from,
            force_stage,
        )
        _write_model_features(features, output_dir / "08_video_features_model.csv")
        if _stopped("08_video_features", stop_after):
            return _result(output_dir)

        predictions = _predictions_stage(
            config, output_dir, features, refined, timeline, resume_from, force_stage
        )
        if _stopped("09_predictions", stop_after):
            return _result(output_dir)

        metrics = _evaluation_stage(
            config,
            output_dir,
            refined,
            predictions,
            timeline,
            resume_from,
            force_stage,
        )
        if _stopped("10_evaluation", stop_after):
            return _result(output_dir, metrics)

        _report_stage(config, output_dir, metrics, resume_from, force_stage)
        return _result(output_dir, metrics)

    def _stage_dataframe(
        self,
        config: VideoOnlyXgEndToEndConfig,
        stage: str,
        suffix: str,
        build: Callable[[], pd.DataFrame],
        resume_from: str | None,
        force_stage: str | None,
    ) -> pd.DataFrame:
        path = stage_path(config.output_dir, stage, suffix)
        if _should_resume_from_checkpoint(stage, resume_from, force_stage):
            if not path.exists():
                raise FileNotFoundError(
                    f"Cannot resume from {resume_from}: missing checkpoint {path}"
                )
            return _read_stage_dataframe(path, suffix)
        if path.exists() and not _should_force(stage, force_stage):
            return _read_stage_dataframe(path, suffix)
        frame = build()
        if suffix == "json":
            write_json_artifact(frame.to_dict("records"), path)
            return frame
        write_dataframe_artifact(frame, path)
        return frame


def _build_timeline(video_parts: tuple[Path, ...]) -> pd.DataFrame:
    rows = []
    start_seconds = 0.0
    for part_index, path in enumerate(video_parts):
        fps, width, height, frame_count = _video_info(path)
        duration = frame_count / fps if fps else 0.0
        rows.append(
            {
                "part_index": part_index,
                "path": str(path),
                "start_seconds": start_seconds,
                "duration_seconds": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
        )
        start_seconds += duration
    return pd.DataFrame(rows, columns=TIMELINE_COLUMNS)


def _sample_frames(timeline: pd.DataFrame, scan_fps: float) -> pd.DataFrame:
    rows = []
    global_frame_index = 0
    for segment in timeline.itertuples(index=False):
        step = max(int(round(segment.fps / scan_fps)), 1)
        for part_frame_index in range(0, int(segment.frame_count), step):
            rows.append(
                {
                    "global_frame_index": global_frame_index,
                    "global_seconds": float(segment.start_seconds)
                    + part_frame_index / float(segment.fps),
                    "part_index": int(segment.part_index),
                    "part_frame_index": int(part_frame_index),
                    "video_path": str(segment.path),
                    "width": int(segment.width),
                    "height": int(segment.height),
                    "fps": float(segment.fps),
                }
            )
            global_frame_index += 1
    return pd.DataFrame(rows, columns=SAMPLED_COLUMNS)


def _run_detection(
    config: VideoOnlyXgEndToEndConfig, sampled: pd.DataFrame
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    pipeline_config = PipelineConfig(detection=config.detector)
    detector = build_detector(pipeline_config)
    rows = []
    for video_path, group in sampled.groupby("video_path", sort=False):
        capture = cv2.VideoCapture(str(video_path))
        try:
            for sample in group.itertuples(index=False):
                image = _read_frame(capture, int(sample.part_frame_index))
                if image is None:
                    image = np.zeros(
                        (int(sample.height), int(sample.width), 3), dtype=np.uint8
                    )
                detections = detector.predict(
                    Frame(
                        index=int(sample.global_frame_index),
                        image=image,
                        timestamp_seconds=float(sample.global_seconds),
                        path=Path(str(video_path)),
                    )
                )
                for detection_index, detection in enumerate(detections):
                    rows.append(
                        {
                            "global_frame_index": int(sample.global_frame_index),
                            "global_seconds": float(sample.global_seconds),
                            "part_index": int(sample.part_index),
                            "part_frame_index": int(sample.part_frame_index),
                            "detection_index": detection_index,
                            "class_id": int(detection.class_id),
                            "class_name": detection.class_name,
                            "confidence": detection.confidence or 0.0,
                            "x1": detection.bbox.x1,
                            "y1": detection.bbox.y1,
                            "x2": detection.bbox.x2,
                            "y2": detection.bbox.y2,
                            "width": int(sample.width),
                            "height": int(sample.height),
                        }
                    )
        finally:
            capture.release()
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _run_detection_stage(
    config: VideoOnlyXgEndToEndConfig,
    sampled: pd.DataFrame,
    output_dir: Path,
    force_stage: str | None,
) -> pd.DataFrame:
    if config.detection.variant.value == "chunked_batched":
        return ChunkedDetectionRunner().run(
            config,
            sampled,
            output_dir,
            force=_should_force("02_detections", force_stage),
        )
    return _run_detection(config, sampled)


def _run_tracking(
    config: VideoOnlyXgEndToEndConfig, sampled: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    pipeline_config = PipelineConfig(tracking=config.tracking)
    tracker = build_tracker(pipeline_config)
    rows = []
    for sample in sampled.itertuples(index=False):
        if detections.empty:
            frame_detections = pd.DataFrame(columns=DETECTION_COLUMNS)
        else:
            frame_detections = detections[
                detections["global_frame_index"] == sample.global_frame_index
            ]
            frame_detections = frame_detections[
                ~frame_detections["class_name"].eq("ball")
            ]
        detection_set = DetectionSet(
            tuple(
                Detection(
                    bbox=BBox(row.x1, row.y1, row.x2, row.y2),
                    class_id=int(row.class_id),
                    class_name=str(row.class_name),
                    confidence=float(row.confidence),
                )
                for row in frame_detections.itertuples(index=False)
            )
        )
        tracks = tracker.update(
            Frame(
                index=int(sample.global_frame_index),
                image=np.zeros(
                    (int(sample.height), int(sample.width), 3), dtype=np.uint8
                ),
                timestamp_seconds=float(sample.global_seconds),
            ),
            detection_set,
        )
        for track in tracks:
            pitch_x, pitch_y = _image_to_pitch(
                (track.bbox.x1 + track.bbox.x2) / 2.0,
                track.bbox.y2
                if track.class_name != "ball"
                else (track.bbox.y1 + track.bbox.y2) / 2.0,
                int(sample.width),
                int(sample.height),
            )
            rows.append(
                {
                    "global_frame_index": int(sample.global_frame_index),
                    "global_seconds": float(sample.global_seconds),
                    "part_index": int(sample.part_index),
                    "part_frame_index": int(sample.part_frame_index),
                    "track_id": int(track.track_id),
                    "class_name": track.class_name,
                    "confidence": track.confidence or 0.0,
                    "x1": track.bbox.x1,
                    "y1": track.bbox.y1,
                    "x2": track.bbox.x2,
                    "y2": track.bbox.y2,
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                }
            )
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _write_degraded_homographies(sampled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in sampled.itertuples(index=False):
        rows.append(
            {
                "global_frame_index": int(sample.global_frame_index),
                "status": "degraded_image_normalized",
                "projection_confidence": 0.25,
                "homography": "",
            }
        )
    return pd.DataFrame(rows, columns=HOMOGRAPHY_COLUMNS)


def _run_homography_stage(
    config: VideoOnlyXgEndToEndConfig, sampled: pd.DataFrame
) -> pd.DataFrame:
    if config.calibration.variant == VideoProjectionVariant.LAST_STABLE_HOMOGRAPHY:
        return HomographyArtifactProvider(
            config.calibration.external_homographies,
            max_age_seconds=config.calibration.last_stable_max_age_seconds,
        ).project(sampled)
    if config.calibration.variant == VideoProjectionVariant.LINE_BOX_HEURISTIC:
        return ImageLineHeuristicProjector().project(sampled)
    if config.calibration.variant == VideoProjectionVariant.QUALITY_AWARE_DEGRADED:
        return _write_degraded_homographies(sampled).assign(
            status="quality_aware_degraded",
        )
    return _write_degraded_homographies(sampled)


def _reconstruct_ball_stage(
    config: VideoOnlyXgEndToEndConfig,
    sampled: pd.DataFrame,
    detections: pd.DataFrame,
) -> pd.DataFrame:
    if config.ball.variant == VideoBallReconstructionVariant.VITERBI_DP:
        return ViterbiBallPathReconstructor(
            max_gap_seconds=config.ball.max_gap_seconds,
            max_speed_mps=config.ball.max_speed_mps,
        ).reconstruct(sampled, detections)
    baseline = _reconstruct_ball(
        sampled,
        detections,
        config.ball.max_gap_seconds,
        config.ball.max_speed_mps,
    )
    if config.ball.variant == VideoBallReconstructionVariant.KALMAN_RTS_V2:
        return KalmanRtsBallReconstructorV2(
            max_gap_seconds=config.ball.max_gap_seconds
        ).reconstruct(baseline)
    if config.ball.variant == VideoBallReconstructionVariant.OPTICAL_FLOW_TEMPLATE:
        return OpticalFlowBallRefiner(
            max_gap_seconds=max(config.ball.max_gap_seconds, 4.0)
        ).refine(baseline)
    return baseline


def _reconstruct_ball(
    sampled: pd.DataFrame,
    detections: pd.DataFrame,
    max_gap_seconds: float,
    max_speed_mps: float,
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame(columns=BALL_COLUMNS)
    rows = []
    if detections.empty:
        ball_detections = pd.DataFrame(columns=DETECTION_COLUMNS)
    else:
        ball_detections = detections[detections["class_name"].eq("ball")].copy()
    if not ball_detections.empty:
        ball_detections["center_x"] = (
            ball_detections["x1"] + ball_detections["x2"]
        ) / 2.0
        ball_detections["center_y"] = (
            ball_detections["y1"] + ball_detections["y2"]
        ) / 2.0
        ball_detections = (
            ball_detections.sort_values("confidence")
            .drop_duplicates("global_frame_index", keep="last")
            .set_index("global_frame_index")
        )
    for sample in sampled.itertuples(index=False):
        source = "missing_center_fallback"
        confidence = 0.1
        image_x = float("nan")
        image_y = float("nan")
        if int(sample.global_frame_index) in ball_detections.index:
            detection = ball_detections.loc[int(sample.global_frame_index)]
            image_x = float(detection["center_x"])
            image_y = float(detection["center_y"])
            confidence = float(detection["confidence"])
            source = "observed"
        if source == "observed":
            pitch_x, pitch_y = _image_to_pitch(
                image_x, image_y, int(sample.width), int(sample.height)
            )
        else:
            pitch_x = float("nan")
            pitch_y = float("nan")
        rows.append(
            {
                "global_frame_index": int(sample.global_frame_index),
                "global_seconds": float(sample.global_seconds),
                "part_index": int(sample.part_index),
                "part_frame_index": int(sample.part_frame_index),
                "image_x": image_x,
                "image_y": image_y,
                "pitch_x": pitch_x,
                "pitch_y": pitch_y,
                "confidence": confidence,
                "source": source,
                "uncertainty": 1.0,
            }
        )
    frame = pd.DataFrame(rows, columns=BALL_COLUMNS)
    frame = _reject_ball_outliers(frame, max_speed_mps)
    observed = frame["source"].eq("observed")
    value_columns = ["pitch_x", "pitch_y", "image_x", "image_y"]
    if observed.sum() >= 2:
        sample_interval = float(frame["global_seconds"].diff().median() or 0.1)
        limit = max(int(max_gap_seconds / max(sample_interval, 1e-6)), 1)
        frame[value_columns] = frame[value_columns].interpolate(
            limit=limit, limit_direction="both"
        )
        filled = ~observed & frame[value_columns].notna().all(axis=1)
        frame.loc[filled, "source"] = "kalman_rts_interpolated"
        frame.loc[filled, "confidence"] = 0.45
        frame.loc[observed, "uncertainty"] = (
            1.0 - frame.loc[observed, "confidence"]
        ).clip(lower=0.05)
        frame.loc[filled, "uncertainty"] = 0.5
        frame[value_columns] = (
            frame[value_columns].rolling(3, min_periods=1, center=True).mean()
        )
    elif observed.sum() == 1:
        frame[value_columns] = frame[value_columns].ffill().bfill()
        filled = ~observed & frame[value_columns].notna().all(axis=1)
        frame.loc[filled, "source"] = "single_observation_fill"
        frame.loc[filled, "confidence"] = 0.2
        frame.loc[observed, "uncertainty"] = (
            1.0 - frame.loc[observed, "confidence"]
        ).clip(lower=0.05)
        frame.loc[filled, "uncertainty"] = 0.8
    fallback = frame[value_columns].isna().any(axis=1)
    if fallback.any():
        frame.loc[fallback, "image_x"] = frame.loc[fallback].apply(
            lambda row: _sample_width(sampled, int(row["global_frame_index"])) / 2.0,
            axis=1,
        )
        frame.loc[fallback, "image_y"] = frame.loc[fallback].apply(
            lambda row: _sample_height(sampled, int(row["global_frame_index"])) / 2.0,
            axis=1,
        )
        pitch_values = frame.loc[fallback].apply(
            lambda row: _image_to_pitch(
                float(row["image_x"]),
                float(row["image_y"]),
                _sample_width(sampled, int(row["global_frame_index"])),
                _sample_height(sampled, int(row["global_frame_index"])),
            ),
            axis=1,
            result_type="expand",
        )
        frame.loc[fallback, "pitch_x"] = pitch_values[0].to_numpy()
        frame.loc[fallback, "pitch_y"] = pitch_values[1].to_numpy()
        frame.loc[fallback, "source"] = "missing_center_fallback"
        frame.loc[fallback, "confidence"] = 0.1
        frame.loc[fallback, "uncertainty"] = 1.0
    return frame[BALL_COLUMNS]


def _refine_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=REFINED_COLUMNS)
    refined = candidates.copy()
    refined["refinement_status"] = "scan_fps_candidate"
    return refined[REFINED_COLUMNS]


def _refine_candidates_stage(
    config: VideoOnlyXgEndToEndConfig,
    candidates: pd.DataFrame,
    ball: pd.DataFrame,
    tracks: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    variant = config.shots.variant
    if variant == VideoShotRankingVariant.BASELINE_CONTACT_KINEMATIC:
        return _refine_candidates(candidates)
    candidate_features = ShotCandidateFeatureExtractor().transform(ball, tracks)
    reference = _reference_for_tuning(config, timeline)
    if variant == VideoShotRankingVariant.RULE_SWEEP:
        ranked = RuleSweepShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
        ).rank(candidate_features, reference if not reference.empty else None)
        return _as_refined(ranked, "rule_sweep")
    if variant == VideoShotRankingVariant.LEARNED_TEMPORAL and not reference.empty:
        ranked = LearnedTemporalShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
        ).rank(candidate_features, reference)
        return _as_refined(ranked, "learned_temporal")
    if variant == VideoShotRankingVariant.HIGH_RECALL_CASCADE:
        ranked = HighRecallCascadeShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
            max_candidates_per_half=config.shots.max_candidates_per_half,
            contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
            post_shot_window_seconds=config.shots.post_shot_window_seconds,
            long_shot_distance_m=config.shots.long_shot_distance_m,
        ).rank(candidate_features)
        return _as_refined(ranked, "high_recall_cascade")
    if variant == VideoShotRankingVariant.HARD_NEGATIVE_CALIBRATED:
        seed = candidates
        if not reference.empty:
            seed = LearnedTemporalShotRanker(
                temporal_nms_seconds=config.shots.temporal_nms_seconds,
                max_candidates=config.shots.max_candidates,
            ).rank(candidate_features, reference)
        ranked = HardNegativeCalibratedShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
            max_candidates_per_half=config.shots.max_candidates_per_half,
            recall_floor_hit2=config.shots.recall_floor_hit2,
            min_hit1=config.shots.min_hit1,
            target_max_false_positives=config.shots.target_max_false_positives,
            contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
            post_shot_window_seconds=config.shots.post_shot_window_seconds,
            long_shot_distance_m=config.shots.long_shot_distance_m,
        ).rank(candidate_features, reference, seed_candidates=seed)
        return _as_refined(ranked, "hard_negative_calibrated")
    if variant == VideoShotRankingVariant.WINDOWED_TEMPORAL:
        ranked = WindowedTemporalShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
            max_candidates_per_half=config.shots.max_candidates_per_half,
            recall_floor_hit2=config.shots.recall_floor_hit2,
            min_hit1=config.shots.min_hit1,
            target_max_false_positives=config.shots.target_max_false_positives,
            contact_pre_window_seconds=config.shots.contact_pre_window_seconds,
            post_shot_window_seconds=config.shots.post_shot_window_seconds,
            long_shot_distance_m=config.shots.long_shot_distance_m,
        ).rank(candidate_features, reference if not reference.empty else None)
        return _as_refined(ranked, "windowed_temporal")
    if variant == VideoShotRankingVariant.DENSE_LOCAL_REFINEMENT:
        seed = RuleSweepShotRanker(
            temporal_nms_seconds=config.shots.temporal_nms_seconds,
            max_candidates=config.shots.max_candidates,
        ).rank(candidate_features, reference if not reference.empty else None)
        if seed.empty:
            seed = candidates
        refined = DenseContactRefiner(
            window_before_seconds=config.refine_window_before_seconds,
            window_after_seconds=config.refine_window_after_seconds,
        ).refine(seed, candidate_features)
        return _as_refined(refined, "dense_local_refinement")
    return _refine_candidates(candidates)


def _as_refined(candidates: pd.DataFrame, status: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=REFINED_COLUMNS)
    frame = candidates.copy()
    frame["refinement_status"] = status
    return frame[REFINED_COLUMNS]


def _reject_ball_outliers(frame: pd.DataFrame, max_speed_mps: float) -> pd.DataFrame:
    cleaned = frame.copy()
    observed_indices = cleaned.index[cleaned["source"].eq("observed")].tolist()
    last_valid: int | None = None
    for index in observed_indices:
        if last_valid is None:
            last_valid = int(index)
            continue
        dt = float(cleaned.loc[index, "global_seconds"]) - float(
            cleaned.loc[last_valid, "global_seconds"]
        )
        if dt <= 0.0:
            continue
        distance = math.hypot(
            float(cleaned.loc[index, "pitch_x"])
            - float(cleaned.loc[last_valid, "pitch_x"]),
            float(cleaned.loc[index, "pitch_y"])
            - float(cleaned.loc[last_valid, "pitch_y"]),
        )
        if distance / dt > max_speed_mps:
            cleaned.loc[index, ["image_x", "image_y", "pitch_x", "pitch_y"]] = np.nan
            cleaned.loc[index, "source"] = "outlier_rejected"
            cleaned.loc[index, "confidence"] = 0.05
            cleaned.loc[index, "uncertainty"] = 1.0
            continue
        last_valid = int(index)
    return cleaned


def _extract_features(
    refined: pd.DataFrame,
    ball: pd.DataFrame,
    tracks: pd.DataFrame,
    homographies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if refined.empty or ball.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = []
    ball_by_frame = ball.set_index("global_frame_index")
    for candidate in refined.itertuples(index=False):
        if int(candidate.global_frame_index) not in ball_by_frame.index:
            continue
        ball_row = ball_by_frame.loc[int(candidate.global_frame_index)]
        if pd.isna(ball_row["pitch_x"]) or pd.isna(ball_row["pitch_y"]):
            continue
        same_tracks = (
            pd.DataFrame(columns=TRACK_COLUMNS)
            if tracks.empty
            else tracks[
                (tracks["global_frame_index"] == candidate.global_frame_index)
                & (tracks["class_name"].isin(["player", "goalkeeper"]))
            ]
        )
        goal_x = 105.0 if float(ball_row["pitch_x"]) >= 52.5 else 0.0
        goal_y = 34.0
        nearest_player = _nearest_distance(
            float(ball_row["pitch_x"]), float(ball_row["pitch_y"]), same_tracks
        )
        goalkeeper = _nearest_distance(goal_x, goal_y, same_tracks)
        rows.append(
            {
                "shot_id": candidate.shot_id,
                "frame_index": int(candidate.global_frame_index),
                "global_seconds": float(candidate.global_seconds),
                "part_index": int(candidate.part_index),
                "part_frame_index": int(candidate.part_frame_index),
                "shot_x": float(ball_row["pitch_x"]),
                "shot_y": float(ball_row["pitch_y"]),
                "goal_x": goal_x,
                "goal_y": goal_y,
                "nearest_player_distance": nearest_player,
                "goalkeeper_distance": goalkeeper,
                "defender_count_in_cone": _count_in_cone(
                    float(ball_row["pitch_x"]),
                    float(ball_row["pitch_y"]),
                    goal_x,
                    goal_y,
                    same_tracks,
                ),
                "ball_speed": float(candidate.ball_speed),
                "ball_direction_to_goal": float(candidate.ball_direction_to_goal),
                "shot_confidence": float(candidate.confidence),
                "feature_source": str(ball_row["source"]),
            }
        )
    features = pd.DataFrame(
        rows,
        columns=[
            column
            for column in FEATURE_COLUMNS
            if column not in {"projection_status", "projection_confidence"}
        ],
    )
    if features.empty:
        features["projection_status"] = []
        features["projection_confidence"] = []
        return features[FEATURE_COLUMNS]
    if homographies is None:
        features["projection_status"] = "degraded_image_normalized"
        features["projection_confidence"] = 0.25
        return features[FEATURE_COLUMNS]
    return ProjectionQualityAnnotator().annotate(features, homographies)[
        FEATURE_COLUMNS
    ]


def _write_model_features(features: pd.DataFrame, path: Path) -> None:
    frame = (
        features[MODEL_FEATURE_COLUMNS]
        if not features.empty
        else pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    )
    write_dataframe_artifact(frame, path)


def _run_xg_models(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    features: pd.DataFrame,
    refined: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    model_features = output_dir / "08_video_features_model.csv"
    if features.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    run_video_only_xg_experiment(
        features_path=model_features,
        output_dir=output_dir / "09_xg_by_method",
        reference_path=None,
        group_id=config.name,
        model_kinds=config.xg.models,
    )
    rows = []
    for model_kind in config.xg.models:
        prediction_path = (
            output_dir / "09_xg_by_method" / model_kind.value / "video_only_shots.csv"
        )
        predictions = pd.read_csv(prediction_path)
        for row in predictions.itertuples(index=False):
            rows.append(
                {
                    "shot_id": row.shot_id,
                    "frame_index": int(row.frame_index),
                    "method": model_kind.value,
                    "xg": float(row.xg),
                }
            )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    calibrated = _calibrated_xg_predictions(
        config, features, refined, timeline, predictions
    )
    if calibrated.empty:
        return predictions
    return pd.concat([predictions, calibrated], ignore_index=True)[PREDICTION_COLUMNS]


def _predictions_stage(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    features: pd.DataFrame,
    refined: pd.DataFrame,
    timeline: pd.DataFrame,
    resume_from: str | None,
    force_stage: str | None,
) -> pd.DataFrame:
    path = output_dir / "09_predictions.csv"
    if _should_resume_from_checkpoint("09_predictions", resume_from, force_stage):
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot resume from {resume_from}: missing checkpoint {path}"
            )
        return read_dataframe_artifact(path)
    if path.exists() and not _should_force("09_predictions", force_stage):
        return read_dataframe_artifact(path)
    predictions = _run_xg_models(config, output_dir, features, refined, timeline)
    write_dataframe_artifact(predictions, path)
    return predictions


def _calibrated_xg_predictions(
    config: VideoOnlyXgEndToEndConfig,
    features: pd.DataFrame,
    refined: pd.DataFrame,
    timeline: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    variant = config.xg.calibration_variant
    if variant == VideoXgCalibrationVariant.NONE or features.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    reference = _reference_for_tuning(config, timeline)
    reference_by_candidate = _reference_by_candidate(refined, reference)
    if variant == VideoXgCalibrationVariant.COEFFICIENT_FIT:
        return FormulaCoefficientCalibrator().predict(features, reference_by_candidate)
    if variant == VideoXgCalibrationVariant.ISOTONIC_PLATT:
        return IsotonicXgCalibrator().calibrate(predictions, reference_by_candidate)
    if variant == VideoXgCalibrationVariant.QUALITY_AWARE_ENSEMBLE:
        return QualityAwareXgEnsemble().predict(predictions, features)
    if variant == VideoXgCalibrationVariant.NEURAL_VIDEO_XG:
        return NeuralVideoXgCalibrator().predict(features, reference_by_candidate)
    if variant == VideoXgCalibrationVariant.DATABALLPY_SIMPLE_XG:
        return DataBallPySimpleXgBaseline().predict(features)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _evaluate(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    refined: pd.DataFrame,
    predictions: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict[str, float]:
    if (
        config.evaluation.reference_events is None
        or not config.evaluation.reference_events.exists()
    ):
        write_dataframe_artifact(
            pd.DataFrame(columns=PER_SHOT_EVAL_COLUMNS),
            output_dir / "10_per_shot_eval.csv",
        )
        write_dataframe_artifact(
            pd.DataFrame(columns=METHOD_METRICS_COLUMNS),
            output_dir / "10_method_metrics.csv",
        )
        return {"predicted_shots": float(len(refined)), "reference_shots": 0.0}
    reference = _reference_shots(config.evaluation.reference_events, timeline)
    per_eval = _match_predictions(reference, refined, predictions)
    write_dataframe_artifact(per_eval, output_dir / "10_per_shot_eval.csv")
    method_metrics = _method_metrics(per_eval)
    write_dataframe_artifact(method_metrics, output_dir / "10_method_metrics.csv")
    metrics = {
        "predicted_shots": float(len(refined)),
        "reference_shots": float(len(reference)),
    }
    detection_errors = _detection_time_errors(reference, refined)
    for tolerance in config.evaluation.tolerances_seconds:
        hits = sum(abs(error) <= tolerance for error in detection_errors)
        metrics[f"hit@{tolerance}s"] = (
            float(hits / len(reference)) if len(reference) else 0.0
        )
    metrics["temporal_mae_seconds"] = (
        float(np.mean(np.abs(detection_errors))) if detection_errors else 0.0
    )
    return metrics


def _evaluation_stage(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    refined: pd.DataFrame,
    predictions: pd.DataFrame,
    timeline: pd.DataFrame,
    resume_from: str | None,
    force_stage: str | None,
) -> dict[str, float]:
    path = output_dir / "10_metrics.json"
    if _should_resume_from_checkpoint("10_evaluation", resume_from, force_stage):
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot resume from {resume_from}: missing checkpoint {path}"
            )
        return dict(read_json_artifact(path))
    if path.exists() and not _should_force("10_evaluation", force_stage):
        return dict(read_json_artifact(path))
    metrics = _evaluate(config, output_dir, refined, predictions, timeline)
    write_json_artifact(metrics, path)
    return metrics


def _report_stage(
    config: VideoOnlyXgEndToEndConfig,
    output_dir: Path,
    metrics: dict[str, float],
    resume_from: str | None,
    force_stage: str | None,
) -> None:
    path = output_dir / "final_report.md"
    if _should_resume_from_checkpoint("final_report", resume_from, force_stage):
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot resume from {resume_from}: missing checkpoint {path}"
            )
        return
    if path.exists() and not _should_force("final_report", force_stage):
        return
    _write_report(config, output_dir, metrics)


def _reference_shots(path: Path, timeline: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(path)
    shots = (
        events[events["type"].eq("Shot")].copy().sort_values(["period", "timestamp"])
    )
    starts = {
        int(row.part_index) + 1: float(row.start_seconds)
        for row in timeline.itertuples(index=False)
    }
    rows = []
    for _, row in shots.iterrows():
        seconds = _timestamp_seconds(str(row["timestamp"]))
        period = int(row["period"])
        rows.append(
            {
                "reference_id": str(row["id"]),
                "period": period,
                "reference_seconds": starts.get(period, 0.0) + seconds,
                "reference_xg": float(row["shot_statsbomb_xg"]),
                "is_goal": 1.0 if str(row["shot_outcome"]) == "Goal" else 0.0,
                "team": str(row["team"]),
            }
        )
    return pd.DataFrame(rows)


def _reference_for_tuning(
    config: VideoOnlyXgEndToEndConfig, timeline: pd.DataFrame
) -> pd.DataFrame:
    if (
        config.evaluation.reference_events is None
        or not config.evaluation.reference_events.exists()
    ):
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
    return _reference_shots(config.evaluation.reference_events, timeline)


def _reference_by_candidate(
    refined: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    if refined.empty or reference.empty:
        return pd.DataFrame(columns=["shot_id", "reference_xg", "is_goal"])
    rows = []
    for candidate in refined.itertuples(index=False):
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


def _match_predictions(
    reference: pd.DataFrame, refined: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    if reference.empty or refined.empty or predictions.empty:
        return pd.DataFrame(columns=PER_SHOT_EVAL_COLUMNS)
    rows = []
    for ref in reference.itertuples(index=False):
        if refined.empty:
            continue
        nearest_idx = (refined["global_seconds"] - ref.reference_seconds).abs().idxmin()
        shot = refined.loc[nearest_idx]
        shot_predictions = predictions[predictions["shot_id"].eq(shot["shot_id"])]
        for prediction in shot_predictions.itertuples(index=False):
            rows.append(
                {
                    "reference_id": ref.reference_id,
                    "shot_id": shot["shot_id"],
                    "method": prediction.method,
                    "reference_seconds": ref.reference_seconds,
                    "predicted_seconds": float(shot["global_seconds"]),
                    "time_error_seconds": float(
                        shot["global_seconds"] - ref.reference_seconds
                    ),
                    "reference_xg": ref.reference_xg,
                    "is_goal": ref.is_goal,
                    "predicted_xg": prediction.xg,
                    "abs_xg_error": abs(prediction.xg - ref.reference_xg),
                    "team": ref.team,
                }
            )
    return pd.DataFrame(rows, columns=PER_SHOT_EVAL_COLUMNS)


def _method_metrics(per_eval: pd.DataFrame) -> pd.DataFrame:
    if per_eval.empty:
        return pd.DataFrame(columns=METHOD_METRICS_COLUMNS)
    rows = []
    for method, group in per_eval.groupby("method"):
        rows.append(
            {
                "method": method,
                "matched_shots": float(len(group)),
                "mae_vs_reference_xg": float(group["abs_xg_error"].mean()),
                "rmse_vs_reference_xg": float(
                    np.sqrt(
                        ((group["predicted_xg"] - group["reference_xg"]) ** 2).mean()
                    )
                ),
                "total_predicted_xg": float(group["predicted_xg"].sum()),
                "total_reference_xg": float(group["reference_xg"].sum()),
                "total_xg_error": float(
                    group["predicted_xg"].sum() - group["reference_xg"].sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_METRICS_COLUMNS)


def _write_report(
    config: VideoOnlyXgEndToEndConfig, output_dir: Path, metrics: dict[str, float]
) -> None:
    output_dir.joinpath("previews").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("failure_cases").mkdir(parents=True, exist_ok=True)
    metrics_lines = [
        f"- `{key}`: {value:.4f}" for key, value in sorted(metrics.items())
    ]
    report = [
        "# Video-Only xG End-to-End Report",
        "",
        f"Run: `{config.name}`",
        "",
        "## Metrics",
        "",
        *metrics_lines,
        "",
        "## Runtime Contract",
        "",
        "StatsBomb/SoccerNet references are loaded only in the evaluation stage after predictions are written.",
        "Homography currently uses `degraded_image_normalized` fallback unless a calibration backend is enabled.",
    ]
    (output_dir / "final_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    write_dataframe_artifact(
        pd.DataFrame(columns=["team", "method", "predicted_xg", "reference_xg"]),
        output_dir / "team_summary.csv",
    )


def _result(
    output_dir: Path, metrics: dict[str, float] | None = None
) -> VideoOnlyXgRunResult:
    artifacts = tuple(path for path in sorted(output_dir.rglob("*")) if path.is_file())
    return VideoOnlyXgRunResult(
        output_dir=output_dir, artifacts=artifacts, metrics=metrics or {}
    )


def _write_run_config(config: VideoOnlyXgEndToEndConfig, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


def _stopped(stage: str, stop_after: str | None) -> bool:
    return stop_after == stage


def _should_force(stage: str, force_stage: str | None) -> bool:
    if force_stage is None:
        return False
    return STAGES.index(stage) >= STAGES.index(force_stage)


def _should_resume_from_checkpoint(
    stage: str, resume_from: str | None, force_stage: str | None
) -> bool:
    if resume_from is None or _should_force(stage, force_stage):
        return False
    return STAGES.index(stage) <= STAGES.index(resume_from)


def _read_stage_dataframe(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == "json":
        return pd.DataFrame(read_json_artifact(path))
    return read_dataframe_artifact(path)


def _video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return fps, width, height, frame_count
    finally:
        capture.release()


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    if not ok:
        return None
    return np.asarray(image, dtype=np.uint8)


def _image_to_pitch(
    image_x: float, image_y: float, image_width: int, image_height: int
) -> tuple[float, float]:
    return (
        min(max(image_x / max(image_width, 1), 0.0), 1.0) * 105.0,
        min(max(image_y / max(image_height, 1), 0.0), 1.0) * 68.0,
    )


def _nearest_distance(x: float, y: float, tracks: pd.DataFrame) -> float | None:
    if tracks.empty:
        return None
    distances = np.hypot(tracks["pitch_x"] - x, tracks["pitch_y"] - y)
    return float(distances.min())


def _count_in_cone(
    shot_x: float, shot_y: float, goal_x: float, goal_y: float, tracks: pd.DataFrame
) -> int:
    if tracks.empty:
        return 0
    dx = goal_x - shot_x
    dy = goal_y - shot_y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0
    count = 0
    for track in tracks.itertuples(index=False):
        t = ((track.pitch_x - shot_x) * dx + (track.pitch_y - shot_y) * dy) / length_sq
        if not 0.0 < t < 1.0:
            continue
        projected_x = shot_x + t * dx
        projected_y = shot_y + t * dy
        if math.hypot(track.pitch_x - projected_x, track.pitch_y - projected_y) <= 2.5:
            count += 1
    return count


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _validate_stage_name(stage: str | None, argument_name: str) -> None:
    if stage is not None and stage not in STAGES:
        raise ValueError(f"{argument_name} must be one of {', '.join(STAGES)}.")


def _detection_time_errors(
    reference: pd.DataFrame, refined: pd.DataFrame
) -> list[float]:
    if reference.empty or refined.empty:
        return []
    errors = []
    for ref in reference.itertuples(index=False):
        nearest_idx = (refined["global_seconds"] - ref.reference_seconds).abs().idxmin()
        shot = refined.loc[nearest_idx]
        errors.append(float(shot["global_seconds"] - ref.reference_seconds))
    return errors


def _sample_width(sampled: pd.DataFrame, global_frame_index: int) -> int:
    row = sampled[sampled["global_frame_index"] == global_frame_index].iloc[0]
    return int(row["width"])


def _sample_height(sampled: pd.DataFrame, global_frame_index: int) -> int:
    row = sampled[sampled["global_frame_index"] == global_frame_index].iloc[0]
    return int(row["height"])
