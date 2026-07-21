from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from tactifoot_vision.config.schemas import DetectionConfig, TrackingConfig
from tactifoot_vision.enums import (
    BallReconstructionMethod,
    VideoBallReconstructionVariant,
    VideoDetectionVariant,
    VideoProjectionVariant,
    VideoShotDetectorKind,
    VideoShotRankingVariant,
    VideoXgCalibrationVariant,
    XgModelKind,
)


class VideoOnlyDetectionStageConfig(BaseModel):
    variant: VideoDetectionVariant = VideoDetectionVariant.LEGACY
    batch_size: int = Field(8, ge=1)
    chunk_size: int = Field(500, ge=1)
    benchmark_max_frames: int | None = Field(None, ge=1)


class VideoOnlyBallConfig(BaseModel):
    method: BallReconstructionMethod = BallReconstructionMethod.KALMAN_RTS
    variant: VideoBallReconstructionVariant = (
        VideoBallReconstructionVariant.BASELINE_KALMAN
    )
    max_gap_seconds: float = Field(2.0, ge=0.0)
    max_speed_mps: float = Field(38.0, gt=0.0)


class VideoOnlyCalibrationConfig(BaseModel):
    enabled: bool = False
    fallback: str = "image_normalized"
    variant: VideoProjectionVariant = VideoProjectionVariant.DEGRADED_IMAGE_NORMALIZED
    external_homographies: Path | None = None
    last_stable_max_age_seconds: float = Field(3.0, ge=0.0)


class VideoOnlyShotDetectionConfig(BaseModel):
    kind: VideoShotDetectorKind = VideoShotDetectorKind.CONTACT_KINEMATIC
    variant: VideoShotRankingVariant = (
        VideoShotRankingVariant.BASELINE_CONTACT_KINEMATIC
    )
    min_candidate_confidence: float = Field(0.25, ge=0.0, le=1.0)
    temporal_nms_seconds: float = Field(8.0, ge=0.0)
    max_candidates: int = Field(80, ge=1)
    contact_distance_m: float = Field(2.5, gt=0.0)
    recall_floor_hit2: float = Field(0.78, ge=0.0, le=1.0)
    min_hit1: float = Field(0.45, ge=0.0, le=1.0)
    target_max_false_positives: int = Field(30, ge=0)
    max_candidates_per_half: int = Field(25, ge=1)
    post_shot_window_seconds: float = Field(1.0, gt=0.0)
    contact_pre_window_seconds: float = Field(0.5, gt=0.0)
    long_shot_distance_m: float = Field(40.0, gt=0.0)


class VideoOnlyXgModelConfig(BaseModel):
    models: tuple[XgModelKind, ...] = (
        XgModelKind.VIDEO_GEOMETRY,
        XgModelKind.VIDEO_FREEZE_CONTEXT,
        XgModelKind.VIDEO_KINEMATIC_CONTEXT,
    )
    calibration_variant: VideoXgCalibrationVariant = VideoXgCalibrationVariant.NONE


class VideoOnlyEvaluationConfig(BaseModel):
    reference_events: Path | None = None
    tolerances_seconds: tuple[float, ...] = (0.5, 1.0, 2.0)


class VideoOnlyAblationConfig(BaseModel):
    baseline_run_dir: Path | None = None
    output_dir: Path | None = None
    baseline_runtime_seconds: float = Field(1.0, gt=0.0)
    target_scan_fps: float = Field(10.0, gt=0.0)
    smoke_first_minutes_per_half: float = Field(5.0, gt=0.0)


class VideoOnlyXgEndToEndConfig(BaseModel):
    name: str = "video_only_xg_end_to_end"
    video_parts: tuple[Path, ...]
    scan_fps: float = Field(10.0, gt=0.0)
    refine_fps: float = Field(30.0, gt=0.0)
    refine_window_before_seconds: float = Field(1.5, ge=0.0)
    refine_window_after_seconds: float = Field(1.0, ge=0.0)
    output_dir: Path
    detector: DetectionConfig = Field(default_factory=lambda: DetectionConfig())
    detection: VideoOnlyDetectionStageConfig = Field(
        default_factory=lambda: VideoOnlyDetectionStageConfig()
    )
    tracking: TrackingConfig = Field(default_factory=lambda: TrackingConfig())
    calibration: VideoOnlyCalibrationConfig = Field(
        default_factory=lambda: VideoOnlyCalibrationConfig()
    )
    ball: VideoOnlyBallConfig = Field(default_factory=lambda: VideoOnlyBallConfig())
    shots: VideoOnlyShotDetectionConfig = Field(
        default_factory=lambda: VideoOnlyShotDetectionConfig()
    )
    xg: VideoOnlyXgModelConfig = Field(default_factory=lambda: VideoOnlyXgModelConfig())
    evaluation: VideoOnlyEvaluationConfig = Field(
        default_factory=lambda: VideoOnlyEvaluationConfig()
    )
    ablation: VideoOnlyAblationConfig = Field(
        default_factory=lambda: VideoOnlyAblationConfig()
    )


def load_video_only_xg_end_to_end_config(
    path: str | Path,
) -> VideoOnlyXgEndToEndConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")
    return VideoOnlyXgEndToEndConfig.model_validate(data)
