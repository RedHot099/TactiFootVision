from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tactifoot_vision.enums import (
    BallReconstructionMethod,
    DatasetFormat,
    DatasetSource,
    DetectionBackend,
    Device,
    ExperimentKind,
    HomographyMethod,
    KeypointBackend,
    Sam2OutputBoxMode,
    ShotDetectorKind,
    TeamAssignmentClusterer,
    TeamAssignmentCropMethod,
    TeamAssignmentEmbedding,
    TeamAssignmentReducer,
    TrackingBackend,
    XgModelKind,
)


class PathsConfig(BaseModel):
    input: Path | None = None
    output_dir: Path = Path("results")
    model_dir: Path = Path("models")


class DetectionConfig(BaseModel):
    backend: DetectionBackend = DetectionBackend.FAKE
    checkpoint: Path | None = None
    confidence: float = Field(0.3, ge=0.0, le=1.0)
    nms: float = Field(0.5, ge=0.0, le=1.0)
    classes: dict[str, int] = Field(
        default_factory=lambda: {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
    )
    include_labels: tuple[str, ...] | None = None
    per_class_confidence: dict[str, float] | None = None
    device: str | None = None

    @property
    def id_to_name(self) -> dict[int, str]:
        return {value: key for key, value in self.classes.items()}


class DetectionTrainingConfig(BaseModel):
    data: Path
    epochs: int = Field(50, ge=1)
    batch_size: int = Field(8, ge=1)
    image_size: int = Field(640, ge=32)
    learning_rate: float | None = Field(None, gt=0.0)
    output_dir: Path | None = None
    run_name: str | None = None
    base_model: str | None = None
    dataset_format: DatasetFormat = DatasetFormat.YOLO
    dataset_source: DatasetSource = DatasetSource.FILESYSTEM
    converted_dataset_dir: Path | None = None
    valid_fraction: float = Field(0.2, ge=0.0, lt=1.0)
    every_nth_frame: int = Field(1, ge=1)
    max_sequences: int | None = Field(None, ge=1)
    symlink_images: bool = True
    device: str | None = None
    grad_accum_steps: int = Field(2, ge=1)
    num_workers: int | None = Field(None, ge=0)
    multi_scale: bool | None = None
    early_stopping: bool = False
    early_stopping_patience: int | None = Field(None, ge=1)
    early_stopping_min_delta: float | None = Field(None, ge=0.0)
    early_stopping_use_ema: bool | None = None
    save_checkpoint_path: Path | None = None


class SAM2Config(BaseModel):
    checkpoint: Path
    model_config_path: Path
    device: Device = Device.AUTO
    max_side: int | None = Field(None, ge=128)
    max_objects: int | None = Field(None, ge=1)
    reseed_interval: int | None = Field(None, ge=1)
    reseed_iou: float = Field(0.3, ge=0.0, le=1.0)
    output_box_mode: Sam2OutputBoxMode = Sam2OutputBoxMode.MASK
    mask_threshold: float = 0.0
    mask_filter_distance: float = Field(0.0, ge=0.0)
    mask_open: int = Field(0, ge=0)
    mask_close: int = Field(0, ge=0)
    bbox_ema_alpha: float = Field(0.0, ge=0.0, le=1.0)
    min_mask_area: float = Field(100.0, ge=0.0)


class TrackingConfig(BaseModel):
    backend: TrackingBackend = TrackingBackend.FAKE
    frame_rate: int = Field(25, ge=1)
    track_activation_threshold: float = Field(0.25, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(30, ge=1)
    minimum_matching_threshold: float = Field(0.8, ge=0.0, le=1.0)
    minimum_consecutive_frames: int = Field(1, ge=1)
    sam2: SAM2Config | None = None


class KeypointsConfig(BaseModel):
    backend: KeypointBackend = KeypointBackend.NONE
    checkpoint: Path | None = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ProjectionConfig(BaseModel):
    enabled: bool = False
    min_keypoint_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_keypoints: int = Field(4, ge=4)
    smoothing_window: int = Field(15, ge=1)
    project_ball: bool = True


class TeamAssignmentConfig(BaseModel):
    enabled: bool = False
    crop_ratio: float = Field(0.6, gt=0.0, le=1.0)
    crop_method: TeamAssignmentCropMethod = TeamAssignmentCropMethod.CENTER
    embedding: TeamAssignmentEmbedding = TeamAssignmentEmbedding.COLOR_HISTOGRAM
    reducer: TeamAssignmentReducer = TeamAssignmentReducer.NONE
    clusterer: TeamAssignmentClusterer = TeamAssignmentClusterer.KMEANS
    clusters: int = Field(2, ge=2)
    batch_size: int = Field(32, ge=1)
    device: Device = Device.AUTO
    cache_dir: Path | None = None
    random_state: int = 0


class ExportConfig(BaseModel):
    pipeline_csv: Path | None = None
    mot: Path | None = None
    statsbomb: Path | None = None


class BallReconstructionConfig(BaseModel):
    method: BallReconstructionMethod = BallReconstructionMethod.LINEAR
    max_speed_pixels_per_frame: float | None = Field(None, gt=0.0)


class ShotDetectionConfig(BaseModel):
    kind: ShotDetectorKind = ShotDetectorKind.KINEMATIC
    window_before: int = Field(64, ge=0)
    window_after: int = Field(32, ge=0)
    max_candidates: int = Field(1, ge=0)
    min_speed_pixels_per_frame: float = Field(0.0, ge=0.0)
    use_soccernet_metadata: bool = False


class XgEstimatorConfig(BaseModel):
    model: XgModelKind = XgModelKind.GEOMETRY
    image_width: int = Field(1920, ge=1)
    image_height: int = Field(1080, ge=1)
    attacking_goal_x: float | None = None
    penalty_xg: float = Field(0.76, ge=0.0, le=1.0)


class VideoXgConfig(BaseModel):
    ball: BallReconstructionConfig = Field(
        default_factory=lambda: BallReconstructionConfig()
    )
    shots: ShotDetectionConfig = Field(default_factory=lambda: ShotDetectionConfig())
    xg: XgEstimatorConfig = Field(default_factory=lambda: XgEstimatorConfig())
    group_by_game_id: bool = True
    write_shots_csv: bool = True
    write_summary_json: bool = True


class HomographyComparisonConfig(BaseModel):
    split: str = "valid"
    methods: tuple[HomographyMethod, ...] = (
        HomographyMethod.CURRENT_YOLOPOSE_7PT,
        HomographyMethod.ORACLE_GSR_LINES_RANSAC,
    )
    external_homographies: tuple[Path, ...] = ()
    confidence_iterations: int = Field(200, ge=0)
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "median_error_m": 0.35,
            "p90_error_m": 0.25,
            "success@2m": 0.15,
            "availability": 0.15,
            "temporal_jitter": 0.10,
        }
    )


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "tactifoot_pipeline"
    paths: PathsConfig = Field(default_factory=lambda: PathsConfig())
    detection: DetectionConfig = Field(default_factory=lambda: DetectionConfig())
    tracking: TrackingConfig = Field(default_factory=lambda: TrackingConfig())
    keypoints: KeypointsConfig = Field(default_factory=lambda: KeypointsConfig())
    projection: ProjectionConfig = Field(default_factory=lambda: ProjectionConfig())
    team_assignment: TeamAssignmentConfig = Field(
        default_factory=lambda: TeamAssignmentConfig()
    )
    export: ExportConfig = Field(default_factory=lambda: ExportConfig())

    @field_validator("paths", mode="before")
    @classmethod
    def _default_paths(cls, value: object) -> object:
        return value or {}


class ExperimentConfig(BaseModel):
    name: str
    kind: ExperimentKind
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    video_xg: VideoXgConfig = Field(default_factory=VideoXgConfig)
    homography_comparison: HomographyComparisonConfig = Field(
        default_factory=lambda: HomographyComparisonConfig()
    )
    max_frames: int | None = Field(None, ge=1)
    output_dir: Path = Path("results/experiments")
    soccernet_root: Path | None = None
    sequence_names: tuple[str, ...] | None = None
    max_sequences: int | None = Field(None, ge=1)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    write_mot: bool = True
    write_metrics_json: bool = True
