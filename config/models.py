from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, Literal
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class DetectionModelType(str, Enum):
    YOLO = "yolo"
    RFDETR = "rfdetr"


class KeypointModelType(str, Enum):
    YOLO_POSE = "yolo_pose"


class FrameVisualizationStyle(str, Enum):
    STANDARD = "standard"
    VIDEO_GAME = "video_game"


class PathsConfig(BaseModel):
    input_video: Path
    output_video: Path = Path("data/output/output_video.mp4")
    model_dir: Path = Path("models/")
    statsbomb_input_csv: Path = Path("data/output/statsbomb_merged.csv")
    pipeline_input_csv: Path = Path("data/output/pipeline_output.csv")
    merged_output_csv: Path = Path("data/output/merged_output.csv")

    @field_validator(
        "input_video",
        "model_dir",
        "statsbomb_input_csv",
        "pipeline_input_csv",
        "merged_output_csv",
        mode="before",
    )
    @classmethod
    def check_path_string(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        raise ValueError("Path must be a non-empty string or a Path object")


class DetectionConfig(BaseModel):
    model_type: DetectionModelType
    checkpoint_path: Optional[Path] = None
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.5, ge=0.0, le=1.0)
    classes: Dict[str, int] = Field(
        default_factory=lambda: {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
    )
    include_labels: Optional[list[str]] = None

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def check_checkpoint_path_string(cls, v: Any) -> Optional[Path]:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        if v is None or (isinstance(v, str) and not v):
            return None
        raise TypeError("checkpoint_path must be a string, Path, or None")

    @field_validator("include_labels", mode="before")
    @classmethod
    def _normalize_include_labels(cls, v: Any) -> Optional[list[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return [stripped] if stripped else None
        if isinstance(v, (list, tuple, set)):
            cleaned: list[str] = []
            for item in v:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        cleaned.append(stripped)
                else:
                    raise TypeError(
                        "include_labels entries must be strings if provided"
                    )
            return cleaned or None
        raise TypeError("include_labels must be a string, sequence of strings, or None")


class TrackingConfig(BaseModel):
    enabled: bool = True
    backend: Literal["bytetrack", "sam2"] = "bytetrack"
    track_activation_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    lost_track_buffer: Optional[int] = Field(None, ge=1)
    minimum_matching_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    frame_rate: Optional[int] = Field(None, ge=1)
    minimum_consecutive_frames: Optional[int] = Field(None, ge=1)
    sam2: Optional["SAM2Config"] = None


class SAM2Config(BaseModel):
    checkpoint_path: Optional[Path] = None
    config_path: Optional[Path] = None
    mask_filter_distance: float = Field(300.0, ge=0.0)
    reseed_interval: Optional[int] = Field(None, ge=1)
    reseed_iou_threshold: float = Field(0.3, ge=0.0, le=1.0)

    @field_validator("checkpoint_path", "config_path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Optional[Path]:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        if v is None or (isinstance(v, str) and not v):
            return None
        raise TypeError("Value must be a string, Path, or None")


class KeypointsConfig(BaseModel):
    enabled: bool = True
    model_type: KeypointModelType = KeypointModelType.YOLO_POSE
    checkpoint_path: Optional[Path] = None
    confidence_threshold: float = 0.5

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def check_kp_checkpoint_path_string(cls, v: Any) -> Optional[Path]:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        if v is None or (isinstance(v, str) and not v):
            return None
        raise TypeError("checkpoint_path must be a string, Path, or None")


class GeometryConfig(BaseModel):
    min_keypoint_confidence_for_homography: float = Field(0.5, ge=0.0, le=1.0)
    homography_smoothing_window: int = Field(5, ge=1)
    target_pitch_length: float = Field(100.0, gt=0)
    target_pitch_width: float = Field(100.0, gt=0)
    ball_outlier_threshold_percent: float = Field(5.0, gt=0, le=100)


class PitchVisualizerConfig(BaseModel):
    enabled: bool = True
    pitch_color: str = "#22312b"
    line_color: str = "#ffffff"
    line_thickness: int = 1
    path_color: str = "#ffff00"
    path_thickness: int = 1
    player_dot_radius: int = 5
    ball_dot_radius: int = 4
    player_color_default: str = "#FFFFFF"
    ball_color: str = "#FFFF00"
    team_color_0: str = "#00BFFF"
    team_color_1: str = "#FF1493"
    canvas_width_px: int = 400
    canvas_padding_px: int = 10
    overlay: bool = True
    overlay_width_fraction: float = Field(0.25, ge=0.1, le=0.5)
    overlay_position: Literal[
        "bottom-center",
        "bottom-left",
        "bottom-right",
        "top-center",
        "top-left",
        "top-right",
    ] = "bottom-center"
    overlay_padding: int = 10
    overlay_alpha: float = Field(0.8, ge=0.0, le=1.0)
    draw_projected_pitch: bool = True
    draw_segmentation_masks: bool = False
    draw_bounding_boxes: bool = True
    draw_keypoints: bool = True
    draw_pitch_detection: bool = True
    frame_style: FrameVisualizationStyle = FrameVisualizationStyle.STANDARD


class ProcessingConfig(BaseModel):
    period: Literal[1, 2] = 1
    period_start_time_seconds: float = Field(0.0, ge=0)


class SiglipTeamClassificationOptions(BaseModel):
    model_name: str = "google/siglip-base-patch16-224"
    batch_size: int = Field(32, ge=1)
    pooling: Literal["mean", "cls"] = "mean"
    use_umap: bool = True
    umap_components: int = Field(3, ge=1)
    umap_neighbors: int = Field(15, ge=2)
    umap_min_dist: float = Field(0.1, ge=0.0, le=1.0)
    umap_metric: str = "euclidean"
    umap_random_state: Optional[int] = None
    color_space: Literal["rgb", "hsv"] = "rgb"
    color_hist_bins: int = Field(16, ge=0, le=256)
    color_hist_weight: float = Field(0.2, ge=0.0)


class TeamClassificationConfig(BaseModel):
    enabled: bool = False
    warmup_frames: int = Field(150, ge=0)
    sample_stride: int = Field(5, ge=1)
    max_samples: int = Field(200, ge=2)
    crop_scale: float = Field(0.6, gt=0.0, le=1.0)
    consecutive_frames: int = Field(3, ge=1)
    embedding_model: str = "resnet18"
    num_clusters: int = Field(2, ge=2)
    center_crop_ratio: float = Field(0.3, gt=0.0, le=3)
    method: Literal["resnet", "siglip"] = "resnet"
    siglip: SiglipTeamClassificationOptions = Field(
        default_factory=SiglipTeamClassificationOptions
    )
    device: Optional[str] = None


# Shared training parameters moved to a common base
class TrainingSharedSettings(BaseModel):
    ultralytics_assets_tag: str = "v8.0.0"
    imgsz: int = 640
    learning_rate: float = 0.001
    optimizer: str = "auto"
    device: Optional[str] = None
    plots: bool = True
    project_name: Optional[str] = None
    run_name: Optional[str] = None


class TrainingDetectionConfig(TrainingSharedSettings):
    base_model: Optional[str] = None
    dataset_path: Path
    dataset_format: Literal["yolo", "coco"] = "yolo"
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 2

    @field_validator("dataset_path", mode="before")
    @classmethod
    def check_dataset_path(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        raise ValueError("dataset_path must be a non-empty string or a Path object")

    @field_validator("ultralytics_assets_tag", mode="before")
    @classmethod
    def check_assets_tag_string(cls, v: Any) -> str:
        if isinstance(v, str) and v:
            return v
        if v is None or (isinstance(v, str) and not v):
            raise ValueError("ultralytics_assets_tag cannot be empty if provided")
        raise TypeError("ultralytics_assets_tag must be a non-empty string")

    @field_validator("base_model", mode="before")
    @classmethod
    def check_base_model_string(cls, v: Any) -> Optional[str]:
        if isinstance(v, str) and not v:
            return None
        if v is None or isinstance(v, str):
            return v
        raise TypeError("base_model must be a string or None")


class TrainingKeypointsConfig(TrainingSharedSettings):
    dataset_path: Path
    base_model: Optional[str] = None
    epochs: int = 100
    batch_size: int = 16

    @field_validator("dataset_path", mode="before")
    @classmethod
    def check_kp_dataset_path(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v:
            return Path(v)
        raise ValueError(
            "dataset_path cannot be empty and must be a valid path string or Path object"
        )

    @field_validator("ultralytics_assets_tag", mode="before")
    @classmethod
    def check_kp_assets_tag_string(cls, v: Any) -> str:
        if isinstance(v, str) and v:
            return v
        if v is None or (isinstance(v, str) and not v):
            raise ValueError("ultralytics_assets_tag cannot be empty if provided")
        raise TypeError("ultralytics_assets_tag must be a non-empty string")

    @field_validator("base_model", mode="before")
    @classmethod
    def check_kp_base_model_string(cls, v: Any) -> Optional[str]:
        if isinstance(v, str) and not v:
            return None
        if v is None or isinstance(v, str):
            return v
        raise TypeError("base_model must be a string or None")


class TrainingConfig(BaseModel):
    detection: Optional[TrainingDetectionConfig] = None
    keypoints: Optional[TrainingKeypointsConfig] = None


class Config(BaseModel):
    project_name: str = "tactifoot_vision"
    logging_level: str = "INFO"
    paths: PathsConfig
    detection: Optional[DetectionConfig] = None
    keypoints: Optional[KeypointsConfig] = None
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    visualization: PitchVisualizerConfig = Field(default_factory=PitchVisualizerConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    training: Optional[TrainingConfig] = None
    team_classification: Optional[TeamClassificationConfig] = None

    @model_validator(mode="before")
    @classmethod
    def resolve_paths_in_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    class Config:
        extra = "forbid"
