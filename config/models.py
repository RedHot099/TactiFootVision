# config/models.py
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, Literal
import yaml
import logging

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class DetectionModelType(str, Enum):
    YOLO = "yolo"
    RFDETR = "rfdetr"


class KeypointModelType(str, Enum):
    YOLO_POSE = "yolo_pose"


class PathsConfig(BaseModel):
    input_video: Path
    output_video: Path = Path("data/output/output_video.mp4")
    model_dir: Path = Path("models/")

    @field_validator("input_video", "model_dir", mode="before")
    @classmethod
    def check_path_string(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("Path cannot be empty")
        return v


class DetectionConfig(BaseModel):
    model_type: DetectionModelType
    checkpoint_path: Optional[Path] = None
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.5, ge=0.0, le=1.0)
    classes: Dict[str, int] = Field(
        default_factory=lambda: {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
    )

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def check_checkpoint_path_string(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("checkpoint_path cannot be empty")
        return v


class TrackingConfig(BaseModel):
    enabled: bool = True
    track_activation_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    lost_track_buffer: Optional[int] = Field(None, ge=1)
    minimum_matching_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    frame_rate: Optional[int] = Field(None, ge=1)
    minimum_consecutive_frames: Optional[int] = Field(None, ge=1)


class KeypointsConfig(BaseModel):
    enabled: bool = True
    model_type: KeypointModelType = KeypointModelType.YOLO_POSE
    checkpoint_path: Optional[Path] = None
    confidence_threshold: float = 0.5

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def check_kp_checkpoint_path_string(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("checkpoint_path cannot be empty")
        return v


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


class ProcessingConfig(BaseModel):
    period: Literal[1, 2] = 1  # Which period to process
    period_start_time_seconds: float = Field(
        0.0, ge=0
    )  # Start time of this period in seconds


class TrainingDetectionConfig(BaseModel):
    base_model: str
    dataset_path: Path
    dataset_format: Literal["yolo", "coco"] = "yolo"
    ultralytics_assets_tag: str = "v8.0.0"
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 2
    imgsz: int = 640
    learning_rate: float = 0.001
    optimizer: str = "auto"
    project_name: str = "tactifoot_training"
    run_name: str = "detect_run"
    device: Optional[str] = None
    plots: bool = True

    @field_validator(
        "dataset_path", "base_model", "ultralytics_assets_tag", mode="before"
    )
    @classmethod
    def check_string_non_empty(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("String value cannot be empty")
        return v

    @model_validator(mode="after")
    def check_dataset_path_format(self) -> "TrainingDetectionConfig":
        if self.dataset_format == "yolo":
            if (
                not self.dataset_path.is_file()
                or self.dataset_path.suffix.lower() != ".yaml"
            ):
                pass
        elif self.dataset_format == "coco":
            if not self.dataset_path.is_dir():
                pass
        return self


class TrainingKeypointsConfig(BaseModel):
    dataset_path: Path
    base_model: str
    ultralytics_assets_tag: str = "v8.0.0"
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    learning_rate: float = 0.001
    optimizer: str = "auto"
    project_name: str = "tactifoot_kp_training"
    run_name: str = "yolo_pose_run"
    device: Optional[str] = None
    plots: bool = True

    @field_validator("dataset_path", mode="before")
    @classmethod
    def check_kp_dataset_path_string(cls, v: Any) -> Any:
        if isinstance(v, str) and not v:
            raise ValueError("dataset_path cannot be empty")
        return Path(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_yaml_and_kpt_shape(self) -> "TrainingKeypointsConfig":
        yaml_path = self.dataset_path
        if not yaml_path.is_file() or yaml_path.suffix.lower() != ".yaml":
            raise ValueError(f"dataset_path must be a .yaml file, got: {yaml_path}")
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if "kpt_shape" not in data:
                pass
        except FileNotFoundError:
            raise ValueError(f"data.yaml file not found at: {yaml_path}")
        except Exception as e:
            raise ValueError(f"Could not process data.yaml file: {yaml_path}") from e
        return self


class TrainingConfig(BaseModel):
    detection: Optional[TrainingDetectionConfig] = None
    keypoints: Optional[TrainingKeypointsConfig] = None


class Config(BaseModel):
    project_name: str = "tactifoot_vision"
    logging_level: str = "INFO"
    paths: PathsConfig
    detection: DetectionConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    keypoints: KeypointsConfig = Field(default_factory=KeypointsConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    visualization: PitchVisualizerConfig = Field(default_factory=PitchVisualizerConfig)
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig
    )  # Added processing config
    training: Optional[TrainingConfig] = None

    class Config:
        extra = "forbid"
