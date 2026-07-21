from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


SynLocSplit = Literal["train", "val", "valid", "test", "challenge"]
SynLocImageVersion = Literal["4K", "fullres", "fullhd"]
SynLocDetectorType = Literal["yolo", "rfdetr"]
SynLocPointStrategy = Literal["bottom_center", "learned_offset"]
SynLocAuxiliaryTask = Literal["gamestate-2024", "gamestate-2025"]
BehindCameraPolicy = Literal["drop", "clip"]


def _pathify(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise TypeError("Expected a non-empty path string or Path.")


class SynLocDatasetConfig(BaseModel):
    root: Path
    split: SynLocSplit = "val"
    image_version: SynLocImageVersion = "4K"
    use_tiles: bool = True
    tile_size: int = Field(1280, ge=128)
    tile_overlap: int = Field(256, ge=0)
    person_only: bool = True
    auxiliary_roots: list[Path] = Field(default_factory=list)
    auxiliary_tasks: list[SynLocAuxiliaryTask] = Field(default_factory=list)
    max_aux_images_per_split: int | None = Field(default=None, ge=1)

    @field_validator("root", mode="before")
    @classmethod
    def _validate_root(cls, value: object) -> Path:
        return _pathify(value)

    @field_validator("auxiliary_roots", mode="before")
    @classmethod
    def _validate_auxiliary_roots(cls, value: object) -> list[Path]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise TypeError("auxiliary_roots must be a list of paths.")
        return [_pathify(item) for item in value]

    @field_validator("image_version", mode="before")
    @classmethod
    def _normalize_image_version(cls, value: object) -> SynLocImageVersion:
        if value in (None, "", "fullres"):
            return "4K"
        if value == "4k":
            return "4K"
        if value in {"4K", "fullhd"}:
            return value
        raise ValueError("image_version must be one of: 4K, fullres, fullhd.")

    @model_validator(mode="after")
    def _validate_tile_params(self) -> "SynLocDatasetConfig":
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size.")
        if self.auxiliary_tasks and self.auxiliary_roots and len(self.auxiliary_tasks) != len(self.auxiliary_roots):
            raise ValueError("auxiliary_tasks and auxiliary_roots must have the same length.")
        return self


class SynLocDetectorConfig(BaseModel):
    model_type: SynLocDetectorType = "yolo"
    checkpoint_path: Optional[Path] = None
    base_model: Optional[str] = None
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.5, ge=0.0, le=1.0)
    tile_size: int = Field(1280, ge=128)
    tile_overlap: int = Field(256, ge=0)
    person_class_ids: list[int] = Field(default_factory=lambda: [0])
    class_names: list[str] = Field(default_factory=lambda: ["person"])
    device: Optional[str] = None
    train_imgsz: int = Field(1280, ge=128)
    inference_imgsz: int = Field(1280, ge=128)
    tta_scales: list[int] = Field(default_factory=list)
    max_detections: int = Field(300, ge=1)
    class_filter: list[str] = Field(default_factory=lambda: ["player"])

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _validate_checkpoint(cls, value: object) -> Optional[Path]:
        if value is None or value == "":
            return None
        return _pathify(value)

    @field_validator("class_filter", mode="before")
    @classmethod
    def _normalize_class_filter(cls, value: object) -> list[str]:
        if value is None:
            return ["player"]
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else ["player"]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("class_filter must be a string or list of strings.")
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("class_filter entries must be strings.")
            stripped = item.strip()
            if stripped:
                result.append(stripped)
        return result or ["player"]


class SynLocProjectionConfig(BaseModel):
    point_strategy: SynLocPointStrategy = "bottom_center"
    world_nms_radius_m: float = Field(0.75, ge=0.0)
    image_nms_iou: float = Field(0.6, ge=0.0, le=1.0)
    clip_to_pitch: bool = False
    point_regressor_checkpoint: Path | None = None
    clip_margin_m: float = Field(0.0, ge=0.0)
    behind_camera_policy: BehindCameraPolicy = "drop"

    @field_validator("point_regressor_checkpoint", mode="before")
    @classmethod
    def _validate_point_regressor_checkpoint(cls, value: object) -> Path | None:
        if value is None or value == "":
            return None
        return _pathify(value)


class SynLocSubmissionConfig(BaseModel):
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    split: SynLocSplit = "challenge"
    output_dir: Path = Path("results/synloc/submissions")
    archive_name: Optional[str] = None
    zip_name: Optional[str] = None
    position_from_keypoint_index: Optional[int] = None
    topk_per_image: int | None = Field(default=None, ge=1)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: object) -> Path:
        return _pathify(value)

    @model_validator(mode="after")
    def _normalize_archive_names(self) -> "SynLocSubmissionConfig":
        if self.zip_name and not self.archive_name:
            self.archive_name = self.zip_name
        elif self.archive_name and not self.zip_name:
            self.zip_name = self.archive_name
        return self


class SynLocPrediction(BaseModel):
    image_id: int
    category_id: int = 1
    score: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(default_factory=list, min_length=4, max_length=4)
    image_point_xy: list[float] = Field(default_factory=list, min_length=2, max_length=2)
    position_on_pitch_xyz: list[float] = Field(
        default_factory=list, min_length=3, max_length=3
    )
    source_tile_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    source_scale: int | None = None
    world_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _default_world_confidence(self) -> "SynLocPrediction":
        if self.world_confidence is None:
            self.world_confidence = float(self.score)
        return self

    def bbox_xywh(self) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in self.bbox_xyxy]
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    def to_result_dict(
        self,
        prediction_id: int,
        *,
        position_from_keypoint_index: Optional[int] = None,
    ) -> dict[str, object]:
        result = {
            "id": int(prediction_id),
            "image_id": int(self.image_id),
            "category_id": int(self.category_id),
            "score": float(self.score),
            "bbox": self.bbox_xywh(),
            "area": float(self.bbox_xywh()[2] * self.bbox_xywh()[3]),
        }
        if position_from_keypoint_index is None:
            result["position_on_pitch"] = [float(v) for v in self.position_on_pitch_xyz]
        else:
            x, y = [float(v) for v in self.image_point_xy]
            # Two-keypoint format to stay compatible with the official convenience path.
            result["keypoints"] = [0.0, 0.0, 0.0, x, y, 1.0]
        return result


class SynLocConfig(BaseModel):
    dataset: SynLocDatasetConfig
    detector: SynLocDetectorConfig = Field(default_factory=SynLocDetectorConfig)
    projection: SynLocProjectionConfig = Field(default_factory=SynLocProjectionConfig)
    submission: SynLocSubmissionConfig = Field(default_factory=SynLocSubmissionConfig)
    results_path: Optional[Path] = None
    visuals_dir: Optional[Path] = None
    point_regressor_checkpoint: Optional[Path] = None
    model_dir: Path = Path("models")

    @field_validator(
        "results_path",
        "visuals_dir",
        "point_regressor_checkpoint",
        "model_dir",
        mode="before",
    )
    @classmethod
    def _validate_optional_paths(cls, value: object) -> Optional[Path]:
        if value is None or value == "":
            return None
        return _pathify(value)

    @model_validator(mode="after")
    def _sync_regressor_checkpoint(self) -> "SynLocConfig":
        if self.point_regressor_checkpoint is None and self.projection.point_regressor_checkpoint is not None:
            self.point_regressor_checkpoint = self.projection.point_regressor_checkpoint
        elif self.point_regressor_checkpoint is not None and self.projection.point_regressor_checkpoint is None:
            self.projection.point_regressor_checkpoint = self.point_regressor_checkpoint
        return self
