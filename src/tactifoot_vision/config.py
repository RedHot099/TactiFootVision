"""YAML configuration for the inference pipeline.

A config names components by their registry keys, so the same file format covers
every backend::

    detector: {type: yolo, weights: ../models/football_yolo11m.pt, conf: 0.3}
    keypoints: {type: yolo_pose, weights: ../models/pitch_yolov8n_pose.pt}
    tracker: {type: bytetrack}
    teams: {embedder: siglip}

Relative paths are resolved against the YAML file's folder.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from tactifoot_vision.models import Model
    from tactifoot_vision.pipeline import Pipeline
    from tactifoot_vision.teams import TeamClassifier


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    """A registered model (``tactifoot_vision.models.available_models()``)."""

    type: str
    weights: Path | str | None = None
    conf: float | None = Field(None, ge=0, le=1)
    device: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> "Model":
        from tactifoot_vision.models import load_model

        explicit = {"conf": self.conf, "device": self.device}
        kwargs = self.options | {k: v for k, v in explicit.items() if v is not None}
        return load_model(self.type, self.weights, **kwargs)


class TrackerConfig(_Section):
    """A registered tracker (``bytetrack``, ``sam2``) and its keyword arguments."""

    type: str = "bytetrack"
    options: dict[str, Any] = Field(default_factory=dict)


class TeamsConfig(_Section):
    embedder: str = "siglip"
    n_teams: int = Field(2, ge=2)
    reducer: str | None = "umap"
    device: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> "TeamClassifier":
        from tactifoot_vision.teams import TeamClassifier

        return TeamClassifier(
            embedder=self.embedder,
            n_teams=self.n_teams,
            reducer=self.reducer,
            device=self.device,
            **self.options,
        )


class PitchConfig(_Section):
    length: float = Field(105.0, gt=0)
    width: float = Field(68.0, gt=0)


class HomographyConfig(_Section):
    min_confidence: float = Field(0.5, ge=0, le=1)
    smoothing_window: int = Field(3, ge=1)
    ransac_threshold: float = Field(10.0, gt=0)
    max_age: int | None = Field(50, ge=0)


class PipelineConfig(_Section):
    """Everything :class:`~tactifoot_vision.pipeline.Pipeline` needs, loadable from YAML.

    ``options`` holds the remaining ``Pipeline`` keyword arguments
    (``include_classes``, ``team_sample_stride``, ``ball_max_speed``, ...).
    """

    detector: ModelConfig
    keypoints: ModelConfig | None = None
    tracker: TrackerConfig | None = Field(default_factory=TrackerConfig)
    teams: TeamsConfig | None = None
    pitch: PitchConfig = Field(default_factory=PitchConfig)
    homography: HomographyConfig = Field(default_factory=HomographyConfig)
    options: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> "Pipeline":
        from tactifoot_vision.pipeline import Pipeline
        from tactifoot_vision.pitch import HomographyEstimator, SoccerPitch
        from tactifoot_vision.tracking import create_tracker

        pitch = SoccerPitch(self.pitch.length, self.pitch.width)
        return Pipeline(
            detector=self.detector.build(),
            keypoint_model=self.keypoints.build() if self.keypoints else None,
            tracker=create_tracker(self.tracker.type, **self.tracker.options)
            if self.tracker
            else None,
            team_classifier=self.teams.build() if self.teams else None,
            pitch=pitch,
            homography=HomographyEstimator(pitch, **self.homography.model_dump()),
            **self.options,
        )


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a pipeline YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return PipelineConfig.model_validate(_resolve_paths(data, path.parent.absolute()))


def _resolve_paths(data: Any, base: Path) -> Any:
    """Make ``weights`` and ``*_path`` / ``*_dir`` / checkpoint / config values absolute."""
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            if (
                isinstance(value, str)
                and _is_path_key(key)
                and not Path(value).is_absolute()
            ):
                candidate = base / value
                # Bare names ("yolo11n.pt") and package-relative configs
                # ("configs/sam2.1/...") are left for the backend to resolve.
                if candidate.exists() or value.startswith(("./", "../")):
                    value = str(candidate)
            resolved[key] = _resolve_paths(value, base)
        return resolved
    if isinstance(data, list):
        return [_resolve_paths(v, base) for v in data]
    return data


def _is_path_key(key: str) -> bool:
    return key in {"weights", "checkpoint", "config", "repo_dir"} or key.endswith(
        ("_path", "_dir")
    )
