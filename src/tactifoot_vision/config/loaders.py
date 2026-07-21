from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from tactifoot_vision.config.schemas import ExperimentConfig, PipelineConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")
    return data


def load_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    return schema.model_validate(load_yaml(path))


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return load_config(path, PipelineConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


def dump_config(config: BaseModel, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.model_dump(mode="json"), file, sort_keys=False)
    return output
