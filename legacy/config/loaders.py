# config/loaders.py
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

# Use relative import within the same package
from .models import Config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def load_config(config_path: Optional[Path] = None) -> Config:
    """Loads and validates configuration from a YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        logger.info(f"Using default config: {config_path}")
    elif not isinstance(config_path, Path):
        config_path = Path(config_path)

    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        # Basic path resolution (relative to config file's directory)
        config_dir = config_path.parent
        _resolve_paths_recursive(config_data, config_dir)

        validated_config = Config(**config_data)
        logger.info("Configuration loaded and validated successfully.")
        return validated_config

    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML file: {config_path}")
        raise e
    except ValidationError as e:
        logger.exception(f"Configuration validation error: {config_path}")
        # Log details of validation errors
        # logger.error(e.json(indent=2)) # Requires pydantic v2 style
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error loading config: {config_path}")
        raise e


def _resolve_paths_recursive(data: object, base_dir: Path):
    """Recursively resolves string paths ending in '_path' or '_dir'."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and (
                key.endswith("_path") or key.endswith("_dir")
            ):
                path = Path(value)
                if not path.is_absolute():
                    data[key] = str((base_dir / path).resolve())  # Store as string
            else:
                _resolve_paths_recursive(value, base_dir)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _resolve_paths_recursive(item, base_dir)
