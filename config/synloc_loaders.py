from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .synloc_models import SynLocConfig

logger = logging.getLogger(__name__)


def load_synloc_config(config_path: Optional[Path] = None) -> SynLocConfig:
    if config_path is None:
        raise ValueError("config_path is required for SynLoc.")
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"SynLoc config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"SynLoc config is empty: {config_path}")

    _resolve_paths_recursive(payload, config_path.parent.resolve())
    try:
        return SynLocConfig(**payload)
    except ValidationError:
        logger.exception("SynLoc config validation failed.")
        raise


def _resolve_paths_recursive(data: object, base_dir: Path) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and (
                key.endswith("_path")
                or key.endswith("_dir")
                or key in {"root", "output_dir", "model_dir", "point_regressor_checkpoint"}
            ):
                path = Path(value)
                if not path.is_absolute():
                    data[key] = str((base_dir / path).resolve())
            elif isinstance(value, list) and key in {"auxiliary_roots"}:
                resolved: list[str] = []
                for item in value:
                    path = Path(item)
                    if not path.is_absolute():
                        path = (base_dir / path).resolve()
                    resolved.append(str(path))
                data[key] = resolved
            else:
                _resolve_paths_recursive(value, base_dir)
    elif isinstance(data, list):
        for item in data:
            _resolve_paths_recursive(item, base_dir)
