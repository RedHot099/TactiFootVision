import abc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv

from config.models import DetectionConfig, TrainingDetectionConfig

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    def __init__(
        self,
        detection_config: DetectionConfig,
        training_config: Optional[TrainingDetectionConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        self.detection_config = detection_config
        self.training_config = training_config
        self.model_dir = (model_dir or Path("models")).resolve()
        self.model = None

        try:
            if self.detection_config.checkpoint_path:
                resolved_path = self._resolve_path(
                    self.detection_config.checkpoint_path
                )
                logger.info(f"Initializing model from checkpoint: {resolved_path}")
                if not resolved_path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint file not found: {resolved_path}"
                    )
                self.model = self._load_model_from_checkpoint(resolved_path)
                logger.info(f"Model loaded successfully from {resolved_path}")

            elif self.training_config and self.training_config.base_model:
                logger.info(
                    f"Initializing model from base name: {self.training_config.base_model}"
                )
                self.model = self._init_model_from_name(self.training_config.base_model)
                logger.info(
                    f"Model initialized successfully from {self.training_config.base_model}"
                )
            else:
                err_msg = (
                    "Cannot initialize handler: detection.checkpoint_path is null, "
                    "and either training config or training.detection.base_model is missing."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)

        except Exception as e:
            logger.exception("Failed to load or initialize model during handler setup.")
            raise e

    def _resolve_path(self, path_str_or_obj: str | Path) -> Path:
        path = Path(path_str_or_obj)
        if not path.is_absolute():
            resolved_path = (self.model_dir / path).resolve()
            logger.debug(f"Handler resolved relative path {path} to {resolved_path}")
            return resolved_path
        return path

    @abc.abstractmethod
    def _load_model_from_checkpoint(self, checkpoint_path: Path):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_model_from_name(self, model_name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> sv.Detections:
        # Base check can remain, but method must be overridden
        if self.model is None:
            logger.error("Model not loaded or initialized. Cannot detect.")
            return sv.Detections.empty()
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        # Base check can remain, but method must be overridden
        if not self.training_config:
            logger.error("Training configuration not provided. Cannot start training.")
            raise ValueError("Training configuration is required to run train().")
        if self.model is None:
            logger.error("Model not loaded or initialized. Cannot train.")
            raise RuntimeError("Model is not available for training.")
        raise NotImplementedError
