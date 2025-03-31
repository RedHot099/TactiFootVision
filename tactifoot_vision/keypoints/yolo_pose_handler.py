# tactifoot_vision/keypoints/yolo_pose_handler.py
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import supervision as sv

from ultralytics import YOLO
from ultralytics.utils.downloads import download

from config.models import KeypointsConfig, TrainingKeypointsConfig
from tactifoot_vision.data.dataset_parsers import convert_coco_to_yolo_pose

logger = logging.getLogger(__name__)


class YOLOPoseHandler:
    def __init__(
        self,
        config: KeypointsConfig,
        training_config: Optional[TrainingKeypointsConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        self.config = config
        self.training_config = training_config
        self.model_dir = (model_dir or Path("models")).resolve()
        self.device = self._get_device()
        self.model: Optional[YOLO] = None

        try:
            if config.checkpoint_path:
                resolved_path = self._resolve_path(config.checkpoint_path)
                logger.info(
                    f"Initializing YOLO-Pose model from checkpoint: {resolved_path}"
                )
                if not resolved_path.is_file():
                    raise FileNotFoundError(
                        f"YOLO-Pose checkpoint file not found: {resolved_path}"
                    )
                self.model = YOLO(resolved_path, task="pose")
                logger.info(f"YOLO-Pose model loaded successfully from {resolved_path}")
            elif training_config and training_config.base_model:
                logger.info(
                    f"Initializing YOLO-Pose model from base name: {training_config.base_model}"
                )
                self.model = self._init_model_from_name(training_config.base_model)
                logger.info(
                    f"YOLO-Pose model initialized successfully from {training_config.base_model}"
                )
            else:
                err_msg = "Cannot initialize handler: keypoints.checkpoint_path is null, and either training config or training.keypoints.base_model is missing."
                logger.error(err_msg)
                raise ValueError(err_msg)
        except Exception as e:
            logger.exception("Failed to load or initialize YOLO-Pose model.")
            raise e

    def _get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_path(self, path_str_or_obj: str | Path) -> Path:
        path = Path(path_str_or_obj)
        if not path.is_absolute():
            resolved_path = (self.model_dir / path).resolve()
            logger.debug(
                f"YOLO-Pose Handler resolved relative path {path} to {resolved_path}"
            )
            return resolved_path
        return path

    def _init_model_from_name(self, model_name: str) -> YOLO:
        logger.debug(f"YOLO-Pose: Initializing from name {model_name}")
        target_path = self.model_dir / model_name
        logger.info(f"Target path for model weights: {target_path}")
        if not target_path.exists():
            logger.info(
                f"Model '{model_name}' not found locally. Attempting download..."
            )
            assets_tag = (
                self.training_config.ultralytics_assets_tag
                if self.training_config
                else "v8.0.0"
            )
            base_url = (
                f"https://github.com/ultralytics/assets/releases/download/{assets_tag}/"
            )
            model_url = base_url + model_name
            logger.info(f"Constructed download URL: {model_url}")
            try:
                download(model_url, dir=self.model_dir, unzip=False)
                logger.info(
                    f"Successfully downloaded '{model_name}' to {self.model_dir}"
                )
            except Exception as download_exc:
                logger.error(
                    f"Failed to download from '{model_url}'. Error: {download_exc}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Could not download YOLO-Pose model: {model_name}"
                ) from download_exc
        else:
            logger.info(f"Model '{model_name}' found locally at {target_path}.")

        if not target_path.is_file():
            raise FileNotFoundError(
                f"Model file not found at {target_path} after check/download."
            )
        try:
            logger.info(
                f"Initializing YOLO-Pose model using weights file: {target_path}"
            )
            model = YOLO(target_path, task="pose")
            logger.info(
                f"Successfully initialized YOLO-Pose model with weights from {target_path}"
            )
            return model
        except Exception as e:
            logger.error(
                f"Failed to initialize YOLO-Pose from file {target_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"YOLO-Pose model initialization from file failed: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> Optional[Tuple[sv.KeyPoints, np.ndarray]]:
        if self.model is None:
            logger.error("YOLO-Pose model not loaded. Cannot detect.")
            return None
        try:
            results = self.model.predict(
                frame, conf=self.config.confidence_threshold, verbose=False
            )

            # Check if results, boxes, and keypoints are present and valid
            if (
                not results
                or results[0].boxes is None
                or results[0].keypoints is None
                or len(results[0].boxes) == 0
                or results[0].keypoints.shape[1]
                == 0  # Check if keypoints tensor is empty
            ):
                return None  # No pitch or keypoints detected

            # Assume the first detection is the pitch object
            pitch_bbox_xyxy = results[0].boxes.xyxy[0].cpu().numpy()
            kpts_data = results[0].keypoints.data.cpu().numpy()

            if kpts_data.shape[-1] == 3:
                xy = kpts_data[..., :2]
                confidence = kpts_data[..., 2]
            elif kpts_data.shape[-1] == 2:
                xy = kpts_data
                det_conf = results[0].boxes.conf.cpu().numpy()
                if det_conf is not None and len(det_conf) == xy.shape[0]:
                    confidence = np.repeat(det_conf[:, np.newaxis], xy.shape[1], axis=1)
                else:
                    confidence = np.ones((xy.shape[0], xy.shape[1]))
            else:
                logger.warning(f"Unexpected keypoint data shape: {kpts_data.shape}")
                return None

            if xy.ndim == 2:
                xy = xy[np.newaxis, ...]
            if (
                confidence.ndim == 1
                and xy.ndim == 3
                and confidence.shape[0] == xy.shape[1]
            ):
                confidence = confidence[np.newaxis, ...]
            elif (
                confidence.ndim == 1
                and xy.ndim == 3
                and confidence.shape[0] == xy.shape[0]
            ):
                confidence = np.repeat(confidence[:, np.newaxis], xy.shape[1], axis=1)

            keypoints_sv = sv.KeyPoints(xy=xy, confidence=confidence)
            return keypoints_sv, pitch_bbox_xyxy  # Return both

        except Exception:
            logger.exception("Error during YOLO-Pose keypoint detection")
            return None

    def train(self):
        if not self.training_config:
            raise ValueError("Training configuration is required.")
        if self.model is None:
            raise RuntimeError("Model is not available for training.")

        train_cfg = self.training_config
        logger.info("Starting YOLO-Pose training/fine-tuning...")

        yolo_pose_yaml_path = None
        source_coco_path = Path(train_cfg.source_coco_dataset_path)
        if not source_coco_path.is_dir():
            raise FileNotFoundError(
                f"Source COCO dataset directory not found: {source_coco_path}"
            )
        try:
            yolo_pose_yaml_path = convert_coco_to_yolo_pose(source_coco_path)
            logger.info(
                f"Using converted YOLO-Pose dataset from: {yolo_pose_yaml_path.parent}"
            )
        except Exception as e:
            logger.error(
                f"Failed to convert COCO dataset to YOLO-Pose format: {e}",
                exc_info=True,
            )
            raise RuntimeError("Dataset conversion failed.") from e

        try:
            train_args = {
                "data": str(yolo_pose_yaml_path),
                "epochs": train_cfg.epochs,
                "batch": train_cfg.batch_size,
                "imgsz": train_cfg.imgsz,
                "optimizer": train_cfg.optimizer,
                "project": train_cfg.project_name,
                "name": train_cfg.run_name,
                "device": train_cfg.device,
                "plots": train_cfg.plots,
                "lr0": train_cfg.learning_rate,
            }
            train_args = {k: v for k, v in train_args.items() if v is not None}

            logger.info(f"Starting Ultralytics pose training with args: {train_args}")
            results = self.model.train(**train_args)
            logger.success("YOLO-Pose training completed.")
            try:
                save_dir = results.save_dir
                best_weights_path = Path(save_dir) / "weights" / "best.pt"
                logger.info(f"Best weights saved at: {best_weights_path}")
            except AttributeError:
                logger.info(
                    "Could not determine exact path of best weights from results."
                )

        except Exception as e:
            logger.error(
                f"An error occurred during YOLO-Pose training: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"YOLO-Pose training failed: {e}") from e
