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
        # Ensure model_dir is resolved correctly relative to where config is loaded/script run
        # Assuming loader handles resolution or script passes absolute path
        self.model_dir = model_dir if model_dir else Path("models").resolve()
        self.device = self._get_device()
        self.model: Optional[YOLO] = None

        try:
            # Use kp_config (main config) for checkpoint loading preference
            kp_config = config
            if kp_config.checkpoint_path:
                # Loader should resolve this path relative to the config file
                resolved_path = kp_config.checkpoint_path
                logger.info(
                    f"Initializing YOLO-Pose model from checkpoint: {resolved_path}"
                )
                if not resolved_path.is_file():
                    # Try resolving relative to model_dir as a fallback if loader didn't fully resolve
                    resolved_path_fallback = (self.model_dir / resolved_path).resolve()
                    if resolved_path_fallback.is_file():
                        resolved_path = resolved_path_fallback
                        logger.info(
                            f"Resolved checkpoint relative to model_dir: {resolved_path}"
                        )
                    else:
                        raise FileNotFoundError(
                            f"YOLO-Pose checkpoint file not found: {kp_config.checkpoint_path} (tried {resolved_path} and {resolved_path_fallback})"
                        )
                self.model = YOLO(resolved_path, task="pose")
                logger.info(f"YOLO-Pose model loaded successfully from {resolved_path}")
            # Use training_config for base_model if no checkpoint specified
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

    # Removed _resolve_path method - assuming loader handles this primarily

    def _init_model_from_name(self, model_name: str) -> YOLO:
        logger.debug(f"YOLO-Pose: Initializing from name {model_name}")
        # Use the resolved model_dir passed during initialization
        target_path = self.model_dir / model_name
        logger.info(f"Target path for model weights: {target_path}")
        if not target_path.exists():
            logger.info(
                f"Model '{model_name}' not found locally. Attempting download..."
            )
            assets_tag = (
                self.training_config.ultralytics_assets_tag
                if self.training_config
                else "v8.0.0"  # Fallback tag
            )
            # Ensure assets_tag is treated as string
            assets_tag_str = (
                str(assets_tag) if isinstance(assets_tag, Path) else assets_tag
            )

            base_url = f"https://github.com/ultralytics/assets/releases/download/{assets_tag_str}/"
            model_url = base_url + model_name
            logger.info(f"Constructed download URL: {model_url}")
            try:
                # Download to the specified model_dir
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

            if (
                not results
                or results[0].boxes is None
                or results[0].keypoints is None
                or len(results[0].boxes) == 0
                or results[0].keypoints.shape[1] == 0
            ):
                return None

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
            return keypoints_sv, pitch_bbox_xyxy

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

        # --- Corrected Logic ---
        # Directly use the dataset_path which should point to the data.yaml
        yolo_pose_yaml_path = train_cfg.dataset_path
        if (
            not yolo_pose_yaml_path.is_file()
            or yolo_pose_yaml_path.suffix.lower() != ".yaml"
        ):
            # Loader should have resolved the path, check if it's valid
            raise FileNotFoundError(
                f"YOLO-Pose dataset YAML file not found or invalid: {yolo_pose_yaml_path}"
            )
        logger.info(f"Using YOLO-Pose dataset definition from: {yolo_pose_yaml_path}")
        # --- End Corrected Logic ---

        try:
            train_args = {
                "data": str(yolo_pose_yaml_path),  # Use the direct path to data.yaml
                "epochs": train_cfg.epochs,
                "batch": train_cfg.batch_size,
                "imgsz": train_cfg.imgsz,
                "optimizer": train_cfg.optimizer,
                "project": str(train_cfg.project_name),  # Ensure project path is string
                "name": train_cfg.run_name,
                "device": train_cfg.device,
                "plots": train_cfg.plots,
                "lr0": train_cfg.learning_rate,
            }
            train_args = {k: v for k, v in train_args.items() if v is not None}

            logger.info(f"Starting Ultralytics pose training with args: {train_args}")
            results = self.model.train(**train_args)
            logger.info("YOLO-Pose training completed.")
            try:
                # Ensure save_dir is treated as Path if it exists
                save_dir_val = getattr(results, "save_dir", None)
                if save_dir_val:
                    save_dir = Path(save_dir_val)
                    best_weights_path = save_dir / "weights" / "best.pt"
                    logger.info(f"Best weights saved at: {best_weights_path}")
                else:
                    logger.info("Could not determine save_dir from training results.")
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
