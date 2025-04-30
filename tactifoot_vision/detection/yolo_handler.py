# tactifoot_vision/detection/yolo_handler.py
import logging
from pathlib import Path
import numpy as np
import supervision as sv

from ultralytics import YOLO
from ultralytics.utils.downloads import download

from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


class YOLOHandler(BaseHandler):
    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> YOLO:
        logger.debug(f"YOLO: Loading from checkpoint {checkpoint_path}")
        try:
            model = YOLO(checkpoint_path)
            return model
        except Exception as e:
            logger.error(
                f"Failed to load YOLO model from checkpoint {checkpoint_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"YOLO model loading from checkpoint failed: {e}") from e

    def _init_model_from_name(self, model_name: str) -> YOLO:
        logger.debug(f"YOLO: Initializing from name {model_name}")
        target_path = self.model_dir / model_name
        logger.info(f"Target path for model weights: {target_path}")

        if not target_path.exists():
            logger.info(
                f"Model '{model_name}' not found locally. Attempting download..."
            )
            model_url = None
            # Simple check if model_name looks like a URL
            if model_name.startswith(("http://", "https://")):
                model_url = model_name
            elif model_name.endswith(".pt") and self.training_config:
                assets_tag = self.training_config.ultralytics_assets_tag
                base_url = f"https://github.com/ultralytics/assets/releases/download/{assets_tag}/"
                model_url = base_url + model_name
                logger.info(f"Constructed download URL: {model_url}")
            else:
                logger.error(
                    f"Cannot determine download source for model name: {model_name}"
                )
                raise ValueError(f"Invalid model name for download: {model_name}")

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
                    f"Could not download YOLO model: {model_name}"
                ) from download_exc
        else:
            logger.info(f"Model '{model_name}' found locally at {target_path}.")

        if not target_path.is_file():
            raise FileNotFoundError(
                f"Model file not found at {target_path} after check/download."
            )

        try:
            logger.info(f"Initializing YOLO model using weights file: {target_path}")
            model = YOLO(target_path, task="detect")
            logger.info(
                f"Successfully initialized YOLO model with weights from {target_path}"
            )
            return model
        except Exception as e:
            logger.error(
                f"Failed to initialize YOLO from file {target_path}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"YOLO model initialization from file failed: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> sv.Detections:
        try:
            results = self.model.predict(
                frame,
                conf=self.detection_config.confidence_threshold,
                iou=self.detection_config.nms_threshold,
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(results)
        except Exception:
            logger.exception("Error during YOLO detection")
            return sv.Detections.empty()

    def train(self):
        train_cfg = self.training_config
        logger.info("Starting YOLO training/fine-tuning...")
        try:
            dataset_yaml_path = Path(train_cfg.dataset_path)
            if not dataset_yaml_path.is_file():
                raise FileNotFoundError(
                    f"Dataset YAML file not found: {dataset_yaml_path}"
                )

            train_args = {
                "data": str(dataset_yaml_path),
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

            logger.info(f"Starting Ultralytics training with args: {train_args}")
            results = self.model.train(**train_args)
            logger.info("YOLO training completed.")
            try:
                save_dir = results.save_dir
                best_weights_path = Path(save_dir) / "weights" / "best.pt"
                logger.info(f"Best weights saved at: {best_weights_path}")
            except AttributeError:
                logger.info(
                    "Could not determine exact path of best weights from results."
                )

        except Exception as e:
            logger.error(f"An error occurred during YOLO training: {e}", exc_info=True)
            raise RuntimeError(f"YOLO training failed: {e}") from e
