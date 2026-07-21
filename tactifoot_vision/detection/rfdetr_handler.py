# tactifoot_vision/detection/rfdetr_handler.py
import logging
from pathlib import Path
import numpy as np
import supervision as sv
from PIL import Image
import cv2

from rfdetr import RFDETRBase

from .base_handler import BaseHandler
from tactifoot_vision.data.dataset_parsers import convert_yolo_to_coco

logger = logging.getLogger(__name__)


class RFDETRHandler(BaseHandler):
    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> RFDETRBase:
        logger.debug(f"RFDETR: Initializing with pretrain_weights={checkpoint_path}")
        try:
            model = RFDETRBase(pretrain_weights=str(checkpoint_path))
            logger.info(f"RFDETR model initialized using checkpoint: {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(
                f"Failed to initialize RFDETR model from checkpoint {checkpoint_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"RFDETR model loading failed: {e}") from e

    def _init_model_from_name(self, model_name: str) -> RFDETRBase:
        logger.debug(f"RFDETR: Initializing from name {model_name}")
        try:
            model = RFDETRBase(pretrain_weights=model_name)
            logger.info("RFDETR base model instantiated.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize RFDETR base model: {e}", exc_info=True)
            raise RuntimeError(f"RFDETR model initialization failed: {e}") from e

    def detect(self, frame: np.ndarray) -> sv.Detections:
        if self.model is None:
            logger.error("Model not loaded or initialized. Cannot detect.")
            return sv.Detections.empty()
        try:
            if hasattr(self.model, "eval"):
                self.model.eval()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            detections = self.model.predict(
                pil_image, threshold=self.detection_config.confidence_threshold
            )
            if not isinstance(detections, sv.Detections):
                logger.warning("RF-DETR output may need conversion to sv.Detections.")
                return sv.Detections.empty()
            return detections
        except Exception:
            logger.exception("Error during RF-DETR detection")
            return sv.Detections.empty()

    def train(self):
        if not self.training_config:
            raise ValueError("Training configuration is required.")
        if self.model is None:
            raise RuntimeError("Model is not available for training.")

        train_cfg = self.training_config
        logger.info("Starting RF-DETR training/fine-tuning...")
        try:
            coco_dataset_dir_for_training = None
            dataset_input_path = Path(train_cfg.dataset_path)

            if train_cfg.dataset_format == "coco":
                if not dataset_input_path.is_dir():
                    raise FileNotFoundError(
                        f"COCO dataset directory not found: {dataset_input_path}"
                    )
                coco_dataset_dir_for_training = dataset_input_path
                logger.info(
                    f"Using COCO dataset directly from: {coco_dataset_dir_for_training}"
                )
            elif train_cfg.dataset_format == "yolo":
                if (
                    not dataset_input_path.is_file()
                    or dataset_input_path.suffix.lower() != ".yaml"
                ):
                    raise FileNotFoundError(
                        f"YOLO data.yaml file not found or invalid: {dataset_input_path}"
                    )
                coco_dataset_dir_for_training = convert_yolo_to_coco(dataset_input_path)
                logger.info(
                    f"Using converted COCO dataset from: {coco_dataset_dir_for_training}"
                )
            else:
                raise ValueError(
                    f"Unsupported dataset_format: {train_cfg.dataset_format}"
                )

            train_args = {
                "dataset_dir": str(coco_dataset_dir_for_training),
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "grad_accum_steps": train_cfg.grad_accum_steps,
                "lr": train_cfg.learning_rate,
                "output_dir": str(self._resolve_training_output_dir(train_cfg)),
                "project": train_cfg.project_name,
                "run": train_cfg.run_name,
                "resolution": train_cfg.imgsz,
                "class_names": ["person"],
                "num_classes": 1,
                "device": train_cfg.device,
                "run_test": False,
            }
            train_args = {key: value for key, value in train_args.items() if value is not None}

            logger.info(f"Starting RFDETR model.train() with args: {train_args}")
            self.model.train(**train_args)
            logger.info("RF-DETR training process finished.")

        except Exception as e:
            logger.error(
                f"An error occurred during RF-DETR training: {e}", exc_info=True
            )
            raise RuntimeError(f"RF-DETR training failed: {e}") from e

    def _resolve_training_output_dir(self, train_cfg) -> Path:
        if train_cfg.project_name and train_cfg.run_name:
            return Path(train_cfg.project_name) / train_cfg.run_name
        if train_cfg.project_name:
            return Path(train_cfg.project_name)
        if train_cfg.run_name:
            return Path(train_cfg.run_name)
        return Path("output")
