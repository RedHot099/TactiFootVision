# tactifoot_vision/detection/rfdetr_seg_handler.py
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from .base_handler import BaseHandler
from tactifoot_vision.data.dataset_parsers import convert_yolo_to_coco

logger = logging.getLogger(__name__)


class RFDETRSegHandler(BaseHandler):
    """
    Handler for RF-DETR Segmentation (Preview) model.

    Relies on `rfdetr.RFDETRSegPreview` which exposes a similar API to RFDETRBase:
    - predict(image: PIL.Image, threshold: float) -> supervision.Detections
    - train(dataset_dir=..., epochs=..., batch_size=..., grad_accum_steps=..., lr=..., output_dir=...)
    """

    def __init__(self, *args, **kwargs):
        # Lazy import to avoid hard dependency during module import
        try:
            from rfdetr import RFDETRSegPreview  # noqa: F401
        except Exception as e:  # pragma: no cover
            logger.warning(
                "rfdetr package with RFDETRSegPreview not available yet. "
                "Ensure dependencies are installed in your environment."
            )
        super().__init__(*args, **kwargs)

    def _load_model_from_checkpoint(self, checkpoint_path: Path):
        try:
            from rfdetr import RFDETRSegPreview
            logger.debug(
                f"RFDETRSegPreview: Initializing with pretrain_weights={checkpoint_path}"
            )
            model = RFDETRSegPreview(pretrain_weights=str(checkpoint_path))
            logger.info(
                f"RFDETRSegPreview model initialized using checkpoint: {checkpoint_path}"
            )
            return model
        except Exception as e:
            logger.error(
                f"Failed to initialize RFDETRSegPreview from checkpoint {checkpoint_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"RFDETRSegPreview model loading failed: {e}") from e

    def _init_model_from_name(self, model_name: str):
        try:
            from rfdetr import RFDETRSegPreview
            logger.debug(f"RFDETRSegPreview: Initializing from name {model_name}")
            # For preview, name is not used; instantiate default model
            model = RFDETRSegPreview()
            logger.info("RFDETRSegPreview model instantiated (preview baseline).")
            return model
        except Exception as e:
            logger.error(
                f"Failed to initialize RFDETRSegPreview base model: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"RFDETRSegPreview model initialization failed: {e}"
            ) from e

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
                logger.warning(
                    "RF-DETR Seg preview returned unexpected output; expected supervision.Detections."
                )
                return sv.Detections.empty()
            detections = self._apply_per_class_thresholds(detections)
            detections = self._apply_nms(detections)
            return detections
        except Exception:
            logger.exception("Error during RF-DETR Seg detection")
            return sv.Detections.empty()

    def train(self):
        if not self.training_config:
            raise ValueError("Training configuration is required.")
        if self.model is None:
            raise RuntimeError("Model is not available for training.")

        train_cfg = self.training_config
        logger.info("Starting RF-DETR Seg (preview) training...")
        try:
            coco_dataset_dir_for_training: Optional[Path] = None
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
                # For segmentation, request masks from conversion if present
                coco_dataset_dir_for_training = convert_yolo_to_coco(
                    dataset_input_path, force_reconvert=False, force_masks=True
                )
                logger.info(
                    f"Using converted COCO dataset from: {coco_dataset_dir_for_training}"
                )
            else:
                raise ValueError(
                    f"Unsupported dataset_format: {train_cfg.dataset_format}"
                )

            output_dir = getattr(train_cfg, "output_dir", None)
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = Path(train_cfg.project_name or "output")
                if train_cfg.run_name:
                    output_dir = output_dir / train_cfg.run_name
            if not output_dir.is_absolute():
                output_dir = (self.model_dir / output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            # RF-DETR Seg preview expects a Roboflow-style COCO layout; pass both dataset_dir and coco_path.
            train_args = {
                "dataset_dir": str(coco_dataset_dir_for_training),
                "coco_path": str(coco_dataset_dir_for_training),
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "grad_accum_steps": train_cfg.grad_accum_steps,
                "lr": train_cfg.learning_rate,
                "dataset_file": "roboflow",
                "output_dir": str(output_dir),
            }
            if train_cfg.num_workers is not None:
                train_args["num_workers"] = train_cfg.num_workers
            if train_cfg.multi_scale is not None:
                train_args["multi_scale"] = train_cfg.multi_scale

            logger.info(f"Starting RFDETRSegPreview.train() with args: {train_args}")
            self.model.train(**train_args)
            logger.info("RF-DETR Seg training process finished.")

            best_checkpoint = output_dir / "checkpoint_best_total.pth"
            if not best_checkpoint.exists():
                fallback_candidates = [
                    output_dir / "checkpoint_best_ema.pth",
                    output_dir / "checkpoint_best_regular.pth",
                    output_dir / "checkpoint.pth",
                ]
                for candidate in fallback_candidates:
                    if candidate.exists():
                        best_checkpoint = candidate
                        break

            if train_cfg.save_checkpoint_path:
                destination = Path(train_cfg.save_checkpoint_path)
                if not destination.is_absolute():
                    destination = (self.model_dir / destination).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if best_checkpoint.exists():
                    shutil.copy2(best_checkpoint, destination)
                    logger.info(
                        f"Best RF-DETR Seg checkpoint copied to: {destination} "
                        f"(source: {best_checkpoint})"
                    )
                else:
                    logger.warning(
                        f"Could not find RF-DETR Seg checkpoint to copy. "
                        f"Expected at least: {best_checkpoint}"
                    )
            else:
                logger.info(f"Best RF-DETR Seg checkpoint kept at: {best_checkpoint}")

        except Exception as e:
            logger.error(
                f"An error occurred during RF-DETR Seg training: {e}", exc_info=True
            )
            raise RuntimeError(f"RF-DETR Seg training failed: {e}") from e
