# scripts/train_detector.py
import argparse
import sys
from pathlib import Path

from loguru import logger

from config.loaders import load_config
from config.models import DetectionModelType
from tactifoot_vision.detection.yolo_handler import YOLOHandler
from tactifoot_vision.detection.rfdetr_handler import RFDETRHandler
from tactifoot_vision.utils.logging_config import setup_logging

project_root = Path(__file__).resolve().parents[1]


def main(config_path: Path):
    try:
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        config = load_config(config_path)

        setup_logging(level=config.logging_level)
        logger.info(f"Starting training script: {config.project_name}")
        logger.debug(f"Config loaded from: {config_path}")

        # *** Check if training.detection config exists ***
        if not config.training or not config.training.detection:
            logger.error(
                "Training config ('training.detection') not found in config file."
            )
            sys.exit(1)
        train_cfg = config.training.detection

        # *** Check if top-level detection config exists ***
        if config.detection is None:
            logger.error(
                "Top-level 'detection' section not found in config file, but required for training."
            )
            sys.exit(1)
        detect_cfg = config.detection  # Use a shorter name

        # Check for base model vs checkpoint consistency
        if detect_cfg.checkpoint_path is None:
            if not train_cfg.base_model:
                logger.error(
                    "Config error: detection.checkpoint_path is null, but training.detection.base_model is missing."
                )
                sys.exit(1)
            logger.info(
                f"Configuring handler for training from scratch using base name: {train_cfg.base_model}"
            )
        else:
            logger.info(
                f"Configuring handler for fine-tuning from checkpoint: {detect_cfg.checkpoint_path}"
            )
            # Path resolution is handled by the loader now

        logger.info(f"Initializing handler for training: {detect_cfg.model_type.value}")
        # Path resolution for model_dir is handled by the loader
        model_dir_abs = config.paths.model_dir

        try:
            if detect_cfg.model_type == DetectionModelType.YOLO:
                handler = YOLOHandler(
                    detection_config=detect_cfg,  # Pass the validated sub-config
                    training_config=train_cfg,
                    model_dir=model_dir_abs,
                )
            elif detect_cfg.model_type == DetectionModelType.RFDETR:
                handler = RFDETRHandler(
                    detection_config=detect_cfg,  # Pass the validated sub-config
                    training_config=train_cfg,
                    model_dir=model_dir_abs,
                )
            else:
                raise ValueError(f"Unsupported model type: {detect_cfg.model_type}")

            if handler.model is None:
                logger.error(
                    "Model failed to load/initialize during handler setup. Exiting."
                )
                sys.exit(1)
            logger.success("Detection handler initialized for training.")

        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
            # *** Fix logging ***
            logger.error(
                f"Failed to initialize handler for training: {e}", exc_info=True
            )
            sys.exit(1)

        logger.info("Starting training process via handler...")
        try:
            handler.train()
            logger.success("Training script finished successfully.")
        except (
            NotImplementedError,
            FileNotFoundError,
            ValueError,
            RuntimeError,
            ImportError,
        ) as e:
            # *** Fix logging ***
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)
        except Exception:
            # *** Fix logging ***
            logger.exception("An unexpected error occurred during training.")
            sys.exit(1)

    except FileNotFoundError as e:
        # *** Fix logging ***
        logger.error(f"File not found error during setup: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        # *** Fix logging ***
        logger.error(f"Configuration or value error during setup: {e}", exc_info=True)
        sys.exit(1)
    except Exception:
        # *** Fix logging ***
        logger.exception("An unexpected error occurred during training script setup.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Detection Model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root
        / "run_config"
        / "default_config.yaml",  # Point to default if needed
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
