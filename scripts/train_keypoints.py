# scripts/train_keypoints.py
import argparse
import sys
from pathlib import Path

from loguru import logger

from config.loaders import load_config
from config.models import KeypointModelType

# Import the new handler
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler

# Remove SimpleBaselineResNet handler import
# from tactifoot_vision.keypoints.handler import KeypointHandler
from tactifoot_vision.utils.logging_config import setup_logging

project_root = Path(__file__).resolve().parents[1]


def main(config_path: Path):
    try:
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        config = load_config(config_path)

        setup_logging(level=config.logging_level)
        logger.info(f"Starting keypoint training script: {config.project_name}")
        logger.debug(f"Config loaded from: {config_path}")

        # *** Check if training.keypoints config exists ***
        if not config.training or not config.training.keypoints:
            logger.error("Keypoint training config ('training.keypoints') not found.")
            sys.exit(1)
        train_cfg = config.training.keypoints

        # *** Check if top-level keypoints config exists ***
        if config.keypoints is None:
            logger.error(
                "Top-level 'keypoints' section not found in config file, but required for training."
            )
            sys.exit(1)
        kp_config = config.keypoints  # Use shorter name

        # Check for base model vs checkpoint consistency
        if kp_config.checkpoint_path is None:
            if not train_cfg.base_model:
                logger.error(
                    "Config error: keypoints.checkpoint_path is null, but training.keypoints.base_model is missing."
                )
                sys.exit(1)
            logger.info(
                f"Configuring handler for training from scratch using base name: {train_cfg.base_model}"
            )
        else:
            logger.info(
                f"Configuring handler for fine-tuning from checkpoint: {kp_config.checkpoint_path}"
            )
            # Path resolution handled by loader

        logger.info(
            f"Initializing handler for keypoint training: {kp_config.model_type.value}"
        )
        model_dir_abs = config.paths.model_dir

        handler = None
        try:
            if kp_config.model_type == KeypointModelType.YOLO_POSE:
                handler = YOLOPoseHandler(
                    config=kp_config,  # Pass validated sub-config
                    training_config=train_cfg,
                    model_dir=model_dir_abs,
                )
            else:
                raise ValueError(
                    f"Unsupported keypoint model type for training: {kp_config.model_type}"
                )

            if handler.model is None:
                logger.error(
                    "Model failed to load/initialize during handler setup. Exiting."
                )
                sys.exit(1)
            logger.success("Keypoint handler initialized for training.")

        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
            # *** Fix logging ***
            logger.error(
                f"Failed to initialize handler for training: {e}", exc_info=True
            )
            sys.exit(1)

        # --- Run Training ---
        logger.info("Starting training process via handler...")
        try:
            handler.train()
            logger.success("Keypoint training script finished successfully.")
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
    parser = argparse.ArgumentParser(description="Train a Keypoint Detection Model.")
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
