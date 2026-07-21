from pathlib import Path
from shutil import copy2

from tactifoot_vision.config.schemas import DetectionTrainingConfig
from tactifoot_vision.datasets import SoccerNetTrackingDataset
from tactifoot_vision.enums import DatasetFormat, DatasetSource

RFDETR_CHECKPOINT_CANDIDATES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)


def build_rfdetr_train_args(
    config: DetectionTrainingConfig, output_dir: Path
) -> dict[str, object]:
    dataset_dir = prepare_rfdetr_dataset(config, output_dir)

    args: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "coco_path": str(dataset_dir),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "dataset_file": "roboflow",
        "output_dir": str(output_dir),
    }
    if config.learning_rate is not None:
        args["lr"] = config.learning_rate
    if config.num_workers is not None:
        args["num_workers"] = config.num_workers
    if config.multi_scale is not None:
        args["multi_scale"] = config.multi_scale
    if config.early_stopping:
        args["early_stopping"] = True
        if config.early_stopping_patience is not None:
            args["early_stopping_patience"] = config.early_stopping_patience
        if config.early_stopping_min_delta is not None:
            args["early_stopping_min_delta"] = config.early_stopping_min_delta
        if config.early_stopping_use_ema is not None:
            args["early_stopping_use_ema"] = config.early_stopping_use_ema
    return args


def prepare_rfdetr_dataset(config: DetectionTrainingConfig, output_dir: Path) -> Path:
    if config.dataset_source == DatasetSource.SOCCERNET_TRACKING:
        converted_dir = config.converted_dataset_dir or output_dir / "converted_coco"
        SoccerNetTrackingDataset(config.data).to_coco(
            converted_dir,
            valid_fraction=config.valid_fraction,
            every_nth_frame=config.every_nth_frame,
            max_sequences=config.max_sequences,
            symlink_images=config.symlink_images,
        )
        return converted_dir
    if config.dataset_format == DatasetFormat.YOLO:
        raise NotImplementedError("YOLO to COCO conversion is not ported to src yet")
    if not config.data.is_dir():
        raise FileNotFoundError(f"COCO dataset directory not found: {config.data}")
    return config.data


def find_best_rfdetr_checkpoint(output_dir: Path) -> Path | None:
    for name in RFDETR_CHECKPOINT_CANDIDATES:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


def copy_best_checkpoint_if_requested(
    *, best_checkpoint: Path | None, destination: Path | None
) -> Path | None:
    if best_checkpoint is None or destination is None:
        return best_checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(best_checkpoint, destination)
    return destination
