"""TactiFoot Vision: football video analysis from dataset to tactical map.

    import tactifoot_vision as tv

    ds = tv.load_dataset("data/datasets/football_yolo/data.yaml")    # data preparation
    aug = tv.augment.Compose([tv.augment.HorizontalFlip(), ...])      # augmentation
    run = tv.train("yolo", ds, weights="yolo11n.pt", epochs=10)      # training
    result = tv.Pipeline(detector=run.load()).run("match.mp4")       # inference
    tv.viz.render_video(result, "match.mp4", "annotated.mp4")        # visualisation

Submodules are imported lazily, so ``import tactifoot_vision`` stays fast and
heavy dependencies (torch, ultralytics, rfdetr, transformers) load on first use.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("tactifoot-vision")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

_SUBMODULES = (
    "augment",
    "config",
    "data",
    "evaluation",
    "models",
    "pipeline",
    "pitch",
    "teams",
    "tracking",
    "utils",
    "viz",
)

# Shortcuts for the most common entry points: name -> (module, attribute).
_EXPORTS: dict[str, tuple[str, str]] = {
    "Dataset": ("data", "Dataset"),
    "load_dataset": ("data", "load_dataset"),
    "VideoReader": ("data", "VideoReader"),
    "load_model": ("models", "load_model"),
    "train": ("models", "train"),
    "TrainConfig": ("models", "TrainConfig"),
    "evaluate": ("evaluation", "evaluate"),
    "Pipeline": ("pipeline", "Pipeline"),
    "PipelineResult": ("pipeline", "PipelineResult"),
    "SoccerPitch": ("pitch", "SoccerPitch"),
    "show": ("viz", "show"),
    "setup_logging": ("utils", "setup_logging"),
}

__all__ = [
    "__version__",
    "augment",
    "config",
    "data",
    "evaluation",
    "models",
    "pipeline",
    "pitch",
    "teams",
    "tracking",
    "utils",
    "viz",
    "Dataset",
    "load_dataset",
    "VideoReader",
    "load_model",
    "train",
    "TrainConfig",
    "evaluate",
    "Pipeline",
    "PipelineResult",
    "SoccerPitch",
    "show",
    "setup_logging",
]

if TYPE_CHECKING:
    from tactifoot_vision import (
        augment,
        config,
        data,
        evaluation,
        models,
        pipeline,
        pitch,
        teams,
        tracking,
        utils,
        viz,
    )
    from tactifoot_vision.data import Dataset, VideoReader, load_dataset
    from tactifoot_vision.evaluation import evaluate
    from tactifoot_vision.models import TrainConfig, load_model, train
    from tactifoot_vision.pipeline import Pipeline, PipelineResult
    from tactifoot_vision.pitch import SoccerPitch
    from tactifoot_vision.utils import setup_logging
    from tactifoot_vision.viz import show
else:

    def __getattr__(name: str) -> Any:
        if name in _SUBMODULES:
            module = importlib.import_module(f"{__name__}.{name}")
        elif name in _EXPORTS:
            module_name, attribute = _EXPORTS[name]
            module = getattr(
                importlib.import_module(f"{__name__}.{module_name}"), attribute
            )
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = module
        return module

    def __dir__() -> list[str]:
        return sorted(set(globals()) | set(__all__))
