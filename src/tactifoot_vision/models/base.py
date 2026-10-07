"""The one interface every trainable model implements, plus its registry."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import supervision as sv
from pydantic import BaseModel, ConfigDict, Field

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.registry import Registry
from tactifoot_vision.utils import next_run_dir, resolve_device

if TYPE_CHECKING:
    from tactifoot_vision.evaluation.metrics import DetectionMetrics, KeypointMetrics

logger = logging.getLogger(__name__)

MODELS: "Registry[Model]" = Registry("model")

# Backend arguments that place the run; Model.train sets them from TrainConfig.
_RUN_FOLDER_OPTIONS = ("output_dir", "project", "name", "dataset_dir", "data")


class TrainConfig(BaseModel):
    """Training settings understood by every backend.

    Unknown keyword arguments are kept (``extra="allow"``) and passed verbatim to
    the backend, e.g. ``mosaic=0.0`` for Ultralytics or ``grad_accum_steps=4``
    for RF-DETR. Read them with ``config.backend_options``. A config is
    immutable; ``config.model_copy(update={...})`` derives another one.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    epochs: int = Field(10, ge=1)
    batch_size: int = Field(8, ge=1)
    imgsz: int = Field(640, ge=32)
    lr: float | None = Field(None, gt=0, description="None keeps the backend default")
    device: str | None = None
    workers: int = Field(4, ge=0)
    patience: int | None = Field(
        None, ge=1, description="Early-stopping patience in epochs"
    )
    seed: int = 0
    output_dir: Path = Path("runs")
    name: str | None = Field(
        None, description="Run folder name; defaults to the model name"
    )
    exist_ok: bool = Field(
        False, description="Reuse output_dir/name instead of creating name2, name3, ..."
    )

    @property
    def backend_options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


@dataclass
class TrainResult:
    """Outcome of :meth:`Model.train`."""

    model_type: str  # registry name, e.g. "yolo"
    weights: Path  # best checkpoint
    run_dir: Path
    history: pd.DataFrame  # one row per epoch (column names follow the backend)
    # Validation metrics of the best checkpoint, same keys for every backend:
    # map50_95, map50 (+ map75, precision, recall when reported; pose_* for keypoints).
    metrics: dict[str, float]

    def load(self, **options: Any) -> "Model":
        """Instantiate the trained model."""
        return load_model(self.model_type, self.weights, **options)


class Model(ABC):
    """A vision model that can be loaded, trained, run and evaluated the same way.

    Detection models return ``sv.Detections`` whose ``data["class_name"]`` holds
    the class names; keypoint models return ``sv.KeyPoints`` with one instance
    per detected object, sorted by confidence (best first).

    Subclasses implement ``_load``, ``_train``, ``class_names`` and ``predict``,
    may extend ``_check_options``, and register themselves in ``MODELS`` under
    ``name``.
    """

    name: ClassVar[str]
    task: ClassVar[Task]
    # Backend option -> the TrainConfig field that sets it; Model.train refuses the option.
    _train_config_aliases: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        if not 0 <= conf <= 1:
            raise ValueError(f"conf must be in [0, 1], got {conf}")
        self.conf = conf
        self.device = resolve_device(device)
        self.weights: str | Path | None = weights
        self._load(weights)

    # ---------------------------------------------------------------- backend
    @abstractmethod
    def _load(self, weights: str | Path | None) -> None:
        """(Re)build the underlying model from a checkpoint path or model name.

        ``None`` means the backend's default pretrained weights.
        """

    @abstractmethod
    def _train(
        self, dataset: Dataset, config: TrainConfig, run_dir: Path
    ) -> TrainResult:
        """Fine-tune on ``dataset``, writing everything to ``run_dir`` (already created)."""

    def _check_options(self, config: TrainConfig) -> None:
        """Reject bad training settings; called before the run folder is created.

        Backend options that would move the run folder, or that are the
        backend's name for a :class:`TrainConfig` field
        (:attr:`_train_config_aliases`), are always refused. Backends extend
        this (calling ``super()``) to reject options they do not know and
        settings they cannot use, so a typo costs no run folder.
        """
        options = config.backend_options
        reserved = sorted(set(options) & set(_RUN_FOLDER_OPTIONS))
        if reserved:
            raise ValueError(
                f"Backend option(s) {', '.join(reserved)} would move the run folder; "
                "set output_dir, name and exist_ok instead"
            )
        aliases = [k for k in options if k in self._train_config_aliases]
        if aliases:
            fields = ", ".join(
                f"{k} -> {self._train_config_aliases[k]}" for k in sorted(aliases)
            )
            raise ValueError(
                f"Set these through the TrainConfig field instead of the {self.name} "
                f"option: {fields}"
            )

    @property
    @abstractmethod
    def class_names(self) -> list[str]: ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> sv.Detections | sv.KeyPoints:
        """Run on one BGR image."""

    # ------------------------------------------------------------ public API
    def __call__(self, image: np.ndarray) -> sv.Detections | sv.KeyPoints:
        return self.predict(image)

    def train(
        self, dataset: Dataset, config: TrainConfig | None = None, **overrides: Any
    ) -> TrainResult:
        """Fine-tune on ``dataset``; afterwards this model holds the best weights.

        Pass a :class:`TrainConfig` and/or keyword overrides (``epochs=5``, ...).
        """
        if dataset.task != self.task:
            raise ValueError(
                f"{self.name} is a {self.task.value} model but the dataset is {dataset.task.value}"
            )
        if not dataset["train"] or not dataset["valid"]:
            raise ValueError(
                "Training needs non-empty 'train' and 'valid' splits; "
                "create them with dataset.resplit(train=0.8, valid=0.2)"
            )
        base = config.model_dump() if config is not None else {}
        cfg = TrainConfig(**(base | overrides))
        if cfg.name is None:
            cfg = cfg.model_copy(update={"name": self.name})
        self._check_options(cfg)
        run_dir = next_run_dir(Path(cfg.output_dir).absolute() / cfg.name, cfg.exist_ok)
        logger.info(
            "Training %s on %r for %d epochs in %s",
            self.name,
            dataset,
            cfg.epochs,
            run_dir,
        )
        result = self._train(dataset, cfg, run_dir)
        self.weights = result.weights
        self._load(result.weights)
        return result

    def evaluate(
        self, dataset: Dataset, split: str = "valid", **kwargs: Any
    ) -> "DetectionMetrics | KeypointMetrics":
        """Score on a dataset split with backend-independent metrics.

        See :func:`tactifoot_vision.evaluation.evaluate`.
        """
        from tactifoot_vision.evaluation import evaluate

        return evaluate(self, dataset, split=split, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weights={self.weights!r}, conf={self.conf}, device={self.device!r})"


def load_model(name: str, weights: str | Path | None = None, **options: Any) -> Model:
    """Create a registered model, e.g. ``load_model("yolo", "yolo11n.pt", conf=0.3)``."""
    return MODELS.create(name, weights, **options)


def available_models() -> list[str]:
    return MODELS.names()


def train(
    model: str | Model,
    dataset: Dataset,
    config: TrainConfig | None = None,
    *,
    weights: str | Path | None = None,
    **overrides: Any,
) -> TrainResult:
    """Train a model given by name (``"yolo"``, ``"rfdetr"``, ...) or instance.

    ``weights`` picks the starting checkpoint when ``model`` is a name; the
    model is then loaded on the training ``device`` (from ``overrides`` or
    ``config``), so nothing is loaded on another GPU first.
    """
    if isinstance(model, str):
        device = overrides.get("device", config.device if config else None)
        options = {"device": device} if device is not None else {}
        model = load_model(model, weights, **options)
    elif weights is not None:
        raise ValueError("Pass weights only together with a model name")
    return model.train(dataset, config, **overrides)
