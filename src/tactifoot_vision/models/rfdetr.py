"""RF-DETR detection backend (``rfdetr``): Roboflow's real-time DETR on a DINOv2 backbone.

Class ids in RF-DETR outputs are the raw ``category_id`` values of the COCO json
the model was trained on, and the library's own ``RFDETR.class_names`` assumes
Roboflow's 1-based layout. This backend derives the mapping from the checkpoint
instead (see :func:`_label_names`) and returns ``class_id`` as an index into
:attr:`RFDETRDetector.class_names`, like the YOLO backends.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.models.base import MODELS, Model, TrainConfig, TrainResult
from tactifoot_vision.utils import cache_dir

logger = logging.getLogger(__name__)

# size -> (rfdetr detector class, rfdetr model config class)
SIZES: dict[str, tuple[str, str]] = {
    "nano": ("RFDETRNano", "RFDETRNanoConfig"),
    "small": ("RFDETRSmall", "RFDETRSmallConfig"),
    "medium": ("RFDETRMedium", "RFDETRMediumConfig"),
    "base": ("RFDETRBase", "RFDETRBaseConfig"),
    "large": ("RFDETRLarge", "RFDETRLargeConfig"),
}


# Checkpoint args that must match the chosen size. A mismatch either breaks
# loading or, worse, loads silently (missing decoder layers are skipped).
_ARCHITECTURE_FIELDS = (
    "encoder",
    "hidden_dim",
    "dec_layers",
    "patch_size",
    "out_feature_indexes",
    "projector_scale",
)


@MODELS.register("rfdetr")
class RFDETRDetector(Model):
    """RF-DETR box detector (sizes nano, small, medium, base, large).

    ``weights=None`` loads the library's COCO-pretrained weights for ``size``
    (downloaded once to ``~/.cache/tactifoot_vision/rfdetr``); a path loads a fine-tuned
    checkpoint whose class names are read from the checkpoint.

    Example::

        model = tv.load_model("rfdetr", "models/football_rfdetr_base.pth", conf=0.4)
        detections = model(frame)  # data["class_name"]: "player", "ball", ...
    """

    name = "rfdetr"
    task = Task.DETECT
    _train_config_aliases = {
        "num_workers": "workers",
        "resolution": "imgsz",
        "early_stopping_patience": "patience",
    }

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        size: str | None = None,
        conf: float = 0.5,
        resolution: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            weights: checkpoint path, a hosted RF-DETR weight name
                (``"rf-detr-base.pth"``) or ``None`` for the COCO weights of ``size``.
            size: architecture: nano, small, medium, base or large. ``None`` reads
                it from a fine-tuned checkpoint and means base for pretrained weights.
            conf: minimum confidence of returned objects.
            resolution: square input size; ``None`` uses the checkpoint's training
                resolution. Must be a multiple of patch size x windows (56 for
                base/large, 32 for the others).
            device: ``"cpu"``, ``"cuda"`` or ``"mps"``; ``None`` picks CUDA if available.
        """
        if size is not None and size not in SIZES:
            raise ValueError(f"Unknown RF-DETR size {size!r}; use one of {list(SIZES)}")
        self.size = size
        self.resolution = resolution
        super().__init__(weights, conf=conf, device=device)

    # ---------------------------------------------------------------- loading
    def _load(self, weights: str | Path | None) -> None:
        import rfdetr
        import rfdetr.config
        from rfdetr.util.coco_classes import COCO_CLASSES

        if weights is None:
            weights = _default(
                getattr(rfdetr.config, SIZES[self.size or "base"][1]),
                "pretrain_weights",
            )
        path = _resolve_weights(weights)
        args = _checkpoint_args(path)
        if self.size is None:
            self.size = _infer_size(args, path)
        _check_architecture(args, self.size, path)
        detector_name, config_name = SIZES[self.size]
        config_cls = getattr(rfdetr.config, config_name)

        resolution = self.resolution or args.get("resolution")
        resolution = int(resolution or _default(config_cls, "resolution"))
        block = _block_size(config_cls)
        if resolution % block:
            raise ValueError(
                f"RF-DETR {self.size} needs a resolution divisible by {block}, got {resolution}"
            )
        self._rf = getattr(rfdetr, detector_name)(
            pretrain_weights=str(path),
            resolution=resolution,
            device=_rfdetr_device(self.device),
        )
        label_names = _label_names(args, COCO_CLASSES)
        labels = sorted(label_names)
        self._class_names = [label_names[label] for label in labels]
        self._label_to_index = np.full(labels[-1] + 1, -1, dtype=int)
        self._label_to_index[labels] = np.arange(len(labels))

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    # -------------------------------------------------------------- inference
    def predict(self, image: np.ndarray) -> sv.Detections:
        """Detect objects in one BGR image."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self._rf.predict(rgb, threshold=self.conf)
        labels = detections.class_id.astype(int)
        index = np.full(len(labels), -1, dtype=int)
        known = (labels >= 0) & (labels < len(self._label_to_index))
        index[known] = self._label_to_index[labels[known]]
        if (index < 0).any():
            # Heads have spare logits (e.g. COCO id gaps); they carry no class.
            logger.debug("Dropping %d detections without a class", (index < 0).sum())
        detections = detections[index >= 0]
        detections.class_id = index[index >= 0]
        detections.data["class_name"] = np.asarray(self._class_names, dtype=str)[
            detections.class_id
        ]
        return detections

    # --------------------------------------------------------------- training
    def _check_options(self, config: TrainConfig) -> None:
        super()._check_options(config)
        _check_train_options(config.backend_options)
        _rfdetr_device(config.device or self.device)

    def _train(
        self, dataset: Dataset, config: TrainConfig, run_dir: Path
    ) -> TrainResult:
        options = config.backend_options
        # RF-DETR appends to an existing log.txt (exist_ok=True re-runs); the
        # history must describe this run only.
        (run_dir / "log.txt").unlink(missing_ok=True)
        # RF-DETR always builds a test loader; reuse the validation images if needed.
        has_test = bool(dataset["test"])
        export = dataset if has_test else dataset.with_split("test", dataset["valid"])
        coco_root = export.to_coco(run_dir / "dataset")

        model_config = self._rf.model_config
        block = model_config.patch_size * model_config.num_windows
        resolution = max(block, round(config.imgsz / block) * block)
        if resolution != config.imgsz:
            logger.info(
                "RF-DETR %s trains at resolution %d (imgsz %d rounded to a multiple of %d)",
                self.size,
                resolution,
                config.imgsz,
                block,
            )
        kwargs: dict[str, Any] = {
            "dataset_dir": str(coco_root),
            "output_dir": str(run_dir),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "num_workers": config.workers,
            "resolution": resolution,
            "seed": config.seed,
            "device": _rfdetr_device(config.device or self.device),
            "run_test": has_test,
        }
        if config.lr is not None:
            kwargs["lr"] = config.lr
        if config.patience is not None:
            kwargs |= {
                "early_stopping": True,
                "early_stopping_patience": config.patience,
            }
        kwargs |= options
        logger.info("RF-DETR training arguments: %s", kwargs)

        self._rf.train(**kwargs)
        # Predict at the resolution the weights were trained at.
        self.resolution = int(kwargs["resolution"])
        best = run_dir / "checkpoint_best_total.pth"
        if not best.is_file():
            raise RuntimeError(f"RF-DETR training finished without writing {best}")
        history = _read_log(run_dir / "log.txt")
        return TrainResult(
            model_type=self.name,
            weights=best,
            run_dir=run_dir,
            history=history,
            metrics=_final_metrics(history),
        )


# ------------------------------------------------------------------ helpers
def _default(config_cls: Any, field: str) -> Any:
    return config_cls.model_fields[field].default


def _block_size(config_cls: Any) -> int:
    return _default(config_cls, "patch_size") * _default(config_cls, "num_windows")


def _resolve_weights(weights: str | Path) -> Path:
    """Local checkpoint path; hosted RF-DETR weight names are downloaded to the cache."""
    from rfdetr.main import HOSTED_MODELS

    name = str(weights)
    path = Path(name)
    if path.is_file():
        return path
    if name not in HOSTED_MODELS:
        raise FileNotFoundError(
            f"RF-DETR checkpoint not found: {path} "
            f"(hosted weight names: {', '.join(sorted(HOSTED_MODELS))})"
        )
    target = cache_dir("rfdetr") / name
    if not target.is_file():
        from rfdetr.util.files import download_file

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        logger.info("Downloading RF-DETR weights %s to %s", name, target)
        download_file(HOSTED_MODELS[name], str(partial))
        partial.rename(target)
    return target


def _checkpoint_args(path: Path) -> dict[str, Any]:
    """Training arguments stored in an RF-DETR checkpoint (empty if absent)."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    args = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
    if args is None:
        return {}
    return dict(args) if isinstance(args, Mapping) else dict(vars(args))


def _mismatches(args: Mapping[str, Any], size: str) -> list[str]:
    """Architecture fields of a checkpoint that differ from the ``size`` defaults."""
    import rfdetr.config

    config_cls = getattr(rfdetr.config, SIZES[size][1])
    return [
        f"{field}={args[field]!r} (expected {_default(config_cls, field)!r})"
        for field in _ARCHITECTURE_FIELDS
        if field in args and args[field] != _default(config_cls, field)
    ]


def _infer_size(args: Mapping[str, Any], path: str | Path) -> str:
    if not any(args.get(field) is not None for field in _ARCHITECTURE_FIELDS):
        # No architecture recorded: trust a hosted name such as "rf-detr-nano.pth".
        return next((s for s in SIZES if f"-{s}" in Path(path).name), "base")
    fitting = [s for s in SIZES if not _mismatches(args, s)]
    if not fitting:
        raise ValueError(
            f"{path} does not match any RF-DETR {'/'.join(SIZES)} architecture "
            f"of rfdetr's installed version: {', '.join(_mismatches(args, 'base'))}"
        )
    return fitting[0]


def _check_architecture(args: Mapping[str, Any], size: str, path: Path) -> None:
    """Fail early (instead of on a state-dict shape mismatch) when ``size`` is wrong."""
    wrong = _mismatches(args, size)
    if wrong:
        fitting = [s for s in SIZES if not _mismatches(args, s)]
        hint = f"; pass size={fitting[0]!r}" if fitting else ""
        raise ValueError(
            f"{path} is not an RF-DETR {size} checkpoint: {', '.join(wrong)}{hint}"
        )


def _label_names(
    args: Mapping[str, Any], coco_classes: Mapping[int, str]
) -> dict[int, str]:
    """Map the label ids an RF-DETR checkpoint predicts to class names.

    * no ``class_names`` (the hosted COCO weights): labels are COCO category ids.
    * a dict: already ``{label: name}``.
    * a list (fine-tuned on a Roboflow-style COCO folder): labels are the raw
      ``category_id`` values. RF-DETR sets ``num_classes`` to the number of
      categories but keeps only names whose supercategory is not ``"none"``;
      Roboflow exports put that ``"none"`` parent first (id 0), so the offset is
      ``num_classes - len(class_names)`` (0 for our own ``to_coco`` exports).
    """
    names = args.get("class_names")
    if not names:
        return {int(k): str(v) for k, v in coco_classes.items()}
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    names = [str(n) for n in names]
    offset = max(int(args.get("num_classes") or len(names)) - len(names), 0)
    return {i + offset: name for i, name in enumerate(names)}


def _rfdetr_device(device: str) -> str:
    """RF-DETR accepts only ``cpu``, ``cuda`` (current device) or ``mps``."""
    if device in {"cpu", "cuda", "mps"}:
        return device
    if device in {"cuda:0", "0"}:
        return "cuda"
    raise ValueError(
        f"RF-DETR cannot run on {device!r}: use 'cpu', 'cuda' or 'mps' and pick the GPU "
        "with CUDA_VISIBLE_DEVICES"
    )


def _check_train_options(options: dict[str, Any]) -> None:
    """RF-DETR silently ignores unknown training arguments; catch typos like ``epoch=5``."""
    import inspect

    from rfdetr.config import TrainConfig as RFTrainConfig
    from rfdetr.main import populate_args

    known = set(RFTrainConfig.model_fields) | set(
        inspect.signature(populate_args).parameters
    )
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"Unknown RF-DETR training options: {', '.join(unknown)}")


_COCO_STATS = ("map50_95", "map50", "map75")


def _read_log(path: Path) -> pd.DataFrame:
    """Per-epoch history from RF-DETR's ``log.txt`` (one JSON object per line).

    Epochs are 1-based. Scalar values become columns; COCO box stats (``*coco_eval_bbox`` lists) are
    expanded to ``<prefix>map50_95``, ``<prefix>map50`` and ``<prefix>map75``.
    """
    if not path.is_file():
        raise RuntimeError(f"RF-DETR training finished without writing {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = {}
        for key, value in json.loads(line).items():
            if key.endswith("coco_eval_bbox") and isinstance(value, list):
                prefix = key.removesuffix("coco_eval_bbox")
                row |= {
                    f"{prefix}{name}": v
                    for name, v in zip(_COCO_STATS, value, strict=False)
                }
            elif isinstance(value, int | float | str | bool) or value is None:
                row[key] = value
        rows.append(row)
    history = pd.DataFrame(rows)
    if "epoch" in history:
        history["epoch"] += (
            1  # RF-DETR counts from 0; report epochs 1..N like Ultralytics
        )
    return history


def _final_metrics(history: pd.DataFrame) -> dict[str, float]:
    """Validation metrics of the epoch behind ``checkpoint_best_total.pth``.

    RF-DETR keeps the best epoch of the regular and of the EMA model and copies
    the EMA one only if it is strictly better, so the same rule applies here.
    """
    best: tuple[float, str, int] | None = None
    for prefix in ("test_", "ema_test_"):
        column = f"{prefix}map50_95"
        if column not in history or history[column].isna().all():
            continue
        row = int(history[column].idxmax())
        value = float(history.at[row, column])
        if best is None or value > best[0]:
            best = (value, prefix, row)
    if best is None:
        return {}
    _, prefix, row = best
    metrics = {
        name: float(history.at[row, f"{prefix}{name}"])
        for name in _COCO_STATS
        if f"{prefix}{name}" in history
    }
    return metrics
