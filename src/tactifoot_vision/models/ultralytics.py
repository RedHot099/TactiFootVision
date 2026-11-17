"""Ultralytics YOLO backend: box detection (``yolo``)."""

import base64
import contextlib
import io
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import supervision as sv
from tqdm.auto import tqdm

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.enums import DetectionKey, ModelType, Resize
from tactifoot_vision.models.base import (
    MODELS,
    Model,
    TrainConfig,
    TrainResult,
    check_class_conf,
)
from tactifoot_vision.utils import cache_dir

logger = logging.getLogger(__name__)

# Ultralytics' letterbox grey, the colour of the bands around a training image.
_PAD_VALUE = 114


class _UltralyticsModel(Model):
    """Loading, settings and training shared by the Ultralytics backends."""

    default_weights: ClassVar[str]
    # Ultralytics' names for TrainConfig fields that differ from them (the
    # others, such as imgsz or epochs, are the same name and so the field itself).
    _train_config_aliases = {"batch": "batch_size", "lr0": "lr"}

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | None = None,
        resize: Resize | str = Resize.LETTERBOX,
        device: str | None = None,
    ) -> None:
        """
        Args:
            weights: checkpoint path or an Ultralytics model name (``"yolo11s.pt"``,
                downloaded by Ultralytics); ``None`` uses ``default_weights``.
            conf: minimum confidence of returned objects.
            iou: IoU threshold of non-maximum suppression.
            imgsz: inference size; ``None`` keeps the size the model was trained at.
            resize: how the image becomes the model's input; outputs are always in
                image pixels. Match it to how the checkpoint saw its training images:

                * ``"letterbox"``: Ultralytics' inference default, the aspect ratio
                  kept and the image padded only to a multiple of the stride
                  (960 x 544 for a 16:9 frame at 960).
                * ``"square"``: the aspect ratio kept and the image padded to a
                  centred square with Ultralytics' grey (114), as Ultralytics'
                  training letterboxes every image. For checkpoints that learnt
                  from those bands and miss without them. The square is
                  resized to ``imgsz``, by default the size the checkpoint
                  was trained at.
            device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...; ``None`` picks CUDA if available.
        """
        self.iou = iou
        self.imgsz = imgsz
        self.resize = Resize.parse(resize, "resize")
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights: str | Path | None) -> None:
        from ultralytics import YOLO

        if weights is None:
            weights = self.default_weights
            self.weights = weights
        model = YOLO(str(_checkpoint_path(weights)))
        if model.task != self.task.value:
            raise ValueError(
                f"{weights} is a {model.task!r} checkpoint but {type(self).__name__} "
                f"needs a {self.task.value!r} one"
            )
        self._yolo = model

    @property
    def class_names(self) -> list[str]:
        names = self._yolo.names
        return [names[i] for i in sorted(names)]

    def _run(self, image: np.ndarray) -> tuple[Any, np.ndarray]:
        """The Ultralytics result and the ``(x, y)`` shift from its pixels to
        ``image``'s: ``image_xy = result_xy + shift``."""
        options: dict[str, Any] = {
            "conf": self._lowest_conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
        }
        size = self.imgsz
        shift = np.zeros(2, np.float32)
        if self.resize == Resize.SQUARE:
            size = size or _trained_size(self._yolo)
            height, width = image.shape[:2]
            side = max(height, width)
            top, left = (side - height) // 2, (side - width) // 2
            canvas = np.full((side, side, *image.shape[2:]), _PAD_VALUE, image.dtype)
            canvas[top : top + height, left : left + width] = image
            image = canvas
            shift = np.float32([-left, -top])
        if size is not None:
            options["imgsz"] = size
        return self._yolo.predict(image, **options)[0], shift

    def _check_options(self, config: TrainConfig) -> None:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        super()._check_options(config)
        unknown = sorted(set(config.backend_options) - set(DEFAULT_CFG_DICT))
        if unknown:
            raise ValueError(
                f"Unknown Ultralytics training options: {', '.join(unknown)}"
            )

    def _train(
        self, dataset: Dataset, config: TrainConfig, run_dir: Path
    ) -> TrainResult:
        # Ultralytics appends to an existing results.csv (exist_ok=True re-runs);
        # the history must describe this run only.
        (run_dir / "results.csv").unlink(missing_ok=True)
        data_yaml = dataset.to_yolo(run_dir / "dataset")
        args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.imgsz,
            "device": config.device or self.device,
            "workers": config.workers,
            "seed": config.seed,
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
        }
        if config.lr is not None:
            args["lr0"] = config.lr
        if config.patience is not None:
            args["patience"] = config.patience
        args |= config.backend_options
        logger.info("Ultralytics training arguments: %s", args)

        # Ultralytics' AMP check loads "yolo11n.pt" from the working directory and
        # downloads it there if missing. Run from the run folder, with a link to a
        # cached copy, so nothing is downloaded twice or dropped in the caller's folder.
        probe = run_dir / _AMP_CHECK_WEIGHTS
        if args.get("amp", True) and not probe.exists():
            probe.unlink(missing_ok=True)  # a dangling link left by an interrupted run
            probe.symlink_to(_cached_asset(_AMP_CHECK_WEIGHTS))
        try:
            with contextlib.chdir(run_dir), _console(self._yolo, config.progress):
                self._yolo.train(**args)
        finally:
            if probe.is_symlink():
                probe.unlink()
        trainer = self._yolo.trainer
        save_dir = Path(trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError(f"Ultralytics training finished without writing {best}")
        return TrainResult(
            model_type=self.name,
            weights=best,
            run_dir=save_dir,
            history=_read_results_csv(save_dir / "results.csv"),
            metrics=_standard_metrics(trainer.metrics or {}),
        )


@contextlib.contextmanager
def _console(yolo: Any, progress: bool) -> Iterator[None]:
    """What an Ultralytics training run prints: its own log, or one progress bar.

    Ultralytics' log draws a bar per epoch and per validation, sized for a
    terminal: in a notebook the lines run past the cell and each redraw wipes
    the epoch header. With ``progress`` its log is lowered to warnings (which
    also turns its bars off) and :class:`_TrainingBar` shows the run instead.
    """
    if not progress:
        yield
        return
    log = logging.getLogger("ultralytics")
    level = log.level
    bar = _TrainingBar()
    events = {
        "on_train_start": bar.start,
        "on_train_epoch_start": bar.epoch_start,
        "on_train_batch_end": bar.batch_end,
        "on_fit_epoch_end": bar.epoch_end,
        "on_train_end": bar.finish,
    }
    for event, callback in events.items():
        yolo.add_callback(event, callback)
    # Never louder than the user set it (YOLO_VERBOSE=False means errors only).
    log.setLevel(max(log.getEffectiveLevel(), logging.WARNING))
    try:
        yield
    finally:
        log.setLevel(level)
        bar.close()  # still open only when training failed: tqdm marks it so
        for event, callback in events.items():
            yolo.callbacks[event].remove(callback)


# Ultralytics' distribution focal loss tracks the box loss; without it the
# live plot (box loss, class loss, metrics) fits one row.
_HIDDEN_LOSSES = ("dfl",)


class _TrainingBar:
    """One bar over every batch of a run, fed by Ultralytics' trainer callbacks:
    the epoch and the training loss from the first batch, the validation mAP50
    after each epoch. In a notebook, :func:`tv.viz.plot_training` of the
    history so far is drawn under the bar and redrawn after each epoch."""

    def __init__(self) -> None:
        self._bar: Any = None
        self._map50 = ""
        self._curves: Any = None  # the notebook display that holds the plot
        self._epoch = 0  # the epoch under way, from "on_train_epoch_start"

    def start(self, trainer: Any) -> None:
        self._epoch = getattr(trainer, "start_epoch", 0)
        batches = len(trainer.train_loader)
        epochs = trainer.epochs - getattr(trainer, "start_epoch", 0)  # a resumed run
        self._bar = tqdm(total=epochs * batches, desc="Training", unit="batch")
        self._show(trainer)

    def epoch_start(self, trainer: Any) -> None:
        # Ultralytics may change the run under way: a smaller batch after an
        # out-of-memory error, a redone epoch after NaN losses, epochs from a
        # time limit. Count the batches left from here.
        self._epoch = trainer.epoch
        if self._bar is not None:
            left = trainer.epochs - trainer.epoch
            self._bar.total = self._bar.n + left * len(trainer.train_loader)
            self._bar.refresh()

    def batch_end(self, trainer: Any) -> None:
        if self._bar is not None:
            self._show(trainer, refresh=False)  # drawn by the update
            self._bar.update(1)

    def epoch_end(self, trainer: Any) -> None:
        # The box metric: a detection run reports mAP50(B); a run of another
        # Ultralytics task adds its own after it (masks, keypoints), not shown.
        metrics = trainer.metrics or {}
        if "metrics/mAP50(B)" in metrics:
            self._map50 = f"{metrics['metrics/mAP50(B)']:.3f}"
        if self._bar is not None:
            self._show(trainer)
        self._plot(trainer)

    def _plot(self, trainer: Any) -> None:
        # Ultralytics appends the epoch to results.csv before "on_fit_epoch_end";
        # the final validation of best.pt calls it once more and adds no row.
        csv = Path(getattr(trainer, "csv", ""))
        if not _in_notebook() or not csv.is_file():
            return
        from IPython.display import display

        from tactifoot_vision.viz import plot_training

        figure = plot_training(
            _read_results_csv(csv), skip_losses=_HIDDEN_LOSSES, transparent=True
        )
        png = io.BytesIO()
        figure.savefig(png, format="png", dpi=100, transparent=True)
        image = _TransparentImage(png.getvalue())
        if self._curves is None:
            self._curves = display(image, display_id=True)
        else:
            self._curves.update(image)

    def finish(self, trainer: Any) -> None:
        """Training ended, perhaps early (patience, time limit): a full bar."""
        if self._bar is not None:
            self._bar.total = self._bar.n
        self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _show(self, trainer: Any, refresh: bool = True) -> None:
        # Not trainer.epoch: the final validation of best.pt runs one past the
        # last epoch, which after an early stop is not the last planned one.
        parts = [f"epoch {self._epoch + 1}/{trainer.epochs}"]
        # The running mean of each loss: a dict by name (a tensor before 8.4).
        losses = getattr(trainer, "tloss", None)
        if losses is not None:
            values = losses.values() if isinstance(losses, Mapping) else losses
            parts.append(f"loss {sum(float(v) for v in values):.3f}")
        if self._map50:
            parts.append(f"val mAP50 {self._map50}")
        self._bar.set_postfix_str(", ".join(parts), refresh=refresh)


class _TransparentImage:
    """A PNG that a notebook shows without a background, at most as wide as the cell.

    Shown as HTML: Colab's dark theme puts a white background under image
    outputs, which an inline style overrides. The PNG is the fallback for
    viewers that do not render HTML. (A figure in the notebook's own format
    may be JPEG, which has no transparency.)
    """

    def __init__(self, png: bytes) -> None:
        self.png = png

    def __repr__(self) -> str:  # the text/plain output: no memory address
        return "Training curves"

    def _repr_png_(self) -> bytes:
        return self.png

    def _repr_html_(self) -> str:
        data = base64.b64encode(self.png).decode("ascii")
        style = "background: transparent; max-width: 100%; height: auto"
        return f'<img src="data:image/png;base64,{data}" style="{style}">'


def _in_notebook() -> bool:
    """Whether this runs in a Jupyter kernel (Colab too), where figures display."""
    if "IPython" not in sys.modules:  # a script or the CLI: no kernel to ask
        return False
    from IPython import get_ipython

    return hasattr(get_ipython(), "kernel")  # a terminal IPython has none


@MODELS.register(ModelType.YOLO)
class YOLODetector(_UltralyticsModel):
    """Ultralytics YOLO box detector (YOLOv8, YOLO11, ... detection checkpoints).

    ``predict`` returns ``sv.Detections`` with ``data["class_name"]``; ``class_id``
    indexes :attr:`class_names`.

    Example::

        model = tv.load_model("yolo", "models/detector_yolo11l.pt", conf=0.3)
        detections = model(frame)
    """

    name = ModelType.YOLO.value
    task = Task.DETECT
    default_weights = "yolo11n.pt"

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        class_conf: Mapping[str, float] | None = None,
        iou: float = 0.7,
        imgsz: int | None = None,
        resize: Resize | str = Resize.LETTERBOX,
        device: str | None = None,
    ) -> None:
        """
        Args:
            weights: checkpoint path or an Ultralytics model name (``"yolo11s.pt"``,
                downloaded by Ultralytics); ``None`` uses ``default_weights``.
            conf: minimum confidence of returned objects.
            class_conf: minimum confidence per class name, replacing ``conf``
                for those classes, e.g. ``{"ball": 0.05}`` so that weak ball
                candidates are kept.
            iou: IoU threshold of non-maximum suppression.
            imgsz: inference size; ``None`` keeps the size the model was trained at.
            resize: ``"letterbox"`` or ``"square"``, as for every Ultralytics
                backend: how the image becomes the model's input.
            device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...; ``None`` picks CUDA if available.
        """
        self.class_conf = check_class_conf(class_conf)
        super().__init__(
            weights, conf=conf, iou=iou, imgsz=imgsz, resize=resize, device=device
        )

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Detect objects in one BGR image."""
        result, shift = self._run(image)
        detections = sv.Detections.from_ultralytics(result)
        detections.xyxy = detections.xyxy + np.tile(shift, 2)
        detections.data[DetectionKey.CLASS_NAME] = np.asarray(
            self.class_names, dtype=str
        )[detections.class_id.astype(int)]
        return self._apply_class_conf(detections)


def _trained_size(yolo: Any) -> int:
    """The square input size a checkpoint was trained at (``imgsz`` of its training arguments)."""
    size = getattr(yolo, "overrides", {}).get("imgsz")
    if isinstance(size, list | tuple) and len(set(size)) == 1:
        size = size[0]
    if not isinstance(size, int):
        raise ValueError(
            f"resize='square' needs a square training size, but the checkpoint "
            f"records imgsz={size!r}; pass imgsz"
        )
    return size


_AMP_CHECK_WEIGHTS = "yolo11n.pt"


def _checkpoint_path(weights: str | Path) -> Path:
    """A local checkpoint, or an Ultralytics release asset by bare name (downloaded once).

    Only names Ultralytics publishes (``yolo11n.pt``, ...) are downloaded; any
    other missing file is ``FileNotFoundError``.
    """
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    path = Path(weights)
    if path.is_file():
        return path
    if path.parent != Path(".") or path.name not in GITHUB_ASSETS_NAMES:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return _cached_asset(path.name)


def _cached_asset(name: str) -> Path:
    """An Ultralytics release asset (e.g. ``yolo11n.pt``), downloaded once to the cache."""
    from ultralytics.utils.downloads import attempt_download_asset

    return Path(attempt_download_asset(cache_dir("ultralytics") / name))


_METRIC_NAMES = {
    "precision": "precision",
    "recall": "recall",
    "mAP50": "map50",
    "mAP50-95": "map50_95",
}
_METRIC_PREFIXES = {"B": "", "M": "mask_"}


def _standard_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """``metrics/mAP50-95(B)`` -> ``map50_95``, ``metrics/mAP50(M)`` -> ``mask_map50``, ..."""
    metrics = {}
    for key, value in raw.items():
        match = re.fullmatch(r"metrics/(.+)\((\w)\)", str(key))
        if match and match[1] in _METRIC_NAMES and match[2] in _METRIC_PREFIXES:
            metrics[_METRIC_PREFIXES[match[2]] + _METRIC_NAMES[match[1]]] = float(value)
    return metrics


def _read_results_csv(path: Path) -> pd.DataFrame:
    """Per-epoch history written by Ultralytics (``results.csv``), column names stripped."""
    if not path.is_file():
        raise RuntimeError(f"Ultralytics training finished without writing {path}")
    history = pd.read_csv(path)
    history.columns = history.columns.str.strip()
    return history
