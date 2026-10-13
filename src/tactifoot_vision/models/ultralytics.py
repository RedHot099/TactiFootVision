"""Ultralytics YOLO backends: box detection (``yolo``) and keypoints (``yolo_pose``)."""

import contextlib
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.models.base import MODELS, Model, TrainConfig, TrainResult
from tactifoot_vision.utils import cache_dir

logger = logging.getLogger(__name__)

Resize = Literal["letterbox", "square", "stretch"]
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
        resize: Resize = "letterbox",
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
                  from those bands, such as the SoccerNet pitch keypoint model
                  ``pitch_yolo11m_pose_sn960.pt``: with ``"letterbox"`` it finds
                  the pitch on 20% of SoccerNet GSR frames, with ``"square"`` on all.
                * ``"stretch"``: resized to ``imgsz`` x ``imgsz``, ignoring the
                  aspect ratio, for checkpoints trained on images stretched to a
                  square, such as Roboflow exports with "Resize: Stretch" (the
                  Roboflow pitch model ``pitch_yolov8n_pose.pt``).
            device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...; ``None`` picks CUDA if available.
        """
        if resize not in get_args(Resize):
            raise ValueError(
                f"resize must be one of {', '.join(get_args(Resize))}, got {resize!r}"
            )
        self.iou = iou
        self.imgsz = imgsz
        self.resize = resize
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

    def _run(
        self, image: np.ndarray, conf: float | None = None
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """The Ultralytics result and the ``(x, y)`` scale and shift from its pixels
        to ``image``'s: ``image_xy = result_xy * scale + shift``."""
        options: dict[str, Any] = {
            "conf": self.conf if conf is None else conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
        }
        size = self.imgsz
        scale, shift = np.ones(2, np.float32), np.zeros(2, np.float32)
        height, width = image.shape[:2]
        if self.resize == "stretch":
            size = size or _trained_size(self._yolo, self.resize)
            image = cv2.resize(image, (size, size))
            scale = np.float32([width / size, height / size])
        elif self.resize == "square":
            size = size or _trained_size(self._yolo, self.resize)
            side = max(height, width)
            top, left = (side - height) // 2, (side - width) // 2
            canvas = np.full((side, side, *image.shape[2:]), _PAD_VALUE, image.dtype)
            canvas[top : top + height, left : left + width] = image
            image = canvas
            shift = np.float32([-left, -top])
        if size is not None:
            options["imgsz"] = size
        return self._yolo.predict(image, **options)[0], scale, shift

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
            with contextlib.chdir(run_dir):
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


@MODELS.register("yolo")
class YOLODetector(_UltralyticsModel):
    """Ultralytics YOLO box detector (YOLOv8, YOLO11, ... detection checkpoints).

    ``predict`` returns ``sv.Detections`` with ``data["class_name"]``; ``class_id``
    indexes :attr:`class_names`.

    Example::

        model = tv.load_model("yolo", "models/football_yolo11m.pt", conf=0.3)
        detections = model(frame)
        # The ball is a few pixels wide: a larger input and a lower threshold for it.
        model = tv.load_model(
            "yolo", "models/football_yolo11m.pt", conf=0.3, imgsz=1920,
            class_conf={"ball": 0.15},
        )
    """

    name = "yolo"
    task = Task.DETECT
    default_weights = "yolo11n.pt"

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | None = None,
        resize: Resize = "letterbox",
        class_conf: Mapping[str, float] | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            weights, conf, iou, imgsz, resize, device: as for every Ultralytics backend.
            class_conf: minimum confidence per class name, replacing ``conf``
                for the classes it names (e.g. ``{"ball": 0.15}``). Other
                classes keep exactly the detections they get with ``conf``
                alone (non-maximum suppression is per class), unless the
                lower threshold lets more than Ultralytics' ``max_det`` (300)
                candidates through, which football frames do not reach.
        """
        self.class_conf = dict(class_conf or {})
        for name, value in self.class_conf.items():
            if not 0 <= value <= 1:
                raise ValueError(f"class_conf[{name!r}] must be in [0, 1], got {value}")
        super().__init__(
            weights, conf=conf, iou=iou, imgsz=imgsz, resize=resize, device=device
        )
        unknown = []
        if self.class_conf:
            unknown = sorted(set(self.class_conf) - set(self.class_names))
        if unknown:
            raise ValueError(
                f"class_conf names unknown class(es) {', '.join(unknown)}; "
                f"the model has {', '.join(self.class_names)}"
            )

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Detect objects in one BGR image."""
        floor = min([self.conf, *self.class_conf.values()])
        result, scale, shift = self._run(image, conf=floor)
        detections = sv.Detections.from_ultralytics(result)
        detections.xyxy = detections.xyxy * np.tile(scale, 2) + np.tile(shift, 2)
        class_id = detections.class_id.astype(int)
        detections.data["class_name"] = np.asarray(self.class_names, dtype=str)[
            class_id
        ]
        if self.class_conf:
            thresholds = np.array(
                [self.class_conf.get(name, self.conf) for name in self.class_names]
            )
            detections = detections[detections.confidence >= thresholds[class_id]]
        return detections


@MODELS.register("yolo_pose")
class YOLOPoseModel(_UltralyticsModel):
    """Ultralytics YOLO pose model, e.g. pitch landmarks (32 keypoints per pitch).

    ``predict`` returns ``sv.KeyPoints`` holding every detected instance, sorted by
    box confidence (best first), with per-keypoint confidence. ``data`` carries
    ``class_name`` and the instance boxes (``xyxy``, ``box_confidence``).

    Example::

        # The SoccerNet pitch model learnt on square canvases at 960.
        model = tv.load_model(
            "yolo_pose", "models/pitch_yolo11m_pose_sn960.pt", conf=0.1, resize="square"
        )
        keypoints = model(frame)
        best = keypoints.xy[0]  # (32, 2) pixel coordinates of the best pitch
    """

    name = "yolo_pose"
    task = Task.POSE
    default_weights = "yolov8n-pose.pt"

    def predict(self, image: np.ndarray) -> sv.KeyPoints:
        """Detect keypoint instances in one BGR image."""
        result, scale, shift = self._run(image)
        if result.keypoints is None or len(result.boxes) == 0:
            return _keypoints_from_arrays(
                np.zeros((0, 0, 3), np.float32),
                np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, int),
                self.class_names,
            )
        keypoints = result.keypoints.data.cpu().numpy()
        keypoints[..., :2] = keypoints[..., :2] * scale + shift
        return _keypoints_from_arrays(
            keypoints,
            result.boxes.xyxy.cpu().numpy() * np.tile(scale, 2) + np.tile(shift, 2),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            self.class_names,
        )


def _trained_size(yolo: Any, resize: str) -> int:
    """The square input size a checkpoint was trained at (``imgsz`` of its training arguments)."""
    size = getattr(yolo, "overrides", {}).get("imgsz")
    if isinstance(size, list | tuple) and len(set(size)) == 1:
        size = size[0]
    if not isinstance(size, int):
        raise ValueError(
            f"resize={resize!r} needs a square training size, but the checkpoint "
            f"records imgsz={size!r}; pass imgsz"
        )
    return size


def _keypoints_from_arrays(
    keypoints: np.ndarray,
    xyxy: np.ndarray,
    box_confidence: np.ndarray,
    class_id: np.ndarray,
    class_names: list[str],
) -> sv.KeyPoints:
    """Build ``sv.KeyPoints`` sorted by instance confidence from raw pose outputs.

    ``keypoints`` is ``(M, K, 3)`` (x, y, confidence) or ``(M, K, 2)``; without a
    per-keypoint confidence every keypoint gets its instance's confidence.
    No instances give ``xy`` of shape ``(0, 0, 2)``, the only empty shape
    ``sv.KeyPoints`` accepts.
    """
    if len(keypoints) == 0:
        keypoints = keypoints.reshape(0, 0, keypoints.shape[-1])
    order = np.argsort(-box_confidence, kind="stable")
    keypoints, box_confidence = keypoints[order], box_confidence[order]
    if keypoints.shape[-1] == 3:
        confidence = keypoints[..., 2]
    else:
        confidence = np.repeat(box_confidence[:, None], keypoints.shape[1], axis=1)
    class_id = class_id[order].astype(int)
    return sv.KeyPoints(
        xy=keypoints[..., :2].astype(np.float32),
        confidence=confidence.astype(np.float32),
        class_id=class_id,
        data={
            "class_name": np.asarray(class_names, dtype=str)[class_id],
            "xyxy": xyxy[order].astype(np.float32).reshape(-1, 4),
            "box_confidence": box_confidence.astype(np.float32),
        },
    )


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
_METRIC_PREFIXES = {"B": "", "P": "pose_", "M": "mask_"}


def _standard_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """``metrics/mAP50-95(B)`` -> ``map50_95``, ``metrics/mAP50(P)`` -> ``pose_map50``, ..."""
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
