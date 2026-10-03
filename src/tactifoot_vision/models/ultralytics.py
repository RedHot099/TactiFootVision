"""Ultralytics YOLO backends: box detection (``yolo``) and keypoints (``yolo_pose``)."""

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.models.base import MODELS, Model, TrainConfig, TrainResult
from tactifoot_vision.utils import cache_dir

logger = logging.getLogger(__name__)


class _UltralyticsModel(Model):
    """Loading, settings and training shared by the Ultralytics backends."""

    default_weights: ClassVar[str]

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            weights: checkpoint path or an Ultralytics model name (``"yolo11s.pt"``,
                downloaded by Ultralytics); ``None`` uses ``default_weights``.
            conf: minimum confidence of returned objects.
            iou: IoU threshold of non-maximum suppression.
            imgsz: inference size; ``None`` keeps the size the model was trained at.
            device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...; ``None`` picks CUDA if available.
        """
        self.iou = iou
        self.imgsz = imgsz
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights: str | Path | None) -> None:
        from ultralytics import YOLO

        if weights is None:
            weights = self.default_weights
            self.weights = weights
        path = Path(weights)
        if not path.is_file():
            if path.parent != Path("."):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            path = _cached_asset(path.name)  # a bare name such as "yolo11n.pt"
        model = YOLO(str(path))
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

    def _run(self, image: np.ndarray) -> Any:
        options: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
        }
        if self.imgsz is not None:
            options["imgsz"] = self.imgsz
        return self._yolo.predict(image, **options)[0]

    def _check_options(self, options: dict[str, Any]) -> None:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        super()._check_options(options)

        unknown = sorted(set(options) - set(DEFAULT_CFG_DICT))
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
    """

    name = "yolo"
    task = Task.DETECT
    default_weights = "yolo11n.pt"

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Detect objects in one BGR image."""
        detections = sv.Detections.from_ultralytics(self._run(image))
        detections.data["class_name"] = np.asarray(self.class_names, dtype=str)[
            detections.class_id.astype(int)
        ]
        return detections


@MODELS.register("yolo_pose")
class YOLOPoseModel(_UltralyticsModel):
    """Ultralytics YOLO pose model, e.g. pitch landmarks (32 keypoints per pitch).

    ``predict`` returns ``sv.KeyPoints`` holding every detected instance, sorted by
    box confidence (best first), with per-keypoint confidence. ``data`` carries
    ``class_name`` and the instance boxes (``xyxy``, ``box_confidence``).

    Example::

        model = tv.load_model("yolo_pose", "models/pitch_yolov8n_pose.pt")
        keypoints = model(frame)
        best = keypoints.xy[0]  # (32, 2) pixel coordinates of the best pitch
    """

    name = "yolo_pose"
    task = Task.POSE
    default_weights = "yolov8n-pose.pt"

    def predict(self, image: np.ndarray) -> sv.KeyPoints:
        """Detect keypoint instances in one BGR image."""
        result = self._run(image)
        if result.keypoints is None or len(result.boxes) == 0:
            return _keypoints_from_arrays(
                np.zeros((0, 0, 3), np.float32),
                np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, int),
                self.class_names,
            )
        return _keypoints_from_arrays(
            result.keypoints.data.cpu().numpy(),
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            self.class_names,
        )


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
