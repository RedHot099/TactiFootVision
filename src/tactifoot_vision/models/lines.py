"""Pitch line backend (``lines``): a heatmap network that finds the painted pitch markings.

For every :class:`~tactifoot_vision.enums.PitchMarking` (17 straight lines,
3 circle arcs) the network predicts a heatmap of the marking's visible part;
:meth:`PitchLineModel.predict` decodes it into points along each marking
(:class:`~tactifoot_vision.pitch.fit.LinePoints`), which
:class:`~tactifoot_vision.pitch.HomographyEstimator` fits the camera to.
The architecture and decoder are in ``_line_net`` (torch, loaded only when
a model is created).

The package ships no weights. Train them with
``tv.train("lines", tv.load_pitch_lines(calibration, gamestate))`` (see
:meth:`PitchLineModel._train` for the recipe) and load the result with
``tv.load_model("lines", "runs/lines/weights/final.pt")``.
"""

import hashlib
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.enums import LineBackbone, ModelType, PitchMarking, Split, Task
from tactifoot_vision.models.base import MODELS, Model, TrainConfig, TrainResult
from tactifoot_vision.pitch.fit import LinePoints

logger = logging.getLogger(__name__)

# The recipe of the shipped configuration (ADR 0005); TrainConfig fields left at
# their defaults take these values.
DEFAULT_LR = 6e-4
DEFAULT_WEIGHT_DECAY = 1e-4
_OPTIONS = ("weight_decay", "stop")


class TrainingStopped(RuntimeError):
    """Training ended early at an epoch checkpoint (the ``stop`` option said so); train again to resume."""

    def __init__(self, run_dir: Path, epoch: int) -> None:
        super().__init__(
            f"Training stopped after epoch {epoch}; call train again with the same "
            f"output_dir and name and exist_ok=True to resume from {run_dir / 'weights' / 'last.pt'}"
        )
        self.run_dir = run_dir
        self.epoch = epoch


@MODELS.register(ModelType.LINES)
class PitchLineModel(Model):
    """Pitch marking heatmaps decoded into points along each marking.

    ``weights=None`` builds a new network (an ImageNet-pretrained encoder
    when ``pretrained``) of ``backbone`` (``resnet18`` or ``resnet34``),
    output ``stride`` (2 or 4) and input ``width`` (the frame keeps 16:9),
    for training; a checkpoint carries its own shape, so these three must
    then be left unset. ``conf`` is the heatmap probability a point needs.

    Example::

        model = tv.load_model("lines", "runs/lines/weights/final.pt")
        points = model(frame)  # LinePoints: marking index, pixel, probability
    """

    name = ModelType.LINES.value
    task = Task.LINES

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.2,
        device: str | None = None,
        backbone: LineBackbone | str | None = None,
        stride: int | None = None,
        width: int | None = None,
        pretrained: bool = True,
    ) -> None:
        shape = {"backbone": backbone, "stride": stride, "width": width}
        self._shape = {k: v for k, v in shape.items() if v is not None}
        if weights is not None and self._shape:
            raise ValueError(
                f"{', '.join(self._shape)} set the shape of a new network; "
                "a checkpoint carries its own, so leave them unset with weights"
            )
        self.pretrained = pretrained
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights: str | Path | None) -> None:
        from tactifoot_vision.models import _line_net

        self._lib = _line_net
        spec = _line_net.NetSpec(**self._shape)
        self.net = _line_net.load_net(weights, spec, self.pretrained, self.device)
        self.spec = self.net.spec

    @property
    def class_names(self) -> list[str]:
        return [m.value for m in PitchMarking]

    def predict(self, image: np.ndarray) -> LinePoints:
        """Points along the markings in one BGR frame (frame pixels)."""
        return self._lib.predict(self.net, [image], self.conf, self.device)[0]

    # ------------------------------------------------------------- training
    def _check_options(self, config: TrainConfig) -> None:
        super()._check_options(config)
        unknown = sorted(set(config.backend_options) - set(_OPTIONS))
        if unknown:
            raise ValueError(
                f"Unknown lines option(s) {', '.join(unknown)}; the backend takes {', '.join(_OPTIONS)}"
            )
        stop = config.backend_options.get("stop")
        if stop is not None and not callable(stop):
            raise ValueError("stop must be a function of the finished epoch count")
        if config.imgsz != self.spec.width:
            raise ValueError(
                f"imgsz {config.imgsz} differs from the network's input width {self.spec.width}; "
                "a new network of another width is load_model('lines', width=...)"
            )
        if config.patience is not None:
            raise ValueError(
                "The lines backend trains a fixed number of epochs; leave patience unset"
            )

    def _train(
        self, dataset: Dataset, config: TrainConfig, run_dir: Path
    ) -> TrainResult:
        """The recipe: focal loss on Gaussian line heatmaps (sigma one output cell), AdamW
        (``lr`` default 6e-4, ``weight_decay`` 1e-4), cosine schedule after a 500-step warm-up,
        bf16 on CUDA, gradient clip 5; augmentation: a random zoom, a horizontal mirror
        (marking names swapped by each sample's view), colour jitter, blur.

        One checkpoint per epoch (``run_dir/weights/last.pt``); a run folder
        that holds one resumes from it. The final weights are the last
        epoch's (``weights/final.pt``); the valid split is only scored (its
        loss each epoch), never used to pick weights.

        Before the first epoch, ``run_dir/samples.jsonl`` lists every sample
        (split, image, weight, view and a hash of its line points), and
        ``run_dir/manifest.json`` records the recipe, the data (counts and
        the hashes of both splits and of the sample list) and the session.
        A run folder is resumed only when its manifest holds the same recipe
        and data; a checkpoint without a manifest is refused. After each
        session the manifest gets the session's end, its wall time and the
        weights' hash.
        """
        from tactifoot_vision.utils import resolve_device

        options = config.backend_options
        device = resolve_device(config.device) if config.device else self.device
        lr = config.lr if config.lr is not None else DEFAULT_LR
        weight_decay = float(options.get("weight_decay", DEFAULT_WEIGHT_DECAY))
        stop: Callable[[int], bool] | None = options.get("stop")
        train, valid = dataset[Split.TRAIN], dataset[Split.VALID]
        manifest_path = run_dir / "manifest.json"
        recipe = {
            "epochs": config.epochs, "batch_size": config.batch_size, "lr": lr,
            "weight_decay": weight_decay, "seed": config.seed, "workers": config.workers,
            "spec": asdict(self.spec), "pretrained": self.pretrained, "start_weights": _str(self.weights),
        }  # fmt: skip
        index = _sample_index({Split.TRAIN: train, Split.VALID: valid})
        data = {
            "name": dataset.name,
            "train": len(train),
            "valid": len(valid),
            "train_sha256": _samples_sha(train),
            "valid_sha256": _samples_sha(valid),
            "samples_file": "samples.jsonl",
            "samples_sha256": hashlib.sha256(index.encode()).hexdigest(),
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["recipe"] != recipe or previous["data"] != data:
                raise ValueError(
                    f"{run_dir} holds a run of another recipe or data; pick another name"
                )
            sessions = previous["sessions"]
        elif (run_dir / "weights" / "last.pt").exists():
            raise ValueError(
                f"{run_dir} holds a checkpoint without a manifest: nothing says which "
                "recipe and data it was trained on; pick another name"
            )
        else:
            sessions = []
            (run_dir / "samples.jsonl").write_text(index)
        session = {
            "started": _now(),
            "device": device,
            "host": platform.node(),
            **_versions(),
        }
        manifest = {
            "model": self.name,
            "decoder_version": self._lib.DECODER_VERSION,
            "recipe": recipe,
            "data": data,
            "sessions": [*sessions, session],
            "weights": None,
            "weights_sha256": None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        if self.weights is None:  # a new network: its random layers come from the seed
            self.net = self._lib.load_net(
                None, self.spec, self.pretrained, device, config.seed
            )
        started = time.time()
        weights, log = self._lib.fit(
            self.net, train, valid, epochs=config.epochs, batch_size=config.batch_size, lr=lr,
            workers=config.workers, seed=config.seed, device=device, run_dir=run_dir,
            weight_decay=weight_decay, stop=stop,
        )  # fmt: skip
        session |= {
            "finished": _now(),
            "wall_s": round(time.time() - started, 1),
            "epochs_done": len(log),
        }
        manifest |= {
            "sessions": [*sessions, session],
            "weights": None if weights is None else weights.name,
            "weights_sha256": None if weights is None else _file_sha(weights),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        if weights is None:
            raise TrainingStopped(run_dir, len(log))
        history = pd.DataFrame(log)
        last = log[-1] if log else {}
        return TrainResult(
            model_type=self.name,
            weights=weights,
            run_dir=run_dir,
            history=history,
            metrics={k: float(last[k]) for k in ("loss", "valid_loss") if k in last},
        )


def _sample_index(splits: dict) -> str:
    """One JSON line per sample: split, image, weight, view and the sha256 of its line points."""
    lines = []
    for split, samples in splits.items():
        for s in samples:
            a = s.annotations
            h = hashlib.sha256()
            for name in sorted(a.lines or {}):
                h.update(
                    name.encode() + np.asarray(a.lines[name], np.float32).tobytes()
                )
            lines.append(json.dumps({"split": str(split), "image": str(s.image_path), "weight": a.weight,
                                     "view": str(a.view), "lines_sha256": h.hexdigest()}))  # fmt: skip
    return "\n".join(lines) + "\n"


def _samples_sha(samples: list) -> str:
    """A hash of a split: every image path with its labels, weight and view."""
    h = hashlib.sha256()
    for s in samples:
        a = s.annotations
        h.update(f"{s.image_path}|{a.weight}|{a.view}\n".encode())
        for name in sorted(a.lines or {}):
            h.update(name.encode() + np.asarray(a.lines[name], np.float32).tobytes())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions() -> dict[str, Any]:
    import torch
    import torchvision

    from tactifoot_vision import __version__

    out = {
        "package": __version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
    try:
        here = Path(__file__).parent
        commit = subprocess.run(["git", "-C", str(here), "describe", "--always", "--dirty", "--abbrev=12"],
                                capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        out["commit"] = commit
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _str(weights: str | Path | None) -> str | None:
    return None if weights is None else str(weights)
