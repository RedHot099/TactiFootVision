"""HOTA by TrackEval, the reference implementation, from the optional ``eval`` extra."""

import builtins
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
import supervision as sv

type Frame = tuple[np.ndarray, np.ndarray]
"""Track ids and xyxy boxes of one class in one frame."""


class Hota(NamedTuple):
    """HOTA and its detection (DetA) and association (AssA) parts.

    Each is the mean over TrackEval's IoU thresholds 0.05, 0.10, ..., 0.95.
    """

    hota: float
    deta: float
    assa: float


NO_HOTA = Hota(float("nan"), float("nan"), float("nan"))


def hota_by_class(
    clips: Sequence[tuple[dict[str, Sequence[Frame]], dict[str, Sequence[Frame]]]],
) -> tuple[dict[str, Hota], Hota]:
    """HOTA of each class over ``clips``, and overall.

    Each clip is ``(truth, predicted)``, ``truth[name][i]`` against
    ``predicted[name][i]``; a class is scored in the clips that have it. As
    TrackEval does, the clips of a class are combined first, then the classes:
    both add up the detections.
    """
    metric = _trackeval_hota()
    with _numpy_aliases():
        by_class: dict[str, dict[int, Any]] = {}
        for i, (truth, predicted) in enumerate(clips):
            for name in truth:
                by_class.setdefault(name, {})[i] = metric.eval_sequence(
                    _sequence(truth[name], predicted[name])
                )
        if not by_class:
            return {}, Hota(0.0, 0.0, 0.0)
        results = {name: metric.combine_sequences(r) for name, r in by_class.items()}
        overall = metric.combine_classes_det_averaged(results)
    return {name: _summary(r) for name, r in results.items()}, _summary(overall)


def _trackeval_hota() -> Any:
    try:
        from trackeval.metrics import HOTA
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != "trackeval":
            raise  # TrackEval is installed but broken: say what is missing
        raise ImportError(
            "HOTA needs TrackEval, from the eval extra: run `uv sync --extra eval` "
            "(or `pip install -e '.[eval]'` in a checkout), or pass hota=False"
        ) from error
    return HOTA()


@contextmanager
def _numpy_aliases() -> Iterator[None]:
    """Lend numpy the ``np.float`` and ``np.int`` that TrackEval still uses (numpy 1.24 removed them).

    The aliases are set on the numpy module itself, so other threads see them
    while HOTA runs.
    """
    missing = [name for name in ("float", "int") if name not in vars(np)]
    for name in missing:
        setattr(np, name, getattr(builtins, name))
    try:
        yield
    finally:
        for name in missing:
            delattr(np, name)


def _sequence(truth: Sequence[Frame], predicted: Sequence[Frame]) -> dict[str, Any]:
    """TrackEval's input to a metric: track ids numbered from 0 and the IoU in each frame."""
    gt_ids, num_gt_ids = _renumbered([ids for ids, _ in truth])
    tracker_ids, num_tracker_ids = _renumbered([ids for ids, _ in predicted])
    return {
        "num_timesteps": len(truth),
        "num_gt_ids": num_gt_ids,
        "num_tracker_ids": num_tracker_ids,
        "num_gt_dets": sum(map(len, gt_ids)),
        "num_tracker_dets": sum(map(len, tracker_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": [
            sv.box_iou_batch(true_xyxy, xyxy)
            for (_, true_xyxy), (_, xyxy) in zip(truth, predicted, strict=True)
        ],
    }


def _renumbered(frames: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """The ids of each frame numbered 0, 1, ... in order of appearance, and how many there are."""
    index: dict[int, int] = {}
    renumbered = [
        np.array([index.setdefault(int(i), len(index)) for i in ids], dtype=int)
        for ids in frames
    ]
    return renumbered, len(index)


def _summary(result: dict[str, Any]) -> Hota:
    return Hota(*(float(np.mean(result[name])) for name in ("HOTA", "DetA", "AssA")))
