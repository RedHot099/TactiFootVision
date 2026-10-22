"""Class-masked YOLO training: images whose source labels only some classes.

A source that boxes players but not referees must not teach the detector that
referees are background. Every training image gets a mask over the classes
(true = annotated); the classification BCE of a class outside the mask is
zeroed at every anchor of that image, positives and negatives alike. Box and
DFL losses are unchanged.

Mosaic and mixup build one training image from several. Their partner images
are drawn from images with the same mask, and the mask of the combined image
is the AND of the masks of every image that went into it (a class counts as
annotated only if every tile annotates it). With same-mask partners the AND
keeps the first image's mask, so mosaics lose no supervision.

Imported only when a ``yolo`` training run gets ``annotated_classes``; it
imports Ultralytics at module level.
"""

import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model

logger = logging.getLogger(__name__)


def read_class_masks(
    table: str | Path, class_names: list[str]
) -> dict[str, np.ndarray]:
    """Per-image class masks from a table with columns ``image`` and ``annotated_classes``.

    ``annotated_classes`` is a comma-separated list of class names. Keys are the
    images' real paths (symlinks resolved), so an exported dataset that links
    to the same files finds them. Returns ``{real path: bool mask (C,)}``.
    """
    path = Path(table)
    if not path.is_file():
        raise FileNotFoundError(f"annotated_classes table not found: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"image", "annotated_classes"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks column(s) {', '.join(sorted(missing))}")
    masks: dict[str, np.ndarray] = {}
    parsed: dict[str, np.ndarray] = {}  # few distinct class lists, many rows
    for image, classes in zip(frame["image"], frame["annotated_classes"], strict=True):
        mask = parsed.get(classes)
        if mask is None:
            names = [c.strip() for c in str(classes).split(",") if c.strip()]
            unknown = sorted(set(names) - set(class_names))
            if unknown:
                raise ValueError(
                    f"{path}: unknown class(es) {', '.join(unknown)}; "
                    f"the dataset's classes are {', '.join(class_names)}"
                )
            mask = parsed[classes] = np.isin(class_names, names)
        key = os.path.realpath(image)
        if key in masks and not np.array_equal(masks[key], mask):
            raise ValueError(f"{path}: conflicting annotated classes for {key}")
        masks[key] = mask
    return masks


class ClassMaskedDataset(YOLODataset):
    """YOLO training dataset whose samples carry ``class_mask`` (bool, one per class).

    Images missing from ``class_masks`` count as fully annotated.
    """

    def __init__(self, *args: Any, class_masks: dict[str, np.ndarray], **kwargs: Any):
        super().__init__(*args, **kwargs)
        full = np.ones(len(self.data["names"]), bool)
        real = [os.path.realpath(f) for f in self.im_files]
        matched = sum(r in class_masks for r in real)
        if class_masks and not matched:
            raise ValueError(
                "No training image is in the annotated_classes table; "
                "its image paths must name the dataset's image files"
            )
        self.class_masks = np.stack([class_masks.get(r, full) for r in real])
        groups, self.mask_group = np.unique(
            self.class_masks, axis=0, return_inverse=True
        )
        self.mask_group = self.mask_group.reshape(-1)
        self._group_members = [
            np.flatnonzero(self.mask_group == g) for g in range(len(groups))
        ]
        self._fetched: list[int] | None = None
        logger.info(
            "Class mask: %d of %d training images in the table, %d with masked "
            "classes, %d mask groups",
            matched,
            len(real),
            int((~self.class_masks).any(axis=1).sum()),
            len(groups),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._fetched = []
        try:
            labels = super().__getitem__(index)
            fetched = self._fetched
        finally:
            self._fetched = None
        labels["class_mask"] = self.class_masks[fetched].all(axis=0)
        return labels

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        # Inside __getitem__ the first call is the sample itself; later calls
        # are mosaic / mixup partners, drawn from the first image's mask group.
        if self._fetched is not None:
            if self._fetched:
                index = self._partner(self._fetched[0])
            self._fetched.append(index)
        return super().get_image_and_label(index)

    def _partner(self, first: int) -> int:
        """A random image with ``first``'s mask, from the mosaic buffer when it has one."""
        group = self.mask_group[first]
        buffered = [i for i in self.buffer if self.mask_group[i] == group]
        if buffered:
            return random.choice(buffered)
        return int(random.choice(self._group_members[group]))

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        masks = np.stack([sample.pop("class_mask") for sample in batch])
        collated = YOLODataset.collate_fn(batch)
        collated["class_mask"] = torch.from_numpy(masks)
        return collated


class ClassMaskedLoss(v8DetectionLoss):
    """``v8DetectionLoss`` whose cls BCE is zeroed for each image's masked classes.

    Reads ``batch["class_mask"]`` (``(B, C)``, true = annotated); without it
    the loss is the stock one.
    """

    def __init__(self, model: Any, tal_topk: int = 10) -> None:
        super().__init__(model, tal_topk)
        self._stock_bce = self.bce
        self.bce = self._masked_bce
        self._class_mask: torch.Tensor | None = None

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.get("class_mask")
        self._class_mask = None if mask is None else mask.to(self.device)
        try:
            return super().__call__(preds, batch)
        finally:
            self._class_mask = None

    def _masked_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._stock_bce(pred, target)  # (B, anchors, C)
        if self._class_mask is None:
            return loss
        return loss * self._class_mask[:, None, :].to(loss.dtype)


class ClassMaskedTrainer(DetectionTrainer):
    """``DetectionTrainer`` with class-masked training data and loss.

    Validation is unchanged (the validation set is fully annotated). Only the
    training model gets the masked loss; the EMA copy that is saved in the
    checkpoints keeps the stock one, so checkpoints load without this module.
    """

    def __init__(self, *args: Any, class_masks: dict[str, np.ndarray], **kwargs: Any):
        self.class_masks = class_masks
        super().__init__(*args, **kwargs)

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        cfg = self.args
        stride = max(
            int(unwrap_model(self.model).stride.max() if self.model else 0), 32
        )
        return ClassMaskedDataset(  # the arguments of build_yolo_dataset in train mode
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=True,
            hyp=cfg,
            rect=cfg.rect,
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix=colorstr("train: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction,
            class_masks=self.class_masks,
        )

    def _setup_train(self) -> None:
        super()._setup_train()  # creates the EMA copy first
        model = unwrap_model(self.model)
        model.criterion = ClassMaskedLoss(model)
        logger.info("Class-masked classification loss on the training model")


def masked_trainer(table: str | Path, class_names: list[str]) -> Callable[..., Any]:
    """Trainer factory for ``YOLO.train(trainer=...)`` masking by ``table``."""
    masks = read_class_masks(table, class_names)

    def make(*args: Any, **kwargs: Any) -> ClassMaskedTrainer:
        return ClassMaskedTrainer(*args, class_masks=masks, **kwargs)

    return make
