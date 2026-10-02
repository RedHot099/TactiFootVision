"""Offline augmentation: grow a dataset's training split with augmented copies."""

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from tactifoot_vision.augment.base import Transform
from tactifoot_vision.data._files import check_output_location, unique_stems
from tactifoot_vision.data.dataset import SPLITS, Dataset, Sample, canonical_split
from tactifoot_vision.utils import ensure_dir

logger = logging.getLogger(__name__)


def augment_dataset(
    dataset: Dataset,
    transform: Transform,
    out_dir: str | Path,
    copies: int = 1,
    splits: Sequence[str] = ("train",),
    seed: int = 0,
    image_format: str = "jpg",
) -> Dataset:
    """Add ``copies`` augmented variants of every image in ``splits``.

    Augmented images are written to
    ``<out_dir>/<split>/images/<original stem>_aug<k>.<image_format>``
    (``k = 0 .. copies-1``); their labels stay in memory like every other
    sample, so export the result with ``to_yolo`` / ``to_coco`` for training.

    Each listed split keeps its original samples first, followed by the
    augmented ones in the same image order. Splits not listed are untouched:
    never augment validation or test data, or the metrics stop measuring
    performance on real images.

    Args:
        dataset: source dataset (not modified).
        transform: augmentation to apply, e.g. a :class:`Compose`.
        out_dir: folder for the augmented images.
        copies: augmented variants per original image.
        splits: splits to augment.
        seed: the output is identical for the same ``seed``, dataset and
            transform; image ``i`` of a split always gets the same random
            stream, whatever other splits are augmented.
        image_format: file extension of the written images (``jpg``, ``png``...).

    Returns:
        A new dataset named ``"<name>-aug"``.
    """
    if copies < 1:
        raise ValueError(f"copies must be >= 1, got {copies}")
    split_names = list(dict.fromkeys(canonical_split(s) for s in splits))
    missing = [s for s in split_names if not dataset[s]]
    if missing:
        raise ValueError(
            f"Cannot augment split(s) {missing}: {dataset.name!r} has images in "
            f"{dataset.split_names}"
        )
    suffix = image_format.lower().lstrip(".")
    out_dir = Path(out_dir).resolve()
    check_output_location(
        out_dir,
        [out_dir / split / "images" for split in split_names],
        (s.image_path for s in dataset),
    )

    result = dataset
    for split in split_names:
        originals = dataset[split]
        images_dir = ensure_dir(out_dir / split / "images")
        augmented = []
        stems = unique_stems(s.image_path.stem for s in originals)
        progress = tqdm(
            zip(originals, stems, strict=True),
            total=len(originals),
            desc=f"augment {split}",
        )
        for index, (sample, stem) in enumerate(progress):
            image = sample.read_image()
            annotations = sample.annotations
            # Hand-built annotations may lack the dataset's flip_idx; HorizontalFlip needs it.
            if annotations.flip_idx is None and dataset.flip_idx is not None:
                annotations = replace(annotations, flip_idx=dataset.flip_idx)
            for k in range(copies):
                seed_seq = np.random.SeedSequence(
                    seed, spawn_key=(SPLITS.index(split), index, k)
                )
                new_image, new_annotations = transform(
                    image, annotations, np.random.default_rng(seed_seq)
                )
                path = images_dir / f"{stem}_aug{k}.{suffix}"
                if not cv2.imwrite(str(path), new_image):
                    raise RuntimeError(f"Could not write augmented image {path}")
                augmented.append(
                    Sample(
                        image_path=path,
                        width=new_image.shape[1],
                        height=new_image.shape[0],
                        annotations=new_annotations,
                    )
                )
        logger.info(
            "Augmented %s: %d originals + %d augmented images in %s",
            split,
            len(originals),
            len(augmented),
            images_dir,
        )
        result = result.with_split(split, originals + augmented)
    return replace(result, name=f"{dataset.name}-aug")
