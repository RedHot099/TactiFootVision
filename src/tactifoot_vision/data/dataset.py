"""In-memory index of an annotated image dataset (detection or pose)."""

import logging
import numbers
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.data.annotations import Annotations, Task

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {"val": "valid", "validation": "valid"}
LinkMode = Literal["symlink", "hardlink", "copy"]


def canonical_split(name: str) -> str:
    name = SPLIT_ALIASES.get(name, name)
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; use one of {SPLITS}")
    return name


def check_flip_idx(
    flip_idx: tuple[int, ...] | None, num_keypoints: int | None, source: str = ""
) -> None:
    """Refuse a ``flip_idx`` that does not have one entry per keypoint.

    Checked when a dataset is loaded, so a bad ``data.yaml`` fails there and
    not in :class:`~tactifoot_vision.augment.HorizontalFlip` after an
    augmentation run has written some images.
    """
    if flip_idx is None or num_keypoints is None or len(flip_idx) == num_keypoints:
        return
    where = f"{source}: " if source else ""
    raise ValueError(
        f"{where}flip_idx has {len(flip_idx)} entries but there are "
        f"{num_keypoints} keypoints; it needs one entry per keypoint"
    )


@dataclass(frozen=True)
class Sample:
    """One image on disk with its labels (pixel coordinates)."""

    image_path: Path
    width: int
    height: int
    annotations: Annotations

    def read_image(self) -> np.ndarray:
        """Load the image as a BGR ``uint8`` array (OpenCV convention)."""
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image {self.image_path}")
        return image


@dataclass
class Dataset:
    """Annotated images grouped into ``train`` / ``valid`` / ``test`` splits.

    A ``Dataset`` is a lightweight index: images stay on disk, labels live in
    memory. All operations return new datasets, and ``to_yolo`` / ``to_coco``
    write whatever format a training backend needs.

    Load one with :func:`load_dataset` (YOLO ``data.yaml`` or COCO folder).
    """

    task: Task
    class_names: list[str]
    splits: dict[str, list[Sample]] = field(default_factory=dict)
    num_keypoints: int | None = None
    flip_idx: tuple[int, ...] | None = None
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.task = Task(self.task)
        self.splits = {canonical_split(k): list(v) for k, v in self.splits.items()}
        if self.task is Task.POSE and self.num_keypoints is None:
            raise ValueError("Pose datasets need num_keypoints")
        check_flip_idx(self.flip_idx, self.num_keypoints)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Load a YOLO (``data.yaml`` or its folder) or COCO (split folders) dataset."""
        path = Path(path)
        if path.is_file() and path.suffix in {".yaml", ".yml"}:
            return cls.from_yolo(path)
        if path.is_dir():
            if (path / "data.yaml").is_file():
                return cls.from_yolo(path / "data.yaml")
            if any(
                (path / s / "_annotations.coco.json").is_file()
                for s in ("train", "valid", "val", "test")
            ):
                return cls.from_coco(path)
        raise FileNotFoundError(
            f"{path} is neither a YOLO data.yaml (or its folder) nor a COCO dataset folder"
        )

    @classmethod
    def from_yolo(cls, data_yaml: str | Path) -> "Dataset":
        from tactifoot_vision.data import yolo

        return yolo.read_dataset(Path(data_yaml))

    @classmethod
    def from_coco(
        cls, root: str | Path, flip_idx: tuple[int, ...] | None = None
    ) -> "Dataset":
        from tactifoot_vision.data import coco

        return coco.read_dataset(Path(root), flip_idx=flip_idx)

    # ------------------------------------------------------------------- access
    def __getitem__(self, split: str) -> list[Sample]:
        """The samples of ``split`` (a new list; empty if the split is missing)."""
        return list(self.splits.get(canonical_split(split), []))

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.splits.values())

    def __iter__(self) -> Iterator[Sample]:
        for split in self.split_names:
            yield from self.splits[split]

    @property
    def split_names(self) -> list[str]:
        return [s for s in SPLITS if self.splits.get(s)]

    def read(self, split: str, index: int) -> tuple[np.ndarray, Annotations]:
        """Return ``(image_bgr, annotations)`` of one sample."""
        sample = self.splits.get(canonical_split(split), [])[index]
        return sample.read_image(), sample.annotations.copy()

    def summary(self) -> pd.DataFrame:
        """Images, objects and per-class object counts for every split."""
        rows = []
        for split in self.split_names:
            samples = self.splits[split]
            class_ids = np.concatenate(
                [s.annotations.class_ids for s in samples] or [np.zeros(0, np.int64)]
            )
            counts = np.bincount(class_ids, minlength=len(self.class_names))
            rows.append(
                {"split": split, "images": len(samples), "objects": int(class_ids.size)}
                | {name: int(counts[i]) for i, name in enumerate(self.class_names)}
            )
        return pd.DataFrame(rows).set_index("split") if rows else pd.DataFrame()

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s}={len(self.splits[s])}" for s in self.split_names)
        kp = f", keypoints={self.num_keypoints}" if self.num_keypoints else ""
        return (
            f"Dataset(name={self.name!r}, task={self.task.value}, "
            f"classes={self.class_names}{kp}, {sizes or 'empty'})"
        )

    # ------------------------------------------------------ derived datasets
    def with_split(self, split: str, samples: list[Sample]) -> "Dataset":
        """Return a copy where ``split`` holds ``samples``."""
        return replace(
            self, splits={**self.splits, canonical_split(split): list(samples)}
        )

    def subset(
        self, size: int | float | Mapping[str, int | float], seed: int = 0
    ) -> "Dataset":
        """Randomly keep ``size`` images per split.

        ``size`` is an image count (an integer, NumPy integers included), a
        fraction (a float in ``[0, 1]``) or a mapping
        ``{split: count_or_fraction}``; splits missing from the mapping are
        kept whole.
        """
        rng = random.Random(seed)
        sizes = (
            {canonical_split(k): v for k, v in size.items()}
            if isinstance(size, Mapping)
            else {}
        )
        splits = {}
        for split, samples in self.splits.items():
            want = sizes.get(split) if isinstance(size, Mapping) else size
            if want is None:
                splits[split] = list(samples)
                continue
            count = _subset_count(want, len(samples))
            splits[split] = rng.sample(samples, min(count, len(samples)))
        return replace(self, splits=splits)

    def resplit(
        self, train: float = 0.8, valid: float = 0.2, test: float = 0.0, seed: int = 0
    ) -> "Dataset":
        """Pool every image and split again with the given fractions.

        Split boundaries are rounded cumulatively and the rounding remainder
        goes to the last split with a non-zero fraction, so a split asked to
        be empty (``test=0``) stays empty.
        """
        fractions = {"train": train, "valid": valid, "test": test}
        if any(f < 0 for f in fractions.values()):
            raise ValueError(f"Split fractions must not be negative, got {fractions}")
        total = train + valid + test
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1, got {total}")
        pool = list(self)
        random.Random(seed).shuffle(pool)
        last = max(i for i, f in enumerate(fractions.values()) if f > 0)
        splits, begin, cumulative = {}, 0, 0.0
        for position, (name, fraction) in enumerate(fractions.items()):
            cumulative += fraction
            end = len(pool) if position >= last else round(len(pool) * cumulative)
            splits[name] = pool[begin:end]
            begin = end
        return replace(self, splits={k: v for k, v in splits.items() if v})

    def merge(self, other: "Dataset") -> "Dataset":
        """Concatenate the splits of two datasets with identical classes.

        Pose datasets must also share their keypoint layout: the keypoint
        count and ``flip_idx``. An export writes one ``flip_idx`` for every
        sample, so a different mapping would change what the other dataset's
        keypoints mean once reloaded.
        """
        if other.task != self.task or other.class_names != self.class_names:
            raise ValueError(
                "Can only merge datasets with the same task and class names"
            )
        if other.num_keypoints != self.num_keypoints:
            raise ValueError(
                "Can only merge datasets with the same keypoint layout, got "
                f"{self.num_keypoints} and {other.num_keypoints} keypoints"
            )
        if other.flip_idx != self.flip_idx:
            raise ValueError(
                "Can only merge datasets with the same keypoint layout, got flip_idx "
                f"{self.flip_idx} and {other.flip_idx}"
            )
        splits = {s: self[s] + other[s] for s in SPLITS if self[s] or other[s]}
        return replace(self, splits=splits)

    # ----------------------------------------------------------------- export
    def to_yolo(self, out_dir: str | Path, link: LinkMode = "symlink") -> Path:
        """Write the dataset in Ultralytics YOLO layout and return its ``data.yaml``."""
        from tactifoot_vision.data import yolo

        return yolo.write_dataset(self, Path(out_dir), link=link)

    def to_coco(self, out_dir: str | Path, link: LinkMode = "symlink") -> Path:
        """Write the dataset in Roboflow/RF-DETR COCO layout and return its root folder.

        Layout: ``<out_dir>/<split>/_annotations.coco.json`` next to the images.
        """
        from tactifoot_vision.data import coco

        return coco.write_dataset(self, Path(out_dir), link=link)


def _subset_count(want: object, available: int) -> int:
    """Images to keep for a ``subset`` size: an integer count or a fraction in ``[0, 1]``."""
    if isinstance(want, numbers.Integral) and not isinstance(want, bool):
        if want < 0:
            raise ValueError(f"A subset count must be >= 0, got {want}")
        return int(want)
    if isinstance(want, numbers.Real) and not isinstance(want, bool):
        if not 0 <= want <= 1:
            raise ValueError(
                f"A subset fraction must be in [0, 1], got {want}; pass an int for a count"
            )
        return round(available * float(want))
    raise ValueError(f"A subset size must be a number, got {want!r}")


def load_dataset(path: str | Path) -> Dataset:
    """Load a YOLO (``data.yaml`` or its folder) or COCO dataset. See :class:`Dataset`."""
    return Dataset.load(path)
