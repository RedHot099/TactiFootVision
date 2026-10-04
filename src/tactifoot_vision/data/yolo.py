"""Ultralytics YOLO dataset format (detection and pose)."""

import logging
import os
from pathlib import Path

import numpy as np
import yaml

from tactifoot_vision.data._files import (
    IMAGE_SUFFIXES,
    check_link_mode,
    check_output_location,
    clipped_box,
    image_size,
    link_file,
    prepare_output,
    unique_names,
)
from tactifoot_vision.data.annotations import NOT_LABELLED, VISIBLE, Annotations, Task
from tactifoot_vision.data.dataset import Dataset, LinkMode, Sample, canonical_split

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- reading
def read_dataset(data_yaml: Path) -> Dataset:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text()) or {}
    names = config.get("names")
    if not names:
        raise ValueError(f"{data_yaml} has no 'names'")
    class_names = (
        [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    )

    kpt_shape = config.get("kpt_shape")
    num_keypoints, kpt_dims = (
        (int(kpt_shape[0]), int(kpt_shape[1])) if kpt_shape else (None, None)
    )
    flip_idx = tuple(config["flip_idx"]) if config.get("flip_idx") else None

    data_yaml = data_yaml.absolute()
    root = Path(config.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root

    splits: dict[str, list[Sample]] = {}
    loaded: dict[tuple[Path, ...], str] = {}
    for key in ("train", "val", "valid", "test"):
        if not config.get(key):
            continue
        split = canonical_split(key)
        if split in splits:
            continue
        images = tuple(_list_images(config[key], root))
        if images in loaded:  # Roboflow exports often reuse the valid images as "test"
            logger.info("Split %r repeats split %r; skipping it", split, loaded[images])
            continue
        loaded[images] = split
        splits[split] = [
            _read_sample(image, len(class_names), num_keypoints, kpt_dims, flip_idx)
            for image in images
        ]

    return Dataset(
        task=Task.POSE if num_keypoints else Task.DETECT,
        class_names=class_names,
        splits=splits,
        num_keypoints=num_keypoints,
        flip_idx=flip_idx,
        name=data_yaml.parent.name,
    )


def _list_images(entry: str | list[str], root: Path) -> list[Path]:
    """Resolve a split entry (folder, image list .txt, image, or a list of those)."""
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for item in entries:
        path = Path(item) if Path(item).is_absolute() else root / item
        if not path.exists() and str(item).startswith("../"):
            path = root / str(item)[3:]  # Roboflow's "../train/images" convention
        if path.is_dir():
            images += [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        elif path.suffix == ".txt" and path.is_file():
            lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            images += [
                p if p.is_absolute() else path.parent / p for p in map(Path, lines)
            ]
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"YOLO split path not found: {path}")
    return sorted(images)


def label_path(image: Path) -> Path:
    """Ultralytics convention: swap the last ``images`` folder for ``labels``."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(str(image).rsplit(sa, 1))).with_suffix(".txt")


def _read_sample(
    image: Path,
    num_classes: int,
    num_keypoints: int | None,
    kpt_dims: int | None,
    flip_idx: tuple[int, ...] | None,
) -> Sample:
    width, height = image_size(image)
    labels = label_path(image)
    rows = []
    if labels.is_file():
        rows = [ln.split() for ln in labels.read_text().splitlines() if ln.strip()]

    boxes, class_ids, keypoints = [], [], []
    scale = np.array([width, height, width, height], dtype=np.float32)
    for row in rows:
        values = np.asarray(row[1:], dtype=np.float32)
        class_id = int(float(row[0]))
        if not 0 <= class_id < num_classes:
            raise ValueError(
                f"{labels}: class id {class_id} is out of range for the "
                f"{num_classes} classes in data.yaml"
            )
        class_ids.append(class_id)
        if num_keypoints is not None:
            cx, cy, w, h = values[:4]
            if len(values) != 4 + num_keypoints * kpt_dims:
                raise ValueError(
                    f"{labels}: expected {4 + num_keypoints * kpt_dims} values after the class "
                    f"for kpt_shape [{num_keypoints}, {kpt_dims}], got {len(values)}"
                )
            kp = values[4:].reshape(num_keypoints, kpt_dims)
            xy = kp[:, :2] * [width, height]
            if kpt_dims == 3:
                visibility = kp[:, 2]
            else:
                visibility = np.where(
                    (kp[:, :2] > 0).any(axis=1), VISIBLE, NOT_LABELLED
                )
            keypoints.append(np.column_stack([xy, visibility]))
        elif len(values) == 4:
            cx, cy, w, h = values
        else:  # segmentation polygon "x1 y1 x2 y2 ..." -> its bounding box
            polygon = values.reshape(-1, 2)
            (x1, y1), (x2, y2) = polygon.min(axis=0), polygon.max(axis=0)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        boxes.append(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * scale)

    if not rows:
        annotations = Annotations.empty(num_keypoints, flip_idx)
    else:
        annotations = Annotations(
            boxes=np.asarray(boxes),
            class_ids=np.asarray(class_ids),
            keypoints=np.asarray(keypoints) if num_keypoints is not None else None,
            flip_idx=flip_idx,
        )
    return Sample(image_path=image, width=width, height=height, annotations=annotations)


# ------------------------------------------------------------------- writing
def write_dataset(dataset: Dataset, out_dir: Path, link: LinkMode = "symlink") -> Path:
    check_link_mode(link)
    splits = dataset.split_names
    managed = [
        out_dir / split / sub for split in splits for sub in ("images", "labels")
    ]
    check_output_location(out_dir, managed, (s.image_path for s in dataset))
    targets = {
        split: [
            out_dir / split / "images" / name
            for name in unique_names([s.image_path for s in dataset[split]])
        ]
        for split in splits
    }
    images = [target for split in splits for target in targets[split]]
    data_yaml = out_dir / "data.yaml"
    prepare_output(out_dir, managed, [*images, *map(label_path, images), data_yaml])
    for split in splits:
        for sample, target in zip(dataset[split], targets[split], strict=True):
            link_file(sample.image_path, target, link)
            label_path(target).write_text(_format_labels(sample))

    config: dict = {"path": str(out_dir.resolve())}
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if split in splits:
            config[key] = f"{split}/images"
    config |= {"nc": len(dataset.class_names), "names": list(dataset.class_names)}
    if dataset.num_keypoints:
        config["kpt_shape"] = [dataset.num_keypoints, 3]
        if dataset.flip_idx:
            config["flip_idx"] = list(dataset.flip_idx)
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    logger.info("Wrote YOLO dataset (%d images) to %s", len(dataset), out_dir)
    return data_yaml


def _format_labels(sample: Sample) -> str:
    ann = sample.annotations
    width, height = sample.width, sample.height
    lines = []
    for i in range(len(ann)):
        box = clipped_box(ann.boxes[i], width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        fields = [str(int(ann.class_ids[i])), *(f"{v:.6f}" for v in (cx, cy, w, h))]
        if ann.keypoints is not None:
            for x, y, v in ann.keypoints[i]:
                nx, ny = x / width, y / height
                # Unlabelled and out-of-frame points become "0 0 0": Ultralytics
                # rejects coordinates outside [0, 1].
                if v <= NOT_LABELLED or not (0 <= nx <= 1 and 0 <= ny <= 1):
                    fields += ["0", "0", "0"]
                else:
                    fields += [f"{nx:.6f}", f"{ny:.6f}", str(int(v))]
        lines.append(" ".join(fields))
    return "\n".join(lines) + ("\n" if lines else "")
