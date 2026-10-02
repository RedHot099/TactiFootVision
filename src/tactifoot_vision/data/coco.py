"""COCO dataset format in the Roboflow / RF-DETR folder layout.

<root>/<split>/_annotations.coco.json   (+ the split's images next to it)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactifoot_vision.data._files import (
    check_output_location,
    clipped_box,
    link_file,
    prepare_output,
    unique_names,
)
from tactifoot_vision.data.annotations import NOT_LABELLED, Annotations, Task
from tactifoot_vision.data.dataset import Dataset, LinkMode, Sample, canonical_split

logger = logging.getLogger(__name__)

ANNOTATION_FILE = "_annotations.coco.json"
SUPERCATEGORY = "objects"  # anything but "none": RF-DETR drops "none" categories


# ------------------------------------------------------------------- reading
def read_dataset(root: Path, flip_idx: tuple[int, ...] | None = None) -> Dataset:
    root = root.absolute()
    splits: dict[str, list[Sample]] = {}
    class_names: list[str] | None = None
    num_keypoints: int | None = None
    for folder in ("train", "valid", "val", "test"):
        path = root / folder / ANNOTATION_FILE
        split = canonical_split(folder)
        if not path.is_file() or split in splits:
            continue
        coco = json.loads(path.read_text())
        names, category_index = _categories(coco)
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(
                f"{path} has classes {names}, other splits have {class_names}"
            )
        num_keypoints = num_keypoints or _num_keypoints(coco)
        flip_idx = flip_idx or _flip_idx(coco)
        splits[split] = _read_split(
            path.parent, coco, category_index, num_keypoints, flip_idx
        )
    if class_names is None:
        raise FileNotFoundError(f"No <split>/{ANNOTATION_FILE} under {root}")
    return Dataset(
        task=Task.POSE if num_keypoints else Task.DETECT,
        class_names=class_names,
        splits=splits,
        num_keypoints=num_keypoints,
        flip_idx=flip_idx,
        name=root.name,
    )


def _categories(coco: dict) -> tuple[list[str], dict[int, int]]:
    """Class names and category id -> class index, without Roboflow's placeholder category."""
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    parents = {c.get("supercategory") for c in categories}
    kept = [
        c
        for c in categories
        if not (
            c.get("supercategory") == "none"
            and c["name"] in parents
            and len(categories) > 1
        )
    ]
    return [c["name"] for c in kept], {c["id"]: i for i, c in enumerate(kept)}


def _num_keypoints(coco: dict) -> int | None:
    for category in coco.get("categories", []):
        if category.get("keypoints"):
            return len(category["keypoints"])
    for annotation in coco.get("annotations", []):
        if annotation.get("keypoints"):
            return len(annotation["keypoints"]) // 3
    return None


def _flip_idx(coco: dict) -> tuple[int, ...] | None:
    # Not part of COCO; our writer stores it on the category so pose data round-trips.
    for category in coco.get("categories", []):
        if category.get("flip_idx"):
            return tuple(category["flip_idx"])
    return None


def _read_split(
    folder: Path,
    coco: dict,
    category_index: dict[int, int],
    num_keypoints: int | None,
    flip_idx: tuple[int, ...] | None,
) -> list[Sample]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if annotation["category_id"] in category_index:
            by_image[annotation["image_id"]].append(annotation)

    samples = []
    for image in coco.get("images", []):
        objects = by_image.get(image["id"], [])
        if objects:
            boxes = np.array([a["bbox"] for a in objects], dtype=np.float32)
            boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
            keypoints = None
            if num_keypoints:
                keypoints = np.array(
                    [a.get("keypoints") or [0.0] * num_keypoints * 3 for a in objects],
                    dtype=np.float32,
                ).reshape(len(objects), num_keypoints, 3)
            annotations = Annotations(
                boxes=boxes,
                class_ids=[category_index[a["category_id"]] for a in objects],
                keypoints=keypoints,
                flip_idx=flip_idx,
            )
        else:
            annotations = Annotations.empty(num_keypoints, flip_idx)
        samples.append(
            Sample(
                image_path=folder / image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=annotations,
            )
        )
    return samples


# ------------------------------------------------------------------- writing
def write_dataset(dataset: Dataset, out_dir: Path, link: LinkMode = "symlink") -> Path:
    splits = dataset.split_names
    managed = [out_dir / s for s in splits]
    check_output_location(out_dir, managed, (s.image_path for s in dataset))
    prepare_output(out_dir, managed)
    categories = []
    for i, name in enumerate(dataset.class_names):
        category = {"id": i, "name": name, "supercategory": SUPERCATEGORY}
        if dataset.num_keypoints:
            category |= {
                "keypoints": [str(k) for k in range(dataset.num_keypoints)],
                "skeleton": [],
            }
            if dataset.flip_idx:
                category["flip_idx"] = list(dataset.flip_idx)
        categories.append(category)

    for split in splits:
        samples = dataset[split]
        images, annotations = [], []
        names = unique_names([s.image_path for s in samples])
        # Ids start at 1: pycocotools (RF-DETR's evaluator) treats id 0 as "no match".
        for image_id, (sample, name) in enumerate(zip(samples, names, strict=True), 1):
            link_file(sample.image_path, out_dir / split / name, link)
            images.append(
                {
                    "id": image_id,
                    "file_name": name,
                    "width": sample.width,
                    "height": sample.height,
                }
            )
            annotations += _annotations(sample, image_id, first_id=len(annotations) + 1)
        coco = {
            "info": {"description": dataset.name},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        (out_dir / split / ANNOTATION_FILE).write_text(json.dumps(coco))
    logger.info("Wrote COCO dataset (%d images) to %s", len(dataset), out_dir)
    return out_dir


def _annotations(sample: Sample, image_id: int, first_id: int) -> list[dict]:
    ann = sample.annotations
    result = []
    for i in range(len(ann)):
        box = clipped_box(ann.boxes[i], sample.width, sample.height)
        if box is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        record = {
            "id": first_id + len(result),
            "image_id": image_id,
            "category_id": int(ann.class_ids[i]),
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0,
        }
        if ann.keypoints is not None:
            keypoints = ann.keypoints[i].copy()
            keypoints[keypoints[:, 2] <= NOT_LABELLED] = 0.0
            record["keypoints"] = [float(v) for v in keypoints.reshape(-1)]
            record["num_keypoints"] = int((keypoints[:, 2] > NOT_LABELLED).sum())
        result.append(record)
    return result
