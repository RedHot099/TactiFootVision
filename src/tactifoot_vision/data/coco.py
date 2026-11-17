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
from tactifoot_vision.data.annotations import Annotations, Task
from tactifoot_vision.data.dataset import Dataset, Sample
from tactifoot_vision.enums import LinkMode, Split

logger = logging.getLogger(__name__)

ANNOTATION_FILE = "_annotations.coco.json"
SUPERCATEGORY = "objects"  # anything but "none": RF-DETR drops "none" categories


# ------------------------------------------------------------------- reading
def read_dataset(root: Path) -> Dataset:
    root = root.absolute()
    splits: dict[str, list[Sample]] = {}
    class_names: list[str] | None = None
    for folder in ("train", "valid", "val", "test"):
        path = root / folder / ANNOTATION_FILE
        split = Split.parse(folder)
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
        if _has_keypoints(coco):
            raise ValueError(
                f"{path} holds keypoints (a pose dataset); the package reads "
                "detection datasets only"
            )
        splits[split] = _read_split(path, coco, category_index)
    if class_names is None:
        raise FileNotFoundError(f"No <split>/{ANNOTATION_FILE} under {root}")
    return Dataset(
        task=Task.DETECT,
        class_names=class_names,
        splits=splits,
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


def _has_keypoints(coco: dict) -> bool:
    return any(c.get("keypoints") for c in coco.get("categories", [])) or any(
        a.get("keypoints") for a in coco.get("annotations", [])
    )


def _read_split(
    path: Path,
    coco: dict,
    category_index: dict[int, int],
) -> list[Sample]:
    folder = path.parent
    known = {c["id"] for c in coco.get("categories", [])}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if annotation["category_id"] not in known:
            raise ValueError(
                f"{path}: annotation {annotation.get('id')} has category_id "
                f"{annotation['category_id']}, which is not among its categories"
            )
        if annotation["category_id"] in category_index:
            by_image[annotation["image_id"]].append(annotation)

    samples = []
    for image in coco.get("images", []):
        objects = by_image.get(image["id"], [])
        if objects:
            boxes = np.array([a["bbox"] for a in objects], dtype=np.float32)
            boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
            annotations = Annotations(
                boxes=boxes,
                class_ids=[category_index[a["category_id"]] for a in objects],
            )
        else:
            annotations = Annotations.empty()
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
def write_dataset(
    dataset: Dataset, out_dir: Path, link: LinkMode | str = LinkMode.SYMLINK
) -> Path:
    link = LinkMode.parse(link, "link")
    splits = dataset.split_names
    managed = [out_dir / s for s in splits]
    check_output_location(out_dir, managed, (s.image_path for s in dataset))
    names = {s: unique_names([x.image_path for x in dataset[s]]) for s in splits}
    files = [out_dir / s / name for s in splits for name in names[s]]
    files += [out_dir / s / ANNOTATION_FILE for s in splits]
    prepare_output(out_dir, managed, files)
    categories = []
    for i, name in enumerate(dataset.class_names):
        categories.append({"id": i, "name": name, "supercategory": SUPERCATEGORY})

    for split in splits:
        samples = dataset[split]
        images, annotations = [], []
        # Ids start at 1: pycocotools (RF-DETR's evaluator) treats id 0 as "no match".
        pairs = zip(samples, names[split], strict=True)
        for image_id, (sample, name) in enumerate(pairs, 1):
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
        result.append(record)
    return result
