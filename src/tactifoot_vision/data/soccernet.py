"""SoccerNet Game State Reconstruction (GSR) as a 4-class detection dataset.

Layout read (the SoccerNet download, unzipped)::

    <root>/<split>/SNGS-xxx/Labels-GameState.json
    <root>/<split>/SNGS-xxx/img1/000001.jpg ...

Every object annotation (``supercategory == "object"``) becomes a box of the
class named by its category: ``ball``, ``goalkeeper``, ``player`` or
``referee`` (GSR's ``attributes.role`` repeats the category for these four).
Category ``other`` (staff, ball boys, people GSR does not assign a role) is
dropped: it is none of the four classes, and it is rare (399 of ~749k
objects in GSR valid). A detection on such a person therefore counts as a
false positive. Pitch and camera annotations are not objects and are skipped.
"""

import json
import logging
from pathlib import Path

import numpy as np

from tactifoot_vision.data.annotations import Annotations, Task
from tactifoot_vision.data.dataset import SPLITS, Dataset, Sample, canonical_split

logger = logging.getLogger(__name__)

GSR_CLASSES = ["ball", "goalkeeper", "player", "referee"]
GSR_DROPPED = ("other",)
LABELS_FILE = "Labels-GameState.json"


def is_gsr_root(path: Path) -> bool:
    """Whether ``path`` holds GSR split folders (``<split>/SNGS-*/Labels-GameState.json``)."""
    return any(
        next((path / split).glob(f"*/{LABELS_FILE}"), None) is not None
        for split in ("train", "valid", "test")
        if (path / split).is_dir()
    )


def read_dataset(
    root: Path,
    splits: tuple[str, ...] = SPLITS,
    stride: int = 1,
    tracking_root: Path | None = None,
) -> Dataset:
    """Read GSR clips into a detection :class:`Dataset`.

    Args:
        root: folder holding the ``train`` / ``valid`` / ``test`` split folders.
        splits: splits to read; a split folder that does not exist is an error.
        stride: keep 1-based frames ``1, 1 + stride, 1 + 2 * stride, ...`` of
            every clip (consecutive 25 fps frames are near duplicates).
        tracking_root: a SoccerNet-Tracking 2023 folder
            (``<split>/SNMOT-xxx/img1``). GSR train and test are built on the
            same clips (``SNGS-060`` is ``SNMOT-060``, same images byte for
            byte, same boxes); a frame whose GSR image is missing is read from
            there, so a download that kept every 5th GSR image still gives
            every labelled frame.

    Every :class:`Sample` carries ``meta``: ``sequence`` (clip name, e.g.
    ``SNGS-021``), ``game`` (GSR's ``info.game_id``, an index shared by the
    clips of one match; ``""`` when absent), ``frame`` (1-based frame
    number), ``image_source`` (``gsr`` or ``tracking``) and ``other_dropped``
    (the frame's ``other`` objects left out), all as strings.

    Raises:
        FileNotFoundError: a split folder or a kept frame's image is missing
            (SoccerNet downloads that keep every 5th image need ``stride=5``).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    dataset_splits: dict[str, list[Sample]] = {}
    for split in splits:
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"GSR split folder {folder} does not exist")
        samples = []
        for labels in sorted(folder.glob(f"*/{LABELS_FILE}")):
            tracking = None if tracking_root is None else tracking_root / split
            samples += _read_clip(labels, stride, tracking)
        dataset_splits[canonical_split(split)] = samples
        logger.info(
            "GSR %s: %d frames (stride %d) from %s", split, len(samples), stride, folder
        )
    return Dataset(
        task=Task.DETECT,
        class_names=list(GSR_CLASSES),
        splits=dataset_splits,
        name=f"gsr-{root.name}",
    )


def _read_clip(
    labels_path: Path, stride: int, tracking: Path | None = None
) -> list[Sample]:
    labels = json.loads(labels_path.read_text())
    info = labels.get("info", {})
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    class_of = {
        cid: GSR_CLASSES.index(name)
        for cid, name in categories.items()
        if name in GSR_CLASSES
    }
    image_dir = labels_path.parent / info.get("im_dir", "img1")
    sequence = info.get("name", labels_path.parent.name)
    game = str(info.get("game_id", ""))

    images = {}
    for image in labels["images"]:
        frame = int(Path(image["file_name"]).stem)
        if (frame - 1) % stride == 0:
            images[image["image_id"]] = (frame, image)
    boxes: dict[str, list[list[float]]] = {key: [] for key in images}
    class_ids: dict[str, list[int]] = {key: [] for key in images}
    dropped = dict.fromkeys(images, 0)
    for item in labels["annotations"]:
        if item.get("supercategory") != "object" or item["image_id"] not in images:
            continue
        category = item["category_id"]
        if category not in class_of:
            name = categories.get(category, category)
            if name in GSR_DROPPED:
                dropped[item["image_id"]] += 1
                continue
            raise ValueError(
                f"{labels_path}: object category {name!r} is not one of "
                f"{GSR_CLASSES} nor dropped ({', '.join(GSR_DROPPED)})"
            )
        box = item["bbox_image"]
        boxes[item["image_id"]].append(
            [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]]
        )
        class_ids[item["image_id"]].append(class_of[category])

    samples = []
    for key, (frame, image) in sorted(images.items(), key=lambda kv: kv[1][0]):
        path, source = image_dir / image["file_name"], "gsr"
        if not path.is_file() and tracking is not None:
            clip = tracking / sequence.replace("SNGS-", "SNMOT-")
            path, source = clip / "img1" / image["file_name"], "tracking"
        if not path.is_file():
            raise FileNotFoundError(
                f"GSR image {path} is missing; if the download keeps only some "
                "frames, read with a stride that selects them (e.g. stride=5) "
                "or give the SoccerNet-Tracking folder (tracking_root)"
            )
        width, height = int(image["width"]), int(image["height"])
        xyxy = np.asarray(boxes[key], dtype=np.float32).reshape(-1, 4)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
        keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        samples.append(
            Sample(
                image_path=path,
                width=width,
                height=height,
                annotations=Annotations(
                    boxes=xyxy[keep], class_ids=np.asarray(class_ids[key])[keep]
                ),
                meta={
                    "sequence": sequence,
                    "game": game,
                    "frame": str(frame),
                    "image_source": source,
                    "other_dropped": str(dropped[key]),
                },
            )
        )
    return samples
