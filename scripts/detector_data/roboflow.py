"""Roboflow Universe football sets (Hugging Face mirrors) -> 4-class YOLO sets with manifest and tag report.

    uv run python -m scripts.detector_data.roboflow

* ``martinjolif/football-player-detection`` = Roboflow ``football-players-detection-3zvbc`` fork (4 classes, 1:1).
* ``martinjolif/football-ball-detection`` = Roboflow ``football-ball-detection-rejhg`` fork (ball only).

Both are frames of DFL Bundesliga Data Shootout clips (file stems ``<clip>_<half>_...``). Roboflow ``.rf.<hash>``
augmentation copies are collapsed to one image per source stem and near-duplicate consecutive video frames are thinned
(dHash + thumbnail correlation, keeping the first in split/name order; both counted, ``--no-thin`` keeps everything).
Benchmark near-duplicates are tagged (``near_dup_of``), never dropped.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, load_dataset

RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
DFL_STEM = re.compile(r"^([0-9a-f]{6}_\d+)_(?:mp4-)?(\d+)_(?:jpg|png)$")

SOURCES = {
    "roboflow_players": {
        "raw": common.ROOT / "raw/roboflow_football_player_detection/dl/data/data.yaml",
        "annotated": ["ball", "goalkeeper", "player", "referee"],
        "manifest": {
            "source": "Roboflow Universe football-players-detection (fork football-project-pifbc/"
            "football-players-detection-3zvbc-yyhdl v1 of roboflow-jvuqo/football-players-detection-3zvbc), "
            "via the Hugging Face mirror martinjolif/football-player-detection",
            "mapping": {
                "ball": "ball",
                "goalkeeper": "goalkeeper",
                "player": "player",
                "referee": "referee",
            },
        },
    },
    "roboflow_ball": {
        "raw": common.ROOT / "raw/roboflow_football_ball_detection/dl/data/data.yaml",
        "annotated": ["ball"],
        "manifest": {
            "source": "Roboflow Universe football-ball-detection (fork football-project-pifbc/"
            "football-ball-detection-rejhg-tl9ep v1 of roboflow-jvuqo/football-ball-detection-rejhg), via the "
            "Hugging Face mirror martinjolif/football-ball-detection",
            "mapping": {"ball": "ball"},
        },
    },
}
COMMON_MANIFEST = {
    "source": "roboflow",
    "licence_tag": "CC BY 4.0",
    "licence": "CC BY 4.0 (mirror card and Roboflow data.yaml); frames are DFL Bundesliga Data Shootout broadcast "
    "clips (Kaggle competition data), so the upstream footage terms are unclear: research use",
    "access_route": "hf download martinjolif/<name> --repo-type dataset (no key); the Roboflow API needs a key",
    "provenance": "unverifiable game identity (DFL clip ids only); no SoccerNet games; benchmark near-dups tagged",
}


def source_stem(path: Path) -> str:
    """Image stem without Roboflow's ``.rf.<hash>`` augmentation suffix."""
    return RF_SUFFIX.sub("", path.stem)


def remap(dataset: Dataset) -> Dataset:
    """Same images, class ids re-indexed by name into ``common.CLASS_NAMES``."""
    lookup = np.array(
        [common.CLASS_NAMES.index(n) for n in dataset.class_names], np.int64
    )
    splits = {
        split: [
            Sample(
                s.image_path,
                s.width,
                s.height,
                Annotations(s.annotations.boxes, lookup[s.annotations.class_ids]),
            )
            for s in dataset[split]
        ]
        for split in dataset.split_names
    }
    return Dataset(dataset.task, list(common.CLASS_NAMES), splits, name=dataset.name)


def dedup_rf_copies(dataset: Dataset) -> tuple[Dataset, int]:
    """Keep the first image (split order, then name) of every ``.rf.`` source stem."""
    seen: set[str] = set()
    drop: set[str] = set()
    for split in dataset.split_names:
        for sample in sorted(dataset[split], key=lambda s: s.image_path.name):
            stem = source_stem(sample.image_path)
            if stem in seen:
                drop.add(str(sample.image_path))
            seen.add(stem)
    return common.drop_paths(dataset, drop), len(drop)


def clip_frame(path: Path) -> tuple[str, int | None]:
    """DFL clip id (``<video>_<half>``) and frame index from a Roboflow stem, e.g. ``08fd33_0_mp4-0002_jpg``."""
    match = DFL_STEM.match(source_stem(path))
    if not match:
        return source_stem(path), None
    return match.group(1), int(match.group(2))


def convert(name: str, spec: dict, references: dict, thin: bool = True) -> dict:
    dataset = remap(load_dataset(spec["raw"]))
    raw_images = len(dataset)
    rf_copies = 0
    if thin:
        dataset, rf_copies = dedup_rf_copies(dataset)
    dataset, thinned = common.thin_video_frames(dataset, thin)
    tags, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], references
    )
    for path, tag in tags.items():
        clip, frame = clip_frame(Path(path))
        tag |= {"clip": clip, "frame": frame}
    manifest = (
        spec["manifest"]
        | COMMON_MANIFEST
        | {
            "raw_images": raw_images,
            "thinning": thin,
            "rf_augmentation_copies_thinned": rf_copies,
            "near_duplicate_frames_thinned": thinned,
            "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first frame kept"
            if thin
            else "none (--no-thin)",
            "clips": sorted({t["clip"] for t in tags.values()}),
        }
    )
    out = common.ROOT / "sets" / name
    common.export_set(
        dataset, out, manifest, spec["annotated"], tags, report, link="hardlink"
    )
    return {
        "images": len(dataset),
        "rf": rf_copies,
        "thinned": thinned,
        "near_dup_tagged": report["images_tagged"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thin", action="store_true", help="keep consecutive near-duplicate frames"
    )
    args = parser.parse_args()
    references = common.benchmark_fingerprints()
    results = {
        name: convert(name, spec, references, thin=not args.no_thin)
        for name, spec in SOURCES.items()
    }
    print(json.dumps(results))


if __name__ == "__main__":
    main()
