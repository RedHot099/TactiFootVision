"""Roboflow Universe football sets (Hugging Face mirrors) -> 4-class YOLO sets with manifest and leak report.

    uv run python -m scripts.detector_data.roboflow

* ``martinjolif/football-player-detection`` = Roboflow ``football-players-detection-3zvbc`` fork (4 classes, 1:1).
* ``martinjolif/football-ball-detection`` = Roboflow ``football-ball-detection-rejhg`` fork (ball only).

Both are frames of DFL Bundesliga Data Shootout clips (file stems ``<clip>_<half>_...``). Roboflow ``.rf.<hash>``
augmentation copies are collapsed to one image per source stem, then near-duplicate video frames are removed
(dHash + thumbnail correlation, keeping the first in split/name order), then benchmark near-duplicates are dropped.
"""

import json
import re
from pathlib import Path

import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, load_dataset

RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")

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
    "licence": "CC BY 4.0 (mirror card and Roboflow data.yaml); frames are DFL Bundesliga Data Shootout broadcast "
    "clips (Kaggle competition data), so the upstream footage terms are unclear: research use",
    "access_route": "hf download martinjolif/<name> --repo-type dataset (no key); the Roboflow API needs a key",
    "provenance": "unverifiable game identity (DFL clip ids only); no SoccerNet games; benchmark dHash check run",
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


def drop_paths(dataset: Dataset, drop: set[str]) -> Dataset:
    splits = {
        split: [s for s in dataset[split] if str(s.image_path) not in drop]
        for split in dataset.split_names
    }
    return Dataset(dataset.task, dataset.class_names, splits, name=dataset.name)


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
    return drop_paths(dataset, drop), len(drop)


def dedup_frames(dataset: Dataset) -> tuple[Dataset, int]:
    """Drop near-duplicate pictures (adjacent video frames) inside the set."""
    samples = [
        s
        for split in dataset.split_names
        for s in sorted(dataset[split], key=lambda s: s.image_path.name)
    ]
    found = common.Fingerprints.of([s.image_path for s in samples])
    drop = {str(samples[i].image_path) for i in common.dedup_within(found)}
    return drop_paths(dataset, drop), len(drop)


def convert(name: str, spec: dict, references: dict) -> dict:
    dataset = remap(load_dataset(spec["raw"]))
    raw_images = len(dataset)
    dataset, rf_copies = dedup_rf_copies(dataset)
    dataset, near_dups = dedup_frames(dataset)
    kept, report = common.leak_check(dataset, references)
    provenance = {
        str(s.image_path): {
            "clip": source_stem(s.image_path).split("_")[0],
            "source_split": split,
        }
        for split in kept.split_names
        for s in kept[split]
    }
    manifest = (
        spec["manifest"]
        | COMMON_MANIFEST
        | {
            "raw_images": raw_images,
            "rf_augmentation_copies_dropped": rf_copies,
            "near_duplicate_frames_dropped": near_dups,
            "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first frame kept",
            "clips": sorted({p["clip"] for p in provenance.values()}),
        }
    )
    out = common.ROOT / "sets" / name
    common.export_set(
        kept, out, manifest, spec["annotated"], provenance, report, link="hardlink"
    )
    return {
        "images": len(kept),
        "rf": rf_copies,
        "near_dups": near_dups,
        "leak_dropped": len(report["dropped"]),
    }


def main() -> None:
    references = common.benchmark_fingerprints()
    results = {name: convert(name, spec, references) for name, spec in SOURCES.items()}
    print(json.dumps(results))


if __name__ == "__main__":
    main()
