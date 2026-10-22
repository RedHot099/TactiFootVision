"""Soccana_player_ball_detection_v1 (Hugging Face, 3 classes) -> YOLO set with manifest and tag report.

    uv run python -m scripts.detector_data.soccana [--no-thin]

Input: ``raw/soccana/x/V1/{images,labels}/{train,test}`` from ``V1.zip`` (Ultralytics layout, classes
``0 player, 1 referee, 2 ball``). The card says the images come from SoccerNet, game renders and public match footage,
with SAHI tiles (160-1280 px) and augmentations, and states no licence. The survey advised avoiding it (unknown
provenance); under the collect-everything rule it is kept and tagged instead: besides the benchmark near-dup check,
every image is matched against all SoccerNet-v3 frames, and a hit carries the v3 game and its ``xg_role``.
Goalkeepers are labelled ``player`` (no goalkeeper class), so goalkeeper is not annotated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task, load_dataset

RAW = common.ROOT / "raw/soccana/x/V1"
WORK = common.ROOT / "work/soccana"
OUT = common.ROOT / "sets/soccana"
SOURCE_NAMES = ["player", "referee", "ball"]

MANIFEST = {
    "source": "soccana",
    "licence_tag": "not stated",
    "description": "Soccana_player_ball_detection_v1 (Adit-jain on Hugging Face), 25K-image subset card, V1.zip",
    "licence": "no licence on the card or repo; mixes SoccerNet frames, game renders and public footage: research use, "
    "unclear terms",
    "access_route": "hf download Adit-jain/Soccana_player_ball_detection_v1 V1.zip --repo-type dataset (no gate)",
    "mapping": {
        "player (goalkeepers included)": "player",
        "referee": "referee",
        "ball": "ball",
    },
    "caveats": [
        "goalkeepers are labelled player: mask goalkeeper for this source",
        "contains SAHI tiles, augmented and synthetic (game render) images, not flagged per image",
        "provenance unknown: SoccerNet frames are matched against v3 (game, xg_role) and the benchmarks (near_dup_of)",
    ],
    "provenance": "file-name prefix (e.g. v3_sahi, v7_sahi) as clip; game and xg_role only where a v3 frame matches",
}


def load(raw: Path = RAW) -> Dataset:
    """The V1 folders as a 4-class dataset (a ``data.yaml`` is written under ``WORK``, never into ``raw``)."""
    config = WORK / "data.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(raw),
                "train": "images/train",
                "test": "images/test",
                "names": SOURCE_NAMES,
            }
        )
    )
    source = load_dataset(config)
    lookup = np.array([common.CLASS_NAMES.index(n) for n in SOURCE_NAMES], np.int64)
    splits = {
        split: [
            Sample(
                s.image_path,
                s.width,
                s.height,
                Annotations(s.annotations.boxes, lookup[s.annotations.class_ids]),
            )
            for s in source[split]
        ]
        for split in source.split_names
    }
    return Dataset(Task.DETECT, list(common.CLASS_NAMES), splits, name="soccana")


def prefix(path: Path) -> str:
    """File-name family, e.g. ``v3_sahi`` for ``v3_sahi_00012.jpg``."""
    return path.stem.rstrip("0123456789_-") or path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thin", action="store_true", help="keep near-duplicate images"
    )
    args = parser.parse_args()
    dataset = load()
    raw_images = len(dataset)
    dataset, thinned = common.thin_video_frames(dataset, thin=not args.no_thin)
    images = [s.image_path for s in dataset]
    tags, report = common.tag_near_duplicates(images, common.benchmark_fingerprints())
    v3_fingerprints, v3_rows = common.v3_references()
    v3_hits = common.near_duplicates(common.Fingerprints.of(images), v3_fingerprints)
    report["hits"]["soccernet_v3"] = v3_hits
    best: dict[str, dict] = {}
    for hit in v3_hits:
        if hit["image"] not in best or hit["corr"] > best[hit["image"]]["corr"]:
            best[hit["image"]] = hit
    for path, hit in best.items():
        tag = tags[path]
        tag |= v3_rows[hit["reference"]]  # game + xg_role of the matched v3 frame
        tag["near_dup_hits"] += 1
        if tag["near_dup_of"] is None:
            of, ref_set, _ = common.reference_id(hit["reference"])
            tag |= {
                "near_dup_of": of,
                "near_dup_set": ref_set,
                "near_dup_corr": hit["corr"],
            }
    report["images_matching_v3"] = len(best)
    for path, tag in tags.items():
        tag["clip"] = prefix(Path(path))
    manifest = MANIFEST | {
        "raw_images": raw_images,
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
        "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first kept"
        if not args.no_thin
        else "none (--no-thin)",
        "images_matching_v3": len(best),
    }
    common.export_set(
        dataset,
        OUT,
        manifest,
        ["ball", "player", "referee"],
        tags,
        report,
        link="hardlink",
    )
    print(
        json.dumps(
            {
                "images": len(dataset),
                "thinned": thinned,
                "near_dup_tagged": report["images_tagged"],
                "v3": len(best),
            }
        )
    )


if __name__ == "__main__":
    main()
