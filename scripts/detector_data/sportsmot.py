"""SportsMOT football clips -> player-only YOLO set with manifest and tag report.

    uv run python -m scripts.detector_data.sportsmot

Input: ``raw/sportsmot/<train|val>/<clip>/{gt/gt.txt, seqinfo.ini, img1/*.jpg}`` written by
``download_sportsmot.py`` with frame stride 10 (every 10th frame of 25 fps video). MOT gt rows are
``frame, id, x, y, w, h, conf, class, visibility``; every box is a player on the pitch (goalkeepers included). Referees,
coaches and spectators are deliberately unlabelled and there is no ball, so only ``player`` is annotated (a trainer must
mask the other classes). Near-duplicate frames inside a clip (static shots) are removed with the dHash rule.
"""

import argparse
import configparser
import json
from pathlib import Path

import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task

RAW = common.ROOT / "raw/sportsmot"
OUT = common.ROOT / "sets/sportsmot_football"
STRIDE = 10
SPLITS = {"train": "train", "val": "valid"}
PLAYER = common.CLASS_NAMES.index("player")

MANIFEST = {
    "source": "sportsmot",
    "licence_tag": "CC BY-NC 4.0",
    "description": "SportsMOT (Cui et al., ICCV 2023), football clips only (splits_txt/football.txt), train + val",
    "licence": "CC BY-NC 4.0 (dataset card; HF gate text: non-commercial use only)",
    "access_route": "Hugging Face MCG-NJU/SportsMOT dataset/{train,val}.tar streamed, football clips kept "
    "(scripts/detector_data/download_sportsmot.py); test split has no public ground truth",
    "mapping": {"player (every tracked player, goalkeepers included)": "player"},
    "caveats": [
        "goalkeepers are boxed as player: mask goalkeeper (and referee, ball) for this source",
        "referees are on the pitch but unlabelled: they must not become background for player either; "
        "use this set only with a class mask or for person pretraining",
        "YouTube broadcast clips, 720p",
    ],
    "provenance": "YouTube video id (game) + clip id + frame per image; no SoccerNet games; benchmark near-dups tagged",
}


def read_gt(path: Path) -> dict[int, np.ndarray]:
    """MOT ``gt.txt`` -> ``{frame: (N, 4) xyxy}``."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    frames: dict[int, list] = {}
    for row in rows:
        frame, x, y, w, h = int(row[0]), *row[2:6]
        frames.setdefault(frame, []).append([x, y, x + w, y + h])
    return {f: np.asarray(b, np.float32) for f, b in frames.items()}


def clip_samples(clip_dir: Path) -> tuple[list[Sample], dict[str, dict]]:
    info = configparser.ConfigParser()
    info.read(clip_dir / "seqinfo.ini")
    width, height = int(info["Sequence"]["imWidth"]), int(info["Sequence"]["imHeight"])
    gt = read_gt(clip_dir / "gt" / "gt.txt")
    samples, tags = [], {}
    for image in sorted((clip_dir / "img1").glob("*.jpg")):
        frame = int(image.stem)
        if (frame - 1) % STRIDE:
            continue
        boxes = gt.get(frame, np.zeros((0, 4), np.float32))
        samples.append(
            Sample(
                image, width, height, Annotations(boxes, np.full(len(boxes), PLAYER))
            )
        )
        video = clip_dir.name.rsplit("_c", 1)[0]
        tags[str(image)] = {
            "game": f"youtube:{video.removeprefix('v_')}",
            "clip": clip_dir.name,
            "frame": frame,
        }
    return samples, tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thin", action="store_true", help="keep near-duplicate frames"
    )
    args = parser.parse_args()
    splits: dict[str, list[Sample]] = {}
    tags: dict[str, dict] = {}
    for raw_split, split in SPLITS.items():
        for clip_dir in sorted((RAW / raw_split).iterdir()):
            samples, clip_tags = clip_samples(clip_dir)
            splits.setdefault(split, []).extend(samples)
            tags |= clip_tags
    dataset = Dataset(
        Task.DETECT, list(common.CLASS_NAMES), splits, name="sportsmot_football"
    )
    dataset, thinned = common.thin_video_frames(dataset, thin=not args.no_thin)
    near, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], common.benchmark_fingerprints()
    )
    tags = {path: tags[path] | tag for path, tag in near.items()}
    manifest = MANIFEST | {
        "clips": sorted({t["clip"] for t in tags.values()}),
        "frame_subsampling": f"every {STRIDE}th frame (2.5 fps) at download"
        + (
            ", then dHash <= 4 bits + thumbnail corr >= 0.95 within the set (first kept)"
            if not args.no_thin
            else ""
        ),
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
    }
    common.export_set(dataset, OUT, manifest, ["player"], tags, report, link="hardlink")
    print(
        json.dumps(
            {
                "images": len(dataset),
                "thinned": thinned,
                "near_dup_tagged": report["images_tagged"],
            }
        )
    )


if __name__ == "__main__":
    main()
