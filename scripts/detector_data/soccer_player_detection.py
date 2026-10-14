"""SoccerPlayerDetection_bmvc17_v1 (Lu et al., BMVC 2017) -> player-only YOLO set with manifest and tag report.

    uv run python -m scripts.detector_data.soccer_player_detection [--no-thin]

Input: ``raw/soccer_player_detection/x/SoccerPlayerDetection_bmvc17_v1/{annotation_1.mat, annotation_2.mat,
DataSet_001/, DataSet_002/}`` (extracted from the FootAndBall Drive mirror of the UBC zip). ``annot(i).BBox`` is an
``(N, 4)`` ``x1 y1 x2 y2`` array of players, ``annot(i).ImgName`` the frame file (frame number, 0-based, 30 fps) of
two YouTube broadcast videos. Only players are boxed: ball, goalkeeper and referee are not annotated.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task
from tactifoot_vision.data._files import image_size

RAW = common.ROOT / "raw/soccer_player_detection/x/SoccerPlayerDetection_bmvc17_v1"
OUT = common.ROOT / "sets/soccer_player_detection"
GAMES = {1: "XRfXrj9Bkiw", 2: "R_IlsfVMQBk"}  # readme.txt: Game 1 / Game 2 YouTube ids
PLAYER = common.CLASS_NAMES.index("player")

MANIFEST = {
    "source": "soccer_player_detection",
    "licence_tag": "research only",
    "description": "SoccerPlayerDetection_bmvc17_v1 (Lu, Chen, Little, BMVC 2017), UBC",
    "licence": "readme.txt: 'This dataset is used for research only.' Frames of two YouTube broadcast videos",
    "access_route": "http://www.cs.ubc.ca/labs/lci/datasets/SoccerPlayerDetection_bmvc17_v1.zip, mirrored on Google "
    "Drive 1ctJojwDaWtHEAeDmB-AwEcO3apqT-O-9 (FootAndBall README); fetched with fetch_ranged.sh",
    "mapping": {"BBox (player)": "player"},
    "caveats": [
        "players only: referees and goalkeepers may be unboxed or boxed as players (not documented); mask "
        "goalkeeper, referee and ball for this source",
    ],
    "provenance": "game = YouTube video id, clip = DataSet folder, frame = file number",
}


def read_annotations(mat_path: Path) -> dict[str, np.ndarray]:
    """``{image name: (N, 4) xyxy}`` from an ``annotation_<k>.mat``."""
    from scipy.io import loadmat

    annot = loadmat(mat_path)["annot"]
    out = {}
    for entry in annot.ravel():
        name = str(np.asarray(entry["ImgName"]).ravel()[0])
        boxes = np.asarray(entry["BBox"], np.float32).reshape(-1, 4)
        out[name] = boxes
    return out


def build(raw: Path = RAW) -> tuple[Dataset, dict[str, dict]]:
    samples, tags = [], {}
    for game, video in GAMES.items():
        folder = raw / f"DataSet_{game:03d}"
        for name, boxes in sorted(
            read_annotations(raw / f"annotation_{game}.mat").items()
        ):
            image = folder / name
            if not image.is_file():
                continue
            width, height = image_size(image)
            samples.append(
                Sample(
                    image,
                    width,
                    height,
                    Annotations(boxes, np.full(len(boxes), PLAYER)),
                )
            )
            tags[str(image)] = {
                "game": f"youtube:{video}",
                "clip": folder.name,
                "frame": int(Path(name).stem),
            }
    return Dataset(
        Task.DETECT,
        list(common.CLASS_NAMES),
        {"train": samples},
        name="soccer_player_detection",
    ), tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thin", action="store_true", help="keep near-duplicate consecutive frames"
    )
    args = parser.parse_args()
    dataset, tags = build()
    raw_images = len(dataset)
    dataset, thinned = common.thin_video_frames(dataset, thin=not args.no_thin)
    near, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], common.benchmark_fingerprints()
    )
    tags = {path: tags[path] | tag for path, tag in near.items()}
    manifest = MANIFEST | {
        "raw_images": raw_images,
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
        "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first frame kept"
        if not args.no_thin
        else "none (--no-thin)",
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
