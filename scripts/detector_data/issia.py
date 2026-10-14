"""ISSIA-CNR Soccer (6 static 1080p cameras, one match, ~2 min) -> YOLO set with manifest and tag report.

    uv run python -m scripts.detector_data.issia [--thin]

Input: ``raw/issia/x/{filmrole<k>.avi, Annotation Files/Film Role-0 ID-<k> ...xgtf}`` (ViPER XML) from the
FootAndBall Drive copy. Conventions follow FootAndBall's ``data/issia_utils.py``: annotation frame ``f`` is video
frame ``f + FRAME_DELTA`` (-8), the ball x is mirrored for cameras 2 and 6, and the first 50 annotated frames are
skipped as unreliable. ``Person`` boxes carry no role (players, goalkeepers and referees alike) -> ``player`` meaning
"person"; the ball is a point, boxed as a square of ``BALL_TO_PERSON`` x the frame's median person height (clipped to
``BALL_SIDE``). Frames are taken every ``STRIDE``-th frame of the annotated range; picture-similarity thinning is off
by default here (static cameras: the whole picture barely changes, so it would drop ~4 in 5 frames whose labels do).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task

RAW = common.ROOT / "raw/issia/x"
WORK = common.ROOT / "work/issia/images"
OUT = common.ROOT / "sets/issia"
FRAME_DELTA = -8
SKIP_FIRST = 50
STRIDE = 5
BALL_TO_PERSON = 0.12  # ball diameter (0.22 m) / player height (~1.8 m)
BALL_SIDE = (6.0, 30.0)
MIRRORED_BALL_X = (2, 6)
VIPER = "{http://lamp.cfar.umd.edu/viper#}"
VIPER_DATA = "{http://lamp.cfar.umd.edu/viperdata#}"

MANIFEST = {
    "source": "issia",
    "licence_tag": "no licence found",
    "description": "ISSIA-CNR Soccer dataset (D'Orazio et al., 2009), 6 fixed full-HD cameras",
    "licence": "no licence text found; informal copies only (survey: 'No licence found'). Research use",
    "access_route": "Google Drive 1Pj6syLRShNQWQaunJmAZttUw2jDh8L_f (issia.7z, FootAndBall README), fetch_ranged.sh",
    "mapping": {
        "Person (any role)": "player",
        "BALL point": "ball (square box, size from person height)",
    },
    "caveats": [
        "static side cameras, off-domain for broadcast",
        "'player' is every person (referees and goalkeepers included): mask goalkeeper/referee and treat player as "
        "person for this source",
        "ball boxes are synthesised from a point annotation",
    ],
    "provenance": "game = issia (one match), clip = camera, frame = video frame index",
}


def parse_xgtf(path: Path) -> tuple[dict[int, list], dict[int, list]]:
    """``(persons, balls)``: annotation frame -> list of ``[x1, y1, x2, y2]`` / ``[x, y]``."""
    root = ET.parse(path).getroot()
    persons: dict[int, list] = {}
    balls: dict[int, list] = {}
    for obj in root.iter(f"{VIPER}object"):
        name = obj.get("name")
        for attribute in obj.iter(f"{VIPER}attribute"):
            for item in attribute:
                start, end = (int(v) for v in item.get("framespan").split(":"))
                if name == "Person" and item.tag == f"{VIPER_DATA}bbox":
                    x, y = int(item.get("x")), int(item.get("y"))
                    box = [
                        x,
                        y,
                        x + int(item.get("width")),
                        y + int(item.get("height")),
                    ]
                    for f in range(start, end + 1):
                        persons.setdefault(f, []).append(box)
                elif name == "BALL" and item.tag == f"{VIPER_DATA}point":
                    for f in range(start, end + 1):
                        balls.setdefault(f, []).append(
                            [int(item.get("x")), int(item.get("y"))]
                        )
    return persons, balls


def ball_box(point: list, persons: list, width: int, camera: int) -> list:
    x, y = point
    if camera in MIRRORED_BALL_X:
        x = width - x
    heights = [b[3] - b[1] for b in persons]
    side = (
        float(np.clip(BALL_TO_PERSON * np.median(heights), *BALL_SIDE))
        if heights
        else BALL_SIDE[1] / 2
    )
    return [x - side / 2, y - side / 2, x + side / 2, y + side / 2]


def frame_labels(
    persons: dict, balls: dict, width: int, camera: int
) -> dict[int, tuple[np.ndarray, list[int]]]:
    """Video frame -> ``(boxes, class ids)`` for the annotated frames after the first ``SKIP_FIRST``, every ``STRIDE``."""
    annotated = sorted(persons)
    if not annotated:
        return {}
    first = annotated[0] + SKIP_FIRST
    labels = {}
    for f in annotated:
        if f < first or (f - first) % STRIDE:
            continue
        boxes = [list(map(float, b)) for b in persons[f]]
        ids = [common.CLASS_NAMES.index("player")] * len(boxes)
        for point in balls.get(f, []):
            boxes.append(ball_box(point, persons[f], width, camera))
            ids.append(common.CLASS_NAMES.index("ball"))
        labels[f + FRAME_DELTA] = (np.asarray(boxes, np.float32), ids)
    return labels


def camera_samples(camera: int) -> tuple[list[Sample], dict[str, dict]]:
    (xgtf,) = (RAW / "Annotation Files").glob(f"Film Role-0 ID-{camera} *.xgtf")
    persons, balls = parse_xgtf(xgtf)
    video = cv2.VideoCapture(str(RAW / f"filmrole{camera}.avi"))
    width, height = int(video.get(3)), int(video.get(4))
    labels = frame_labels(persons, balls, width, camera)
    samples, tags = [], {}
    index = 0
    while labels:
        ok, frame = video.read()
        if not ok:
            break
        if index in labels:
            boxes, ids = labels.pop(index)
            target = WORK / f"cam{camera}" / f"{index:06d}.jpg"
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            samples.append(
                Sample(target, width, height, Annotations(boxes, np.asarray(ids)))
            )
            tags[str(target)] = {
                "game": "issia",
                "clip": f"cam{camera}",
                "frame": index,
            }
        index += 1
    return samples, tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thin",
        action="store_true",
        help="also thin by picture similarity (off by default: with static cameras and small players it keeps ~1 in 5)",
    )
    args = parser.parse_args()
    args.no_thin = not args.thin
    samples, tags = [], {}
    for camera in range(1, 7):
        s, t = camera_samples(camera)
        samples += s
        tags |= t
    dataset = Dataset(
        Task.DETECT, list(common.CLASS_NAMES), {"train": samples}, name="issia"
    )
    raw_images = len(dataset)
    dataset, thinned = common.thin_video_frames(dataset, thin=not args.no_thin)
    near, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], common.benchmark_fingerprints()
    )
    tags = {path: tags[path] | tag for path, tag in near.items()}
    manifest = MANIFEST | {
        "frames_sampled": raw_images,
        "frame_subsampling": f"every {STRIDE}th annotated frame (5 fps) after the first {SKIP_FIRST}"
        + (
            "; then dHash <= 4 bits + thumbnail corr >= 0.95 within the set"
            if not args.no_thin
            else ""
        ),
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
        "ball_box_rule": f"square, side = {BALL_TO_PERSON} x median person height, clipped to {list(BALL_SIDE)} px",
    }
    common.export_set(
        dataset, OUT, manifest, ["ball", "player"], tags, report, link="hardlink"
    )
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
