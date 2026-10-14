"""TeamTrack soccer (Scott et al., 2024: 8K fisheye side panorama + drone top view) -> YOLO set with tag report.

    uv run python -m scripts.detector_data.teamtrack

Input: ``raw/teamtrack/dl/teamtrack/soccer_{side,top}/{train,val,test}/{videos/<clip>.mp4, annotations/<clip>.csv}``
(the soccer members of ``teamtrack.zip``, pulled by ``remote_zip.py``). Each csv has three header rows
(``TeamID``, ``PlayerID``, ``Attributes`` = ``bb_height, bb_left, bb_top, bb_width`` per object), a ``frame`` row,
then one row per 1-based frame. Objects are the 22 players (two teams, goalkeepers included) and ``BALL``; an empty
cell means the object is not in the frame. Referees are not annotated. One frame per second (``STRIDE``) is kept;
fixed cameras, so no picture-similarity thinning.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task

RAW = common.ROOT / "raw/teamtrack/dl/teamtrack"
WORK = common.ROOT / "work/teamtrack/images"
OUT = common.ROOT / "sets/teamtrack"
STRIDE = 25  # 25 fps video -> 1 frame per second
VIEWS = ("soccer_side", "soccer_top")
SPLITS = {"train": "train", "val": "valid", "test": "test"}

MANIFEST = {
    "source": "teamtrack",
    "licence_tag": "MIT (Kaggle/GitHub label)",
    "description": "TeamTrack (Scott et al., arXiv 2404.13868), soccer: 8K fisheye side panorama and drone top view",
    "licence": "Kaggle and GitHub: MIT (weak licence basis for data, per open-football-data survey). Research use",
    "access_route": "Google Drive teamtrack.zip (12KOMgl9BFirmPVHbbUTUWacc9vLG94er, TeamTrack README), soccer members "
    "only, read by HTTP range (scripts/detector_data/remote_zip.py)",
    "mapping": {"team players (goalkeepers included)": "player", "BALL": "ball"},
    "caveats": [
        "off-domain fixed cameras (fisheye side panorama 6500x1000 px, drone top view)",
        "goalkeepers are boxed as player and referees are unlabelled: mask goalkeeper and referee for this source",
    ],
    "provenance": "game = recording (e.g. F_20220220_1), clip = 30 s clip, frame = 1-based csv frame",
}


def read_annotations(path: Path) -> dict[int, tuple[np.ndarray, list[int]]]:
    """1-based frame -> ``(xyxy boxes, class ids)``."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    teams, attributes = rows[0][1:], rows[2][1:]
    objects: dict[int, dict[str, int]] = {}
    for column, (team, attribute) in enumerate(zip(teams, attributes, strict=True)):
        objects.setdefault(column // 4, {"ball": team == "BALL"})[attribute] = (
            column + 1
        )
    labels = {}
    for row in rows[3:]:
        if not row or not row[0].strip().isdigit():
            continue
        boxes, ids = [], []
        for obj in objects.values():
            cells = [
                row[obj[a]] if obj[a] < len(row) else ""
                for a in ("bb_left", "bb_top", "bb_width", "bb_height")
            ]
            if any(c.strip() == "" for c in cells):
                continue
            left, top, width, height = map(float, cells)
            if width <= 0 or height <= 0:
                continue
            boxes.append([left, top, left + width, top + height])
            ids.append(common.CLASS_NAMES.index("ball" if obj["ball"] else "player"))
        labels[int(row[0])] = (np.asarray(boxes, np.float32).reshape(-1, 4), ids)
    return labels


def clip_samples(
    view: str, raw_split: str, csv_path: Path
) -> tuple[list[Sample], dict[str, dict]]:
    clip = csv_path.stem
    labels = {
        f: v for f, v in read_annotations(csv_path).items() if (f - 1) % STRIDE == 0
    }
    video = cv2.VideoCapture(str(csv_path.parent.parent / "videos" / f"{clip}.mp4"))
    width, height = int(video.get(3)), int(video.get(4))
    samples, tags = [], {}
    frame = 0
    while labels:
        ok, image = video.read()
        if not ok:
            break
        frame += 1
        if frame not in labels:
            continue
        boxes, ids = labels.pop(frame)
        target = WORK / view / raw_split / f"{clip}_{frame:04d}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        samples.append(
            Sample(target, width, height, Annotations(boxes, np.asarray(ids, np.int64)))
        )
        tags[str(target)] = {
            "game": clip.rsplit("_", 2)[0],
            "clip": f"{view}/{clip}",
            "frame": frame,
            "image_type": view,
        }
    return samples, tags


def main() -> None:
    splits: dict[str, list[Sample]] = {}
    tags: dict[str, dict] = {}
    for view in VIEWS:
        for raw_split, split in SPLITS.items():
            for csv_path in sorted(
                (RAW / view / raw_split / "annotations").glob("*.csv")
            ):
                samples, clip_tags = clip_samples(view, raw_split, csv_path)
                splits.setdefault(split, []).extend(samples)
                tags |= clip_tags
    dataset = Dataset(Task.DETECT, list(common.CLASS_NAMES), splits, name="teamtrack")
    near, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], common.benchmark_fingerprints()
    )
    tags = {path: tags[path] | tag for path, tag in near.items()}
    manifest = MANIFEST | {
        "frame_subsampling": f"every {STRIDE}th frame (1 fps); no picture thinning (fixed cameras)",
        "thinning": False,
        "near_duplicate_frames_thinned": 0,
        "clips": sorted({t["clip"] for t in tags.values()}),
    }
    common.export_set(
        dataset, OUT, manifest, ["ball", "player"], tags, report, link="hardlink"
    )
    print(
        json.dumps({"images": len(dataset), "near_dup_tagged": report["images_tagged"]})
    )


if __name__ == "__main__":
    main()
