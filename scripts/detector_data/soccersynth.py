"""SoccerSynth-Detection (synthetic, Unreal) -> YOLO set with manifest and tag report. Kept separate: synthetic.

    uv run python -m scripts.detector_data.soccersynth

Input: ``raw/soccersynth/dl/SoccerSynth_Detection.zip`` (``train/<n>.png`` + ``train/<n>.txt``, same for ``val``;
YOLO txt with classes ``0`` person, ``1`` ball). Images are read straight from the archive and re-encoded as JPEG
(quality 95) under ``work/soccersynth/images``; nothing is extracted. Every person is a ``player`` box: the source
has no roles, so goalkeeper and referee are not annotated (and ``player`` means "any person" here).
Variant per image: 1920x1080 = ``clear``, other sizes (800x600) = ``motion_blur``.
"""

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task

ZIP = common.ROOT / "raw/soccersynth/dl/SoccerSynth_Detection.zip"
WORK = common.ROOT / "work/soccersynth/images"
OUT = common.ROOT / "sets/soccersynth"
SOURCE_CLASSES = {0: "player", 1: "ball"}
SPLITS = {"train": "train", "val": "valid"}
JPEG_QUALITY = 95

MANIFEST = {
    "source": "soccersynth",
    "licence_tag": "unstated (research)",
    "description": "SoccerSynth-Detection (Qin et al., arXiv 2501.09281), open-starlab",
    "synthetic": True,
    "licence": "repo Apache-2.0; the data licence is not stated (research use)",
    "access_route": "Google Drive file 1UJd9ayx5_cXW28NEE8jlLO9oThleZ-y7 (link in github.com/open-starlab/"
    "SoccerSynth-Detection); plain download hits 'quota exceeded', fetched with HTTP range requests "
    "(scripts/detector_data/fetch_ranged.sh)",
    "mapping": {"0 (every person)": "player", "1 ball": "ball"},
    "provenance": "rendered scenes, no real games; benchmark near-dups tagged anyway; clip = archive split folder, frame = file number",
    "caveats": [
        "'player' is every rendered person: referees or goalkeepers, if rendered, are 'player' boxes, so use this set "
        "for pretraining or with goalkeeper/referee masked and player treated as 'person'",
    ],
    "use": "pretraining / ball and blur ablation; keep apart from real data",
}


def parse_yolo(text: str, width: int, height: int) -> tuple[np.ndarray, list[str]]:
    """Normalised ``cls cx cy w h`` lines -> pixel ``xyxy`` boxes and target class names."""
    boxes, classes = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        if cls not in SOURCE_CLASSES:
            raise ValueError(f"unknown SoccerSynth class {cls}")
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
        classes.append(SOURCE_CLASSES[cls])
    return np.asarray(boxes, np.float32).reshape(-1, 4), classes


def convert_members(zip_path: str, stems: list[str], work: str) -> list[dict]:
    """Re-encode ``<split>/<n>.png`` members to JPEG and return one record per image."""
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for stem in stems:
            split = SPLITS[stem.split("/")[0]]
            target = Path(work) / split / f"{Path(stem).name}.jpg"
            pixels = None
            if not target.is_file():
                pixels = cv2.imdecode(
                    np.frombuffer(archive.read(stem + ".png"), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if pixels is None:
                    records.append({"stem": stem, "unreadable": True})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    cv2.imencode(
                        ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )[1].tobytes()
                )
            height, width = (
                pixels.shape[:2]
                if pixels is not None
                else cv2.imread(str(target)).shape[:2]
            )
            boxes, classes = parse_yolo(
                archive.read(stem + ".txt").decode(), width, height
            )
            records.append(
                {
                    "path": str(target),
                    "split": split,
                    "stem": stem,
                    "width": width,
                    "height": height,
                    "variant": "clear"
                    if (width, height) == (1920, 1080)
                    else "motion_blur",
                    "boxes": boxes.tolist(),
                    "classes": classes,
                }
            )
    return records


def to_dataset(records: list[dict]) -> Dataset:
    splits: dict[str, list[Sample]] = {}
    for r in records:
        ids = [common.CLASS_NAMES.index(c) for c in r["classes"]]
        annotations = Annotations(
            np.asarray(r["boxes"], np.float32).reshape(-1, 4), np.asarray(ids, np.int64)
        )
        splits.setdefault(r["split"], []).append(
            Sample(Path(r["path"]), r["width"], r["height"], annotations)
        )
    return Dataset(Task.DETECT, list(common.CLASS_NAMES), splits, name="soccersynth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thin", action="store_true", help="keep near-duplicate frames"
    )
    args = parser.parse_args()
    with zipfile.ZipFile(ZIP) as archive:
        names = set(archive.namelist())
    stems = sorted(
        n[:-4] for n in names if n.endswith(".png") and n[:-4] + ".txt" in names
    )
    unpaired = sum(1 for n in names if n.endswith(".png")) - len(stems)
    chunks = [stems[i : i + 500] for i in range(0, len(stems), 500)]
    records: list[dict] = []
    with ProcessPoolExecutor(10) as pool:
        for part in pool.map(
            convert_members, [str(ZIP)] * len(chunks), chunks, [str(WORK)] * len(chunks)
        ):
            records += part
    problems = [r for r in records if "path" not in r]
    records = [r for r in records if "path" in r]
    dataset, thinned = common.thin_video_frames(
        to_dataset(records), thin=not args.no_thin
    )
    tags, report = common.tag_near_duplicates(
        [s.image_path for s in dataset], common.benchmark_fingerprints()
    )
    variants: dict[str, int] = {}
    for r in records:
        variants[r["variant"]] = variants.get(r["variant"], 0) + 1
        if r["path"] in tags:
            tags[r["path"]] |= {
                "clip": r["stem"].split("/")[0],
                "frame": int(r["stem"].split("/")[1]),
                "image_type": r["variant"],
            }
    manifest = MANIFEST | {
        "raw_images": len(stems),
        "unpaired_images": unpaired,
        "conversion_problems": problems,
        "variants_before_thinning": variants,
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
        "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first in name order kept"
        if not args.no_thin
        else "none (--no-thin)",
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
