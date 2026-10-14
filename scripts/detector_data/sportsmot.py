"""SportsMOT football clips -> player-only YOLO set with manifest and leak report.

    uv run python -m scripts.detector_data.sportsmot

Input: ``raw/sportsmot/<train|val>/<clip>/{gt/gt.txt, seqinfo.ini, img1/*.jpg}`` written by
``download_sportsmot.py`` with frame stride 10 (every 10th frame of 25 fps video). MOT gt rows are
``frame, id, x, y, w, h, conf, class, visibility``; every box is a player on the pitch (goalkeepers included). Referees,
coaches and spectators are deliberately unlabelled and there is no ball, so only ``player`` is annotated (a trainer must
mask the other classes). Near-duplicate frames inside a clip (static shots) are removed with the dHash rule.
"""

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
    "source": "SportsMOT (Cui et al., ICCV 2023), football clips only (splits_txt/football.txt), train + val",
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
    "provenance": "YouTube video id + clip id per image; no SoccerNet games; benchmark dHash check run",
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
    samples, provenance = [], {}
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
        provenance[str(image)] = {"video": video, "clip": clip_dir.name, "frame": frame}
    return samples, provenance


def main() -> None:
    splits: dict[str, list[Sample]] = {}
    provenance: dict[str, dict] = {}
    for raw_split, split in SPLITS.items():
        for clip_dir in sorted((RAW / raw_split).iterdir()):
            samples, prov = clip_samples(clip_dir)
            splits.setdefault(split, []).extend(samples)
            provenance |= prov
    dataset = Dataset(
        Task.DETECT, list(common.CLASS_NAMES), splits, name="sportsmot_football"
    )
    samples = list(dataset)
    found = common.Fingerprints.of([s.image_path for s in samples])
    drop = {str(samples[i].image_path) for i in common.dedup_within(found)}
    dataset = Dataset(
        Task.DETECT,
        dataset.class_names,
        {
            s: [x for x in dataset[s] if str(x.image_path) not in drop]
            for s in dataset.split_names
        },
        name=dataset.name,
    )
    kept, report = common.leak_check(dataset, common.benchmark_fingerprints())
    manifest = MANIFEST | {
        "clips": sorted({p["clip"] for p in provenance.values()}),
        "frame_subsampling": f"every {STRIDE}th frame (2.5 fps), then dHash <= 4 bits + thumbnail corr >= 0.95 "
        "within the set (first kept)",
        "near_duplicate_frames_dropped": len(drop),
    }
    common.export_set(
        kept, OUT, manifest, ["player"], provenance, report, link="hardlink"
    )
    print(
        json.dumps(
            {
                "images": len(kept),
                "near_dups": len(drop),
                "leak_dropped": len(report["dropped"]),
            }
        )
    )


if __name__ == "__main__":
    main()
