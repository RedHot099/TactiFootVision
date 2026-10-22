"""``football_yolo`` (the Roboflow set the baseline detector was trained on) -> pool set + Ekstraklasa dev split.

    uv run python -I -c "import sys; sys.path.insert(0, '<repo>'); from scripts.detector_data import football_yolo as m; m.main()"

Input: ``data/datasets/football_yolo`` (main checkout; 4 classes in the pool's order; ``test`` is a byte copy of
``valid`` and is skipped). Four file-name families, identified by looking at the frames:

* ``<clip>_mp4-NNNN_jpg.rf.<hash>.jpg``: Ekstraklasa (and Lech's UEFA qualifiers) clips from a LiveTag Pro high
  camera, 1920x1080 video exported by Roboflow **stretched to 640x640**. Train holds 1-6 augmented copies per frame
  (rotation with black corners, salt-and-pepper noise); valid holds the unaugmented frame.
* ``<n>.jpg`` (1000, 2000, ...): one Lech Poznan home game, fixed wide camera, 1280x720.
* ``<a>_<b>.jpg``: SoccerNet game 2016-10-19 Barcelona 4-0 Manchester City (Sky Sport, second half), 1280x720.
* ``jsladjf<n>.jpg``: Germany - Azerbaijan (CCTV5 broadcast, first half), 1280x720.

Conversion:

* ``.rf.`` copies collapse to one image per source stem: the valid copy if there is one (unaugmented), else the train
  copy with the fewest black corners (unrotated), then the least salt-and-pepper noise, then the name;
* stretched frames are **un-stretched** to 1280x720 (pure resize, so the normalised YOLO boxes stay valid; 1280x720
  rather than 1920x1080 because the 640 px rows carry no more detail); the other families are already 1280x720;
* consecutive near-duplicate frames are thinned like ``roboflow.py`` (``--no-thin`` keeps them);
* tags: ``clip`` / ``frame`` from the name, ``image_type`` = family, ``game`` where known, benchmark near-dups
  (GSR valid/test) and SoccerNet-v3 matches as in ``soccana.py``. The ``football_yolo_valid`` reference is this set's
  own valid split, so it is not used here.

Ekstraklasa dev split: whole LiveTag clips, about 20-25% of their images, drawn with a fixed seed
(:func:`choose_holdout`), are exported as this set's ``test`` split, so their pool rows have ``split = "test"``
(``Dataset`` splits are train / valid / test; Roboflow's own test was a copy of valid and is skipped). The Roboflow
split was per image and valid shares clips with train, so it means nothing for a held-out clip; each image's Roboflow
split is kept in the dev record. Those images are
also written as a 16:9 YOLO set under ``bench/ekstraklasa_dev`` and recorded in ``research/ekstraklasa_dev_split.json``.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, load_dataset

RAW = common.MAIN / "data/datasets/football_yolo"
WORK = common.ROOT / "work/football_yolo"
OUT = common.ROOT / "sets/football_yolo"
DEV = common.ROOT / "bench/ekstraklasa_dev"
DEV_JSON = Path(__file__).resolve().parents[2] / "research/ekstraklasa_dev_split.json"
HOLDOUT_SPLIT = "test"  # the set's own test split; Roboflow's test was a copy of valid
HOLDOUT_SHARE = (0.20, 0.25)
HOLDOUT_SEED = 0
UNSTRETCHED = (1280, 720)  # width, height of the un-stretched 16:9 frame
STRETCHED = (640, 640)
JPEG_QUALITY = 95

RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
LIVETAG = re.compile(r"^(?P<clip>.+)_mp4-(?P<frame>\d+)_jpg$")
BARCA_CITY = "europe_uefa-champions-league/2016-2017/2016-10-19 - 21-45 Barcelona 4 - 0 Manchester City"
FAMILIES = {  # family -> (name pattern, clip, game)
    "lech_home_wide": (re.compile(r"^(?P<frame>\d+)$"), "lech_home_wide", None),
    "ucl_barcelona_mancity": (
        re.compile(r"^(?P<frame>\d+)_\d+$"),
        "barcelona_mancity_2016",
        BARCA_CITY,
    ),
    "germany_azerbaijan": (
        re.compile(r"^jsladjf(?P<frame>\d+)$"),
        "germany_azerbaijan_cctv",
        None,
    ),
}

MANIFEST = {
    "source": "football_yolo",
    "licence_tag": "unknown (Roboflow export)",
    "description": "data/datasets/football_yolo: the Roboflow Universe export the baseline YOLO11m was trained on "
    "(4 classes ball / goalkeeper / player / referee)",
    "licence": "no licence file or README in the export (data.yaml has only paths and names); Roboflow Universe "
    "workspace not recorded; frames are Ekstraklasa / UEFA broadcast and tactical-camera footage: research use, "
    "unclear terms",
    "access_route": "local copy in the main checkout (data/datasets/football_yolo, a symlink kept out of git)",
    "mapping": {c: c for c in common.CLASS_NAMES},
    "families": {
        "lech_livetag": "<clip>_mp4-NNNN: Lech Poznan games (Ekstraklasa: Slask, Zaglebie, Piast; UEFA: Kauno "
        "Zalgiris) from a LiveTag Pro high camera, Roboflow-stretched 1920x1080 -> 640x640, un-stretched here",
        "lech_home_wide": "<n>.jpg: one Lech Poznan home game, fixed wide camera, 1280x720 (Ekstraklasa)",
        "ucl_barcelona_mancity": "<a>_<b>.jpg: SoccerNet game "
        + BARCA_CITY
        + " (identified from the "
        "scoreboard: Barcelona v Man City, Sky Sport, 1-0 at 49' to 3-0 at 87'), 1280x720",
        "germany_azerbaijan": "jsladjf<n>.jpg: Germany - Azerbaijan, CCTV5 broadcast, first half, 1280x720",
    },
    "provenance": "clip / frame from the file name; game for the SoccerNet family (visual identification) and the "
    "LiveTag clips (team names in the clip name); no pixel match against SoccerNet-v3 or GSR",
}


# ------------------------------------------------------------------ names
def source_stem(path: Path) -> str:
    """Image stem without Roboflow's ``.rf.<hash>`` suffix."""
    return RF_SUFFIX.sub("", path.stem)


def livetag_game(clip: str) -> str:
    """``football_yolo:<teams>`` from a LiveTag clip name, e.g. ``...-2-polowa-slask-lech-`` -> ``slask-lech``."""
    name = clip.strip("-").lower()
    if "polowa-" in name:
        teams = name.rsplit("polowa-", 1)[1]
    else:
        teams = "-".join(name.split("-")[-4:])  # two two-word club names
    return f"football_yolo:{teams.strip('-')}"


def parse_name(path: Path) -> dict:
    """``{family, clip, frame, game}`` of a football_yolo image from its file name."""
    stem = source_stem(path)
    match = LIVETAG.match(stem)
    if match:
        clip = match.group("clip").strip("-")
        return {
            "family": "lech_livetag",
            "clip": clip,
            "frame": int(match.group("frame")),
            "game": livetag_game(clip),
        }
    for family, (pattern, clip, game) in FAMILIES.items():
        match = pattern.match(stem)
        if match:
            return {
                "family": family,
                "clip": clip,
                "frame": int(match.group("frame")),
                "game": game,
            }
    raise ValueError(f"unknown football_yolo file-name family: {path.name}")


# ------------------------------------------------------------------ copies
def augmentation_marks(image: np.ndarray) -> tuple[int, float]:
    """``(black corners, salt-and-pepper share)``: how visibly a Roboflow copy is augmented.

    A rotated copy has black fill in its corners (0-4 of the 8x8 corner patches dark); noise shows as pixels far
    from their 3x3 median.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = [grey[:8, :8], grey[:8, -8:], grey[-8:, :8], grey[-8:, -8:]]
    black = sum(int(c.mean() < 10) for c in corners)
    noise = float((cv2.absdiff(grey, cv2.medianBlur(grey, 3)) > 60).mean())
    return black, round(noise, 5)


def pick_copy(copies: list[dict]) -> dict:
    """The least augmented of a stem's copies (``{split, path, black, noise}``): valid first (Roboflow does not
    augment valid), then fewest black corners, least noise, name."""
    return min(
        copies,
        key=lambda c: (
            c["split"] != "valid",
            c["black"],
            c["noise"],
            Path(c["path"]).name,
        ),
    )


def collapse_copies(dataset: Dataset) -> tuple[Dataset, dict]:
    """One sample per source stem (:func:`pick_copy`), in the split of the kept copy."""
    groups: dict[str, list[dict]] = {}
    for split in dataset.split_names:
        for sample in dataset[split]:
            stem = source_stem(sample.image_path)
            groups.setdefault(stem, []).append(
                {"split": split, "path": str(sample.image_path), "sample": sample}
            )
    keep: set[str] = set()
    kept_marks = Counter()
    for copies in groups.values():
        if len(copies) > 1:
            for copy in copies:
                image = cv2.imread(copy["path"], cv2.IMREAD_GRAYSCALE)
                copy["black"], copy["noise"] = augmentation_marks(image)
        else:
            copies[0] |= {"black": 0, "noise": 0.0}
        best = pick_copy(copies)
        keep.add(best["path"])
        if len(copies) > 1:
            kept_marks[f"{best['split']}, {best['black']} black corners"] += 1
    drop = {str(s.image_path) for s in dataset} - keep
    report = {
        "source_stems": len(groups),
        "rf_copies_dropped": len(drop),
        "stems_with_copies": sum(len(c) > 1 for c in groups.values()),
        "kept_copy_kind": dict(sorted(kept_marks.items())),
    }
    return common.drop_paths(dataset, drop), report


# ------------------------------------------------------------------ un-stretch
def unstretch_sample(sample: Sample, out: Path, size=UNSTRETCHED) -> Sample:
    """Resize a stretched frame to ``size`` (written to ``out``); boxes scale with it (normalised boxes unchanged)."""
    sx, sy = size[0] / sample.width, size[1] / sample.height
    image = cv2.resize(sample.read_image(), size, interpolation=cv2.INTER_CUBIC)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    boxes = np.asarray(sample.annotations.boxes, float) * np.array([sx, sy, sx, sy])
    return Sample(
        out, size[0], size[1], Annotations(boxes, sample.annotations.class_ids)
    )


def unstretch(dataset: Dataset, work: Path = WORK) -> tuple[Dataset, int]:
    """Every ``STRETCHED`` sample un-stretched into ``work/<split>/<source stem>.jpg``; others unchanged."""
    splits, count = {}, 0
    for split in dataset.split_names:
        samples = []
        for sample in dataset[split]:
            if (sample.width, sample.height) == STRETCHED:
                out = work / split / f"{source_stem(sample.image_path)}.jpg"
                sample = unstretch_sample(sample, out)
                count += 1
            samples.append(sample)
        splits[split] = samples
    return Dataset(dataset.task, dataset.class_names, splits, name=dataset.name), count


# ------------------------------------------------------------------ dev split
def choose_holdout(
    clip_images: dict[str, int], share=HOLDOUT_SHARE, seed: int = HOLDOUT_SEED
) -> list[str]:
    """Whole clips holding ``share[0]`` to ``share[1]`` of the images, drawn in a seeded random order.

    Clips are visited in ``np.random.default_rng(seed).permutation`` of their sorted names; a clip is taken when the
    total stays at or below ``share[1]``, until it reaches ``share[0]``.
    """
    clips = sorted(clip_images)
    total = sum(clip_images.values())
    low, high = share[0] * total, share[1] * total
    chosen, held = [], 0
    for i in np.random.default_rng(seed).permutation(len(clips)):
        n = clip_images[clips[i]]
        if held + n <= high:
            chosen.append(clips[i])
            held += n
        if held >= low:
            return sorted(chosen)
    raise ValueError(f"no clip set holds {share} of {total} images (got {held})")


def move_to_holdout(
    dataset: Dataset, clips: set[str]
) -> tuple[Dataset, dict[str, str]]:
    """Samples of ``clips`` moved to the ``HOLDOUT_SPLIT`` (test) split; returns their original split per image path."""
    splits: dict[str, list[Sample]] = {s: [] for s in dataset.split_names}
    splits[HOLDOUT_SPLIT] = []
    original = {}
    for split in dataset.split_names:
        for sample in dataset[split]:
            if parse_name(sample.image_path)["clip"] in clips:
                splits[HOLDOUT_SPLIT].append(sample)
                original[str(sample.image_path)] = split
            else:
                splits[split].append(sample)
    return Dataset(
        dataset.task, dataset.class_names, splits, name=dataset.name
    ), original


# ------------------------------------------------------------------ main
def load(raw: Path = RAW) -> Dataset:
    """Train and valid of the export (``test`` is checked to be the same files as valid and left out)."""
    test, valid = raw / "test/images", raw / "valid/images"
    if test.is_dir():
        names = sorted(p.name for p in test.iterdir())
        if names != sorted(p.name for p in valid.iterdir()) or any(
            common.sha256_file(test / n) != common.sha256_file(valid / n) for n in names
        ):
            raise ValueError("football_yolo test differs from valid: convert it too")
    dataset = load_dataset(raw / "data.yaml")
    if dataset.class_names != common.CLASS_NAMES:
        raise ValueError(f"unexpected classes {dataset.class_names}")
    return Dataset(
        dataset.task,
        dataset.class_names,
        {s: dataset[s] for s in ("train", "valid")},
        name="football_yolo",
    )


def tag(dataset: Dataset) -> tuple[dict[str, dict], dict]:
    images = [s.image_path for s in dataset]
    references = {
        k: v
        for k, v in common.benchmark_fingerprints().items()
        if k != "football_yolo_valid"  # this set's own valid split
    }
    tags, report = common.tag_near_duplicates(images, references)
    v3_fingerprints, v3_rows = common.v3_references()
    v3_hits = common.near_duplicates(common.Fingerprints.of(images), v3_fingerprints)
    report["hits"]["soccernet_v3"] = v3_hits
    report["images_matching_v3"] = len({h["image"] for h in v3_hits})
    if v3_hits:
        raise ValueError("v3 matches: carry game / xg_role over as in soccana.py")
    roles = common.xg_roles()
    for path, row in tags.items():
        name = parse_name(Path(path))
        row |= {
            "clip": name["clip"],
            "frame": name["frame"],
            "image_type": name["family"],
            "game": name["game"],
            "xg_role": roles.get(name["game"]) if name["game"] else None,
        }
    return tags, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--no-thin", action="store_true", help="keep consecutive near-duplicate frames"
    )
    args = parser.parse_args()
    dataset = load()
    raw_images = len(dataset)
    dataset, copies = collapse_copies(dataset)
    dataset, thinned = common.thin_video_frames(dataset, thin=not args.no_thin)
    dataset, unstretched = unstretch(dataset)

    livetag = Counter(
        parse_name(s.image_path)["clip"]
        for s in dataset
        if parse_name(s.image_path)["family"] == "lech_livetag"
    )
    held = choose_holdout(dict(livetag))
    dataset, original_split = move_to_holdout(dataset, set(held))
    tags, report = tag(dataset)
    family_counts = Counter(t["image_type"] for t in tags.values())
    manifest = MANIFEST | {
        "raw_images": raw_images,
        "rf_copies": copies,
        "test_split": "byte copy of valid, skipped",
        "unstretched": f"{unstretched} images 640x640 -> {UNSTRETCHED[0]}x{UNSTRETCHED[1]} (cv2 INTER_CUBIC, "
        f"JPEG q{JPEG_QUALITY}, work/football_yolo)",
        "thinning": not args.no_thin,
        "near_duplicate_frames_thinned": thinned,
        "frame_subsampling": "dHash <= 4 bits + thumbnail corr >= 0.95 within the set, first kept"
        if not args.no_thin
        else "none (--no-thin)",
        "near_dup_references": "GSR valid/test frames and SoccerNet-v3 (football_yolo valid is this set itself)",
        "images_by_family": dict(sorted(family_counts.items())),
        "dev_holdout": {
            "split": HOLDOUT_SPLIT,
            "clips": held,
            "images": len(original_split),
            "record": str(DEV_JSON),
        },
    }
    common.export_set(
        dataset,
        OUT,
        manifest,
        list(common.CLASS_NAMES),
        tags,
        report,
        link="hardlink",
    )

    # The dev set: the held-out images as a YOLO "valid" split, read by scripts/detector_bench.py.
    dev = Dataset(
        dataset.task,
        dataset.class_names,
        {"valid": dataset[HOLDOUT_SPLIT]},
        name="ekstraklasa_dev",
    )
    data_yaml = dev.to_yolo(DEV)
    labels = sorted((DEV / "valid/labels").glob("*.txt"))
    digest = hashlib.sha256()
    for path in labels:
        digest.update(path.name.encode() + b"\n" + path.read_bytes())
    record = {
        "title": "Ekstraklasa dev split of football_yolo (detector-data 04, run 2)",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": f"whole LiveTag (lech_livetag) clips, {HOLDOUT_SHARE[0]:.0%}-{HOLDOUT_SHARE[1]:.0%} of their "
        f"images, clips visited in np.random.default_rng({HOLDOUT_SEED}).permutation(sorted clips) order "
        "(scripts/detector_data/football_yolo.py choose_holdout)",
        "seed": HOLDOUT_SEED,
        "livetag_images": sum(livetag.values()),
        "livetag_images_per_clip": dict(sorted(livetag.items())),
        "clips": held,
        "games": sorted({livetag_game(c) for c in held}),
        "images": len(original_split),
        "share": round(len(original_split) / sum(livetag.values()), 4),
        "boxes": dev.summary().loc["valid"].astype(int).to_dict(),
        "pool_tag": f"sets/football_yolo rows with split = {HOLDOUT_SPLIT!r}",
        "data_yaml": str(data_yaml),
        "labels_sha256": digest.hexdigest(),
        "caveat": "the baseline detector was trained on these clips (football_yolo train+valid), so its scores here "
        "are in-sample",
        "images_by_roboflow_split": dict(Counter(original_split.values())),
        "files": sorted(
            f"{original_split[str(s.image_path)]}:{Path(s.image_path).name}"
            for s in dev["valid"]
        ),
    }
    DEV_JSON.parent.mkdir(exist_ok=True)
    DEV_JSON.write_text(json.dumps(record, indent=2, default=int) + "\n")
    print(
        json.dumps(
            {
                "images": len(dataset),
                "rf": copies,
                "thinned": thinned,
                "unstretched": unstretched,
                "families": manifest["images_by_family"],
                "near_dup_tagged": report["images_tagged"],
                "holdout": {"clips": held, "images": len(original_split)},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
