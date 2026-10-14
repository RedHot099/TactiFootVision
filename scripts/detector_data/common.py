"""Shared pieces of the detector-data source converters (detector-data 03).

* the 4 target classes and the per-source manifest (licence, route, class coverage, provenance, label hashes);
* the leak rules: xG held-out games (plan.csv roles ``evaluation``/``gate``), "no SoccerNet-v2 game is a GSR game",
  and near-duplicate pictures against the benchmark frames (GSR valid + test, ``football_yolo`` valid);
* near-duplicate removal inside a set (adjacent video frames).

Near-duplicates follow the pitch-keypoints exclusions rule: a 64-bit difference hash (dHash) within
``DUPLICATE_BITS`` bits is only a candidate, because the hash pairs different frames dominated by grass stripes; a
candidate is a duplicate when the 64 x 36 grey thumbnails also correlate at ``SAME_PICTURE_CORR`` or more.
"""

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.data import Dataset

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
HELD_OUT_ROLES = ("evaluation", "gate")
DUPLICATE_BITS = 4
SAME_PICTURE_CORR = 0.95
THUMB_SIZE = (64, 36)  # width, height
GREY = 114  # Ultralytics letterbox grey, used to blank ignore regions

ROOT = Path("/mnt/storage/tactifoot_research/detector_data")
MAIN = Path("/home/kuba/projects/tactifoot_vision")
PLAN_CSV = (
    MAIN
    / ".claude/worktrees/claude-fable-xg-poc/experiments/claude_fable/results/xgv_split_plan/plan.csv"
)
GSR_ROOT = MAIN / "data/SoccerNetGS"
FOOTBALL_YOLO_VALID = MAIN / "data/datasets/football_yolo/valid/images"


# ---------------------------------------------------------------- game rules
def held_out_games(plan_csv: Path = PLAN_CSV) -> set[str]:
    """SoccerNet game paths of the 172 xG held-out matches (roles ``evaluation`` and ``gate``)."""
    with Path(plan_csv).open(newline="") as handle:
        return {
            row["game"]
            for row in csv.DictReader(handle)
            if row["role"] in HELD_OUT_ROLES
        }


def is_gsr_game(game: str) -> bool:
    """Whether a SoccerNet game path could be a GSR / Tracking game.

    GSR games carry only an integer ``game_id`` and are all from the 2019 Swiss Super League (GSR and Tracking
    papers), so a game path matches when its league is Swiss or its season is 2019 or later. SoccerNet-v2 games are
    2014-2017 games of six other leagues.
    """
    parts = game.split("/")
    league = parts[0].lower() if parts else ""
    season = parts[1] if len(parts) > 1 else ""
    return "swiss" in league or "switzerland" in league or season[:4] >= "2019"


def check_games(games: Iterable[str], held_out: set[str]) -> dict:
    """Leak rules for SoccerNet game paths. Raises ``AssertionError`` on a held-out or GSR game."""
    games = sorted(set(games))
    leaked = [g for g in games if g in held_out]
    gsr = [g for g in games if is_gsr_game(g)]
    assert not leaked, f"{len(leaked)} xG held-out games in the set, e.g. {leaked[:3]}"
    assert not gsr, f"{len(gsr)} possible GSR games in the set, e.g. {gsr[:3]}"
    return {"games": len(games), "held_out_games_in_set": 0, "gsr_games_in_set": 0}


# --------------------------------------------------------- image fingerprints
def dhash(grey: np.ndarray) -> int:
    """64-bit difference hash of a grey image: 9 x 8 thumbnail, one bit per horizontal neighbour comparison."""
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(np.packbits(bits, bitorder="little").view(np.uint64)[0])


def fingerprint_array(image: np.ndarray) -> tuple[int, np.ndarray]:
    """``(dhash, 64 x 36 uint8 grey thumbnail)`` of a BGR or grey image."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    thumb = cv2.resize(grey, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return dhash(grey), thumb


def fingerprint(path: str | Path) -> tuple[int, np.ndarray]:
    """Fingerprint of an image file (JPEGs are decoded at 1/4 size, which keeps both measures)."""
    path = str(path)
    flag = (
        cv2.IMREAD_REDUCED_GRAYSCALE_4
        if path.lower().endswith((".jpg", ".jpeg"))
        else cv2.IMREAD_GRAYSCALE
    )
    grey = cv2.imread(path, flag)
    if grey is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return fingerprint_array(grey)


def fingerprints(
    paths: Sequence[str | Path], workers: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(hashes uint64 (N,), thumbnails uint8 (N, 36, 64))`` of many image files."""
    if not paths:
        return np.zeros(0, np.uint64), np.zeros(
            (0, THUMB_SIZE[1], THUMB_SIZE[0]), np.uint8
        )
    if len(paths) < 256:
        results = [fingerprint(p) for p in paths]
    else:
        workers = workers or max(1, (os.cpu_count() or 2) - 2)
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(fingerprint, map(str, paths), chunksize=64))
    hashes = np.array([h for h, _ in results], dtype=np.uint64)
    return hashes, np.stack([t for _, t in results])


@dataclass
class Fingerprints:
    """Fingerprints of a list of images (a reference set or a candidate set)."""

    paths: list[str]
    hashes: np.ndarray
    thumbs: np.ndarray

    @classmethod
    def of(
        cls, paths: Sequence[str | Path], workers: int | None = None
    ) -> "Fingerprints":
        hashes, thumbs = fingerprints(paths, workers)
        return cls([str(p) for p in paths], hashes, thumbs)

    @classmethod
    def cached(cls, cache: Path, paths: Sequence[str | Path]) -> "Fingerprints":
        """Load from ``cache`` (``.npz``) when it holds exactly ``paths``, else compute and save."""
        paths = [str(p) for p in paths]
        if cache.is_file():
            data = np.load(cache, allow_pickle=False)
            if list(data["paths"]) == paths:
                return cls(paths, data["hashes"], data["thumbs"])
        found = cls.of(paths)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, paths=np.array(paths), hashes=found.hashes, thumbs=found.thumbs)
        return found


def hamming_pairs(
    a: np.ndarray, b: np.ndarray, max_bits: int = DUPLICATE_BITS, chunk: int = 256
) -> np.ndarray:
    """``(K, 3)`` rows ``(i, j, bits)`` for every ``a[i]``, ``b[j]`` whose hashes differ in at most ``max_bits`` bits."""
    a, b = np.asarray(a, np.uint64), np.asarray(b, np.uint64)
    rows = [np.zeros((0, 3), np.int64)]
    for start in range(0, len(a), chunk):
        bits = np.bitwise_count(a[start : start + chunk, None] ^ b[None, :])
        i, j = np.nonzero(bits <= max_bits)
        rows.append(np.column_stack([i + start, j, bits[i, j]]).astype(np.int64))
    return np.concatenate(rows)


def unit_thumbs(thumbs: np.ndarray) -> np.ndarray:
    """Thumbnails as zero-mean unit vectors, so a dot product is their Pearson correlation."""
    flat = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.where(norm == 0, 1, norm)


def pair_correlations(
    a: np.ndarray, b: np.ndarray, pairs: np.ndarray, chunk: int = 200_000
) -> np.ndarray:
    """Thumbnail correlation of each ``(i, j)`` pair (``a[i]`` vs ``b[j]``)."""
    ua, ub = unit_thumbs(a), unit_thumbs(b)
    out = np.zeros(len(pairs), np.float32)
    for start in range(0, len(pairs), chunk):
        p = pairs[start : start + chunk]
        out[start : start + chunk] = np.einsum("ij,ij->i", ua[p[:, 0]], ub[p[:, 1]])
    return out


def thumb_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two thumbnails (1 for the same picture at any size or JPEG quality)."""
    return float(pair_correlations(a[None], b[None], np.array([[0, 0]]))[0])


def near_duplicates(
    query: Fingerprints,
    reference: Fingerprints,
    max_bits: int = DUPLICATE_BITS,
    min_corr: float = SAME_PICTURE_CORR,
) -> list[dict]:
    """Query images that show the same picture as a reference image (one row per confirmed pair)."""
    pairs = hamming_pairs(query.hashes, reference.hashes, max_bits)
    corr = pair_correlations(query.thumbs, reference.thumbs, pairs)
    return [
        {
            "image": query.paths[i],
            "reference": reference.paths[j],
            "bits": int(bits),
            "corr": round(float(c), 4),
        }
        for (i, j, bits), c in zip(pairs, corr, strict=True)
        if c >= min_corr
    ]


def dedup_within(
    found: Fingerprints,
    max_bits: int = DUPLICATE_BITS,
    min_corr: float = SAME_PICTURE_CORR,
) -> list[int]:
    """Indices to drop so no kept image is a near-duplicate of an earlier kept one (greedy, in list order)."""
    pairs = hamming_pairs(found.hashes, found.hashes, max_bits)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    corr = pair_correlations(found.thumbs, found.thumbs, pairs)
    pairs = pairs[corr >= min_corr]
    drop: set[int] = set()
    for i, j in pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))][:, :2]:
        if int(i) not in drop:
            drop.add(int(j))
    return sorted(drop)


# ------------------------------------------------------------ benchmark refs
def benchmark_images(
    gsr_root: Path = GSR_ROOT, football_yolo_valid: Path = FOOTBALL_YOLO_VALID
) -> dict:
    """Every benchmark frame: all GSR valid and test frames on disk, and ``football_yolo`` valid."""
    gsr = sorted(
        p
        for split in ("valid", "test")
        for p in (gsr_root / split).glob("*/img1/*.jpg")
    )
    yolo_valid = sorted(
        p for p in football_yolo_valid.iterdir() if p.suffix.lower() == ".jpg"
    )
    return {"gsr_valid_test": gsr, "football_yolo_valid": yolo_valid}


def benchmark_fingerprints(
    cache_dir: Path = ROOT / "leak_refs",
) -> dict[str, Fingerprints]:
    return {
        name: Fingerprints.cached(cache_dir / f"{name}.npz", paths)
        for name, paths in benchmark_images().items()
    }


def leak_check(
    dataset: Dataset, references: dict[str, Fingerprints], extra: dict | None = None
) -> tuple[Dataset, dict]:
    """Drop every image that is a near-duplicate of a benchmark frame; return the kept set and the report."""
    images = [s.image_path for s in dataset]
    found = Fingerprints.of(images)
    report: dict = {
        "rule": (
            f"dHash <= {DUPLICATE_BITS} bits and 64x36 grey thumbnail Pearson >= {SAME_PICTURE_CORR}"
        ),
        "images_checked": len(images),
        "references": {name: len(ref.paths) for name, ref in references.items()},
        "reference_coverage": "every frame on disk (GSR valid: all 25 fps frames; GSR test: every 5th frame, all "
        "that is local; football_yolo valid: all 193)",
        "hits": {},
    }
    dropped: set[str] = set()
    for name, ref in references.items():
        hits = near_duplicates(found, ref)
        report["hits"][name] = hits
        dropped |= {hit["image"] for hit in hits}
    report["dropped"] = sorted(dropped)
    report |= extra or {}
    kept = {
        split: [s for s in dataset[split] if str(s.image_path) not in dropped]
        for split in dataset.split_names
    }
    return Dataset(dataset.task, dataset.class_names, kept, name=dataset.name), report


# ------------------------------------------------------------------- export
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grey_out(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Blank ``xyxy`` regions (people with no target role) so they are neither positives nor background."""
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        x1, y1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = GREY
    return out


def export_set(
    dataset: Dataset,
    out_dir: Path,
    manifest: dict,
    annotated: Sequence[str],
    provenance: dict[str, dict] | None = None,
    leak_report: dict | None = None,
    link: str = "symlink",
) -> dict:
    """Write the YOLO set, ``manifest.json``, ``label_sha256.txt`` and ``leak_report.json`` into ``out_dir``.

    ``annotated`` lists the classes this source labels exhaustively; the other target classes are recorded as
    not annotated, so a trainer must ignore them for this source (a missing referee box is not background).
    ``provenance`` maps an image path to its game / clip / frame record.
    """
    unknown = set(annotated) - set(CLASS_NAMES)
    if unknown or dataset.class_names != CLASS_NAMES:
        raise ValueError(f"Unknown classes {unknown or dataset.class_names}")
    data_yaml = dataset.to_yolo(out_dir, link=link)
    labels = sorted((out_dir).glob("*/labels/*.txt"))
    lines = [f"{sha256_file(p)}  {p.relative_to(out_dir).as_posix()}" for p in labels]
    sha_file = out_dir / "label_sha256.txt"
    sha_file.write_text("".join(f"{line}\n" for line in lines))

    summary = dataset.summary()
    counts = {
        split: {k: int(v) for k, v in row.items()} for split, row in summary.iterrows()
    }
    full = {
        **manifest,
        "classes": CLASS_NAMES,
        "annotated_classes": list(annotated),
        "not_annotated_classes": [c for c in CLASS_NAMES if c not in annotated],
        "not_annotated_rule": "classes listed here are absent from the labels; mask them out of the loss for "
        "this source (an unlabelled one is not background)",
        "data_yaml": str(data_yaml),
        "images": len(dataset),
        "boxes_per_split": counts,
        "boxes_total": {
            c: int(sum(row[c] for row in counts.values())) for c in CLASS_NAMES
        },
        "label_files": len(labels),
        "label_sha256_file": sha_file.name,
        "label_sha256_of_list": sha256_file(sha_file),
        "per_image": [
            {
                "split": split,
                "image": str(s.image_path),
                **(provenance or {}).get(str(s.image_path), {}),
            }
            for split in dataset.split_names
            for s in dataset[split]
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(full, indent=1, default=str))
    if leak_report is not None:
        (out_dir / "leak_report.json").write_text(
            json.dumps(leak_report, indent=1, default=str)
        )
    return full


if __name__ == "__main__":  # build the benchmark fingerprint cache once
    for ref_name, ref in benchmark_fingerprints().items():
        print(ref_name, len(ref.paths))
