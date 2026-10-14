"""Shared pieces of the detector-data source converters (detector-data 03).

Collect everything, tag, split later (owner rule, 2026-10-14): converters drop nothing for leak reasons. Each image
carries tags (see ``README.md`` in this folder for the pool-index schema):

* ``xg_role`` from plan.csv (``training`` / ``evaluation`` / ``gate``; null when the game is not in the plan);
* ``near_dup_of``: the benchmark frame (GSR valid/test frame or ``football_yolo`` valid image) the picture duplicates,
  with ``gsr_split`` when that frame is GSR;
* licence, annotated classes, provenance (game, clip, frame) and per-class box counts.

The only thing a converter may remove is in-set near-duplicates of consecutive video frames ("thinning", counted in
the manifest, ``--no-thin`` keeps them).

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
from tactifoot_vision.data._files import unique_names

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
HELD_OUT_ROLES = ("evaluation", "gate")  # the 172 xG matches the default view excludes
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
def xg_roles(plan_csv: Path = PLAN_CSV) -> dict[str, str]:
    """SoccerNet game path -> xG split-plan role (``training``, ``evaluation`` or ``gate``)."""
    with Path(plan_csv).open(newline="") as handle:
        return {row["game"]: row["role"] for row in csv.DictReader(handle)}


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


def game_tags(games: Iterable[str], roles: dict[str, str]) -> dict:
    """Per-game ``xg_role`` and a summary; asserts no game is a GSR game (a fact for SoccerNet-v2 paths, so a
    failure means the input is not what we think it is)."""
    games = sorted(set(games))
    gsr = [g for g in games if is_gsr_game(g)]
    assert not gsr, (
        f"{len(gsr)} possible GSR games among SoccerNet-v2 paths, e.g. {gsr[:3]}"
    )
    by_role: dict[str, int] = {}
    for g in games:
        key = roles.get(g) or "none"
        by_role[key] = by_role.get(key, 0) + 1
    return {"games": len(games), "games_by_xg_role": by_role, "gsr_games": 0}


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


def reference_id(path: str) -> tuple[str, str, str | None]:
    """``(near_dup_of, near_dup_set, gsr_split)`` for a benchmark frame path.

    GSR frames become ``gsr:<split>/<clip>/<frame>`` (e.g. ``gsr:valid/SNGS-021/000001``); other references become
    ``football_yolo_valid:<file name>``, and SoccerNet-v3 frames ``v3:<work-copy stem>``.
    """
    parts = Path(path).parts
    if (
        len(parts) >= 4
        and parts[-2] == "img1"
        and parts[-4] in ("train", "valid", "test", "challenge")
    ):
        split, clip, frame = parts[-4], parts[-3], Path(parts[-1]).stem
        return f"gsr:{split}/{clip}/{frame}", f"gsr_{split}", split
    if "soccernet_v3" in parts:  # a v3 frame (work copy): see v3_references()
        return f"v3:{Path(path).stem}", "soccernet_v3", None
    return f"football_yolo_valid:{parts[-1]}", "football_yolo_valid", None


def tag_near_duplicates(
    images: Sequence[str | Path], references: dict[str, Fingerprints]
) -> tuple[dict[str, dict], dict]:
    """Tag every image that shows the same picture as a benchmark frame; nothing is removed.

    Returns ``{image path: {near_dup_of, near_dup_set, near_dup_corr, near_dup_hits, gsr_split}}`` (all images,
    ``None`` where there is no hit; the best-correlated hit wins) and a report with every confirmed pair.
    """
    found = Fingerprints.of(images)
    tags = {
        p: {
            "near_dup_of": None,
            "near_dup_set": None,
            "near_dup_corr": None,
            "near_dup_hits": 0,
            "gsr_split": None,
        }
        for p in found.paths
    }
    report: dict = {
        "rule": f"dHash <= {DUPLICATE_BITS} bits and 64x36 grey thumbnail Pearson >= {SAME_PICTURE_CORR}",
        "policy": "tag only (owner rule 2026-10-14: collect everything, split later); no image is dropped",
        "images_checked": len(found.paths),
        "references": {name: len(ref.paths) for name, ref in references.items()},
        "reference_coverage": "every frame on disk (GSR valid: all 25 fps frames; GSR test: every 5th frame, all "
        "that is local; football_yolo valid: all 193)",
        "hits": {},
    }
    for name, ref in references.items():
        hits = near_duplicates(found, ref)
        report["hits"][name] = hits
        for hit in hits:
            tag = tags[hit["image"]]
            tag["near_dup_hits"] += 1
            if tag["near_dup_corr"] is None or hit["corr"] > tag["near_dup_corr"]:
                of, ref_set, gsr_split = reference_id(hit["reference"])
                tag |= {
                    "near_dup_of": of,
                    "near_dup_set": ref_set,
                    "near_dup_corr": hit["corr"],
                    "gsr_split": gsr_split,
                }
    tagged = [t for t in tags.values() if t["near_dup_of"]]
    report["images_tagged"] = len(tagged)
    report["tagged_by_set"] = {
        s: sum(t["near_dup_set"] == s for t in tagged)
        for s in sorted({t["near_dup_set"] for t in tagged})
    }
    return tags, report


def drop_paths(dataset: Dataset, drop: set[str]) -> Dataset:
    """The dataset without the images whose path is in ``drop``."""
    splits = {
        split: [s for s in dataset[split] if str(s.image_path) not in drop]
        for split in dataset.split_names
    }
    return Dataset(dataset.task, dataset.class_names, splits, name=dataset.name)


def thin_video_frames(dataset: Dataset, thin: bool = True) -> tuple[Dataset, int]:
    """Remove in-set near-duplicates (consecutive video frames), keeping the first in split/name order.

    With ``thin=False`` nothing is removed (``--no-thin``). Returns the set and the number removed.
    """
    if not thin:
        return dataset, 0
    samples = [
        s
        for split in dataset.split_names
        for s in sorted(dataset[split], key=lambda s: s.image_path.name)
    ]
    found = Fingerprints.of([s.image_path for s in samples])
    drop = {str(samples[i].image_path) for i in dedup_within(found)}
    return drop_paths(dataset, drop), len(drop)


def v3_references(
    cache_dir: Path = ROOT / "leak_refs",
) -> tuple[Fingerprints, dict[str, dict]]:
    """Fingerprints of every SoccerNet-v3 frame in ``sets/soccernet_v3`` and ``{v3 work path: {game, xg_role}}``.

    For sources that mix in SoccerNet frames of unknown games: a hit gives the image the v3 game and its xG role.
    """
    manifest = json.loads((ROOT / "sets/soccernet_v3/manifest.json").read_text())
    rows = {
        r["source_image"]: {"game": r["game"], "xg_role": r["xg_role"]}
        for r in manifest["per_image"]
    }
    return Fingerprints.cached(cache_dir / "soccernet_v3.npz", sorted(rows)), rows


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


# One row per image in pool_index.parquet; every set's manifest ``per_image`` carries exactly these keys.
SCHEMA: dict[str, str] = {
    "set": "set folder name under sets/ (one per source or source part)",
    "source": "source family, e.g. soccernet_v3, roboflow, soccersynth, sportsmot, gsr",
    "split": "split inside the set's YOLO export (the source's own split, not a training decision)",
    "image": "absolute path of the exported image (sets/<set>/<split>/images/...); its label is the YOLO txt beside",
    "source_image": "path of the image the export links to (raw or work copy)",
    "width": "image width, px",
    "height": "image height, px",
    "game": "game identity: SoccerNet game path, GSR game_id as 'gsr:<id>', or null when unknown",
    "clip": "clip / video id, or null",
    "frame": "frame index inside the clip, or null",
    "image_type": "source-specific kind (v3: action / replay; soccersynth: clear / motion_blur), or null",
    "xg_role": "plan.csv role of the game (training / evaluation / gate), null if not a plan game",
    "gsr_split": "GSR split of the game or of the matched benchmark frame (train / valid / test / challenge), or null",
    "near_dup_of": "benchmark frame this picture duplicates ('gsr:<split>/<clip>/<frame>' or "
    "'football_yolo_valid:<file>'), or null",
    "near_dup_set": "gsr_valid / gsr_test / football_yolo_valid (benchmark), or soccernet_v3 (sources of unknown "
    "provenance are also matched against every v3 frame, so v3 game and xg_role carry over), or null",
    "near_dup_corr": "thumbnail correlation of the best near-dup hit, or null",
    "near_dup_hits": "number of confirmed benchmark near-dup pairs",
    "licence": "short licence tag of the source",
    "synthetic": "rendered rather than real footage",
    "annotated_classes": "comma-separated classes the source labels exhaustively; the others are not annotated and "
    "must be masked out of the loss (never treated as background)",
    "n_ball": "ball boxes in the label",
    "n_goalkeeper": "goalkeeper boxes",
    "n_player": "player boxes",
    "n_referee": "referee boxes",
}
DEFAULT_TAGS = {
    "game": None,
    "clip": None,
    "frame": None,
    "image_type": None,
    "xg_role": None,
    "gsr_split": None,
    "near_dup_of": None,
    "near_dup_set": None,
    "near_dup_corr": None,
    "near_dup_hits": 0,
}


def export_set(
    dataset: Dataset,
    out_dir: Path,
    manifest: dict,
    annotated: Sequence[str],
    tags: dict[str, dict] | None = None,
    report: dict | None = None,
    link: str = "symlink",
) -> dict:
    """Write the YOLO set, ``manifest.json``, ``label_sha256.txt`` and ``leak_report.json`` into ``out_dir``.

    ``manifest`` needs ``source`` (family id), ``licence_tag`` (short) and the long ``licence`` / ``access_route``
    texts. ``annotated`` lists the classes this source labels exhaustively; the others are recorded as not
    annotated. ``tags`` maps a source image path to its tag values (see :data:`SCHEMA`); missing tags are null.
    The manifest's ``per_image`` holds one :data:`SCHEMA` row per image.
    """
    unknown = set(annotated) - set(CLASS_NAMES)
    if unknown or dataset.class_names != CLASS_NAMES:
        raise ValueError(f"Unknown classes {unknown or dataset.class_names}")
    for key in ("source", "licence_tag", "licence", "access_route"):
        if key not in manifest:
            raise ValueError(f"manifest needs {key!r}")
    data_yaml = dataset.to_yolo(out_dir, link=link)
    labels = sorted((out_dir).glob("*/labels/*.txt"))
    lines = [f"{sha256_file(p)}  {p.relative_to(out_dir).as_posix()}" for p in labels]
    sha_file = out_dir / "label_sha256.txt"
    sha_file.write_text("".join(f"{line}\n" for line in lines))

    summary = dataset.summary()
    counts = {
        split: {k: int(v) for k, v in row.items()} for split, row in summary.iterrows()
    }
    rows = []
    for split in dataset.split_names:
        samples = dataset[split]
        names = unique_names([s.image_path for s in samples])
        for sample, name in zip(samples, names, strict=True):
            per_class = np.bincount(
                sample.annotations.class_ids, minlength=len(CLASS_NAMES)
            )
            extra = (tags or {}).get(str(sample.image_path), {})
            unknown_keys = set(extra) - set(DEFAULT_TAGS)
            if unknown_keys:  # set, image, counts, licence ... are the exporter's, never a converter's
                raise ValueError(
                    f"tags outside the schema's tag columns: {sorted(unknown_keys)}"
                )
            rows.append(
                {
                    "set": out_dir.name,
                    "source": manifest["source"],
                    "split": split,
                    "image": str(out_dir.resolve() / split / "images" / name),
                    "source_image": str(sample.image_path),
                    "width": sample.width,
                    "height": sample.height,
                    **DEFAULT_TAGS,
                    **extra,
                    "licence": manifest["licence_tag"],
                    "synthetic": bool(manifest.get("synthetic", False)),
                    "annotated_classes": ",".join(annotated),
                    **{f"n_{c}": int(per_class[i]) for i, c in enumerate(CLASS_NAMES)},
                }
            )
    tagged = [r for r in rows if r["near_dup_of"]]
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
        "images_by_xg_role": _count(rows, "xg_role"),
        "images_by_gsr_split": _count(rows, "gsr_split"),
        "images_near_dup": len(tagged),
        "label_files": len(labels),
        "label_sha256_file": sha_file.name,
        "label_sha256_of_list": sha256_file(sha_file),
        "per_image": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(full, indent=1, default=str))
    if report is not None:
        (out_dir / "leak_report.json").write_text(
            json.dumps(report, indent=1, default=str)
        )
    return full


def _count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key]) if row[key] is not None else "null"
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":  # build the benchmark fingerprint cache once
    for ref_name, ref in benchmark_fingerprints().items():
        print(ref_name, len(ref.paths))
