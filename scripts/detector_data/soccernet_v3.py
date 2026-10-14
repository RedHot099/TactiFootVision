"""SoccerNet-v3 (+ SoccerNet-v3D refined ball boxes) -> 4-class YOLO set with manifest and leak report.

    uv run python -m scripts.detector_data.soccernet_v3 [--limit-games N]

Inputs (downloaded by ``download_soccernet_v3.py``, untrusted):
``raw/soccernet_v3/<game>/Labels-v3.json`` and ``Frames-v3.zip`` (games outside the xG held-out list only), and
``raw/soccernet_v3d/dl/SNv3D.csv`` (v3D release v1.0.0).

Per image: the 4-class boxes of ``tactifoot_vision.data.soccernet_v3``; staff and walls of players are greyed out;
the image is re-encoded as JPEG (quality 95) under ``work/soccernet_v3/images``. Where v3D refined the ball, the
ball box becomes a square of the v3D diameter (``optimized_d``) around the original centre (see ``refine_ball``).
Split = the official SoccerNet split (``imageMetadata.set``).
"""

import argparse
import ast
import csv
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.detector_data import common
from tactifoot_vision.data import Annotations, Dataset, Sample, Task
from tactifoot_vision.data.soccernet_v3 import V3Image, read_labels

RAW = common.ROOT / "raw/soccernet_v3"
V3D_CSV = common.ROOT / "raw/soccernet_v3d/dl/SNv3D.csv"
WORK = common.ROOT / "work/soccernet_v3/images"
OUT = common.ROOT / "sets/soccernet_v3"
JPEG_QUALITY = 95
# v3D refinement is applied only where the triangulation is consistent and the size change is plausible.
MAX_REP_ERROR_PX = 10.0
SIZE_RATIO = (0.5, 2.0)

MANIFEST = {
    "source": "SoccerNet-v3 (Cioppa et al., CVPRW 2022) with SoccerNet-v3D refined ball boxes "
    "(Gutierrez-Perez et al., CVPRW 2025)",
    "licence": "SoccerNet-v3 repo MIT, paper says 'open licence'; frames are broadcast footage, used privately "
    "under the SoccerNet NDA terms (research weights only). SoccerNet-v3D repo GPL-2.0 (annotations).",
    "access_route": "SoccerNet pip 0.1.62 SoccerNetDownloader.downloadGame(files=['Labels-v3.json'|'Frames-v3.zip']) "
    "with the downloader's built-in v3 share password (no NDA password); v3D: GitHub release v1.0.0 SNv3D.csv",
    "mapping": {
        "Player team left/right": "player",
        "Goalkeeper team left/right": "goalkeeper",
        "Main referee, Side referee": "referee",
        "Ball": "ball",
        "Staff members, Wall of players": "greyed out (ignore region)",
        "Yellow card, Red card, Referee flag": "dropped",
    },
    "caveats": [
        "the ball is boxed only while in play: a dead ball can be visible and unlabelled",
        "replays (image_type=replay) are close-ups and other angles; weight or cap them in training",
        "images re-encoded from PNG to JPEG q95",
    ],
    "frame_subsampling": "none needed: one action frame per annotated action plus its replays (no video "
    "neighbours); within-set dHash duplicates are still reported",
}


def refine_index(csv_path: Path = V3D_CSV) -> dict[tuple[str, str], dict]:
    """v3D rows keyed by ``(game path, image name)``."""
    csv.field_size_limit(1 << 30)
    index = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            game = f"{row['league']}/{row['season']}/{row['match']}"
            action = row["main_action"]
            name = (
                f"{action}.png"
                if row["action"]
                else f"{action}_{int(float(row['replay']))}.png"
            )
            index[(game, name)] = {
                "bbox": ast.literal_eval(row["ball_bbox"])
                if row["ball_bbox"]
                else None,
                "d": float(row["optimized_d"]) if row["optimized_d"] else None,
                "rep_error": float(row["rep_error"]) if row["rep_error"] else None,
                "set": row["set"],
            }
    return index


def refine_ball(image: V3Image, row: dict | None) -> str:
    """Replace the v3 ball box by the v3D one in place; returns what happened (for the manifest counts).

    v3D gives the original box (``x, y, w, h``) and an optimised ball diameter in pixels; the refined box is a square
    of that diameter on the original centre. Applied when exactly one v3 ball box matches the v3D box, the
    triangulation reprojection error is at most ``MAX_REP_ERROR_PX`` and the size ratio is within ``SIZE_RATIO``.
    """
    if row is None or row["bbox"] is None or row["d"] is None:
        return "no_v3d"
    x, y, w, h = row["bbox"]
    target = np.array([x, y, x + w, y + h], np.float32)
    balls = [i for i, c in enumerate(image.classes) if c == "ball"]
    match = [i for i in balls if np.allclose(image.boxes[i], target, atol=0.6)]
    if len(match) != 1:
        return "unmatched"
    if row["rep_error"] is None or row["rep_error"] > MAX_REP_ERROR_PX:
        return "rep_error"
    ratio = row["d"] / max((w + h) / 2, 1e-6)
    if not SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]:
        return "size_ratio"
    cx, cy, r = x + w / 2, y + h / 2, row["d"] / 2
    image.boxes[match[0]] = [cx - r, cy - r, cx + r, cy + r]
    return "refined"


def image_target(image: V3Image) -> Path:
    game = image.game.replace("/", "__").replace(" ", "_")
    return (
        WORK
        / image.metadata.get("set", "train")
        / f"{game}__{Path(image.name).stem}.jpg"
    )


def convert_game(labels_path: str, refine: dict) -> list[dict]:
    """Write the game's images (ignore regions greyed) and return one record per image."""
    images = read_labels(labels_path)
    zip_path = Path(labels_path).with_name("Frames-v3.zip")
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(n).name: n for n in archive.namelist() if not n.endswith("/")}
        for image in images:
            status = refine_ball(image, refine.get((image.game, image.name)))
            member = members.get(image.name)
            if member is None:
                records.append({"key": image.key, "missing": True})
                continue
            target = image_target(image)
            if not target.is_file():
                data = np.frombuffer(archive.read(member), np.uint8)
                pixels = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if pixels is None:
                    records.append({"key": image.key, "unreadable": True})
                    continue
                if pixels.shape[:2] != (image.height, image.width):
                    records.append(
                        {"key": image.key, "size_mismatch": list(pixels.shape[:2])}
                    )
                    continue
                pixels = common.grey_out(pixels, image.ignore_boxes)
                target.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(
                    ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                target.write_bytes(encoded.tobytes())
            records.append(
                {
                    "key": image.key,
                    "path": str(target),
                    "split": image.metadata.get("set", "train"),
                    "game": image.game,
                    "v3_name": image.name,
                    "image_type": image.image_type,
                    "half": image.metadata.get("half"),
                    "position_ms": image.metadata.get("position"),
                    "action_label": image.metadata.get("label"),
                    "width": image.width,
                    "height": image.height,
                    "boxes": image.boxes.tolist(),
                    "classes": image.classes,
                    "ignore_regions": len(image.ignore_boxes),
                    "v3d_ball": status,
                }
            )
    return records


def to_dataset(records: list[dict]) -> Dataset:
    splits: dict[str, list[Sample]] = {}
    for r in records:
        ids = [common.CLASS_NAMES.index(c) for c in r["classes"]]
        sample = Sample(
            image_path=Path(r["path"]),
            width=r["width"],
            height=r["height"],
            annotations=Annotations(
                boxes=np.asarray(r["boxes"]), class_ids=np.asarray(ids)
            ),
        )
        splits.setdefault(r["split"], []).append(sample)
    return Dataset(Task.DETECT, list(common.CLASS_NAMES), splits, name="soccernet_v3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-games", type=int, default=None)
    args = parser.parse_args()

    held_out = common.held_out_games()
    all_games = sorted(p.parent for p in RAW.glob("*/*/*/Labels-v3.json"))
    games = [g for g in all_games if g.relative_to(RAW).as_posix() not in held_out]
    # An interrupted or running download has no zip end record, so is_zipfile skips it.
    complete = [g for g in games if zipfile.is_zipfile(g / "Frames-v3.zip")]
    missing_frames = sorted(
        g.relative_to(RAW).as_posix() for g in games if g not in complete
    )
    with_frames = complete[: args.limit_games]
    refine = refine_index()
    records: list[dict] = []
    with ProcessPoolExecutor(8) as pool:
        for game_records in pool.map(
            convert_game,
            [str(g / "Labels-v3.json") for g in with_frames],
            [refine] * len(with_frames),
        ):
            records += game_records
    problems = [r for r in records if "path" not in r]
    records = [r for r in records if "path" in r]
    # Leak rule 1-2: game identity, checked on the label's localpath (never on folder names alone).
    game_report = common.check_games([r["game"] for r in records], held_out)
    assert all(
        (RAW / r["game"]).resolve() in {g.resolve() for g in with_frames}
        for r in records
    ), "an image's localpath differs from its game folder"

    dataset = to_dataset(records)
    kept, leak_report = common.leak_check(
        dataset,
        common.benchmark_fingerprints(),
        extra={
            "game_rules": game_report
            | {
                "games_with_labels": len(all_games),
                "held_out_games_skipped": len(all_games) - len(games),
                "games_converted": len(with_frames),
                "kept_games_without_frames": missing_frames,
            }
        },
    )
    within = common.dedup_within(common.Fingerprints.of([s.image_path for s in kept]))
    leak_report["within_set_duplicates_reported"] = len(within)

    statuses: dict[str, int] = {}
    types: dict[str, int] = {}
    for r in records:
        statuses[r["v3d_ball"]] = statuses.get(r["v3d_ball"], 0) + 1
        types[r["image_type"]] = types.get(r["image_type"], 0) + 1
    provenance = {
        r["path"]: {
            k: r[k]
            for k in (
                "game",
                "v3_name",
                "image_type",
                "half",
                "position_ms",
                "action_label",
                "v3d_ball",
            )
        }
        for r in records
    }
    manifest = MANIFEST | {
        "v3d_ball_refinement": statuses,
        "v3d_rule": f"square of v3D optimized_d around the original centre when rep_error <= {MAX_REP_ERROR_PX} px "
        f"and optimized_d / mean(w, h) in {list(SIZE_RATIO)}",
        "image_types": types,
        "conversion_problems": problems,
        "games": sorted({r["game"] for r in records}),
    }
    common.export_set(
        kept,
        OUT,
        manifest,
        annotated=common.CLASS_NAMES,
        provenance=provenance,
        leak_report=leak_report,
        link="hardlink",
    )
    print(
        json.dumps(
            {"images": len(kept), "dropped": len(leak_report["dropped"]), **statuses}
        )
    )


if __name__ == "__main__":
    main()
