"""The frozen detection benchmark of detector-data (ticket 02): build it, score a detector on it.

    uv run python scripts/detector_bench.py build      # bench sets, manifest, research/detector_bench.json
    uv run python scripts/detector_bench.py evaluate --weights W --split valid --out DIR
    uv run python scripts/detector_bench.py placement --weights W --out DIR
    uv run python scripts/detector_bench.py epochs --weights-dir RUN/yolo11l_1280/weights --out DIR  # every epoch

Benchmark: SoccerNet GSR valid (dev) and test (final, read once per candidate), every `STRIDE`-th frame of every clip
(frames 1, 6, 11, ...: the frames GSR test is downloaded with and the frames of the pitch-keypoints homography
evaluation), read by `tv.data.Dataset.from_gsr` (classes ball / goalkeeper / player / referee; GSR's "other" dropped).
The secondary no-regression set is `football_yolo` valid (the Roboflow set the current detector was trained on).

`evaluate` scores every row of `ROWS` (inference settings) on every set of `SETS` (`--rows`, `--sets` select; GSR
test is scored alone): `gsr` (the frozen benchmark), `roboflow` (football_yolo valid, square) and `ekstraklasa`
(the Ekstraklasa dev clips of ticket 04, 16:9; its labels are checked against the dev record) with `tv.evaluation.evaluate_detector`: per-class AP50 and
AP50-95 (supervision's COCO-style mAP, predictions down to `EVAL_CONF`), and precision / recall at the pipeline's
operating point `OPERATING` (E3: conf 0.3, ball 0.15) with IoU >= `MATCH_IOU` matching. `--split test` writes its
file and prints only the path (GSR test is for ticket 05).

`placement` is the downstream ball measure of the pitch-keypoints report (`pk_ball_evidence.py` part 3, ported): on
every scored GSR valid frame whose truth ball (one ball annotation with a pitch position) lies on the pitch, the most
confident detected ball (>= 0.15, the 1080p frame resized to 720p, imgsz 1920, NMS IoU 0.5) is placed by the frozen
homography of that frame (`kp03_r2_refined`, the chosen method of pitch-keypoints 03, cached in the bench folder);
reported: the share of frames with a placed ball, its error and the share of placed balls > 5 m from the truth, with
2,000-draw clip-bootstrap intervals. The homographies are fixed so that only the detector changes.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import tactifoot_vision as tv
from tactifoot_vision.data import Dataset

CHECKOUT = Path("/home/kuba/projects/tactifoot_vision")
GSR = CHECKOUT / "data/SoccerNetGS"
ROBOFLOW = CHECKOUT / "data/datasets/football_yolo"
BENCH = Path("/mnt/storage/tactifoot_research/detector_data/bench")
# Ekstraklasa dev (ticket 04, run 2): whole held-out LiveTag clips of football_yolo, un-stretched to 16:9
# (scripts/detector_data/football_yolo.py, record research/ekstraklasa_dev_split.json). Not part of the frozen GSR
# benchmark; the baseline trained on these clips, so its scores here are in-sample.
EKSTRAKLASA_DEV = BENCH / "ekstraklasa_dev"
EKSTRAKLASA_JSON = (
    Path(__file__).resolve().parents[1] / "research/ekstraklasa_dev_split.json"
)
SETS = ("gsr", "roboflow", "ekstraklasa")  # scoring row families: <set>/<row>
FREEZE = Path(__file__).resolve().parents[1] / "research/detector_bench.json"
HOMOGRAPHY_SOURCE = (
    CHECKOUT
    / ".claude/worktrees/research-xg-detector/experiments/claude_fable/results/pk_candidates"
    / "y11m_960_e18_square/retune_valid/per_frame.parquet"
)
HOMOGRAPHY_METHOD = "kp03_r2_refined"
HOMOGRAPHY_FILE = "gsr_valid_homographies.parquet"

STRIDE = 5
SPLITS = ("valid", "test")
EVAL_CONF = 0.001
MATCH_IOU = 0.5
OPERATING = {"conf": 0.3, "class_conf": {"ball": 0.15}}
ROWS = {
    "640": {"imgsz": 640, "iou": 0.7, "input_size": None},
    "1280": {"imgsz": 1280, "iou": 0.7, "input_size": None},
    # E3's extractor: 720p video frames, letterboxed to 1920, NMS IoU 0.5
    "e3": {"imgsz": 1920, "iou": 0.5, "input_size": (1280, 720)},
}
PLACEMENT = {
    "input_size": (1280, 720),
    "imgsz": 1920,
    "iou": 0.5,
    "ball_conf": 0.15,
    "far_m": 5.0,
    "near_goal_m": 30.0,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 0,
}
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0  # GSR's pitch: centre-origin metres, 105 x 68 m

logger = logging.getLogger("detector_bench")


# ------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_digest(files: list[Path], root: Path) -> str:
    """sha256 over the sorted files' relative names and contents."""
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\n")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gsr_split(split: str) -> Dataset:
    return Dataset.from_gsr(GSR, splits=(split,), stride=STRIDE)


def counts(dataset: Dataset, split: str) -> dict:
    samples = dataset[split]
    boxes = np.bincount(
        np.concatenate([s.annotations.class_ids for s in samples]),
        minlength=len(dataset.class_names),
    )
    return {
        "frames": len(samples),
        "frames_with_ball": sum(
            int((s.annotations.class_ids == 0).any()) for s in samples
        ),
        "boxes": dict(zip(dataset.class_names, boxes.tolist(), strict=True)),
    }


# ------------------------------------------------------------------ build
def build(args: argparse.Namespace) -> None:
    out: Path = args.out
    manifest: dict = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "detector-data 02",
        "gsr_root": str(GSR),
        "stride": STRIDE,
        "frame_rule": f"1-based GSR frames with (frame - 1) % {STRIDE} == 0",
        "classes": ["ball", "goalkeeper", "player", "referee"],
        "dropped_categories": ["other"],
        "splits": {},
    }
    for split in SPLITS:
        dataset = gsr_split(split)
        # Readable, unique names (every GSR clip has a 000001.jpg) before the YOLO export.
        links = out / "images" / split
        links.mkdir(parents=True, exist_ok=True)
        renamed = []
        for sample in dataset[split]:
            link = (
                links / f"{sample.meta['sequence']}_{int(sample.meta['frame']):06d}.jpg"
            )
            if not link.is_symlink():
                link.symlink_to(sample.image_path)
            elif link.resolve() != sample.image_path.resolve():
                raise SystemExit(f"{link} points elsewhere; remove {out / 'images'}")
            renamed.append(
                tv.data.Sample(
                    link, sample.width, sample.height, sample.annotations, sample.meta
                )
            )
        data_yaml = dataset.with_split(split, renamed).to_yolo(out / "yolo" / split)
        label_files = list((data_yaml.parent / split / "labels").glob("*.txt"))
        sequences = sorted({s.meta["sequence"] for s in dataset[split]})
        clip_labels = {
            seq: sha256_file(GSR / split / seq / "Labels-GameState.json")
            for seq in sequences
        }
        manifest["splits"][split] = counts(dataset, split) | {
            "sequences": len(sequences),
            "games": sorted({s.meta["game"] for s in dataset[split]}, key=int),
            "frames_per_game": pd.Series([s.meta["game"] for s in dataset[split]])
            .value_counts()
            .sort_index()
            .to_dict(),
            "data_yaml": str(data_yaml),
            "yolo_labels_sha256": labels_digest(label_files, data_yaml.parent),
            "yolo_label_files": len(label_files),
            "gsr_labels_sha256": clip_labels,
        }

    roboflow = tv.load_dataset(ROBOFLOW)
    files = list((ROBOFLOW / "valid" / "labels").glob("*.txt"))
    manifest["roboflow_valid"] = counts(roboflow, "valid") | {
        "root": str(ROBOFLOW),
        "labels_sha256": labels_digest(files, ROBOFLOW / "valid" / "labels"),
    }

    table = pd.read_parquet(
        HOMOGRAPHY_SOURCE, columns=["clip", "frame", "method", "homography"]
    )
    table = table[(table.method == HOMOGRAPHY_METHOD) & table.homography.notna()]
    table = table.drop(columns="method").sort_values(
        ["clip", "frame"], ignore_index=True
    )
    table["homography"] = table.homography.map(lambda h: np.asarray(h, float).tolist())
    homography_path = out / HOMOGRAPHY_FILE
    table.to_parquet(homography_path, index=False)
    manifest["placement_homographies"] = {
        "file": str(homography_path),
        "sha256": sha256_file(homography_path),
        "frames": len(table),
        "source": str(HOMOGRAPHY_SOURCE),
        "source_sha256": sha256_file(HOMOGRAPHY_SOURCE),
        "method": HOMOGRAPHY_METHOD,
        "convention": "720p frame pixels -> corner-origin metres (105 x 68)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    freeze = {
        "title": "detector-data benchmark (ticket 02), frozen before any training",
        "manifest": manifest | {"manifest_sha256": sha256_file(out / "manifest.json")},
        "metrics": {
            "ap": "supervision.metrics.MeanAveragePrecision via tv.evaluation.evaluate_detector "
            "(COCO-style AP over IoU 0.50:0.95; images without objects do not count)",
            "eval_conf": EVAL_CONF,
            "operating_point": OPERATING,
            "match_iou": MATCH_IOU,
            "rows": ROWS,
            "placement": PLACEMENT
            | {
                "frames": "scored GSR valid frames whose single truth ball (bbox_pitch) lies on the pitch",
                "detector": "most confident ball, box centre through the frame's homography",
                "check": "baseline reproduces pk_ball_evidence (pitch-keypoints report): placed 0.5019, far 0.2508 on 6,404 frames",
            },
        },
        "rules": "GSR valid is the dev set; GSR test is read once per final candidate (ticket 05) plus the baseline row.",
    }
    FREEZE.parent.mkdir(exist_ok=True)
    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def check_manifest(dataset: Dataset, split: str) -> None:
    """The split read now has the frozen frame and box counts."""
    frozen = json.loads((BENCH / "manifest.json").read_text())["splits"][split]
    now = counts(dataset, split)
    for key in ("frames", "boxes"):
        if now[key] != frozen[key]:
            raise SystemExit(
                f"GSR {split} {key} {now[key]} differ from the manifest's {frozen[key]}"
            )


# ------------------------------------------------------------------ evaluate
def ekstraklasa_dev() -> Dataset:
    """The Ekstraklasa dev set, after checking its labels against the dev record."""
    frozen = json.loads(EKSTRAKLASA_JSON.read_text())["labels_sha256"]
    digest = hashlib.sha256()
    for path in sorted((EKSTRAKLASA_DEV / "valid/labels").glob("*.txt")):
        digest.update(path.name.encode() + b"\n" + path.read_bytes())
    if digest.hexdigest() != frozen:
        raise SystemExit(f"{EKSTRAKLASA_DEV} labels differ from {EKSTRAKLASA_JSON}")
    return tv.load_dataset(EKSTRAKLASA_DEV)


def evaluation_sets(split: str, names: list[str] | None) -> dict[str, tuple]:
    """``{set: (dataset, split)}`` to score; GSR test is scored alone."""
    names = list(SETS) if names is None else names
    if split == "test":
        names = ["gsr"]
    sets: dict[str, tuple] = {}
    if "gsr" in names:
        sets["gsr"] = (gsr_split(split), split)
        check_manifest(sets["gsr"][0], split)
    if "roboflow" in names:
        sets["roboflow"] = (tv.load_dataset(ROBOFLOW), "valid")
    if "ekstraklasa" in names:
        sets["ekstraklasa"] = (ekstraklasa_dev(), "valid")
    return sets


def score(
    weights: Path, sets: dict[str, tuple], rows: list[str] | None, device: str | None
) -> dict:
    """``{"<set>/<row>": metrics}`` of ``weights`` for every selected row of ``ROWS`` on every set."""
    scored = {}
    for row, settings in ROWS.items():
        if rows and row not in rows:
            continue
        model = tv.load_model(
            "yolo",
            weights,
            conf=OPERATING["conf"],
            iou=settings["iou"],
            imgsz=settings["imgsz"],
            device=device,
        )
        for name, (dataset, split) in sets.items():
            clock = time.perf_counter()
            metrics = tv.evaluation.evaluate_detector(
                model,
                dataset,
                split=split,
                conf=EVAL_CONF,
                class_conf=OPERATING["class_conf"],
                match_iou=MATCH_IOU,
                input_size=settings["input_size"],
            )
            scored[f"{name}/{row}"] = {
                "settings": settings,
                "map50_95": metrics.map50_95,
                "map50": metrics.map50,
                "images": metrics.num_images,
                "seconds": round(time.perf_counter() - clock, 1),
                "per_class": metrics.per_class.reset_index().to_dict(orient="records"),
            }
            logger.info("%s/%s done", name, row)
    return scored


def evaluate(args: argparse.Namespace) -> None:
    sets = evaluation_sets(args.split, args.sets)
    result = {
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "split": args.split,
        "sets": list(sets),
        "eval_conf": EVAL_CONF,
        "operating_point": OPERATING,
        "rows": score(args.weights, sets, args.rows, args.device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"detection_{args.split}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    if args.split == "test":
        print(path)
        return
    for key, row in result["rows"].items():
        table = pd.DataFrame(row["per_class"]).set_index("class_name")
        print(f"\n{key}: mAP50-95 {row['map50_95']:.4f} mAP50 {row['map50']:.4f}")
        print(table.round(4).to_string())
    print(path)


# ------------------------------------------------------------------ epochs
def epoch_checkpoints(weights_dir: Path) -> list[tuple[int, Path]]:
    """``(epoch, path)`` of every Ultralytics ``epoch<N>.pt`` (N counts from 0), in epoch order."""
    found = []
    for path in weights_dir.glob("epoch*.pt"):
        number = path.stem.removeprefix("epoch")
        if number.isdigit():
            found.append((int(number), path))
    return sorted(found)


def gsr_every(dataset: Dataset, every: int) -> Dataset:
    """Every ``every``-th GSR valid frame in (clip, frame) order: the bench file-name order, so ``every=4`` is the
    training runs' validation subset."""
    samples = sorted(
        dataset["valid"], key=lambda s: (s.meta["sequence"], int(s.meta["frame"]))
    )
    return dataset.with_split("valid", samples[::every])


def epoch_table(results: dict[int, dict]) -> pd.DataFrame:
    """One row per epoch and ``<set>/<row>``: mAP50, mAP50-95 and per-class AP50."""
    records = []
    for epoch, rows in sorted(results.items()):
        for key, row in rows.items():
            record = {
                "epoch": epoch,
                "row": key,
                "map50": row["map50"],
                "map50_95": row["map50_95"],
            }
            record |= {f"ap50_{c['class_name']}": c["map50"] for c in row["per_class"]}
            records.append(record)
    return pd.DataFrame(records)


def epochs(args: argparse.Namespace) -> None:
    """Score every ``epoch<N>.pt`` of a run folder (each written once to ``<out>/epoch<N>.json``, reused when its
    weights hash matches) and write ``<out>/epochs.csv``."""
    sets = evaluation_sets("valid", args.sets)
    if "gsr" in sets and args.gsr_every > 1:
        sets["gsr"] = (gsr_every(sets["gsr"][0], args.gsr_every), "valid")
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for epoch, path in epoch_checkpoints(args.weights_dir):
        out = args.out / f"epoch{epoch}.json"
        digest = sha256_file(path)
        if out.is_file():
            done = json.loads(out.read_text())
            if done["weights_sha256"] == digest and set(done["rows"]) >= {
                f"{s}/{r}" for s in sets for r in (args.rows or ROWS)
            }:
                results[epoch] = done["rows"]
                continue
        rows = score(path, sets, args.rows, args.device)
        record = {
            "weights": str(path),
            "weights_sha256": digest,
            "gsr_every": args.gsr_every,
            "rows": rows,
        }
        out.write_text(json.dumps(record, indent=2) + "\n")
        results[epoch] = rows
    table = epoch_table(results)
    table.to_csv(args.out / "epochs.csv", index=False)
    print(table.round(4).to_string(index=False))


# ------------------------------------------------------------------ placement
def ball_truth(labels: dict) -> dict[int, np.ndarray]:
    """1-based frame -> truth ball (corner metres); frames with more than one ball annotation are left out."""
    ball_ids = {c["id"] for c in labels["categories"] if c["name"] == "ball"}
    frames = {
        image["image_id"]: int(Path(image["file_name"]).stem)
        for image in labels["images"]
    }
    found: dict[int, list[np.ndarray]] = {}
    for item in labels["annotations"]:
        if (
            item.get("category_id") in ball_ids
            and item.get("bbox_pitch")
            and item["image_id"] in frames
        ):
            p = item["bbox_pitch"]
            xy = np.array(
                [p["x_bottom_middle"] + HALF_LENGTH, p["y_bottom_middle"] + HALF_WIDTH]
            )
            found.setdefault(frames[item["image_id"]], []).append(xy)
    return {frame: xy[0] for frame, xy in found.items() if len(xy) == 1}


def placement_rows(model, homographies: dict) -> pd.DataFrame:
    rows = []
    for labels_path in sorted((GSR / "valid").glob("*/Labels-GameState.json")):
        labels = json.loads(labels_path.read_text())
        clip, game = labels_path.parent.name, str(labels["info"].get("game_id", ""))
        truth = ball_truth(labels)
        for image in labels["images"]:
            frame = int(Path(image["file_name"]).stem)
            if (frame - 1) % STRIDE or frame not in truth:
                continue
            true_xy = truth[frame]
            if (
                abs(true_xy[0] - HALF_LENGTH) > HALF_LENGTH
                or abs(true_xy[1] - HALF_WIDTH) > HALF_WIDTH
            ):
                continue
            matrix = homographies.get((clip, frame))
            xy = np.full(2, np.nan)
            if matrix is not None:
                frame_bgr = cv2.imread(
                    str(labels_path.parent / "img1" / image["file_name"])
                )
                size = PLACEMENT["input_size"]
                if frame_bgr.shape[1::-1] != size:
                    frame_bgr = cv2.resize(
                        frame_bgr, size, interpolation=cv2.INTER_AREA
                    )
                detections = model.predict(frame_bgr)
                balls = detections[detections.data["class_name"] == "ball"]
                if len(balls):
                    best = int(np.argmax(balls.confidence))
                    centre = (balls.xyxy[best, :2] + balls.xyxy[best, 2:]) / 2
                    xy = tv.pitch.frame_to_pitch(centre[None], matrix)[0].astype(float)
            near = (
                min(true_xy[0], 2 * HALF_LENGTH - true_xy[0])
                <= PLACEMENT["near_goal_m"]
            )
            rows.append(
                {
                    "clip": clip,
                    "game": game,
                    "frame": frame,
                    "truth_x": true_xy[0],
                    "truth_y": true_xy[1],
                    "near_goal": near,
                    "homography": matrix is not None,
                    "x": xy[0],
                    "y": xy[1],
                    "error_m": float(np.hypot(*(xy - true_xy))),
                }
            )
    return pd.DataFrame(rows)


def placement_statistics(table: pd.DataFrame) -> dict:
    error = table.error_m.to_numpy(float)
    placed = error[np.isfinite(error)]
    stats = {
        "frames": float(len(table)),
        "placed": float(np.isfinite(error).mean()) if len(table) else np.nan,
    }
    stats["median_m"] = float(np.median(placed)) if len(placed) else np.nan
    for t in (1.0, 2.0, PLACEMENT["far_m"]):
        stats[f"lt{t:g}"] = float((placed < t).mean()) if len(placed) else np.nan
    stats["far"] = (
        float((placed > PLACEMENT["far_m"]).mean()) if len(placed) else np.nan
    )
    return stats


def bootstrap(table: pd.DataFrame) -> dict:
    """Point value and 95% clip-bootstrap percentile interval of every statistic."""
    rng = np.random.default_rng(PLACEMENT["bootstrap_seed"])
    codes, _ = pd.factorize(table["clip"], sort=True)
    positions = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    draws = pd.DataFrame(
        [
            placement_statistics(
                table.iloc[
                    np.concatenate(
                        [
                            positions[i]
                            for i in rng.integers(0, len(positions), len(positions))
                        ]
                    )
                ]
            )
            for _ in range(PLACEMENT["bootstrap_draws"])
        ]
    )
    point = placement_statistics(table)
    return {
        k: {"value": v}
        | (
            {}
            if k == "frames"
            else dict(
                zip(
                    ("ci_low", "ci_high"),
                    np.nanquantile(draws[k], [0.025, 0.975]).tolist(),
                    strict=True,
                )
            )
        )
        for k, v in point.items()
    }


def placement(args: argparse.Namespace) -> None:
    frozen = json.loads((BENCH / "manifest.json").read_text())["placement_homographies"]
    path = BENCH / HOMOGRAPHY_FILE
    if sha256_file(path) != frozen["sha256"]:
        raise SystemExit(f"{path} differs from the frozen homographies")
    table = pd.read_parquet(path)
    homographies = {
        (c, int(f)): np.asarray(h, float).reshape(3, 3)
        for c, f, h in zip(
            table["clip"], table["frame"], table["homography"], strict=True
        )
    }
    model = tv.load_model(
        "yolo",
        args.weights,
        conf=PLACEMENT["ball_conf"],
        iou=PLACEMENT["iou"],
        imgsz=PLACEMENT["imgsz"],
        device=args.device,
    )
    clock = time.perf_counter()
    rows = placement_rows(model, homographies)
    summary = {
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "settings": PLACEMENT,
        "seconds": round(time.perf_counter() - clock, 1),
        "all": bootstrap(rows.reset_index(drop=True)),
        "near_goal": bootstrap(rows[rows.near_goal].reset_index(drop=True)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out / "placement_frames.parquet", index=False)
    (args.out / "placement.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--out", type=Path, default=BENCH)
    e = commands.add_parser("evaluate")
    p = commands.add_parser("placement")
    for command in (e, p):
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", default=None)
    e.add_argument("--split", choices=SPLITS, default="valid")
    e.add_argument("--rows", nargs="*", choices=list(ROWS), default=None)
    e.add_argument("--sets", nargs="*", choices=list(SETS), default=None)
    ep = commands.add_parser("epochs", help="score every epoch<N>.pt of a run folder")
    ep.add_argument("--weights-dir", type=Path, required=True)
    ep.add_argument("--out", type=Path, required=True)
    ep.add_argument("--device", default=None)
    ep.add_argument("--rows", nargs="*", choices=list(ROWS), default=["1280"])
    ep.add_argument(
        "--sets", nargs="*", choices=list(SETS), default=["gsr", "ekstraklasa"]
    )
    ep.add_argument(
        "--gsr-every",
        type=int,
        default=4,
        help="GSR valid frames: every Nth bench frame (4 = the training validation subset, 1 = full)",
    )
    args = parser.parse_args()
    tv.setup_logging("INFO")
    {"build": build, "evaluate": evaluate, "placement": placement, "epochs": epochs}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
