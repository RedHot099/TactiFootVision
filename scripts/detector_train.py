"""Training run 1 of detector-data (ticket 04): the view, the throughput probe, the run.

    uv run python scripts/detector_train.py view                 # runs/run1: lists, data.yaml, class-mask table
    uv run python scripts/detector_train.py probe --model yolo11l --batch 8
    uv run python scripts/detector_train.py train --smoke        # ~200 iterations on a random slice
    uv run python scripts/detector_train.py train                # the full run (see runs/run1/RUN.md)
    uv run python scripts/detector_train.py train --resume       # continue from weights/last.pt

View (ticket 04, "Run 1 decisions"): every pool image except the GSR valid/test games (2, 3, 5, 7, 8, 11: the
frozen benchmark) and except any image tagged as a near-duplicate of a GSR valid/test frame. xG evaluation/gate games
and Soccana are in. Classes outside a source's `annotated_classes` get no classification loss (the package's
`annotated_classes` option of the `yolo` model); the "player means any person" sources (ISSIA, TeamTrack, SportsMOT,
SoccerPlayerDetection) annotate neither goalkeeper nor referee, and the script checks that their tables say so.

Validation during training: every 4th frame of the frozen GSR valid benchmark (bench/, ticket 02), so per-epoch
validation stays cheap; the full benchmark is scored after the run with scripts/detector_bench.py.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import tactifoot_vision as tv

CHECKOUT = Path("/home/kuba/projects/tactifoot_vision")
DATA = Path("/mnt/storage/tactifoot_research/detector_data")
POOL = DATA / "pool_index.parquet"
BENCH_VALID = DATA / "bench/yolo/valid/valid/images"
RUN = DATA / "runs/run1"
VIEW_JSON = Path(__file__).resolve().parents[1] / "research/detector_run1_view.json"

CLASSES = ["ball", "goalkeeper", "player", "referee"]
BENCH_GAMES = {
    "gsr:2",
    "gsr:3",
    "gsr:5",
    "gsr:7",
    "gsr:8",
    "gsr:11",
}  # GSR valid + test
BENCH_DUP_SETS = {"gsr_valid", "gsr_test"}
ANY_PERSON_SOURCES = {"issia", "teamtrack", "sportsmot", "soccer_player_detection"}
VAL_EVERY = 4  # every 4th benchmark frame (bench frames are every 5th GSR frame)
SMOKE = {
    "train_images": 800,
    "epochs": 2,
    "val_every": 40,
    "seed": 0,
}  # 2 x 200 iterations at batch 4

# Run settings, chosen by the probe (see the ticket's comments).
WEIGHTS = {
    "yolo11m": CHECKOUT / "models/yolo11m.pt",
    "yolo11l": CHECKOUT / "models/yolo11l.pt",
}
RUN_SETTINGS = {
    "model": "yolo11l",
    "imgsz": 1280,
    "batch_size": 4,
    # Explicit SGD (what "auto" picks for this many iterations): a resumed AdamW run
    # gave NaN losses in Ultralytics 8.3.205 (stock trainer too); SGD resumes cleanly.
    "optimizer": "SGD",
    "lr": 0.01,
    "epochs": 22,
    "patience": 7,
    "workers": 12,
    "seed": 0,
    "cos_lr": True,
    "close_mosaic": 2,
    "save_period": 5,
    "scale": 0.3,
}

logger = logging.getLogger("detector_train")


# ------------------------------------------------------------------ view
def build_view(pool: pd.DataFrame) -> pd.DataFrame:
    """Every pool image outside the GSR benchmark games and their near-duplicates."""
    excluded = (
        pool.game.isin(BENCH_GAMES).fillna(False)
        | pool.near_dup_set.isin(BENCH_DUP_SETS).fillna(False)
        | pool.gsr_split.isin(["valid", "test"]).fillna(False)
    )
    view = pool[~excluded].copy()
    assert not view.gsr_split.isin(["valid", "test"]).any()
    for source in ANY_PERSON_SOURCES:
        annotated = view.annotated_classes[view.source == source]
        assert len(annotated), f"no {source} images in the view"
        assert annotated.map(
            lambda s: not {"goalkeeper", "referee"} & set(s.split(","))
        ).all(), (
            f"{source}: player means any person, so goalkeeper and referee must not be annotated"
        )
    return view


def counts(view: pd.DataFrame) -> dict:
    """Images and boxes per source, plus the share of images and boxes each class mask removes."""
    per_source = {}
    for source, rows in view.groupby("source"):
        per_source[source] = {
            "images": len(rows),
            "annotated_classes": sorted(
                set(",".join(rows.annotated_classes).split(","))
            ),
            "boxes": {c: int(rows[f"n_{c}"].sum()) for c in CLASSES},
        }
    masked = {
        c: round(
            float(
                (
                    ~view.annotated_classes.str.split(",").map(lambda a, c=c: c in a)
                ).mean()
            ),
            4,
        )
        for c in CLASSES
    }
    any_masked = view.annotated_classes.map(lambda a: set(a.split(",")) != set(CLASSES))
    return {
        "images": len(view),
        "boxes": {c: int(view[f"n_{c}"].sum()) for c in CLASSES},
        "per_source": per_source,
        "masked_share_by_class": masked,
        "images_with_any_masked_class": round(float(any_masked.mean()), 4),
        "gsr_train_images": int((view.source == "soccernet_gsr").sum()),
    }


def check_label_paths(images: list[str]) -> dict:
    """Ultralytics' images/ -> labels/ rule on the listed (symlink) paths: every label file exists."""
    from ultralytics.data.utils import img2label_paths

    labels = img2label_paths(images)
    missing = [lab for lab in labels if not os.path.isfile(lab)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} label files missing, e.g. {missing[:3]}"
        )
    return {"images": len(images), "labels_found": len(labels)}


def write_yaml(path: Path, train: str, val: str) -> None:
    config = {
        "path": str(path.parent),
        "train": train,
        "val": val,
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def view(args: argparse.Namespace) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(POOL)
    train = build_view(pool).sort_values("image")
    val = sorted(str(p) for p in BENCH_VALID.iterdir() if p.suffix == ".jpg")[
        ::VAL_EVERY
    ]
    train_images = train.image.tolist()
    label_check = {
        "train": check_label_paths(train_images),
        "val": check_label_paths(val),
    }

    (RUN / "train.txt").write_text("\n".join(train_images) + "\n")
    (RUN / "val.txt").write_text("\n".join(val) + "\n")
    write_yaml(RUN / "data.yaml", "train.txt", "val.txt")
    table = train[["image", "source", "annotated_classes"]].reset_index(drop=True)
    table.to_parquet(RUN / "annotated_classes.parquet", index=False)

    # Smoke slice: a random sample of the view, a sparser validation subset.
    smoke = RUN / "smoke_data"
    smoke.mkdir(exist_ok=True)
    rng = np.random.default_rng(SMOKE["seed"])
    pick = sorted(rng.choice(train_images, SMOKE["train_images"], replace=False))
    (smoke / "train.txt").write_text("\n".join(pick) + "\n")
    (smoke / "val.txt").write_text(
        "\n".join(val[:: SMOKE["val_every"] // VAL_EVERY]) + "\n"
    )
    write_yaml(smoke / "data.yaml", "train.txt", "val.txt")

    record = {
        "title": "detector-data run 1 training view (ticket 04)",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool_index": str(POOL),
        "pool_images": len(pool),
        "rule": {
            "excluded_games": sorted(BENCH_GAMES),
            "excluded_near_dup_sets": sorted(BENCH_DUP_SETS),
            "excluded_gsr_splits": ["valid", "test"],
            "included": "everything else: xG evaluation/gate games, Soccana, every source split",
            "class_mask": "per image, the source's annotated_classes; classes outside it get no cls loss",
            "any_person_sources": sorted(ANY_PERSON_SOURCES),
        },
        "train": counts(train) | {"list_sha256": sha256_lines(train_images)},
        "val": {
            "source": str(BENCH_VALID),
            "rule": f"every {VAL_EVERY}th frame of the sorted GSR valid benchmark (frames every 5th GSR frame)",
            "images": len(val),
            "list_sha256": sha256_lines(val),
            "files": [Path(v).name for v in val],
        },
        "label_paths": label_check,
        "files": {
            "data_yaml": str(RUN / "data.yaml"),
            "annotated_classes": str(RUN / "annotated_classes.parquet"),
            "smoke_data_yaml": str(smoke / "data.yaml"),
        },
    }
    VIEW_JSON.write_text(json.dumps(record, indent=2) + "\n")
    summary = {k: v for k, v in record["train"].items() if k != "per_source"}
    logger.info("view: %s", json.dumps(summary))
    for source, row in record["train"]["per_source"].items():
        logger.info(
            "  %-24s %6d images  %s  %s",
            source,
            row["images"],
            row["annotated_classes"],
            row["boxes"],
        )
    logger.info("val: %d images; wrote %s and %s", len(val), RUN, VIEW_JSON)


# ------------------------------------------------------------------ probe
def probe(args: argparse.Namespace) -> None:
    """GPU step throughput and peak memory on synthetic batches; dataloader throughput on the smoke slice."""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader

    from tactifoot_vision.models._yolo_class_mask import (
        ClassMaskedDataset,
        ClassMaskedLoss,
        read_class_masks,
    )

    result: dict = {"model": args.model, "batch": args.batch, "imgsz": args.imgsz}
    if not args.loader_only:
        model = YOLO(str(_weights(args.model))).model.float().cuda().train()
        for p in model.parameters():
            p.requires_grad_(True)
        model.args = get_cfg()
        criterion = ClassMaskedLoss(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda")
        n_boxes = 20 * args.batch
        batch = {
            "img": torch.rand(args.batch, 3, args.imgsz, args.imgsz, device="cuda"),
            "batch_idx": torch.arange(args.batch, device="cuda")
            .repeat_interleave(20)
            .float(),
            "cls": torch.randint(0, 4, (n_boxes, 1), device="cuda").float(),
            "bboxes": torch.rand(n_boxes, 4, device="cuda") * 0.05
            + torch.tensor([0.5, 0.5, 0.0, 0.0], device="cuda"),
            "class_mask": torch.ones(
                args.batch, model.model[-1].nc, dtype=torch.bool, device="cuda"
            ),
        }
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(args.steps + 5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda"):
                loss, _ = criterion(model(batch["img"]), batch)
            optimizer.zero_grad()
            scaler.scale(loss.sum()).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if i >= 5:
                times.append(time.perf_counter() - start)
        step = float(np.median(times))
        result |= {
            "gpu_step_s": round(step, 4),
            "gpu_img_per_s": round(args.batch / step, 1),
            "peak_mem_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        }
        del model, optimizer, batch
        torch.cuda.empty_cache()
    if not args.gpu_only:
        smoke = RUN / "smoke_data"
        hyp = get_cfg(overrides={"imgsz": args.imgsz, "scale": RUN_SETTINGS["scale"]})
        data = ClassMaskedDataset(
            img_path=str(smoke / "train.txt"),
            imgsz=args.imgsz,
            batch_size=args.batch,
            augment=True,
            hyp=hyp,
            data={"names": dict(enumerate(CLASSES)), "nc": 4, "channels": 3},
            class_masks=read_class_masks(RUN / "annotated_classes.parquet", CLASSES),
        )
        loader = build_dataloader(data, args.batch, args.workers, shuffle=True)
        it = iter(loader)
        for _ in range(5):
            next(it)
        start = time.perf_counter()
        for _ in range(args.steps):
            next(it)
        elapsed = time.perf_counter() - start
        result |= {
            "workers": args.workers,
            "loader_img_per_s": round(args.steps * args.batch / elapsed, 1),
        }
    logger.info("probe: %s", json.dumps(result))
    with (RUN / "probe.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")


def _weights(model: str) -> Path:
    path = WEIGHTS[model]
    if not path.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(path)
    return path


# ------------------------------------------------------------------ train
def train(args: argparse.Namespace) -> None:
    settings = dict(RUN_SETTINGS)
    model = settings.pop("model")
    name = f"{model}_{settings['imgsz']}"
    data_yaml = RUN / "data.yaml"
    weights: Path = _weights(model)
    if args.smoke:
        data_yaml = RUN / "smoke_data/data.yaml"
        name = "smoke"
        settings |= {
            "epochs": SMOKE["epochs"],
            "patience": None,
            "close_mosaic": 0,
            "val": False,
        }
    if args.resume:
        weights = RUN / name / "weights/last.pt"
        settings["resume"] = True
    started = time.time()
    dataset = tv.load_dataset(data_yaml)
    logger.info("loaded %r in %.0f s", dataset, time.time() - started)
    result = tv.train(
        "yolo",
        dataset,
        weights=weights,
        output_dir=RUN,
        name=name,
        exist_ok=True,
        annotated_classes=RUN / "annotated_classes.parquet",
        **settings,
    )
    logger.info("done: %s %s", result.weights, json.dumps(result.metrics))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    commands.add_parser("view").set_defaults(handler=view)
    p = commands.add_parser("probe")
    p.add_argument("--model", choices=sorted(WEIGHTS), required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--gpu-only", action="store_true")
    p.add_argument("--loader-only", action="store_true")
    p.set_defaults(handler=probe)
    t = commands.add_parser("train")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.set_defaults(handler=train)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    tv.setup_logging("INFO")
    args.handler(args)


if __name__ == "__main__":
    main()
