"""Training runs of detector-data (ticket 04): the view, the throughput probe, the run.

    uv run python scripts/detector_train.py view --run run2a       # runs/run2a: lists, data.yaml, class-mask table
    uv run python scripts/detector_train.py probe --run run1 --model yolo11l --batch 8
    uv run python scripts/detector_train.py train --run run2a --smoke   # ~200 iterations on a random slice
    uv run python scripts/detector_train.py train --run run2a           # the full run (see runs/run2a/RUN.md)
    uv run python scripts/detector_train.py train --run run2a --resume  # continue from weights/last.pt

Each run is an entry of `RUNS`: the pool sets it reads, the images it holds out, per-set oversampling, the extra
validation images, the initial weights and the Ultralytics settings. Every view drops the GSR valid/test games (2, 3,
5, 7, 8, 11: the frozen benchmark) and any image tagged as a near-duplicate of a GSR valid/test frame. Classes outside
a source's `annotated_classes` get no classification loss (the package's `annotated_classes` option of the `yolo`
model); the "player means any person" sources (ISSIA, TeamTrack, SportsMOT, SoccerPlayerDetection, SoccerSynth)
annotate neither goalkeeper nor referee, and the script checks that their tables say so.

* run 1 ("Run 1 decisions"): every image of the sets the pool held on 2026-10-15 (`RUN1_SETS`; SoccerSynth and
  football_yolo came later), from COCO weights. Its view is reproduced exactly (same list hash).
* run 2 (owner, 2026-10-16), two sequential stages: run2a fine-tunes run 1's best.pt on football_yolo alone, without
  the Ekstraklasa dev clips (its `test` split, see scripts/detector_data/football_yolo.py); run2b fine-tunes run2a's
  best.pt on SoccerSynth alone. Every epoch's checkpoint is kept (`save_period` 1);
  `scripts/detector_bench.py epochs` scores them all.
* run2c: run 1's sets plus football_yolo (dev clips out) listed 20 times, from run 1's best.pt; 2a showed that
  football_yolo alone makes the model forget GSR.

Validation during training: every 4th frame of the frozen GSR valid benchmark (bench/, ticket 02), plus for run 2 every
Ekstraklasa dev image (bench/ekstraklasa_dev); the full benchmarks are scored after the run with
scripts/detector_bench.py.
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
EKSTRAKLASA_DEV = DATA / "bench/ekstraklasa_dev/valid/images"
RESEARCH = Path(__file__).resolve().parents[1] / "research"

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
ANY_PERSON_SOURCES = {
    "issia",
    "teamtrack",
    "sportsmot",
    "soccer_player_detection",
    "soccersynth",
}
VAL_EVERY = 4  # every 4th benchmark frame (bench frames are every 5th GSR frame)
SMOKE = {
    "train_images": 800,
    "epochs": 2,
    "val_every": 40,
    "seed": 0,
}  # 2 x 200 iterations at batch 4

WEIGHTS = {
    "yolo11m": CHECKOUT / "models/yolo11m.pt",
    "yolo11l": CHECKOUT / "models/yolo11l.pt",
}
# The sets of the pool when run 1 was built (2026-10-15).
RUN1_SETS = (
    "issia",
    "roboflow_ball",
    "roboflow_players",
    "soccana",
    "soccer_player_detection",
    "soccernet_gsr",
    "soccernet_v3",
    "sportsmot_football",
    "teamtrack",
)
RUNS: dict[str, dict] = {
    "run1": {
        "title": "detector-data run 1 training view (ticket 04)",
        "sets": RUN1_SETS,
        "holdout": {},  # set -> split held out of training
        "oversample": {},  # set -> times listed in train.txt
        "val_extra": None,  # folder of extra validation images
        "init": None,  # None: the model's COCO weights
        "after": None,  # the run whose best.pt is the init (its smoke best.pt in smoke mode)
        # Chosen by the probe (see the ticket's comments).
        "settings": {
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
        },
    },
    # Run 2 (owner, 2026-10-16): two sequential fine-tuning stages, every epoch's checkpoint kept (save_period 1,
    # Ultralytics writes weights/epoch<N>.pt with N from 0) so each stage and epoch can be compared.
    "run2a": {
        "title": "detector-data run 2a training view (ticket 04): football_yolo only, Ekstraklasa dev clips out",
        "sets": ("football_yolo",),
        "holdout": {"football_yolo": "test"},  # the Ekstraklasa dev clips
        "oversample": {},
        "val_extra": EKSTRAKLASA_DEV,
        "init": DATA / "runs/run1/yolo11l_1280/weights/best.pt",
        "after": "run1",
        "settings": {
            "model": "yolo11l",
            "imgsz": 1280,
            "batch_size": 4,
            "optimizer": "SGD",
            "lr": 0.002,
            # 736 images = 184 iterations per epoch; 35 epochs = 6,440 iterations.
            "epochs": 35,
            "patience": None,  # Ultralytics' default 100 > epochs: no early stop
            "workers": 12,
            "seed": 0,
            "cos_lr": True,
            "close_mosaic": 2,
            "save_period": 1,
            "scale": 0.3,
        },
    },
    "run2b": {
        "title": "detector-data run 2b training view (ticket 04): SoccerSynth only, from run 2a",
        "sets": ("soccersynth",),
        "holdout": {},
        "oversample": {},
        "val_extra": EKSTRAKLASA_DEV,
        "init": DATA / "runs/run2a/yolo11l_1280/weights/best.pt",
        "after": "run2a",
        "settings": {
            "model": "yolo11l",
            "imgsz": 1280,
            "batch_size": 4,
            "optimizer": "SGD",
            "lr": 0.002,
            # 50,877 images = 12,720 iterations per epoch (about 50 min at 17-18 img/s).
            "epochs": 5,
            "patience": None,
            "workers": 12,
            "seed": 0,
            "cos_lr": True,
            "close_mosaic": 2,
            "save_period": 1,
            "scale": 0.3,
        },
    },
    # Run 2c (2026-10-16, after 2a): 2a closed the Ekstraklasa gap in 2-3 epochs but forgot GSR from epoch 1, so
    # football_yolo is mixed into the run-1 data instead (oversampled to ~13% of the list), from run 1's best.pt.
    # SoccerSynth stays out: in 2b the mixed val fell every epoch.
    "run2c": {
        "title": "detector-data run 2c training view (ticket 04): run 1 data + football_yolo x20, Ekstraklasa dev out",
        "sets": (*RUN1_SETS, "football_yolo"),
        "holdout": {"football_yolo": "test"},  # the Ekstraklasa dev clips
        "oversample": {"football_yolo": 20},
        "val_extra": EKSTRAKLASA_DEV,
        "init": DATA / "runs/run1/yolo11l_1280/weights/best.pt",
        "after": "run1",
        "settings": {
            "model": "yolo11l",
            "imgsz": 1280,
            "batch_size": 4,
            "optimizer": "SGD",
            "lr": 0.002,
            # 99,478 + 20 x 736 = 114,198 listed images = 28,550 iterations per epoch (about 1 h 45).
            "epochs": 3,
            "patience": None,
            "workers": 12,
            "seed": 0,
            "cos_lr": True,
            "close_mosaic": 1,
            "save_period": 1,
            "scale": 0.3,
        },
    },
}

logger = logging.getLogger("detector_train")


def run_dir(run: str) -> Path:
    return DATA / "runs" / run


def view_json(run: str) -> Path:
    return RESEARCH / f"detector_{run}_view.json"


# ------------------------------------------------------------------ view
def build_view(pool: pd.DataFrame, run: str) -> pd.DataFrame:
    """One row per training image of ``run``: its sets, minus the GSR benchmark games, their near-duplicates and
    the run's held-out splits."""
    spec = RUNS[run]
    missing = set(spec["sets"]) - set(pool.set)
    assert not missing, f"sets {sorted(missing)} are not in the pool"
    held = pd.Series(False, index=pool.index)
    for set_name, split in spec["holdout"].items():
        held |= (pool.set == set_name) & (pool.split == split)
    excluded = (
        ~pool.set.isin(spec["sets"])
        | held
        | pool.game.isin(BENCH_GAMES).fillna(False)
        | pool.near_dup_set.isin(BENCH_DUP_SETS).fillna(False)
        | pool.gsr_split.isin(["valid", "test"]).fillna(False)
    )
    view = pool[~excluded].copy()
    assert not view.gsr_split.isin(["valid", "test"]).any()
    assert not ANY_PERSON_SOURCES - set(pool.source), (
        f"any-person sources missing from the pool: {ANY_PERSON_SOURCES - set(pool.source)}"
    )
    for source in ANY_PERSON_SOURCES & set(view.source):
        annotated = view.annotated_classes[view.source == source]
        assert annotated.map(
            lambda s: not {"goalkeeper", "referee"} & set(s.split(","))
        ).all(), (
            f"{source}: player means any person, so goalkeeper and referee must not be annotated"
        )
    return view


def train_list(view: pd.DataFrame, oversample: dict[str, int]) -> list[str]:
    """The sorted image paths of the view, each listed ``oversample[set]`` times (default once)."""
    times = view.set.map(lambda s: oversample.get(s, 1)).to_numpy()
    return sorted(np.repeat(view.image.to_numpy(), times).tolist())


def counts(view: pd.DataFrame, oversample: dict[str, int] | None = None) -> dict:
    """Images and boxes per source, the images per epoch with oversampling, and the share of images and boxes each
    class mask removes."""
    oversample = oversample or {}
    times = view.set.map(lambda s: oversample.get(s, 1))
    per_source = {}
    for source, rows in view.groupby("source"):
        per_source[source] = {
            "images": len(rows),
            "images_per_epoch": int(times[rows.index].sum()),
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
        "images_per_epoch": int(times.sum()),
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


def val_images(run: str) -> list[str]:
    """Every ``VAL_EVERY``-th frame of the sorted GSR valid benchmark, then the run's extra validation images."""
    val = sorted(str(p) for p in BENCH_VALID.iterdir() if p.suffix == ".jpg")
    extra = RUNS[run]["val_extra"]
    more = (
        sorted(str(p) for p in extra.iterdir() if p.suffix == ".jpg") if extra else []
    )
    return val[::VAL_EVERY] + more


def view(args: argparse.Namespace) -> None:
    spec, out = RUNS[args.run], run_dir(args.run)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(POOL)
    train = build_view(pool, args.run).sort_values("image")
    train_images = train_list(train, spec["oversample"])
    val = val_images(args.run)
    label_check = {
        "train": check_label_paths(sorted(set(train_images))),
        "val": check_label_paths(val),
    }

    (out / "train.txt").write_text("\n".join(train_images) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    write_yaml(out / "data.yaml", "train.txt", "val.txt")
    table = train[["image", "source", "annotated_classes"]].reset_index(drop=True)
    table.to_parquet(out / "annotated_classes.parquet", index=False)

    # Smoke slice: a random sample of the training list, a sparser validation subset.
    smoke = out / "smoke_data"
    smoke.mkdir(exist_ok=True)
    rng = np.random.default_rng(SMOKE["seed"])
    size = min(SMOKE["train_images"], len(train_images))
    pick = sorted(rng.choice(train_images, size, replace=False))
    (smoke / "train.txt").write_text("\n".join(pick) + "\n")
    (smoke / "val.txt").write_text(
        "\n".join(val[:: SMOKE["val_every"] // VAL_EVERY]) + "\n"
    )
    write_yaml(smoke / "data.yaml", "train.txt", "val.txt")

    record = {
        "title": spec["title"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool_index": str(POOL),
        "pool_images": len(pool),
        "rule": {
            "sets": list(spec["sets"]),
            "excluded_games": sorted(BENCH_GAMES),
            "excluded_near_dup_sets": sorted(BENCH_DUP_SETS),
            "excluded_gsr_splits": ["valid", "test"],
            "held_out_splits": spec["holdout"],
            "oversample": spec["oversample"],
            "included": "everything else of the sets: xG evaluation/gate games, Soccana, every source split",
            "class_mask": "per image, the source's annotated_classes; classes outside it get no cls loss",
            "any_person_sources": sorted(ANY_PERSON_SOURCES),
        },
        "init": str(spec["init"] or WEIGHTS[spec["settings"]["model"]]),
        "after": spec["after"],
        "settings": spec["settings"],
        "train": counts(train, spec["oversample"])
        | {"list_sha256": sha256_lines(train_images)},
        "val": {
            "source": [str(BENCH_VALID)]
            + ([str(spec["val_extra"])] if spec["val_extra"] else []),
            "rule": f"every {VAL_EVERY}th frame of the sorted GSR valid benchmark (frames every 5th GSR frame)"
            + (", then every image of the extra folder" if spec["val_extra"] else ""),
            "images": len(val),
            "list_sha256": sha256_lines(val),
            "files": [Path(v).name for v in val],
        },
        "label_paths": label_check,
        "files": {
            "data_yaml": str(out / "data.yaml"),
            "annotated_classes": str(out / "annotated_classes.parquet"),
            "smoke_data_yaml": str(smoke / "data.yaml"),
        },
    }
    view_json(args.run).write_text(json.dumps(record, indent=2, default=str) + "\n")
    summary = {k: v for k, v in record["train"].items() if k != "per_source"}
    logger.info("view: %s", json.dumps(summary))
    for source, row in record["train"]["per_source"].items():
        logger.info(
            "  %-24s %6d images (%6d per epoch)  %s  %s",
            source,
            row["images"],
            row["images_per_epoch"],
            row["annotated_classes"],
            row["boxes"],
        )
    logger.info("val: %d images; wrote %s and %s", len(val), out, view_json(args.run))


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
        smoke = run_dir(args.run) / "smoke_data"
        scale = RUNS[args.run]["settings"]["scale"]
        hyp = get_cfg(overrides={"imgsz": args.imgsz, "scale": scale})
        data = ClassMaskedDataset(
            img_path=str(smoke / "train.txt"),
            imgsz=args.imgsz,
            batch_size=args.batch,
            augment=True,
            hyp=hyp,
            data={"names": dict(enumerate(CLASSES)), "nc": 4, "channels": 3},
            class_masks=read_class_masks(
                run_dir(args.run) / "annotated_classes.parquet", CLASSES
            ),
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
    with (run_dir(args.run) / "probe.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")


def _weights(model: str | Path) -> Path:
    path = WEIGHTS.get(model, Path(model))
    if not path.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(path)
    return path


# ------------------------------------------------------------------ train
def init_weights(run: str, smoke: bool = False) -> Path:
    """The run's initial weights: COCO for run 1, else the previous run's best.pt (in smoke mode the previous run's
    smoke best.pt when its full run has not happened yet, so stages can be smoked in a row)."""
    spec = RUNS[run]
    if spec["init"] is None:
        return _weights(spec["settings"]["model"])
    path = Path(spec["init"])
    if smoke and not path.is_file() and spec["after"]:
        path = run_dir(spec["after"]) / "smoke/weights/best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"initial weights of {run} not found: {path} (run {spec['after']} first)"
        )
    return path


def train(args: argparse.Namespace) -> None:
    spec, out = RUNS[args.run], run_dir(args.run)
    settings = dict(spec["settings"])
    model = settings.pop("model")
    name = f"{model}_{settings['imgsz']}"
    data_yaml = out / "data.yaml"
    weights = init_weights(args.run, args.smoke)
    if args.smoke:
        data_yaml = out / "smoke_data/data.yaml"
        name = "smoke"
        settings |= {
            "epochs": SMOKE["epochs"],
            "patience": None,
            "close_mosaic": 0,
            "val": False,
        }
    if args.resume:
        weights = out / name / "weights/last.pt"
        settings["resume"] = True
    started = time.time()
    dataset = tv.load_dataset(data_yaml)
    logger.info("loaded %r in %.0f s", dataset, time.time() - started)
    logger.info("run %s from %s: %s", args.run, weights, json.dumps(settings))
    result = tv.train(
        "yolo",
        dataset,
        weights=weights,
        output_dir=out,
        name=name,
        exist_ok=True,
        annotated_classes=out / "annotated_classes.parquet",
        **settings,
    )
    logger.info("done: %s %s", result.weights, json.dumps(result.metrics))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    v = commands.add_parser("view")
    v.set_defaults(handler=view)
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
    for command in (v, p, t):
        command.add_argument("--run", choices=sorted(RUNS), required=True)
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
