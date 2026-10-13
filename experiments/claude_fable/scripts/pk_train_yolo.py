"""Train a YOLO-pose pitch keypoint candidate on ticket 01's dataset through the package (pitch-keypoints 03).

    ./run.sh python scripts/pk_train_yolo.py --name NAME --weights yolo11m-pose.pt --imgsz 960 --epochs 60
        [--batch 16] [--val-step 4] [--workers 8] [--set key=value ...]      Ultralytics options over `YOLO_AUGMENT`
        [--box-fraction 0.6]                                                  fixed centred instance boxes
        [--train-limit N --val-on-train]                                      smoke / overfit runs
    ./run.sh python scripts/pk_train_yolo.py --name NAME --resume             continue from <run>/weights/last.pt

The run folder is `<runs>/<name>` (default `/mnt/storage/tactifoot_research/pk_runs`): the run's `data/` (image lists
and data.yaml, `training.training_yaml`), the package's YOLO export (`dataset/`, symlinks), Ultralytics' outputs
(`weights/`, `results.csv`, plots) and `sessions.jsonl`, one line when a session starts and one when it ends (wall
time, git state, arguments), so every GPU session's wall time is on record even when a session is killed.

Training goes through `tv.train("yolo_pose", ...)`; `--resume` calls Ultralytics directly on `last.pt` (the package
has no resume), from the run folder as the package does. Reads no StatsBomb data.
"""

import argparse
import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.pitchkp import training as T


def option(text: str) -> tuple[str, object]:
    """`key=value` with the value parsed as YAML (numbers, booleans, strings)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key, yaml.safe_load(value)


def log_session(run_dir: Path, event: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def train(args: argparse.Namespace, run_dir: Path) -> Path:
    import tactifoot_vision as tv

    data_yaml = T.training_yaml(args.dataset, run_dir / "data", val_step=args.val_step, train_limit=args.train_limit,
                                seed=args.seed)
    dataset = tv.Dataset.from_yolo(data_yaml)
    if args.val_on_train:                     # overfit check: validate on the training images
        dataset = dataset.with_split("valid", dataset["train"])
    if args.box_fraction is not None:         # one fixed centred box per image (training.central_boxes)
        dataset = T.central_boxes(dataset, args.box_fraction)
    options = T.YOLO_AUGMENT | dict(args.set or [])
    result = tv.train("yolo_pose", dataset, weights=args.weights, epochs=args.epochs, batch_size=args.batch,
                      imgsz=args.imgsz, device=args.device, workers=args.workers, seed=args.seed,
                      patience=args.patience, output_dir=args.runs, name=args.name, exist_ok=True, **options)
    return result.weights


def resume(run_dir: Path) -> Path:
    from ultralytics import YOLO

    last = run_dir / "weights/last.pt"
    if not last.is_file():
        raise SystemExit(f"nothing to resume: {last} is missing")
    with contextlib.chdir(run_dir):           # Ultralytics' AMP check looks for yolo11n.pt here (the package links it)
        model = YOLO(str(last))
        model.train(resume=True)
    return Path(model.trainer.save_dir) / "weights/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights", default="yolo11m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--val-step", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-on-train", action="store_true")
    parser.add_argument("--box-fraction", type=float, default=None, help="train with centred boxes of this size")
    parser.add_argument("--set", type=option, action="append", metavar="KEY=VALUE")
    parser.add_argument("--dataset", type=Path, default=T.DATASET)
    parser.add_argument("--runs", type=Path, default=T.RUNS)
    args = parser.parse_args()
    run_dir = args.runs / args.name
    if not args.resume and (run_dir / "weights").exists():
        raise SystemExit(f"{run_dir} has weights; pass --resume or use a new --name")

    started, clock = datetime.now(UTC), time.perf_counter()
    session = {"mode": "resume" if args.resume else "train", "args": {k: str(v) if isinstance(v, Path) else v
                                                                     for k, v in vars(args).items()},
               "git": records.git_state(), "libraries": records.library_versions()}
    log_session(run_dir, {"event": "start", "at": started.isoformat(timespec="seconds")} | session)
    status = "failed"
    try:
        with no_statsbomb_reads():
            best = resume(run_dir) if args.resume else train(args, run_dir)
        status = "finished"
    finally:
        log_session(run_dir, {"event": "end", "status": status, "mode": session["mode"],
                              "at": datetime.now(UTC).isoformat(timespec="seconds"),
                              "seconds": round(time.perf_counter() - clock, 1)})
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
