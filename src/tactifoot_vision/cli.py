"""``tactifoot`` command line: the package API for shell scripts and batch jobs.

tactifoot run --config configs/pipeline.yaml --video match.mp4 --output-dir outputs/match
tactifoot train yolo --data data/datasets/football_yolo --weights yolo11n.pt --epochs 50
tactifoot evaluate rfdetr --weights models/football_rfdetr_base.pth --data data/datasets/football_yolo
tactifoot info
"""

import argparse
import json
import sys
from pathlib import Path

import tactifoot_vision as tv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tactifoot", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the inference pipeline on a video")
    run.add_argument("--config", type=Path, required=True, help="pipeline YAML")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument(
        "--period", type=int, default=1, help="match period for freeze frames"
    )
    run.add_argument(
        "--period-start", type=float, default=0.0, help="match clock at frame 0 (s)"
    )
    run.add_argument(
        "--no-video", action="store_true", help="skip rendering the annotated video"
    )

    train = commands.add_parser("train", help="train a model on a dataset")
    train.add_argument(
        "model", help=f"one of: {', '.join(tv.models.available_models())}"
    )
    train.add_argument(
        "--data", type=Path, required=True, help="YOLO data.yaml or COCO folder"
    )
    train.add_argument(
        "--weights", help="starting checkpoint (default: backend's pretrained)"
    )
    for name, kind in (
        ("epochs", int),
        ("batch-size", int),
        ("imgsz", int),
        ("lr", float),
    ):
        train.add_argument(f"--{name}", type=kind)
    train.add_argument("--device")
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--name")

    evaluate = commands.add_parser("evaluate", help="score a model on a dataset split")
    evaluate.add_argument("model")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", default="valid")

    commands.add_parser("info", help="show version and available backends")

    args = parser.parse_args(argv)
    tv.setup_logging(args.log_level)
    return {"run": _run, "train": _train, "evaluate": _evaluate, "info": _info}[
        args.command
    ](args)


def _run(args: argparse.Namespace) -> int:
    pipeline = tv.config.load_config(args.config).build()
    result = pipeline.run(
        args.video, start=args.start, max_frames=args.max_frames, stride=args.stride
    )
    out = args.output_dir
    result.save(out / "result.pkl")
    result.to_csv(out / "tracks.csv")
    result.to_freeze_frames(args.period, args.period_start).to_csv(
        out / "freeze_frames.csv", index=False
    )
    if not args.no_video:
        tv.viz.render_video(result, args.video, out / "annotated.mp4")
    print(f"Wrote results for {len(result)} frames to {out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "lr": args.lr,
            "device": args.device,
            "output_dir": args.output_dir,
            "name": args.name,
        }.items()
        if value is not None
    }
    result = tv.train(
        args.model, tv.load_dataset(args.data), weights=args.weights, **overrides
    )
    print(
        json.dumps(
            {"weights": str(result.weights), "metrics": result.metrics}, indent=2
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    model = tv.load_model(args.model, args.weights)
    metrics = model.evaluate(tv.load_dataset(args.data), split=args.split)
    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


def _info(args: argparse.Namespace) -> int:
    print(f"tactifoot-vision {tv.__version__}")
    print("models:  ", ", ".join(tv.models.available_models()))
    print("trackers:", ", ".join(tv.tracking.available_trackers()))
    print("teams:   ", ", ".join(tv.teams.available_embedders()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
