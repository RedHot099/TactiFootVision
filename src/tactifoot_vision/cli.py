"""``tactifoot`` command line: the package API for shell scripts and batch jobs.

tactifoot run configs/pipeline.yaml --video match.mp4 --output-dir outputs/match
tactifoot train yolo --data data/datasets/football_yolo --weights yolo11n.pt --epochs 50
tactifoot evaluate rfdetr --weights models/football_rfdetr_base.pth --data data/datasets/football_yolo
tactifoot info

The command only translates its arguments into package calls and declares no
defaults: an option left out keeps the package default. Training flags come
from ``TrainConfig`` and run flags from ``Pipeline.run`` and
``PipelineResult.export``, so new settings appear here without editing this file.
"""

import argparse
import functools
import inspect
import json
import sys
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import tactifoot_vision as tv

_SUPPRESS = argparse.SUPPRESS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tactifoot",
        description=__doc__.split("\n")[0],
        argument_default=_SUPPRESS,
    )
    parser.add_argument("--log-level", dest="level", help="DEBUG, INFO, WARNING, ...")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run the pipeline on a video",
        description="Run a run file on a video and write the run folder. --start, "
        "--max-frames and --stride go to Pipeline.run, --period and --period-start "
        "to PipelineResult.export.",
        argument_default=_SUPPRESS,
    )
    run.add_argument("run_file", type=Path, help="run file (YAML)")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True, help="run folder")
    frame_range = _add_flags(run, _parameters(tv.Pipeline.run, ("video", "progress")))
    period = _add_flags(run, _parameters(tv.PipelineResult.export, ("out_dir",)))
    run.add_argument(
        "--no-video", action="store_true", help="skip rendering annotated.mp4"
    )
    _add_overrides(run, "run file value, e.g. detector.conf=0.4")
    run.set_defaults(handler=functools.partial(_run, frame_range, period))

    models = tv.models.available_models()
    train = commands.add_parser(
        "train", help="train a model on a dataset", argument_default=_SUPPRESS
    )
    train.add_argument("model", choices=models)
    train.add_argument(
        "--data", type=Path, required=True, help="YOLO data.yaml or COCO folder"
    )
    train.add_argument(
        "--weights", help="starting checkpoint (default: the backend's pretrained one)"
    )
    fields = tv.TrainConfig.model_fields
    settings = _add_flags(
        train,
        [(name, f.annotation, f.default, f.description) for name, f in fields.items()],
    )
    _add_overrides(train, "TrainConfig field or backend option, e.g. mosaic=0.0")
    train.set_defaults(handler=functools.partial(_train, train, settings))

    evaluate = commands.add_parser(
        "evaluate", help="score a model on a dataset split", argument_default=_SUPPRESS
    )
    evaluate.add_argument("model", choices=models)
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--device", help="torch device, e.g. cpu or cuda:1")
    split = _add_flags(evaluate, _parameters(tv.models.Model.evaluate, ("dataset",)))
    _add_overrides(evaluate, "evaluation option, e.g. max_images=50")
    evaluate.set_defaults(handler=functools.partial(_evaluate, split))

    info = commands.add_parser("info", help="show version and available backends")
    info.set_defaults(handler=_info)

    args = parser.parse_args(argv)
    tv.setup_logging(**_given(args, ["level"]))
    return args.handler(args)


def _run(frame_range: list[str], period: list[str], args: argparse.Namespace) -> int:
    run_file = tv.run_file.load(args.run_file, _overrides(args))
    pipeline = run_file.build_pipeline()
    result = pipeline.run(args.video, **_given(args, frame_range))
    out = result.export(args.output_dir, **_given(args, period))
    if "no_video" not in args:
        run_file.render_video(result, args.video, out / "annotated.mp4")
    print(f"Wrote {len(result)} frames, {len(result.track_ids)} tracks to {out}")
    return 0


def _train(
    parser: argparse.ArgumentParser, settings: list[str], args: argparse.Namespace
) -> int:
    options = _given(args, settings)
    for key, value in map(tv.run_file.parse_override, _overrides(args)):
        if key in options:
            parser.error(f"{key} is set both by --{_flag(key)} and --set")
        options[key] = value
    result = tv.train(
        args.model,
        tv.load_dataset(args.data),
        **_given(args, ["weights"]),
        **options,
    )
    print(
        json.dumps(
            {"weights": str(result.weights), "metrics": result.metrics}, indent=2
        )
    )
    return 0


def _evaluate(split: list[str], args: argparse.Namespace) -> int:
    model = tv.load_model(args.model, args.weights, **_given(args, ["device"]))
    options = dict(map(tv.run_file.parse_override, _overrides(args)))
    metrics = model.evaluate(
        tv.load_dataset(args.data), **_given(args, split), **options
    )
    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


def _info(args: argparse.Namespace) -> int:
    print(f"tactifoot-vision {tv.__version__}")
    print("models:  ", ", ".join(tv.models.available_models()))
    print("trackers:", ", ".join(tv.tracking.available_trackers()))
    print("teams:   ", ", ".join(tv.teams.available_embedders()))
    return 0


# ------------------------------------------------------------------ helpers
Field = tuple[str, Any, Any, str | None]  # name, annotation, default, description


def _parameters(function: Callable[..., Any], exclude: Iterable[str]) -> list[Field]:
    """Keyword parameters of ``function`` as flag fields (minus ``self`` and ``exclude``)."""
    keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return [
        (p.name, p.annotation, p.default, None)
        for p in inspect.signature(function).parameters.values()
        if p.kind in keyword and p.name not in ("self", *exclude)
    ]


def _add_flags(parser: argparse.ArgumentParser, fields: Iterable[Field]) -> list[str]:
    """Add one ``--kebab-name`` flag per field; return their names.

    The type comes from the annotation (``X | None`` becomes ``X``; ``bool``
    gives ``--flag/--no-flag``) and the help shows the package default.
    """
    names = []
    for name, annotation, default, description in fields:
        kind = _unwrap_optional(annotation)
        text = f"{description} " if description else ""
        options: dict[str, Any] = {
            "dest": name,
            "help": f"{text}(default: {default})".replace("%", "%%"),
        }
        if kind is bool:
            options["action"] = argparse.BooleanOptionalAction
        else:
            options |= {"type": kind, "metavar": name.upper()}
        parser.add_argument(f"--{_flag(name)}", **options)
        names.append(name)
    return names


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _add_overrides(parser: argparse.ArgumentParser, example: str) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help=f"{example}; the value is YAML; repeatable",
    )


def _overrides(args: argparse.Namespace) -> list[str]:
    return getattr(args, "overrides", [])


def _given(args: argparse.Namespace, names: Iterable[str]) -> dict[str, Any]:
    """The options among ``names`` that were given on the command line."""
    return {name: getattr(args, name) for name in names if name in args}


def _flag(name: str) -> str:
    return name.replace("_", "-")


if __name__ == "__main__":
    sys.exit(main())
