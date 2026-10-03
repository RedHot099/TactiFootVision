"""Run files and the ``tactifoot`` command, driven by fake registered models."""

import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv
import yaml

import tactifoot_vision as tv
from tactifoot_vision import run_file
from tactifoot_vision.cli import main
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.models import MODELS, Model, TrainConfig, TrainResult
from tactifoot_vision.pipeline import Pipeline, PipelineResult
from tactifoot_vision.pitch import HomographyEstimator, SoccerPitch
from tactifoot_vision.teams import EMBEDDERS, Embedder, TeamClassifier
from tactifoot_vision.tracking import ByteTrackTracker


@MODELS.register("test_box_detector")
class BoxDetector(Model):
    """One player box around the green pixels; records what training and loading got."""

    task = Task.DETECT
    name = "test_box_detector"
    trained: list[TrainConfig] = []

    def __init__(self, weights=None, *, conf=0.25, device="cpu", iou=0.5):
        self.iou = iou
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        BoxDetector.trained.append(config)
        weights = run_dir / "best.pt"
        weights.write_bytes(b"")
        return TrainResult(self.name, weights, run_dir, pd.DataFrame(), {"map50": 1.0})

    @property
    def class_names(self):
        return ["ball", "player"]

    def predict(self, image):
        ys, xs = np.nonzero(image[..., 1] > 128)
        if len(xs) == 0:
            return sv.Detections(
                xyxy=np.zeros((0, 4), np.float32),
                confidence=np.zeros(0, np.float32),
                class_id=np.zeros(0, int),
                data={"class_name": np.zeros(0, str)},
            )
        return sv.Detections(
            xyxy=np.array(
                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32
            ),
            confidence=np.array([0.9], np.float32),
            class_id=np.array([1]),
            data={"class_name": np.array(["player"])},
        )


class FakeEmbedder(Embedder):
    name = "fake"

    def __init__(self, device=None):
        self.device = device

    def embed(self, crops):
        return np.zeros((len(crops), 2), np.float32)


@pytest.fixture(autouse=True)
def _restore_logging(monkeypatch):
    """``main`` calls ``setup_logging``, which stops propagation (and so caplog)."""
    logger = logging.getLogger("tactifoot_vision")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "level", logger.level)


@pytest.fixture
def fake_siglip(monkeypatch):
    """Make the default team embedder cheap to build."""
    monkeypatch.setitem(EMBEDDERS._factories, "siglip", FakeEmbedder)


def _video(path, frames=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(frames):
        image = np.zeros((48, 64, 3), np.uint8)
        image[20:30, 4 + 4 * i : 12 + 4 * i] = (0, 255, 0)
        writer.write(image)
    writer.release()
    return path


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


def _load(tmp_path, data, overrides=()):
    return run_file.load(_write(tmp_path / "run.yaml", data), overrides)


DETECTOR = {"type": "test_box_detector"}


# --------------------------------------------------------------- run files
def test_run_file_builds_each_section_with_its_constructor(tmp_path, fake_siglip):
    pipeline = _load(
        tmp_path,
        {
            "detector": {"type": "test_box_detector", "conf": 0.4, "iou": 0.3},
            "tracker": {"type": "bytetrack", "lost_track_buffer": 5},
            "teams": {"n_teams": 3, "reducer": None},
            "pitch": {"length": 120, "width": 80},
            "homography": {"smoothing_window": 5},
            "pipeline": {"ball_max_speed": None, "include_classes": ["player"]},
        },
    ).build_pipeline()
    assert (pipeline.detector.conf, pipeline.detector.iou) == (0.4, 0.3)
    assert isinstance(pipeline.tracker, ByteTrackTracker)
    assert pipeline.tracker.lost_track_buffer == 5
    assert pipeline.team_classifier.n_teams == 3
    assert pipeline.team_classifier.reducer is None
    assert pipeline.pitch == SoccerPitch(120, 80) == pipeline.homography.pitch
    assert pipeline.homography._history.maxlen == 5
    assert pipeline.ball_max_speed is None
    assert pipeline.include_classes == ("player",)
    assert pipeline.keypoint_model is None


def test_null_sections_turn_components_off(tmp_path):
    pipeline = _load(
        tmp_path, {"detector": DETECTOR, "keypoints": None, "tracker": None}
    ).build_pipeline()
    assert pipeline.tracker is None and pipeline.keypoint_model is None


def test_sections_are_a_copy(tmp_path):
    loaded = _load(tmp_path, {"detector": DETECTOR})
    loaded.sections["detector"]["type"] = "changed"
    assert loaded.sections == {"detector": DETECTOR}


def _homography_settings(estimator):
    return (
        estimator.pitch,
        estimator.min_confidence,
        estimator.ransac_threshold,
        estimator.max_age,
        estimator._history.maxlen,
    )


def _public(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _team_settings(classifier):
    return _public(classifier) | {"embedder": type(classifier.embedder)}


_PIPELINE_SETTINGS = [
    name
    for name in inspect.signature(Pipeline).parameters
    if name not in run_file._PIPELINE_PARTS
]


def test_empty_sections_build_what_the_bare_constructors_build(tmp_path, fake_siglip):
    sections = {"detector": DETECTOR} | dict.fromkeys(
        ("teams", "pitch", "homography", "pipeline", "tracker"), {}
    )
    sections["tracker"] = {"type": "bytetrack"}
    built = _load(tmp_path, sections).build_pipeline()
    bare = Pipeline(BoxDetector())
    assert _homography_settings(built.homography) == _homography_settings(
        HomographyEstimator()
    )
    assert _team_settings(built.team_classifier) == _team_settings(TeamClassifier())
    assert built.pitch == SoccerPitch()
    assert _public(built.tracker) == _public(ByteTrackTracker())
    for name in _PIPELINE_SETTINGS:
        assert getattr(built, name) == getattr(bare, name), name
    assert type(built.tracker) is type(bare.tracker)  # absent tracker: Pipeline's own


def test_run_file_picks_up_changed_constructor_defaults(tmp_path, monkeypatch):
    defaults = HomographyEstimator.__init__.__kwdefaults__ | {"min_confidence": 0.9}
    monkeypatch.setattr(HomographyEstimator.__init__, "__kwdefaults__", defaults)
    built = _load(
        tmp_path, {"detector": DETECTOR, "homography": {}, "pitch": {}}
    ).build_pipeline()
    assert built.homography.min_confidence == 0.9


def _section(path: str, value: dict) -> dict:
    """A run file with the detector plus ``value`` at the dotted ``path``."""
    *parents, last = path.split(".")
    data = {"detector": dict(DETECTOR)}
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = node.get(last, {}) | value
    return data


@pytest.mark.parametrize(
    ("section", "value", "bad", "valid"),
    [
        ("detector", {"bogus": 1}, "bogus", "iou"),
        ("keypoints", DETECTOR | {"bogus": 1}, "bogus", "conf"),
        ("tracker", {"type": "bytetrack", "bogus": 1}, "bogus", "lost_track_buffer"),
        ("tracker", {"type": "sam2", "bogus": 1}, "bogus", "checkpoint"),
        ("pitch", {"bogus": 1}, "bogus", "length"),
        ("homography", {"bogus": 1}, "bogus", "max_age"),
        ("homography", {"pitch": {}}, "pitch", "max_age"),
        ("pipeline", {"bogus": 1}, "bogus", "ball_class"),
        ("pipeline", {"tracker": None}, "tracker", "keep_masks"),
        ("render", {"bogus": 1}, "bogus", "overlay_alpha"),
        ("render", {"progress": False}, "progress", "fps"),
        ("render.annotator", {"bogus": 1}, "bogus", "style"),
        ("render.annotator", {"pitch": {}}, "pitch", "draw_masks"),
        ("render.radar", {"bogus": 1}, "bogus", "width_px"),
    ],
)
def test_unknown_keys_fail_at_load(tmp_path, section, value, bad, valid):
    with pytest.raises(ValueError, match="valid keys") as error:
        _load(tmp_path, _section(section, value))
    message = str(error.value)
    assert f"Unknown key {bad!r} in section {section!r}" in message
    assert valid in message


def test_teams_accept_embedder_options_which_the_embedder_checks(tmp_path):
    # TeamClassifier takes **embedder_options, so any key passes the load check.
    loaded = _load(
        tmp_path, {"detector": DETECTOR, "teams": {"embedder": "resnet", "bogus": 1}}
    )
    with pytest.raises(TypeError, match="bogus"):
        loaded.build_pipeline()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"detector": DETECTOR, "detectr": {}}, "Unknown run file section 'detectr'"),
        ({"keypoints": None}, "needs a 'detector' section"),
        ({"detector": {"type": "nope"}}, "Unknown model 'nope'.*yolo"),
        ({"detector": {"weights": "x.pt"}}, "needs a 'type' key.*yolo"),
        ({"detector": DETECTOR, "tracker": {"type": "nope"}}, "Unknown tracker.*bytetrack"),
        ({"detector": "yolo"}, "'detector' must be a mapping"),
        ({"detector": DETECTOR, "pitch": None}, "'pitch' must be a mapping"),
        ({"detector": DETECTOR, "render": {"annotator": None}}, "'render.annotator' must"),
    ],
)  # fmt: skip
def test_invalid_structure_fails_at_load(tmp_path, data, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, data)


def test_values_are_checked_by_the_constructors(tmp_path):
    loaded = _load(tmp_path, {"detector": DETECTOR, "homography": {"max_age": -1}})
    with pytest.raises(ValueError, match="max_age"):
        loaded.build_pipeline()


# --------------------------------------------------------------- overrides
def test_overrides_parse_yaml_values_and_create_missing_keys(tmp_path):
    loaded = _load(
        tmp_path,
        {"detector": DETECTOR, "tracker": {"type": "bytetrack"}},
        [
            "detector.conf=0.4",
            "detector.device=cpu",
            "tracker=null",
            "homography.max_age=5",
            "pipeline.include_classes=[player, goalkeeper]",
            "pipeline.keep_masks=true",
            "render.annotator.style=video_game",
        ],
    )
    assert loaded.sections == {
        "detector": DETECTOR | {"conf": 0.4, "device": "cpu"},
        "tracker": None,
        "homography": {"max_age": 5},
        "pipeline": {"include_classes": ["player", "goalkeeper"], "keep_masks": True},
        "render": {"annotator": {"style": "video_game"}},
    }
    assert loaded.build_pipeline().tracker is None


def test_override_paths_are_relative_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "..")
    loaded = _load(tmp_path, {"detector": DETECTOR}, ["detector.weights=./w.pt"])
    assert loaded.sections["detector"]["weights"] == str(tmp_path.parent / "w.pt")


def test_overrides_are_key_checked(tmp_path):
    with pytest.raises(ValueError, match="Unknown key 'bogus' in section 'detector'"):
        _load(tmp_path, {"detector": DETECTOR}, ["detector.bogus=1"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("detector.conf", "key=value"),
        ("=1", "empty key"),
        ("detector..conf=1", "empty key"),
        ("detector.type.x=1", "'detector.type' is 'test_box_detector', not a mapping"),
        ("keypoints.conf=0.3", "'keypoints' is None, not a mapping"),
    ],
)
def test_override_errors(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, {"detector": DETECTOR, "keypoints": None}, [override])


# ------------------------------------------------------------------- paths
def test_relative_paths_resolve_against_the_run_file(tmp_path):
    path = _write(
        tmp_path / "configs" / "run.yaml",
        {
            "detector": DETECTOR | {"weights": "../models/w.pt"},
            "keypoints": DETECTOR | {"weights": "yolov8n-pose.pt"},
            "tracker": {"type": "sam2", "checkpoint": "./ckpt.pt", "config": "/abs/t.yaml"},
            "teams": {"embedder": "fake", "paths": ["./a", {"deep": "../b"}, "configs/c"]},
        },
    )  # fmt: skip
    sections = run_file.load(path).sections
    assert sections["detector"]["weights"] == str(tmp_path / "models" / "w.pt")
    assert sections["keypoints"]["weights"] == "yolov8n-pose.pt"
    assert sections["tracker"]["checkpoint"] == str(tmp_path / "configs" / "ckpt.pt")
    assert sections["tracker"]["config"] == "/abs/t.yaml"
    assert sections["teams"]["paths"] == [
        str(tmp_path / "configs" / "a"),
        {"deep": str(tmp_path / "b")},
        "configs/c",
    ]


def test_loading_imports_no_heavy_library(tmp_path):
    code = f"""
import sys, yaml
from pathlib import Path
import tactifoot_vision as tv
from tactifoot_vision.models import MODELS
from tactifoot_vision.teams import EMBEDDERS
from tactifoot_vision.tracking import TRACKERS
for i, model in enumerate(MODELS.names()):
    path = Path({str(tmp_path)!r}) / f"run{{i}}.yaml"
    path.write_text(yaml.safe_dump({{
        "detector": {{"type": model}},
        "keypoints": {{"type": model}},
        "tracker": {{"type": TRACKERS.names()[i % len(TRACKERS.names())]}},
        "teams": {{"embedder": EMBEDDERS.names()[i % len(EMBEDDERS.names())]}},
        "pitch": {{}}, "homography": {{}}, "pipeline": {{}},
        "render": {{"annotator": {{}}, "radar": {{}}}},
    }}))
    tv.run_file.load(path)
for tracker in TRACKERS.names():
    tv.run_file.load(path, [f"tracker.type={{tracker}}"])
print([m for m in ("torch", "ultralytics", "rfdetr", "transformers", "sam2")
       if m in sys.modules])
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


# -------------------------------------------------------------- run folder
def test_export_writes_the_run_folder_data_files(tmp_path):
    result = _load(tmp_path, {"detector": DETECTOR}).build_pipeline()
    result = result.run(_video(tmp_path / "clip.mp4"), progress=False)
    out = result.export(tmp_path / "new" / "run", period=2, period_start=60.0)
    assert out == tmp_path / "new" / "run"
    assert sorted(p.name for p in out.iterdir()) == [
        "freeze_frames.csv",
        "result.pkl",
        "tracks.csv",
    ]
    assert len(PipelineResult.load(out / "result.pkl")) == 6
    frames = pd.read_csv(out / "freeze_frames.csv")
    assert (frames["period"] == 2).all() and frames["minute"].iloc[0] == 1


def _cli_run(tmp_path, *extra):
    config = _write(tmp_path / "run.yaml", {"detector": DETECTOR})
    video = _video(tmp_path / "clip.mp4")
    out = tmp_path / "out"
    args = ["run", str(config), "--video", str(video), "--output-dir", str(out)]
    assert main([*args, *extra]) == 0
    return out


def test_cli_run_writes_the_run_folder(tmp_path):
    out = _cli_run(tmp_path, "--set", "render.radar=null")
    result = PipelineResult.load(out / "result.pkl")
    assert len(result) == 6 and result.track_ids == [1]
    tracks = pd.read_csv(out / "tracks.csv")
    assert (tracks["class_name"] == "player").all() and len(tracks) == 6
    assert (out / "freeze_frames.csv").is_file()
    video = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 6


def test_cli_run_passes_run_inputs_and_overrides(tmp_path):
    out = _cli_run(
        tmp_path,
        *("--start", "1", "--end", "4", "--stride", "2"),
        *("--period", "2", "--period-start", "90", "--no-video"),
        *("--set", "tracker=null", "--set", "detector.conf=0.5"),
    )
    result = PipelineResult.load(out / "result.pkl")
    assert [f.index for f in result] == [1, 3]
    assert result.track_ids == []  # no tracker
    assert not (out / "annotated.mp4").exists()
    frames = pd.read_csv(out / "freeze_frames.csv")
    assert (frames["period"] == 2).all() and frames["minute"].iloc[0] == 1


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    config = _write(tmp_path / "run.yaml", {"detector": {**DETECTOR, "bogus": 1}})
    video = _video(tmp_path / "clip.mp4")
    args = ["run", str(config), "--video", str(video), "--output-dir", str(tmp_path)]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert err.startswith("tactifoot: error:") and "bogus" in err
    assert "Traceback" not in err
    with pytest.raises(ValueError, match="bogus"):  # DEBUG keeps the traceback
        main(["--log-level", "DEBUG", *args])


def test_cli_reports_runtime_errors_in_one_line(tmp_path, capsys):
    config = _write(tmp_path / "run.yaml", {"detector": DETECTOR})
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a video")
    args = ["run", str(config), "--video", str(video), "--output-dir", str(tmp_path)]
    assert main(args) == 1
    err = capsys.readouterr().err
    last = err.strip().splitlines()[-1]
    assert last.startswith("tactifoot: error:") and "cannot open" in last
    assert "Traceback" not in err


def test_cli_run_checks_the_video_before_loading_models(tmp_path, capsys):
    config = _write(tmp_path / "run.yaml", {"detector": {"type": "no_such_model"}})
    args = ["run", str(config), "--video", str(tmp_path / "missing.mp4")]
    assert main([*args, "--output-dir", str(tmp_path / "out")]) == 1
    assert "video not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_every_run_parameter_has_a_flag(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    run = inspect.signature(Pipeline.run).parameters
    export = inspect.signature(PipelineResult.export).parameters
    names = [n for n in run if n not in ("self", "video", "progress")]
    names += [n for n in export if n not in ("self", "out_dir")]
    assert len(names) == 5
    for name in names:
        assert f"--{name.replace('_', '-')}" in out


# -------------------------------------------------------- train / evaluate
def _sample_value(annotation):
    """A command-line string for a field type and the value it should become."""
    kind = tv.cli._unwrap_optional(annotation)
    return {
        int: ("3", 3),
        float: ("0.5", 0.5),
        str: ("abc", "abc"),
        Path: ("some/dir", Path("some/dir")),
    }[kind]


@pytest.mark.parametrize("field", list(TrainConfig.model_fields))
def test_every_train_config_field_is_a_flag(field, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(tv, "train", lambda *a, **kw: calls.append(kw) or _Trained())
    monkeypatch.setattr(tv, "load_dataset", lambda path: "dataset")
    info = TrainConfig.model_fields[field]
    flag = f"--{field.replace('_', '-')}"
    if info.annotation is bool:
        args, expected = [flag], True
    else:
        text, expected = _sample_value(info.annotation)
        args = [flag, text]
    assert main(["train", "test_box_detector", "--data", "d", *args]) == 0
    assert calls == [{field: expected}]

    with pytest.raises(SystemExit):
        main(["train", "--help"])
    assert f"(default: {info.default})" in " ".join(capsys.readouterr().out.split())


class _Trained:
    weights = Path("best.pt")
    metrics = {"map50": 1.0}


def test_train_rejects_a_key_set_twice(monkeypatch):
    monkeypatch.setattr(tv, "load_dataset", lambda path: "dataset")
    with pytest.raises(SystemExit):
        main(["train", "yolo", "--data", "d", "--epochs", "3", "--set", "epochs=4"])


def test_cli_rejects_unknown_models():
    with pytest.raises(SystemExit):
        main(["train", "nope", "--data", "d"])


def _write_dataset(root):
    names = {"train": ["a", "b"], "valid": ["c"]}
    for split, stems in names.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem in stems:
            image = np.zeros((48, 64, 3), np.uint8)
            image[10:30, 20:30] = (0, 255, 0)
            cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), image)
            (root / split / "labels" / f"{stem}.txt").write_text(
                "1 0.390625 0.416667 0.15625 0.416667\n"
            )
    return _write(
        root / "data.yaml",
        {"train": "train/images", "val": "valid/images", "names": ["ball", "player"]},
    )


def test_cli_train_end_to_end(tmp_path, capsys):
    BoxDetector.trained.clear()
    data = _write_dataset(tmp_path / "dataset")
    args = ["train", "test_box_detector", "--data", str(data), "--epochs", "2"]
    args += ["--output-dir", str(tmp_path / "runs"), "--no-exist-ok"]
    args += ["--set", "mosaic=0.0", "--set", "lr=0.01"]
    assert main(args) == 0
    (config,) = BoxDetector.trained
    assert (config.epochs, config.lr, config.exist_ok) == (2, 0.01, False)
    assert config.backend_options == {"mosaic": 0.0}
    printed = json.loads(capsys.readouterr().out)
    assert printed["weights"] == str(
        tmp_path / "runs" / "test_box_detector" / "best.pt"
    )


def test_cli_evaluate_end_to_end(tmp_path, monkeypatch, capsys):
    from tactifoot_vision import evaluation

    calls = []
    original = evaluation.evaluate_detector

    def spy(model, dataset, **kwargs):
        calls.append((model.device, kwargs))
        return original(model, dataset, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_detector", spy)
    data = _write_dataset(tmp_path / "dataset")
    args = ["evaluate", "test_box_detector", "--weights", "w.pt", "--data", str(data)]
    args += ["--split", "train", "--device", "cpu:0", "--set", "max_images=1"]
    assert main(args) == 0
    assert calls == [("cpu:0", {"split": "train", "max_images": 1})]
    assert json.loads(capsys.readouterr().out)["map50"] == pytest.approx(1.0)


def test_cli_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "yolo" in out and "bytetrack" in out and "siglip" in out
