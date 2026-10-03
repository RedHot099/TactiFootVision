"""Model backends: registry, class-name mapping, output conversion and (with -m model) real weights."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.models import (
    MODELS,
    RFDETRDetector,
    TrainResult,
    YOLODetector,
    YOLOPoseModel,
    available_models,
)
from tactifoot_vision.models import ultralytics as ultralytics_backend
from tactifoot_vision.models.rfdetr import (
    _check_train_options,
    _final_metrics,
    _label_names,
    _read_log,
    _rfdetr_device,
)
from tactifoot_vision.models.ultralytics import (
    _keypoints_from_arrays,
    _read_results_csv,
    _standard_metrics,
)
from tactifoot_vision.utils import next_run_dir

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FOOTBALL = ROOT / "data" / "datasets" / "football_yolo"
KEYPOINTS = ROOT / "data" / "keypoints"
FOOTBALL_CLASSES = ["ball", "goalkeeper", "player", "referee"]


# ------------------------------------------------------------------ registry
def test_backends_are_registered():
    assert {"yolo", "yolo_pose", "rfdetr"} <= set(available_models())
    assert MODELS.get("yolo") is YOLODetector
    assert MODELS.get("yolo_pose") is YOLOPoseModel
    assert MODELS.get("rfdetr") is RFDETRDetector
    assert YOLODetector.task is tv.data.Task.DETECT
    assert YOLOPoseModel.task is tv.data.Task.POSE
    assert RFDETRDetector.task is tv.data.Task.DETECT


def test_importing_models_skips_heavy_libraries():
    code = (
        "import sys, tactifoot_vision.models, tactifoot_vision.evaluation; "
        "print([m for m in ('torch', 'ultralytics', 'rfdetr') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_rfdetr_rejects_unknown_size_before_loading():
    with pytest.raises(ValueError, match="Unknown RF-DETR size 'huge'"):
        RFDETRDetector(size="huge", device="cpu")


# --------------------------------------------------------- RF-DETR class ids
COCO = {1: "person", 2: "bicycle", 3: "car"}


def test_rfdetr_labels_of_coco_weights_are_coco_ids():
    assert _label_names({"num_classes": 2}, COCO) == COCO


def test_rfdetr_labels_of_zero_based_export():
    # Our to_coco export (and the legacy football checkpoint): ids 0..C-1.
    args = {"num_classes": 4, "class_names": FOOTBALL_CLASSES}
    assert _label_names(args, COCO) == dict(enumerate(FOOTBALL_CLASSES))


def test_rfdetr_labels_of_roboflow_export_skip_the_none_category():
    # Roboflow: category 0 is the "none" parent, classes follow with ids 1..C.
    args = {"num_classes": 5, "class_names": FOOTBALL_CLASSES}
    assert _label_names(args, COCO) == {
        i + 1: n for i, n in enumerate(FOOTBALL_CLASSES)
    }


def test_rfdetr_labels_given_as_mapping():
    assert _label_names({"class_names": {"3": "ball"}}, COCO) == {3: "ball"}


def test_rfdetr_device_names():
    assert _rfdetr_device("cuda:0") == "cuda"
    assert _rfdetr_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        _rfdetr_device("cuda:1")


def test_next_run_dir_never_reuses_a_run_unless_asked(tmp_path):
    first = next_run_dir(tmp_path / "run")
    second = next_run_dir(tmp_path / "run")
    assert (first.name, second.name) == ("run", "run2")
    assert second.is_dir()
    assert next_run_dir(tmp_path / "run", exist_ok=True) == first


def test_rfdetr_rejects_unknown_training_options():
    _check_train_options({"grad_accum_steps": 4, "warmup_epochs": 1})
    with pytest.raises(ValueError, match="epoch"):
        _check_train_options({"epoch": 5})


def test_ultralytics_metrics_use_the_shared_keys():
    raw = {
        "metrics/precision(B)": 0.5,
        "metrics/mAP50-95(B)": 0.25,
        "metrics/mAP50(P)": 0.75,
        "val/box_loss": 1.0,
        "fitness": 0.3,
    }
    assert _standard_metrics(raw) == {
        "precision": 0.5,
        "map50_95": 0.25,
        "pose_map50": 0.75,
    }


# ------------------------------------------------------------ RF-DETR history
def _coco_stats(map50_95: float) -> list[float]:
    # pycocotools order: AP, AP50, AP75, then 9 size/recall entries
    return [map50_95, map50_95 + 0.2, map50_95 + 0.1] + [0.0] * 9


def _log_line(epoch: int, regular: float, ema: float) -> str:
    return json.dumps(
        {
            "epoch": epoch,
            "train_loss": 1.0 / (epoch + 1),
            "test_coco_eval_bbox": _coco_stats(regular),
            "ema_test_coco_eval_bbox": _coco_stats(ema),
            "test_results_json": {"class_map": []},
            "train_epoch_time": "0:00:05",
        }
    )


def test_rfdetr_log_becomes_history_and_best_metrics(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("\n".join([_log_line(0, 0.3, 0.2), _log_line(1, 0.4, 0.5)]) + "\n")
    history = _read_log(log)
    assert list(history["epoch"]) == [1, 2]  # 1-based like Ultralytics
    assert {"test_map50_95", "test_map50", "ema_test_map75", "train_loss"} <= set(
        history
    )
    assert "test_results_json" not in history
    metrics = _final_metrics(history)
    assert metrics == pytest.approx({"map50_95": 0.5, "map50": 0.7, "map75": 0.6})


def test_rfdetr_best_metrics_prefer_regular_model_on_ties(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(_log_line(0, 0.3, 0.3) + "\n" + _log_line(1, 0.2, 0.3))
    # Both models peak at 0.3; the regular one (epoch 1) is kept, like RF-DETR does.
    assert _final_metrics(_read_log(log))["map50_95"] == pytest.approx(0.3)


# ------------------------------------------------------- Ultralytics helpers
def test_pose_instances_are_sorted_by_confidence():
    keypoints = np.zeros((3, 4, 3), np.float32)
    keypoints[..., 0] = np.array([0, 1, 2])[:, None]  # x marks the instance
    keypoints[..., 2] = 0.5
    boxes = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = _keypoints_from_arrays(
        keypoints, boxes, np.array([0.2, 0.9, 0.5]), np.zeros(3, int), ["pitch"]
    )
    assert isinstance(result, sv.KeyPoints)
    assert result.xy.shape == (3, 4, 2) and result.confidence.shape == (3, 4)
    assert list(result.xy[:, 0, 0]) == [1, 2, 0]
    assert list(result.data["box_confidence"]) == pytest.approx([0.9, 0.5, 0.2])
    np.testing.assert_array_equal(result.data["xyxy"][0], boxes[1])
    assert list(result.data["class_name"]) == ["pitch"] * 3


def test_pose_keypoints_without_visibility_use_instance_confidence():
    result = _keypoints_from_arrays(
        np.zeros((1, 5, 2), np.float32),
        np.zeros((1, 4)),
        np.array([0.7]),
        np.zeros(1, int),
        ["pitch"],
    )
    np.testing.assert_allclose(result.confidence, np.full((1, 5), 0.7))


def test_empty_pose_output_is_a_valid_keypoints_object():
    result = _keypoints_from_arrays(
        np.zeros((0, 32, 3), np.float32),
        np.zeros((0, 4)),
        np.zeros(0),
        np.zeros(0, int),
        ["pitch"],
    )
    assert len(result) == 0 and result.xy.shape == (0, 0, 2)
    assert result.confidence.shape == (0, 0) and len(result.data["xyxy"]) == 0


def test_ultralytics_results_csv_columns_are_stripped(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("   epoch,  metrics/mAP50(B)\n1,0.5\n")
    history = _read_results_csv(csv)
    assert list(history.columns) == ["epoch", "metrics/mAP50(B)"]
    with pytest.raises(RuntimeError, match="without writing"):
        _read_results_csv(tmp_path / "missing.csv")


# ---------------------------------------------- training with fake backends
def _tiny_dataset(tmp_path: Path) -> tv.Dataset:
    samples = []
    for i in range(3):
        path = tmp_path / "src" / f"img{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((48, 64, 3), 100, np.uint8))
        annotations = tv.data.Annotations(boxes=[[1, 2, 30, 40]], class_ids=[0])
        samples.append(tv.data.Sample(path, 64, 48, annotations))
    return tv.Dataset(
        tv.data.Task.DETECT, ["player"], {"train": samples[:2], "valid": samples[2:]}
    )


class _FakeYOLO:
    """Trains like Ultralytics: appends one row per epoch to ``results.csv``."""

    def train(self, **args):
        save_dir = Path(args["project"]) / args["name"]
        csv = save_dir / "results.csv"
        if not csv.is_file():
            csv.write_text("epoch,metrics/mAP50-95(B)\n")
        rows = len(csv.read_text().splitlines()) - 1
        with csv.open("a") as fh:
            fh.write(f"{rows + 1},0.5\n")
        (save_dir / "weights").mkdir(exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"")
        metrics = {"metrics/mAP50-95(B)": 0.5}
        self.trainer = SimpleNamespace(save_dir=save_dir, metrics=metrics)


class _FakeRFDETR:
    """Trains like RF-DETR: appends one JSON line per epoch to ``log.txt``."""

    model_config = SimpleNamespace(patch_size=16, num_windows=2)

    def train(self, **kwargs):
        run_dir = Path(kwargs["output_dir"])
        with (run_dir / "log.txt").open("a") as fh:
            fh.write(_log_line(0, 0.3, 0.2) + "\n")
        (run_dir / "checkpoint_best_total.pth").write_bytes(b"")


@pytest.fixture
def fake_yolo(monkeypatch) -> YOLODetector:
    monkeypatch.setattr(YOLODetector, "_load", lambda self, weights: None)
    model = YOLODetector(device="cpu")
    model._yolo = _FakeYOLO()
    return model


@pytest.fixture
def fake_rfdetr(monkeypatch) -> RFDETRDetector:
    monkeypatch.setattr(RFDETRDetector, "_load", lambda self, weights: None)
    model = RFDETRDetector(size="base", device="cpu")
    model._rf = _FakeRFDETR()
    return model


@pytest.mark.parametrize("backend", ["fake_yolo", "fake_rfdetr"])
def test_rerunning_into_the_same_run_folder_starts_a_fresh_history(
    backend, tmp_path, request
):
    model = request.getfixturevalue(backend)
    dataset = _tiny_dataset(tmp_path)
    options = {"output_dir": tmp_path / "runs", "exist_ok": True, "amp": False}
    if backend == "fake_rfdetr":
        options.pop("amp")
    first = model.train(dataset, **options)
    second = model.train(dataset, **options)
    assert first.run_dir == second.run_dir
    assert len(first.history) == len(second.history) == 1


def test_ultralytics_replaces_a_dangling_amp_probe_link(
    fake_yolo, tmp_path, monkeypatch
):
    cached = tmp_path / "cache" / "yolo11n.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"")
    monkeypatch.setattr(ultralytics_backend, "_cached_asset", lambda name: cached)
    run_dir = tmp_path / "runs" / "yolo"
    run_dir.mkdir(parents=True)
    (run_dir / "yolo11n.pt").symlink_to(tmp_path / "gone.pt")  # an interrupted run
    fake_yolo.train(
        _tiny_dataset(tmp_path), output_dir=tmp_path / "runs", exist_ok=True
    )
    assert not (run_dir / "yolo11n.pt").is_symlink()


@pytest.mark.parametrize(
    ("backend", "options", "match"),
    [
        ("fake_yolo", {"mosiac": 0.0}, "mosiac"),
        ("fake_yolo", {"project": "elsewhere"}, "project"),
        ("fake_yolo", {"data": "other.yaml"}, "data"),
        ("fake_rfdetr", {"epoch": 5}, "epoch"),
        ("fake_rfdetr", {"dataset_dir": "elsewhere"}, "dataset_dir"),
    ],
)
def test_bad_backend_options_fail_before_the_run_folder_exists(
    backend, options, match, tmp_path, request
):
    model = request.getfixturevalue(backend)
    with pytest.raises(ValueError, match=match):
        model.train(_tiny_dataset(tmp_path), output_dir=tmp_path / "runs", **options)
    assert not (tmp_path / "runs").exists()


# ======================================================= real weights / GPU
@pytest.fixture(scope="module")
def football() -> tv.Dataset:
    """A fixed sample of the football dataset: 100 training, 20 validation images."""
    return tv.load_dataset(FOOTBALL).subset({"train": 100, "valid": 20})


@pytest.fixture(scope="module")
def pitch() -> tv.Dataset:
    return tv.load_dataset(KEYPOINTS)


def _check_detections(detections: sv.Detections, class_names: list[str]) -> None:
    assert isinstance(detections, sv.Detections) and len(detections) > 0
    names = detections.data["class_name"]
    assert set(names) <= set(class_names)
    assert [class_names[i] for i in detections.class_id] == list(names)


@pytest.mark.model
def test_yolo_predicts_named_detections(football):
    model = tv.load_model("yolo", MODELS_DIR / "football_yolo11m.pt")
    assert model.class_names == FOOTBALL_CLASSES
    image, _ = football.read("valid", 0)
    _check_detections(model(image), FOOTBALL_CLASSES)


@pytest.mark.model
def test_yolo_pose_predicts_sorted_pitch_keypoints(pitch):
    model = tv.load_model("yolo_pose", MODELS_DIR / "pitch_yolov8n_pose.pt")
    keypoints = model(pitch.read("valid", 0)[0])
    assert isinstance(keypoints, sv.KeyPoints) and len(keypoints) >= 1
    assert keypoints.xy.shape[1:] == (32, 2)
    assert keypoints.confidence.shape == keypoints.xy.shape[:2]
    box_confidence = keypoints.data["box_confidence"]
    assert list(box_confidence) == sorted(box_confidence, reverse=True)


@pytest.mark.model
def test_rfdetr_finetuned_checkpoint_names_its_classes(football):
    model = tv.load_model("rfdetr", MODELS_DIR / "football_rfdetr_base.pth")
    assert model.class_names == FOOTBALL_CLASSES
    image, annotations = football.read("valid", 0)
    detections = model(image)
    _check_detections(detections, FOOTBALL_CLASSES)
    # The first validation image is mostly players; an off-by-one class map
    # would call them goalkeepers.
    predicted = pd.Series(detections.data["class_name"]).value_counts().idxmax()
    truth = pd.Series(annotations.class_ids).map(dict(enumerate(FOOTBALL_CLASSES)))
    assert predicted == truth.value_counts().idxmax() == "player"


@pytest.mark.model
def test_rfdetr_coco_weights_use_coco_names(football):
    model = tv.load_model("rfdetr", MODELS_DIR / "rfdetr_base.pth")
    assert len(model.class_names) == 80 and model.class_names[0] == "person"
    detections = model(football.read("valid", 0)[0])
    _check_detections(detections, model.class_names)
    assert "person" in set(detections.data["class_name"])


@pytest.mark.model
def test_checkpoints_of_the_wrong_kind_are_rejected():
    with pytest.raises(ValueError, match="'pose' checkpoint"):
        tv.load_model("yolo", MODELS_DIR / "pitch_yolov8n_pose.pt")
    with pytest.raises(ValueError, match="'detect' checkpoint"):
        tv.load_model("yolo_pose", MODELS_DIR / "football_yolo11m.pt")
    with pytest.raises(ValueError, match="pass size='base'"):
        tv.load_model("rfdetr", MODELS_DIR / "football_rfdetr_base.pth", size="nano")
    with pytest.raises(FileNotFoundError):
        tv.load_model("yolo", MODELS_DIR / "missing.pt")


def _check_train_result(
    result: TrainResult, model_type: str, tmp_path: Path, weights_name: str
) -> None:
    assert result.model_type == model_type
    assert result.weights.name == weights_name and result.weights.is_file()
    assert result.run_dir.is_relative_to(tmp_path)
    assert len(result.history) == 1
    assert result.metrics and all(isinstance(v, float) for v in result.metrics.values())


@pytest.mark.model
def test_train_yolo_for_one_epoch(football, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # anything Ultralytics drops lands here
    tiny = football.subset({"train": 16, "valid": 8})
    result = tv.train(
        "yolo",
        tiny,
        weights=MODELS_DIR / "football_yolo11m.pt",
        epochs=1,
        batch_size=4,
        imgsz=320,
        workers=2,
        output_dir=tmp_path / "runs",
        amp=False,  # the AMP check would download yolo11n.pt
        plots=False,
    )
    _check_train_result(result, "yolo", tmp_path, "best.pt")
    assert {"map50_95", "map50", "precision", "recall"} <= set(result.metrics)
    assert (result.run_dir / "dataset" / "data.yaml").is_file()
    model = result.load(conf=0.05)
    assert model.class_names == FOOTBALL_CLASSES
    assert isinstance(model(football.read("valid", 0)[0]), sv.Detections)


@pytest.mark.model
def test_train_yolo_pose_for_one_epoch(pitch, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny = pitch.subset({"train": 16, "valid": 8, "test": 0})
    model = tv.load_model("yolo_pose", MODELS_DIR / "pitch_yolov8n_pose.pt")
    result = model.train(
        tiny,
        epochs=1,
        batch_size=4,
        imgsz=320,
        workers=2,
        output_dir=tmp_path / "runs",
        amp=False,
        plots=False,
    )
    _check_train_result(result, "yolo_pose", tmp_path, "best.pt")
    assert model.weights == result.weights  # the instance now holds the new weights
    keypoints = result.load()(pitch.read("valid", 0)[0])
    assert isinstance(keypoints, sv.KeyPoints) and keypoints.xy.shape[1] == 32


@pytest.mark.model
def test_train_rfdetr_for_one_epoch(football, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny = football.subset({"train": 8, "valid": 4})
    result = tv.train(
        "rfdetr",
        tiny,
        weights=MODELS_DIR / "rfdetr_base.pth",
        epochs=1,
        batch_size=2,
        imgsz=336,
        workers=2,
        output_dir=tmp_path / "runs",
        grad_accum_steps=1,
        tensorboard=False,
    )
    _check_train_result(result, "rfdetr", tmp_path, "checkpoint_best_total.pth")
    assert {"map50_95", "map50", "map75"} <= set(result.metrics)
    assert "test_map50_95" in result.history
    model = result.load(conf=0.01)
    assert model.class_names == FOOTBALL_CLASSES
    detections = model(football.read("valid", 0)[0])
    assert set(detections.data["class_name"]) <= set(FOOTBALL_CLASSES)


@pytest.mark.model
def test_yolo_and_rfdetr_score_on_the_same_scale(football):
    # Measured on the fixture's 20 validation images: YOLO11m mAP50 0.84,
    # the RF-DETR base checkpoint 0.88.
    expected_map50 = {"football_yolo11m.pt": 0.7, "football_rfdetr_base.pth": 0.7}
    models = [
        tv.load_model("yolo", MODELS_DIR / "football_yolo11m.pt"),
        tv.load_model("rfdetr", MODELS_DIR / "football_rfdetr_base.pth"),
    ]
    for model in models:
        metrics = tv.evaluate(model, football)
        assert metrics.num_images == len(football["valid"])
        assert expected_map50[Path(model.weights).name] < metrics.map50 <= 1.0
        assert list(metrics.per_class.index) == FOOTBALL_CLASSES
        assert model.conf != 0.01  # evaluation threshold was restored
