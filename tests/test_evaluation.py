"""Backend-independent evaluation on synthetic datasets with fake models."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

from tactifoot_vision.data import Annotations, Dataset, Sample, Task
from tactifoot_vision.evaluation import (
    DetectionMetrics,
    KeypointMetrics,
    compare_with_statsbomb,
    evaluate,
    evaluate_detector,
    evaluate_keypoints,
)
from tactifoot_vision.models import Model
from tactifoot_vision.pipeline import FrameResult, PipelineResult

CLASSES = ["ball", "player", "referee"]


class FakeModel(Model):
    """Returns canned outputs keyed by the image's first pixel value."""

    name = "fake"

    def __init__(self, task: Task, outputs: dict, class_names: list[str]) -> None:
        self.task = task
        self.outputs = outputs
        self._class_names = class_names
        self.seen_conf: list[float] = []
        super().__init__(None, device="cpu")

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def predict(self, image):
        self.seen_conf.append(self.conf)
        output = self.outputs[int(image[0, 0, 0])]
        if isinstance(output, Exception):
            raise output
        return output


def _write_image(path: Path, index: int, width: int = 200, height: int = 100) -> Path:
    cv2.imwrite(str(path), np.full((height, width, 3), index, np.uint8))
    return path


def _detection_dataset(tmp_path: Path, labels: list[Annotations]) -> Dataset:
    samples = [
        Sample(_write_image(tmp_path / f"{i}.png", i), 200, 100, annotations)
        for i, annotations in enumerate(labels)
    ]
    return Dataset(Task.DETECT, CLASSES, {"valid": samples}, name="toy")


def _detections(boxes, names, confidence=None) -> sv.Detections:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    detections = sv.Detections(
        xyxy=boxes,
        confidence=np.asarray(
            confidence if confidence is not None else [0.9] * len(boxes), np.float32
        ),
        class_id=np.arange(len(boxes)),  # deliberately meaningless ids
    )
    detections.data["class_name"] = np.asarray(names, dtype=str)
    return detections


LABELS = [
    Annotations([[10, 10, 30, 40], [50, 20, 80, 60]], [1, 2]),
    Annotations([[100, 10, 110, 20], [120, 30, 150, 90]], [0, 1]),
]
PERFECT = {
    0: _detections(LABELS[0].boxes, ["player", "referee"]),
    1: _detections(LABELS[1].boxes, ["ball", "player"]),
}


# ------------------------------------------------------------- detection
def test_perfect_detections_score_one(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    model = FakeModel(Task.DETECT, PERFECT, CLASSES)
    metrics = evaluate_detector(model, dataset)
    assert isinstance(metrics, DetectionMetrics)
    assert metrics.as_dict() == pytest.approx(
        {"map50_95": 1.0, "map50": 1.0, "map75": 1.0}
    )
    assert metrics.num_images == 2
    assert list(metrics.per_class.index) == CLASSES
    assert list(metrics.per_class.columns) == [
        "map50_95",
        "map50",
        "map75",
        "instances",
    ]
    assert list(metrics.per_class["instances"]) == [1, 2, 1]
    assert model.seen_conf == [0.01, 0.01] and model.conf == 0.25


def test_missing_detections_score_zero(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    empty = {i: sv.Detections.empty() for i in range(2)}
    metrics = evaluate_detector(FakeModel(Task.DETECT, empty, CLASSES), dataset)
    assert metrics.map50_95 == 0.0 and metrics.map50 == 0.0


def test_classes_are_matched_by_name_not_id(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    # Different class order, plus a class the dataset does not know.
    outputs = {
        0: _detections(
            [*LABELS[0].boxes, [0, 0, 5, 5]],
            ["player", "referee", "person"],
            [0.8, 0.8, 0.99],
        ),
        1: PERFECT[1],
    }
    model = FakeModel(Task.DETECT, outputs, ["person", "referee", "player", "ball"])
    assert evaluate_detector(model, dataset).map50_95 == pytest.approx(1.0)


def test_wrong_class_or_loose_box_lowers_the_score(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    outputs = {
        0: _detections(LABELS[0].boxes, ["referee", "player"]),  # swapped classes
        1: _detections(LABELS[1].boxes + [0, 0, 4, 0], ["ball", "player"]),
    }
    metrics = evaluate_detector(FakeModel(Task.DETECT, outputs, CLASSES), dataset)
    per_class = metrics.per_class
    assert per_class.loc["referee", "map50"] == 0.0
    # ball IoU = 10*10 / (14*10) = 0.71: a match up to IoU 0.7, 5 of 10 thresholds
    assert per_class.loc[
        "ball", ["map50", "map75", "map50_95"]
    ].tolist() == pytest.approx([1, 0, 0.5])
    assert 0.0 < metrics.map50 < 1.0


def test_max_images_and_class_without_instances(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    metrics = evaluate_detector(
        FakeModel(Task.DETECT, PERFECT, CLASSES), dataset, max_images=1
    )
    assert metrics.num_images == 1
    assert metrics.per_class.loc["ball", "instances"] == 0
    assert np.isnan(metrics.per_class.loc["ball", "map50_95"])
    assert metrics.map50_95 == pytest.approx(1.0)


def test_confidence_is_restored_when_prediction_fails(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    model = FakeModel(Task.DETECT, {0: RuntimeError("boom")}, CLASSES)
    model.conf = 0.4
    with pytest.raises(RuntimeError, match="boom"):
        evaluate_detector(model, dataset, conf=0.05)
    assert model.seen_conf == [0.05] and model.conf == 0.4


def test_detector_evaluation_rejects_bad_input(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    unnamed = sv.Detections(
        xyxy=LABELS[0].boxes, confidence=np.ones(2), class_id=np.array([1, 2])
    )
    with pytest.raises(ValueError, match="class_name"):
        evaluate_detector(FakeModel(Task.DETECT, {0: unnamed}, CLASSES), dataset)
    with pytest.raises(ValueError, match="detection model"):
        evaluate_detector(FakeModel(Task.POSE, {}, ["pitch"]), dataset)
    with pytest.raises(ValueError, match="no images"):
        evaluate_detector(FakeModel(Task.DETECT, {}, CLASSES), dataset, split="test")


# ------------------------------------------------------------- keypoints
def _keypoint_dataset(tmp_path: Path) -> Dataset:
    truth = np.array([[10, 10, 2], [20, 20, 1], [30, 30, 0]], np.float32)
    samples = [
        Sample(
            _write_image(tmp_path / f"{i}.png", i, 300, 400),
            300,
            400,
            Annotations([[0, 0, 300, 400]], [0], keypoints=truth[None]),
        )
        for i in range(2)
    ]
    samples.append(  # no labelled instance: not scored
        Sample(
            _write_image(tmp_path / "2.png", 2, 300, 400),
            300,
            400,
            Annotations.empty(num_keypoints=3),
        )
    )
    return Dataset(Task.POSE, ["pitch"], {"valid": samples}, num_keypoints=3)


def _pose(*instances) -> sv.KeyPoints:
    if not instances:
        return sv.KeyPoints.empty()
    xy = np.asarray(instances, np.float32)
    return sv.KeyPoints(xy=xy, confidence=np.ones(xy.shape[:2], np.float32))


def test_keypoint_error_pck_and_detection_rate(tmp_path):
    dataset = _keypoint_dataset(tmp_path)
    outputs = {
        # best instance: errors 3 px, 40 px, (unlabelled point ignored); the
        # second instance would be perfect but only the best one counts
        0: _pose([[13, 10], [20, 60], [500, 500]], [[10, 10], [20, 20], [30, 30]]),
        1: _pose(),  # nothing found
        2: _pose([[0, 0], [0, 0], [0, 0]]),
    }
    model = FakeModel(Task.POSE, outputs, ["pitch"])
    # diagonal 500 px: 0.05 -> 25 px
    metrics = evaluate_keypoints(model, dataset, pck_threshold=0.05)
    assert isinstance(metrics, KeypointMetrics)
    assert metrics.num_images == 2
    assert metrics.detection_rate == 0.5
    assert metrics.mean_error_px == pytest.approx(21.5)
    assert metrics.pck == pytest.approx(0.5)
    table = metrics.per_keypoint
    assert list(table["count"]) == [1, 1, 0]
    assert table.loc[0, "mean_error_px"] == pytest.approx(3.0)
    assert table.loc[1, "pck"] == 0.0
    assert np.isnan(table.loc[2, "mean_error_px"])


def test_keypoint_count_mismatch_is_an_error(tmp_path):
    dataset = _keypoint_dataset(tmp_path)
    model = FakeModel(Task.POSE, {0: _pose([[0, 0], [1, 1]])}, ["pitch"])
    with pytest.raises(ValueError, match="predicts 2 keypoints"):
        evaluate_keypoints(model, dataset)


def test_evaluate_dispatches_on_task(tmp_path):
    detection = _detection_dataset(tmp_path, LABELS)
    assert isinstance(
        evaluate(FakeModel(Task.DETECT, PERFECT, CLASSES), detection), DetectionMetrics
    )
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    keypoints = _keypoint_dataset(pose_dir)
    model = FakeModel(Task.POSE, {0: _pose([[10, 10]] * 3), 1: _pose()}, ["pitch"])
    result = model.evaluate(keypoints, pck_threshold=0.1)
    assert isinstance(result, KeypointMetrics) and result.pck_threshold == 0.1


# ------------------------------------------------------------- statsbomb
def _statsbomb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": [1, 1, 1, 2, 1],
            "minute": [0, 0, 0, 0, 0],
            "second": [1, 1, 2, 1, 1],
            "pitch_location": [[10, 10], "[50.0, 50.0]", [5, 5], [10, 10], None],
            "type": ["player", "goalkeeper", "player", "player", "player"],
        }
    )


def _freeze_frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": [1, 1, 1, 2],
            "minute": [0, 0, 0, 0],
            "second": [1, 1, 1, 3],
            "location": ["[11.0, 10.0]", "[49.0, 53.0]", None, "[5.0, 5.0]"],
            "type": ["player", "goalkeeper", "player", "player"],
            "player_id": [7, 8, 9, 10],
            "frame_bbox": ["[0, 0, 1, 1]"] * 4,
            "confidence": [0.9, 0.8, 0.7, 0.6],
            "visible_area": [None] * 4,
        }
    )


def test_statsbomb_objects_get_their_nearest_detection():
    merged = compare_with_statsbomb(_freeze_frames(), _statsbomb(), period=1)
    # period-2 row and the row without a location are dropped
    assert len(merged) == 3
    assert list(merged["type"]) == ["player", "goalkeeper", "player"]
    assert list(merged["detected_player_id"][:2]) == [7, 8]
    assert list(merged["detected_type"][:2]) == ["player", "goalkeeper"]
    assert merged["detected_location"][0] == "[11.0, 10.0]"
    assert merged["euclidean_distance"][:2].tolist() == pytest.approx(
        [1.0, np.hypot(1, 3)]
    )
    assert merged["detected_confidence"][1] == pytest.approx(0.8)
    # nothing detected in second 2
    assert np.isnan(merged["euclidean_distance"][2])
    assert pd.isna(merged["detected_player_id"][2])
    assert merged["detected_player_id"].dtype == "Int64"


def test_statsbomb_period_selects_rows():
    merged = compare_with_statsbomb(_freeze_frames(), _statsbomb(), period=2)
    assert len(merged) == 1 and merged["euclidean_distance"].isna().all()


def test_statsbomb_comparison_needs_its_columns():
    with pytest.raises(ValueError, match="statsbomb is missing columns"):
        compare_with_statsbomb(_freeze_frames(), _statsbomb().drop(columns="type"))
    with pytest.raises(ValueError, match="freeze_frames is missing columns"):
        compare_with_statsbomb(_freeze_frames().drop(columns="location"), _statsbomb())


def test_statsbomb_comparison_accepts_a_pipeline_result():
    people = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20], [30, 0, 40, 20]], np.float32),
        confidence=np.array([0.9, 0.8], np.float32),
        class_id=np.array([2, 2]),
        tracker_id=np.array([4, 5]),
    )
    people.data["class_name"] = np.array(["player", "player"])
    people.data["pitch_xy"] = np.array([[10.5, 10.0], [52.0, 50.0]])
    people.data["team_id"] = np.array([0, 1])
    frame = FrameResult(
        index=30, timestamp=1.2, detections=people, ball=sv.Detections.empty()
    )
    result = PipelineResult(
        frames=[frame], fps=25.0, frame_size=(1280, 720), class_names=["player"]
    )
    merged = compare_with_statsbomb(result, _statsbomb())
    assert list(merged["detected_player_id"][:2]) == [4, 5]
    assert merged["euclidean_distance"][:2].tolist() == pytest.approx([0.5, 2.0])
    assert json.loads(merged["detected_location"][1]) == [52.0, 50.0]


def test_statsbomb_comparison_of_a_second_half_pipeline_result():
    people = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], np.float32),
        confidence=np.array([0.9], np.float32),
        tracker_id=np.array([4]),
        data={
            "class_name": np.array(["player"]),
            "pitch_xy": np.array([[10.5, 10.0]]),
            "team_id": np.array([0]),
        },
    )
    frame = FrameResult(
        index=30, timestamp=1.2, detections=people, ball=sv.Detections.empty()
    )
    result = PipelineResult([frame], 25.0, (1280, 720), ["player"])
    statsbomb = pd.DataFrame(
        {
            "period": [2, 2, 1],
            "minute": [45, 0, 0],
            "second": [1, 1, 1],
            "pitch_location": [[10, 10], [10, 10], [10, 10]],
            "type": ["player"] * 3,
        }
    )
    merged = compare_with_statsbomb(result, statsbomb, period=2)
    # The video starts at the second half's kick-off: 45:01 on StatsBomb's clock.
    assert merged["euclidean_distance"].tolist()[0] == pytest.approx(0.5)
    assert np.isnan(merged["euclidean_distance"][1])
    assert list(merged["detected_player_id"][:1]) == [4]


# ------------------------------------------------------- review round 2
def test_unknown_class_warning_fires_once_per_evaluation(tmp_path, caplog):
    dataset = _detection_dataset(tmp_path, LABELS)
    outputs = {0: _detections([[0, 0, 5, 5]], ["unicorn"]), 1: PERFECT[1]}
    model = FakeModel(Task.DETECT, outputs, ["unicorn", "ball", "player"])
    for _ in range(2):
        evaluate_detector(model, dataset)
    warnings = [r for r in caplog.records if "unicorn" in r.getMessage()]
    assert len(warnings) == 2


def test_evaluate_rejects_unknown_options(tmp_path):
    dataset = _detection_dataset(tmp_path, LABELS)
    model = FakeModel(Task.DETECT, PERFECT, CLASSES)
    with pytest.raises(ValueError, match="max_imgs.*max_images"):
        model.evaluate(dataset, max_imgs=1)


def _frame(index, timestamp, rows, ball_xy=None):
    """A FrameResult with people ``rows`` of (class name, pitch xy) and an optional ball."""
    names = [name for name, _ in rows]
    people = sv.Detections(
        xyxy=np.zeros((len(rows), 4), np.float32),
        confidence=np.full(len(rows), 0.9, np.float32),
        tracker_id=np.arange(1, len(rows) + 1),
        data={
            "class_name": np.array(names, dtype=str),
            "pitch_xy": np.array([xy for _, xy in rows], dtype=float).reshape(-1, 2),
            "team_id": np.zeros(len(rows), int),
        },
    )
    ball = sv.Detections.empty()
    if ball_xy is not None:
        ball = sv.Detections(
            xyxy=np.zeros((1, 4), np.float32),
            confidence=np.ones(1, np.float32),
            data={"class_name": np.array(["ball"]), "pitch_xy": np.array([ball_xy])},
        )
    return FrameResult(index=index, timestamp=timestamp, detections=people, ball=ball)


def _one_player_at(xy, **extra):
    return pd.DataFrame(
        {"period": [1], "minute": [0], "second": [0], "pitch_location": [xy]}
        | {"type": ["player"]}
        | extra
    )


def test_statsbomb_matches_only_players_and_goalkeepers():
    frame = _frame(
        0, 0.0, [("referee", [59.0, 40.0]), ("player", [10.0, 10.0])], [60.0, 40.0]
    )
    result = PipelineResult([frame], 25.0, (100, 100), ["ball", "player", "referee"])
    merged = compare_with_statsbomb(result, _one_player_at([60.0, 40.0]))
    assert merged["detected_type"].tolist() == ["player"]
    assert merged["euclidean_distance"][0] == pytest.approx(np.hypot(50, 30))


def test_statsbomb_matches_the_frame_closest_to_the_event():
    frames = [
        _frame(0, 0.0, [("player", [60.0, 40.0])]),  # closer object, earlier frame
        _frame(12, 0.48, [("player", [65.0, 40.0])]),
        _frame(24, 0.96, [("player", [61.0, 40.0])]),
    ]
    result = PipelineResult(frames, 25.0, (100, 100), ["player"])
    event = _one_player_at([60.0, 40.0], timestamp_seconds=[0.5])
    merged = compare_with_statsbomb(result, event)
    assert merged["detected_frame_id"].tolist() == [12]
    assert merged["euclidean_distance"][0] == pytest.approx(5.0)
    # Without the event's sub-second time, every frame of the second competes.
    pooled = compare_with_statsbomb(result, _one_player_at([60.0, 40.0]))
    assert pooled["detected_frame_id"].tolist() == [0]
    assert pooled["euclidean_distance"][0] == pytest.approx(0.0)


def test_statsbomb_csv_route_agrees_with_the_result_route(tmp_path):
    frame = _frame(30, 1.2, [("player", [10.5, 10.0])])
    result = PipelineResult([frame], 25.0, (100, 100), ["player"])
    statsbomb = pd.DataFrame(
        {
            "period": [2],
            "minute": [45],
            "second": [1],
            "pitch_location": [[10, 10]],
            "type": ["player"],
        }
    )
    csv = result.export(tmp_path / "run", period=2) / "freeze_frames.csv"
    routes = [
        compare_with_statsbomb(result, statsbomb, period=2),
        compare_with_statsbomb(result.to_freeze_frames(period=2), statsbomb, period=2),
        compare_with_statsbomb(pd.read_csv(csv), statsbomb, period=2),
    ]
    for merged in routes:
        assert merged["euclidean_distance"].tolist() == pytest.approx([0.5])


def test_statsbomb_comparison_of_an_empty_result():
    empty = PipelineResult(
        [FrameResult(0, 0.0, sv.Detections.empty(), sv.Detections.empty())],
        25.0,
        (64, 48),
        [],
    )
    merged = compare_with_statsbomb(empty, _one_player_at([10, 20]))
    assert len(merged) == 1 and merged["euclidean_distance"].isna().all()
    assert "detected_location" in merged and "detected_player_id" in merged
