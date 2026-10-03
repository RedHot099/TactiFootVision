"""Package surface, registry, pitch geometry and the pipeline result contract."""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.models import Model
from tactifoot_vision.pipeline import NO_TEAM, FrameResult, ObjectMasks, PipelineResult
from tactifoot_vision.pitch import (
    HomographyEstimator,
    SoccerPitch,
    frame_to_pitch,
    pitch_to_frame,
)
from tactifoot_vision.registry import Registry


def test_import_is_lazy_and_light():
    code = (
        "import sys, tactifoot_vision as tv; "
        "heavy = [m for m in ('torch', 'ultralytics', 'rfdetr', 'transformers', 'matplotlib') "
        "if m in sys.modules]; print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_public_names_resolve():
    assert set(tv.__all__) == {"__version__", *tv._SUBMODULES, *tv._EXPORTS}
    for name in tv.__all__:
        assert getattr(tv, name) is not None
    with pytest.raises(AttributeError):
        tv.does_not_exist  # noqa: B018


def test_registry():
    registry: Registry[object] = Registry("thing")

    @registry.register("a")
    class A:
        def __init__(self, x=1):
            self.x = x

    assert registry.create("a", x=3).x == 3
    assert "a" in registry and registry.names() == ["a"]
    with pytest.raises(ValueError, match="Unknown thing 'b'"):
        registry.create("b")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(A)


def test_pitch_vertices_and_scaling():
    pitch = SoccerPitch()
    assert pitch.vertices.shape == (32, 2)
    np.testing.assert_allclose(pitch.vertices[[0, 29]], [[0, 0], [105, 68]])
    np.testing.assert_allclose(pitch.vertices[8], [11, 34])  # left penalty spot
    assert all(0 <= a < 32 and 0 <= b < 32 for a, b in pitch.edges)
    # Straight markings scale with the pitch; circle and arc points (30, 31 and
    # 10, 11, 18, 19 where the penalty arc meets the box) keep round arcs.
    statsbomb = SoccerPitch(120, 80)
    straight = [i for i in range(32) if i not in (10, 11, 18, 19, 30, 31)]
    np.testing.assert_allclose(
        statsbomb.vertices[straight] / [120, 80],
        pitch.vertices[straight] / [105, 68],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pitch.vertices[10], [16.5, 34 - (9.15**2 - 5.5**2) ** 0.5], atol=1e-4
    )
    # The centre circle meets the halfway line 9.15 m (in pitch units) from the centre.
    assert statsbomb.vertices[31, 0] - 60 == pytest.approx(120 * 9.15 / 105)
    assert statsbomb.vertices[15, 1] - 40 == pytest.approx(80 * 9.15 / 68)


@pytest.mark.parametrize("size", [(105, 68), (120, 80), (100, 64)])
def test_arc_landmarks_lie_on_the_drawn_arcs(size):
    from tactifoot_vision.viz.radar import pitch_markings

    pitch = SoccerPitch(*size)
    vertices = pitch.vertices
    lines, _ = pitch_markings(pitch)
    circle, left_arc, right_arc = lines[-3:]
    np.testing.assert_allclose(left_arc[[0, -1]], vertices[[10, 11]], atol=1e-4)
    np.testing.assert_allclose(right_arc[[0, -1]], vertices[[18, 19]], atol=1e-4)
    np.testing.assert_allclose(circle[0], vertices[31], atol=1e-4)
    assert left_arc[0, 0] == pytest.approx(pitch.penalty_box_length)


def _homography():
    # A pitch seen by a camera: pitch units -> frame pixels.
    src = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    dst = np.float32([[100, 600], [1800, 620], [1500, 200], [350, 190]])
    return cv2.getPerspectiveTransform(src, dst)


def _keypoints(pitch, to_frame, confidence=0.9, drop=()):
    xy = cv2.perspectiveTransform(pitch.vertices.reshape(-1, 1, 2), to_frame).reshape(
        1, -1, 2
    )
    conf = np.full((1, len(pitch.vertices)), confidence, np.float32)
    conf[0, list(drop)] = 0.0
    return sv.KeyPoints(xy=xy.astype(np.float32), confidence=conf)


def test_homography_recovers_pitch_coordinates():
    pitch = SoccerPitch()
    to_frame = _homography()
    estimator = HomographyEstimator(pitch)
    matrix = estimator.update(_keypoints(pitch, to_frame, drop=range(10)))
    assert matrix is not None and len(estimator.used_indices) == 22
    frame_points = cv2.perspectiveTransform(
        np.float32([[[52.5, 34]]]), to_frame
    ).reshape(1, 2)
    np.testing.assert_allclose(
        frame_to_pitch(frame_points, matrix), [[52.5, 34]], atol=1e-2
    )
    np.testing.assert_allclose(
        pitch_to_frame([[52.5, 34]], matrix), frame_points, atol=1e-2
    )


def test_homography_keeps_last_matrix_until_max_age():
    pitch = SoccerPitch()
    estimator = HomographyEstimator(pitch, max_age=2)
    good = estimator.update(_keypoints(pitch, _homography()))
    too_few = _keypoints(pitch, _homography(), drop=range(29))
    assert estimator.update(too_few) is good
    assert estimator.update(None) is good
    assert estimator.update(None) is None


def _result():
    detections = sv.Detections(
        xyxy=np.float32([[10, 10, 20, 40], [30, 10, 40, 40]]),
        confidence=np.float32([0.9, 0.8]),
        class_id=np.array([2, 1]),
        tracker_id=np.array([7, 8]),
        data={
            "class_name": np.array(["player", "goalkeeper"]),
            "pitch_xy": np.float32([[50, 30], [np.nan, np.nan]]),
            "team_id": np.array([1, NO_TEAM]),
        },
    )
    ball = sv.Detections(
        xyxy=np.float32([[5, 5, 8, 8]]),
        confidence=np.float32([0.5]),
        class_id=np.array([0]),
        data={"class_name": np.array(["ball"]), "pitch_xy": np.float32([[60, 20]])},
    )
    frame = FrameResult(
        index=4, timestamp=65.2, detections=detections, ball=ball, homography=np.eye(3)
    )
    empty = FrameResult(
        index=5,
        timestamp=65.24,
        detections=sv.Detections.empty(),
        ball=sv.Detections.empty(),
    )
    return PipelineResult(
        frames=[frame, empty],
        fps=25.0,
        frame_size=(100, 50),
        class_names=["ball", "goalkeeper", "player", "referee"],
    )


def test_result_tables(tmp_path):
    result = _result()
    table = result.to_dataframe()
    assert table["object"].tolist() == ["person", "person", "ball"]
    assert table["team_id"].tolist() == [1, NO_TEAM, NO_TEAM]
    assert np.isnan(table.loc[1, "pitch_x"])
    assert result.track_ids == [7, 8]
    assert result.frame(4).team_ids.tolist() == [1, NO_TEAM]
    with pytest.raises(KeyError):
        result.frame(99)
    assert result.to_csv(tmp_path / "t.csv").is_file()

    frames = result.to_freeze_frames(period=2, period_start=60.0)
    first = frames.iloc[0]
    assert (first["period"], first["minute"], first["second"]) == (2, 2, 5)
    assert first["timestamp"] == "00:02:05.200"
    assert json.loads(first["location"]) == [50, 30]
    assert frames["location"].iloc[1] is None  # NaN position is not exported
    # StatsBomb's flat [x1, y1, x2, y2, ...] polygon
    assert json.loads(first["visible_area"]) == [0, 0, 100, 0, 100, 50, 0, 50]
    assert frames["type"].tolist() == ["player", "goalkeeper", "ball"]


def test_freeze_frames_of_the_second_half_roll_over_the_minute():
    frames = [
        FrameResult(i, i / 25, sv.Detections.empty(), sv.Detections.empty())
        for i in (0, 1499, 1500)
    ]
    ball = sv.Detections(
        xyxy=np.float32([[0, 0, 2, 2]]),
        confidence=np.float32([0.9]),
        data={"class_name": np.array(["ball"]), "pitch_xy": np.float32([[1, 1]])},
    )
    for frame in frames:
        frame.ball = ball
    result = PipelineResult(frames, 25.0, (100, 50), ["ball"])
    table = result.to_freeze_frames(period=2, period_start=45 * 60)
    assert table[["minute", "second"]].values.tolist() == [[45, 0], [45, 59], [46, 0]]
    assert table["timestamp"].tolist() == [
        "00:45:00.000", "00:45:59.960", "00:46:00.000",
    ]  # fmt: skip
    assert (table["period"] == 2).all()


def test_result_save_load(tmp_path):
    path = _result().save(tmp_path / "result.pkl")
    loaded = PipelineResult.load(path)
    assert len(loaded) == 2 and loaded[0].detections.tracker_id.tolist() == [7, 8]


def test_homography_restarts_its_average_after_a_gap():
    pitch = SoccerPitch()
    estimator = HomographyEstimator(pitch, smoothing_window=3, max_age=10)
    before = _keypoints(pitch, _homography())
    for _ in range(3):
        estimator.update(before)
    estimator.update(None)  # a frame without keypoints
    panned = _homography() @ np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=float)
    matrix = estimator.update(_keypoints(pitch, panned))
    expected = np.linalg.inv(panned)
    np.testing.assert_allclose(matrix, expected / expected[2, 2], atol=1e-6)


def test_homography_rejects_a_model_with_the_wrong_number_of_keypoints():
    keypoints = sv.KeyPoints(
        xy=np.zeros((1, 5, 2), np.float32), confidence=np.ones((1, 5), np.float32)
    )
    with pytest.raises(ValueError, match="5 keypoints"):
        HomographyEstimator().update(keypoints)


def test_freeze_frame_clock_rounds_to_milliseconds():
    from tactifoot_vision.pipeline.result import _format_clock

    assert _format_clock(2.3) == "00:00:02.300"
    assert _format_clock(3599.9996) == "01:00:00.000"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_confidence": -0.1}, "min_confidence must be in"),
        ({"min_confidence": 1.5}, "min_confidence must be in"),
        ({"ransac_threshold": 0}, "ransac_threshold must be > 0"),
        ({"max_age": -1}, "max_age must be None or >= 0"),
        ({"smoothing_window": 0}, "smoothing_window must be >= 1"),
    ],
)
def test_homography_rejects_out_of_range_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HomographyEstimator(**kwargs)


def test_homography_accepts_the_edges_of_its_ranges():
    HomographyEstimator(min_confidence=0, max_age=None)
    HomographyEstimator(min_confidence=1, max_age=0, smoothing_window=1)


@pytest.mark.parametrize("size", [(0, 68), (105, -1)])
def test_pitch_rejects_a_non_positive_size(size):
    with pytest.raises(ValueError, match="must be > 0"):
        SoccerPitch(*size)


class _NoOpModel(Model):
    name = "no_op"
    task = Task.DETECT

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self):
        return []

    def predict(self, image):
        return sv.Detections.empty()


@pytest.mark.parametrize("conf", [-0.01, 1.01])
def test_model_rejects_conf_outside_zero_one(conf):
    with pytest.raises(ValueError, match="conf must be in"):
        _NoOpModel(conf=conf, device="cpu")
    assert _NoOpModel(conf=1.0, device="cpu").conf == 1.0


def test_object_masks_round_trip_through_crops():
    masks = np.zeros((3, 50, 100), dtype=bool)
    masks[0, 10:20, 30:35] = True
    masks[0, 25, 60] = True  # a second fragment widens the crop
    masks[2, 49, 99] = True  # touches the frame corner
    compact = ObjectMasks.from_dense(masks)
    assert [c.shape for c in compact.crops] == [(16, 31), (0, 0), (1, 1)]
    assert compact.origins.tolist() == [[30, 10], [0, 0], [99, 49]]
    np.testing.assert_array_equal(compact.to_dense(100, 50), masks)
