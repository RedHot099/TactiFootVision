"""ball-detection 01: the ball matching rules, feature bins and dump helpers of scripts/ball_errors.py, on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_errors as be  # noqa: E402
import detector_bench  # noqa: E402

BALL = np.array([[100.0, 100.0, 110.0, 110.0]])  # a 10 px ball centred on (105, 105)


def shifted(dx: float, dy: float = 0.0, grow: float = 0.0) -> np.ndarray:
    return BALL + [dx - grow, dy - grow, dx + grow, dy + grow]


def test_radius_is_the_diameter_with_a_5_px_floor():
    boxes = np.array([[0, 0, 10, 14], [0, 0, 3, 3]], float)
    assert be.diameter(boxes).tolist() == [12.0, 3.0]
    assert be.radius(boxes).tolist() == [12.0, 5.0]


@pytest.mark.parametrize(
    ("prediction", "iou_hit", "centre_hit"),
    [
        (shifted(0), True, True),
        (shifted(3), True, True),  # IoU 70 / 130 = 0.54
        (shifted(4), False, True),  # IoU 0.43: a bad box, the centre is 4 px off
        (shifted(10), False, True),  # centre exactly r = 10 px off
        (shifted(10.5), False, False),
        (
            shifted(0, grow=4),
            False,
            True,
        ),  # a box twice too big around the right centre
    ],
)
def test_iou_and_centre_rules(prediction, iou_hit, centre_hit):
    conf = np.array([0.9])
    assert (be.match(prediction, conf, BALL, "iou") >= 0)[0] == iou_hit
    assert (be.match(prediction, conf, BALL, "centre") >= 0)[0] == centre_hit


def test_the_centre_radius_has_a_5_px_floor_for_tiny_balls():
    tiny = np.array([[100.0, 100.0, 103.0, 103.0]])
    near = tiny + [4.5, 0, 4.5, 0]
    far = tiny + [5.5, 0, 5.5, 0]
    assert be.match(near, [0.5], tiny, "centre")[0] == 0
    assert be.match(far, [0.5], tiny, "centre")[0] == -1


def test_matching_is_greedy_most_confident_first_and_one_to_one():
    truth = np.vstack([BALL, BALL + [40, 0, 40, 0]])
    # The weaker prediction sits on ball 0; the stronger one is between both, nearer ball 0, so it takes ball 0.
    predictions = np.vstack([BALL, BALL + [8, 0, 8, 0]])
    conf = np.array([0.2, 0.9])
    assert be.match(predictions, conf, truth, "centre").tolist() == [1, -1]
    # Two predictions on one ball: only one hit, the other is a false ball.
    doubles = np.vstack([BALL, BALL + [1, 0, 1, 0]])
    assert be.match(doubles, [0.3, 0.6], BALL, "iou").tolist() == [1]
    assert be.false_balls(doubles, np.array([0.3, 0.6]), BALL).tolist() == [0]


def test_iou_rule_agrees_with_evaluate_detector_counts():
    rng = np.random.default_rng(0)
    from tactifoot_vision.evaluation import detection

    for _ in range(50):
        truth = rng.uniform(0, 200, (3, 2))
        truth = np.hstack([truth, truth + rng.uniform(6, 20, (3, 2))])
        noise = rng.normal(0, 3, (6, 4))
        predictions = np.vstack([truth, truth]) + noise
        predictions[:, 2:] = np.maximum(predictions[:, 2:], predictions[:, :2] + 1)
        conf = rng.uniform(0, 1, 6)
        counts = detection._match_counts(
            sv.Detections(
                predictions.astype(np.float32),
                confidence=conf.astype(np.float32),
                class_id=np.zeros(6, int),
            ),
            sv.Detections(truth.astype(np.float32), class_id=np.zeros(3, int)),
            np.array([0.15]),
            0.5,
        )
        found = be.hits(
            predictions.astype(np.float32),
            conf.astype(np.float32),
            truth.astype(np.float32),
            (0.15,),
        )
        assert found["iou@0.15"].sum() == counts[0, 0]


def test_hits_drop_predictions_under_the_threshold():
    found = be.hits(BALL, np.array([0.1]), BALL, (0.05, 0.15))
    assert found["iou@0.05"].tolist() == [True]
    assert found["centre@0.15"].tolist() == [False]


def test_size_bins_are_left_closed():
    levels = be.size_bin([7.9, 8, 11.99, 12, 16, 24, 40])
    assert list(levels.astype(str)) == [
        "<8",
        "8-12",
        "8-12",
        "12-16",
        "16-24",
        ">=24",
        ">=24",
    ]


def test_coverage_by_person_boxes():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert be.coverage(box, np.zeros((0, 4))) == 0.0
    assert be.coverage(box, [[-5, -5, 20, 20]]) == 1.0
    assert be.coverage(box, [[5, -5, 20, 20]]) == pytest.approx(0.5)
    assert be.coverage(box, [[5, -5, 20, 20], [-5, 5, 20, 20]]) == pytest.approx(0.75)


def test_track_motion_speed_and_deviation():
    frames = np.arange(1, 32)
    straight = pd.DataFrame(
        {
            "clip": "c",
            "track_id": 1,
            "frame": frames,
            "cx": 3.0 * frames,
            "cy": 0.0,
            "pitch_x": 0.5 * frames,
            "pitch_y": 0.0,
        }
    )
    moved = be.track_motion(straight).set_index("frame")
    assert moved.speed_px.tolist() == pytest.approx([3.0] * 31)  # one-sided at the ends
    assert moved.pitch_speed[15] == pytest.approx(12.5)
    assert moved.deviation_m[16] == pytest.approx(0.0)
    assert np.isnan(moved.deviation_m[5])  # frame 5 - span is not in the track
    lob = straight.assign(pitch_y=-0.05 * (frames - 16.0) ** 2)
    assert be.track_motion(lob).set_index("frame").deviation_m[16] == pytest.approx(5.0)
    gap = straight[straight.frame != 11]
    assert be.track_motion(gap).set_index("frame").speed_px[10] == pytest.approx(3.0)


def table(**columns) -> pd.DataFrame:
    base = pd.DataFrame(columns)
    base["x1"], base["y1"] = 0.0, 0.0
    base["x2"] = base.pop("d")
    base["y2"] = base["x2"]
    return be.add_box_features(base)


def test_feature_levels():
    balls = table(
        d=[6.0, 10.0, 14.0, 20.0],
        laplacian=[1.0, 2.0, 3.0, 4.0],
        coverage=[0.0, 0.2, 0.5, 0.9],
        speed_px=[6.0, 5.0, np.nan, 40.0],
        feet_m=[0.5, 1.5, 3.0, np.nan],
        deviation_m=[3.0, 3.0, 3.0, 0.0],
        pitch_x=[50.0, 0.0, -60.0, np.nan],
        pitch_y=[-30.0, 0.0, 20.0, np.nan],
    )
    assert balls["size"].astype(str).tolist() == ["<8", "8-12", "12-16", "16-24"]
    assert balls.blur.tolist() == ["blur", "sharp", "unknown", "blur"]
    assert balls.occlusion.astype(str).tolist() == ["none", "<0.5", ">=0.5", ">=0.5"]
    assert balls.contact.tolist() == ["<1 m", "1-2 m", ">=2 m", "unknown"]
    assert balls.in_air.tolist() == ["ground", "ground", "in air", "unknown"]
    assert balls.camera.tolist() == [
        "far third",
        "middle third",
        "near third",
        "unknown",
    ]
    assert balls.off_pitch.tolist() == ["on", "on", "off", "unknown"]
    assert balls.near_goal.tolist() == ["<=30 m", ">30 m", "<=30 m", "unknown"]


def test_recall_table_and_miss_shares():
    balls = table(
        d=[6.0, 6.0, 14.0, 14.0],
        laplacian=[1.0, 2.0, 3.0, 4.0],
        coverage=[0.0] * 4,
    )
    balls["clip"] = ["a", "a", "b", "b"]
    balls["iou@0.15"] = [False, False, True, False]
    balls["centre@0.15"] = [True, False, True, True]
    recall = be.recall_table(balls, features=("size",)).set_index("level")
    assert recall.loc["<8", "recall_iou"] == 0.0
    assert recall.loc["12-16", "recall_centre"] == 1.0
    assert recall.loc["<8", "miss_share_iou"] == pytest.approx(2 / 3)
    assert be.bad_box(balls) == {
        "iou_misses": 3,
        "bad_box_only": 2,
        "share": pytest.approx(2 / 3),
        "centre_hit_but_iou_miss_of_all_balls": 0.5,
    }
    shares = be.miss_shares(balls)
    assert shares["iou: size<12"]["share"] == pytest.approx(2 / 3)


def test_intervals():
    low, high = be.wilson(5, 10)
    assert low < 0.5 < high
    assert be.clip_bootstrap(
        np.array(["a", "b"]), np.array([1, 1]), np.array([1, 1])
    ) == (1.0, 1.0)


def test_dump_helpers():
    assert detector_bench.ekstraklasa_clip(
        "-18_25-18_45-IMPORTAN-1-polowa-zaglebie-lech-_mp4-0000_jpg"
    ) == ("18_25-18_45-IMPORTAN-1-polowa-zaglebie-lech")
    assert detector_bench.ekstraklasa_clip(
        "-3_02-3_41-Akcja-Lech-Poznan-Kauno-Zalgiris_mp4-0014_jpg"
    ) == ("3_02-3_41-Akcja-Lech-Poznan-Kauno-Zalgiris")
    candidates = pd.DataFrame(
        {
            "image": ["SNGS-001/000001"] * 3,
            "x1": [30.0, 300.0, 600.0],
            "y1": [30.0, 300.0, 600.0],
            "x2": [45.0, 315.0, 615.0],
            "y2": [45.0, 315.0, 615.0],
            "conf": np.array([0.9, 0.5, 0.1], np.float32),
            "class_id": np.array([2, 0, 0], np.int8),
            "class_name": pd.Categorical(["player", "ball", "ball"]),
        }
    )
    found = detector_bench.dump_detections(candidates)
    top = detector_bench.dump_top_ball(found, (1.5, 1.5))
    assert top("SNGS-001", 1, Path("unused")).tolist() == pytest.approx([205.0, 205.0])
    assert top("SNGS-001", 6, Path("unused")) is None
