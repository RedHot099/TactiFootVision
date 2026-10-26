"""ball-detection 14: the views of the repaired fine-tune (scripts/ball_recipe.py): game 4 out of A, B = A plus the
hard negatives only, the crops (whole referee in, no ball), the black-kit rule and the false balls on referees."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_recipe as br  # noqa: E402
import detector_bench as bench  # noqa: E402

import test_ball_view as tbv  # noqa: E402

GAME4 = {"SNGS-060", "SNGS-061", "SNMOT-060", "SNMOT-061"}


def pool_with_game6() -> pd.DataFrame:
    """test_ball_view's pool (game 4 = SNGS-060/061, a valid and a test frame) plus a game 6 clip and a game 4 twin
    listed under another source."""
    rows = [
        tbv.row(
            "soccernet_gsr",
            f"/gsr/SNGS-097_{f:06d}.jpg",
            game="gsr:6",
            clip="SNGS-097",
            frame=f,
            gsr_split="train",
        )
        for f in range(1, 6)
    ]
    rows.append(tbv.row("soccernet_v3", "/v3/SNMOT-060_000001.jpg"))
    extra = pd.DataFrame(rows)
    frame = pd.concat([tbv.pool(5), extra], ignore_index=True)
    for column in ("game", "clip", "gsr_split", "near_dup_set"):
        frame[column] = frame[column].astype("string")
    frame["frame"] = frame["frame"].astype("Int64")
    return frame


def test_view_a_drops_game_4_and_keeps_game_6():
    view = br.view_a(pool_with_game6(), GAME4)
    assert not view.game.eq("gsr:4").any()
    assert not view.image.str.contains("SNGS-06|SNMOT-06").any()
    assert view.game.eq("gsr:6").sum() == 5
    assert not set(view.image) & {"/gsr/valid.jpg", "/gsr/test.jpg", "/fy/dev.jpg"}


def test_a_must_be_run2c_without_game_4():
    run2c = sorted(
        [
            "/gsr/SNGS-060_000001.jpg",
            "/gsr/SNGS-097_000001.jpg",
            "/v3/a.jpg",
            "/fy/t.jpg",
            "/fy/t.jpg",
        ]
    )
    a = ["/fy/t.jpg", "/fy/t.jpg", "/gsr/SNGS-097_000001.jpg", "/v3/a.jpg"]
    br.check_a(a, run2c, GAME4)
    with pytest.raises(SystemExit):
        br.check_a(a + ["/gsr/SNGS-060_000001.jpg"], run2c, GAME4)
    with pytest.raises(SystemExit):
        br.check_a(a[1:], run2c, GAME4)  # a duplicate line lost


def test_b_is_a_plus_the_negatives_and_lists_are_checked(monkeypatch):
    a = ["/v3/a.jpg", "/fy/t.jpg", "/fy/t.jpg"]
    crops = ["/neg/images/x_k0.jpg", "/neg/images/x_k1.jpg"]
    lists = br.arm_lists(a, crops)
    assert lists["bd14_A"] == sorted(a)
    assert sorted(set(lists["bd14_B"]) - set(lists["bd14_A"])) == crops
    assert len(lists["bd14_B"]) == len(a) + len(crops)
    monkeypatch.setattr(
        bench,
        "split_label_files",
        lambda split: [
            Path(
                f"/gsr/{split}/{'SNGS-021' if split == 'valid' else 'SNGS-116'}/Labels-GameState.json"
            )
        ],
    )
    br.check_view(lists["bd14_B"], GAME4)
    for bad in (
        "/x/SNGS-060_000001.jpg",
        "/x/SNMOT-061_000001.jpg",
        "/x/SNGS-021_000001.jpg",
        "/x/SNGS-116_1.jpg",
    ):
        with pytest.raises(SystemExit):
            br.check_view(lists["bd14_B"] + [bad], GAME4)
    br.check_view(
        ["/x/SNGS-021_000001.jpg"], GAME4, ("test",)
    )  # the validation list may hold games 2+3


def test_crop_holds_the_referee_and_no_ball():
    box = np.array([900.0, 400.0, 940.0, 500.0])
    balls = np.array([[1500.0, 600.0, 1510.0, 610.0]])
    for k in range(20):
        rng = np.random.default_rng([0, k])
        x0, y0, w, h = br.crop_window(box, balls, 1920, 1080, rng)
        assert abs(w / h - 1920 / 1080) < 0.01
        assert 0.5 * 1920 - 1 <= w <= 0.8 * 1920 + 1
        assert x0 <= box[0] and y0 <= box[1] and x0 + w >= box[2] and y0 + h >= box[3]
        assert x0 >= 0 and y0 >= 0 and x0 + w <= 1920 and y0 + h <= 1080
        assert not (
            balls[0, 0] < x0 + w
            and balls[0, 2] > x0
            and balls[0, 1] < y0 + h
            and balls[0, 3] > y0
        )
    # a ball right next to the referee cannot be avoided: no crop
    touching = np.array([[941.0, 450.0, 950.0, 459.0]])
    assert br.crop_window(box, touching, 1920, 1080, np.random.default_rng(0)) is None
    # a referee box taller than the image's height / margin: no crop
    assert (
        br.crop_window(
            np.array([0.0, 0.0, 100.0, 1000.0]),
            np.zeros((0, 4)),
            1920,
            1080,
            np.random.default_rng(0),
        )
        is None
    )


def test_crop_labels_drop_balls_and_clip_people():
    classes = np.array([0, 3, 2, 2])
    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],  # ball: never labelled in a crop
            [100.0, 100.0, 140.0, 200.0],  # referee, inside
            [
                -50.0,
                100.0,
                10.0,
                200.0,
            ],  # player, 10 of 60 px wide inside: 1/6 of its area
            [
                195.0,
                100.0,
                260.0,
                200.0,
            ],  # player, 5 of 65 px inside: < 0.1 of its area
        ]
    )
    rows = br.crop_labels(classes, boxes, (0, 0, 200, 300))
    assert rows[:, 0].tolist() == [3, 2]
    np.testing.assert_allclose(rows[0, 1:], [120 / 200, 150 / 300, 40 / 200, 100 / 300])
    np.testing.assert_allclose(rows[1, 1:], [5 / 200, 150 / 300, 10 / 200, 100 / 300])


def test_black_kit_rule():
    import cv2

    image = np.full((200, 200, 3), (40, 160, 40), np.uint8)  # green pitch (BGR)
    image[50:150, 50:90] = (20, 20, 20)  # a black referee
    image[50:150, 120:160] = (0, 220, 255)  # a yellow one
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black = br.torso_colour(hsv, np.array([50.0, 50.0, 90.0, 150.0]))
    yellow = br.torso_colour(hsv, np.array([120.0, 50.0, 160.0, 150.0]))
    assert br.is_dark(*black, 100.0)
    assert not br.is_dark(*yellow, 100.0)
    assert not br.is_dark(*black, 10.0)  # too small to judge


def labels_json(path: Path) -> None:
    def ann(image_id, x, y, w, h, category, role=None):
        item = {
            "supercategory": "object",
            "image_id": image_id,
            "category_id": category,
            "track_id": 1,
            "bbox_image": {"x_center": x, "y_center": y, "w": w, "h": h},
            "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
            "attributes": {"role": role},
        }
        return item

    data = {
        "info": {"game_id": "3"},
        "images": [{"image_id": i, "file_name": f"{i:06d}.jpg"} for i in (1, 2)],
        "categories": [{"id": 4, "name": "ball"}, {"id": 1, "name": "person"}],
        "annotations": [
            ann(1, 100, 100, 10, 10, 4),
            ann(1, 500, 500, 40, 100, 1, "referee"),
            ann(1, 800, 500, 40, 100, 1, "player"),
            ann(2, 500, 500, 40, 100, 1, "referee"),
        ],
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))


def test_false_balls_on_referees(tmp_path, monkeypatch):
    labels_json(tmp_path / "valid/SNGS-030/Labels-GameState.json")
    monkeypatch.setattr(bench, "GSR", tmp_path)

    def cand(frame, cx, cy, conf, side=10):
        return {
            "frame": frame,
            "x1": cx - side / 2,
            "y1": cy - side / 2,
            "x2": cx + side / 2,
            "y2": cy + side / 2,
            "conf": conf,
            "class_id": 0,
        }

    candidates = pd.DataFrame(
        [
            cand(1, 101, 101, 0.9),  # the true ball
            cand(1, 500, 480, 0.5),  # on the referee
            cand(1, 800, 480, 0.5),  # on a player: false, not on a referee
            cand(1, 500, 520, 0.1),  # on the referee, under 0.15
            cand(2, 505, 500, 0.3),  # frame 2 has no ball: false, on the referee
        ]
    )
    table = br.referee_false_balls(candidates, "SNGS-030")
    assert table.set_index("frame")[["false", "on_referee"]].to_dict("index") == {
        1: {"false": 2, "on_referee": 1},
        2: {"false": 1, "on_referee": 1},
    }
