"""ball-detection 04: the offline ball tracking of scripts/ball_temporal.py on a synthetic clip (a moving ball, a
spare ball behind the goal, a penalty spot and a gap, under a panning camera), the homographies carried by the camera
motion, and the scorecard dump encoding."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_temporal as bt  # noqa: E402

FRAMES = 60
GAP = range(30, 34)  # frames (0-based) in which the moving ball is not detected


def camera(t: int) -> np.ndarray:
    """pitch metres -> frame pixels at frame ``t``: 16 px per metre, panning 6 px per frame and zooming slowly."""
    zoom = 1 + 0.002 * t
    return np.array([[16 * zoom, 0, 100 - 6 * t], [0, 12 * zoom, 80], [0, 0, 1.0]])


def to_frame(t: int, xy) -> np.ndarray:
    p = camera(t) @ np.array([*xy, 1.0])
    return p[:2] / p[2]


def frame_to_pitch(t: int) -> np.ndarray:
    return np.linalg.inv(camera(t))


def motion() -> np.ndarray:
    return np.stack(
        [camera(t + 1) @ np.linalg.inv(camera(t)) for t in range(FRAMES - 1)]
    )


def ball_metres(t: int) -> tuple[float, float]:
    """The live ball: 15 m/s along the pitch, 5 m/s across."""
    return 60 + 0.6 * t, 20 + 0.2 * t


def live_conf(t: int) -> float:
    return 0.6 if t < 3 else 0.08


def box(centre, size: float = 8) -> list[float]:
    x, y = centre
    return [x - size / 2, y - size / 2, x + size / 2, y + size / 2]


def scene() -> pd.DataFrame:
    """Dump rows of the clip: the live ball (confident in the first frames, then weak: 0.08, below the operating
    point), a spare ball behind the goal line (0.9) and the right penalty spot (0.7) in every frame, and a player."""
    rows = []
    for t in range(FRAMES):
        balls = [
            (box(to_frame(t, (106.5, 30))), 0.9),
            (box(to_frame(t, (94, 34))), 0.7),
        ]
        if t not in GAP:
            balls.append((box(to_frame(t, ball_metres(t))), live_conf(t)))
        for xyxy, conf in balls:
            rows.append(
                {"frame": t + 1, "class_id": 0, "conf": conf}
                | dict(zip(bt.COLUMNS, xyxy, strict=True))
            )
        player = to_frame(t, (40, 50))
        rows.append(
            {"frame": t + 1, "class_id": 2, "conf": 0.9}
            | dict(
                zip(
                    bt.COLUMNS,
                    [player[0] - 10, player[1] - 40, player[0] + 10, player[1]],
                    strict=True,
                )
            )
        )
    table = pd.DataFrame(rows)
    table["conf"] = table.conf.astype(np.float32)
    return table


def anchors(every: int = 5) -> dict[int, np.ndarray]:
    return {t: frame_to_pitch(t) for t in range(0, FRAMES, every)}


def test_propagate_carries_anchors_by_the_motion_and_stops_at_breaks():
    found = bt.propagate(motion(), anchors(), reach=10)
    for t, matrix in enumerate(found):
        np.testing.assert_allclose(matrix, bt._normalise(frame_to_pitch(t)), atol=1e-9)
    broken = motion()
    broken[12] = np.nan  # frame 12 -> 13 unknown
    found = bt.propagate(
        broken, {10: frame_to_pitch(10), 20: frame_to_pitch(20)}, reach=10
    )
    np.testing.assert_allclose(
        found[12], bt._normalise(frame_to_pitch(12)), atol=1e-9
    )  # from 10 only
    np.testing.assert_allclose(
        found[13], bt._normalise(frame_to_pitch(13)), atol=1e-9
    )  # from 20 only
    np.testing.assert_allclose(
        found[0], bt._normalise(frame_to_pitch(0)), atol=1e-9
    )  # 10 frames: within reach
    assert found[31] is None and found[-1] is None  # beyond reach of any anchor
    assert bt.propagate(broken, {10: frame_to_pitch(10)}, reach=3)[14] is None


def test_tracking_follows_the_weak_moving_ball_past_a_spare_ball_and_a_spot_and_fills_the_gap():
    candidates = scene()
    result = bt.clip_result(
        candidates, bt.propagate(motion(), anchors()), frame_size=(1920, 1080)
    )
    tracked = bt.chosen_balls(bt.track_ball(result, **bt.BASE))
    top = bt.top_balls(candidates, FRAMES)

    assert (
        tracked.status
        == np.where(np.isin(np.arange(FRAMES), GAP), "interpolated", "measured")
    ).all()
    centres = bt.be.centres(tracked[bt.COLUMNS].to_numpy(float))
    truth = np.array([to_frame(t, ball_metres(t)) for t in range(FRAMES)])
    np.testing.assert_allclose(centres, truth, atol=0.5)
    measured = tracked[tracked.status == "measured"]
    np.testing.assert_allclose(
        measured.conf, [np.float32(live_conf(f - 1)) for f in measured.frame]
    )
    assert tracked.conf[tracked.status == "interpolated"].isna().all()
    # top-1 at the operating point is the spare ball in every frame
    spare = np.array([to_frame(t, (106.5, 30)) for t in range(FRAMES)])
    np.testing.assert_allclose(
        bt.be.centres(top[bt.COLUMNS].to_numpy(float)), spare, atol=1e-3
    )


def test_image_scores_hit_within_the_centre_rule_radius():
    truth = pd.DataFrame(
        {
            "clip": "c",
            "frame": [1, 2, 3],
            "tx1": 0.0,
            "ty1": 0.0,
            "tx2": 10.0,
            "ty2": 10.0,
            "radius": 10.0,
        }
    )
    tracks = pd.DataFrame(
        {
            "clip": "c",
            "frame": [1, 2, 3],
            "status": ["measured", "interpolated", "none"],
        }
        | {
            "x1": [9.0, 30.0, np.nan],
            "y1": [0.0, 0.0, np.nan],
            "x2": [19.0, 40.0, np.nan],
            "y2": [10.0, 10.0, np.nan],
        }
    )
    scored = bt.image_scores(tracks, truth)
    assert scored.hit.tolist() == [True, False, False]
    assert scored.chosen.tolist() == [True, True, False]
    summary = bt.score_summary(scored)
    assert summary["hit"] == pytest.approx(1 / 3) and summary[
        "wrong_of_chosen"
    ] == pytest.approx(0.5)


def reference_dump(folder: Path) -> None:
    """A two-frame ticket 01 dump: frame 1 has two balls and a player, frame 6 one ball; plus an Ekstraklasa image."""
    folder.mkdir()
    images = pd.DataFrame(
        {
            "set": ["gsr", "gsr", "ekstraklasa"],
            "image": ["C/000001", "C/000006", "e1"],
            "clip": ["C", "C", "e"],
            "game": ["2", "2", ""],
            "frame": [1, 6, -1],
            "path": "x",
            "width": 1920,
            "height": 1080,
        }
    )
    rows = [
        ("gsr", "1280", "C/000001", [10, 10, 20, 20], 0.6, 0),
        ("gsr", "1280", "C/000001", [100, 100, 110, 110], 0.05, 0),
        ("gsr", "1280", "C/000001", [300, 300, 330, 380], 0.9, 2),
        ("gsr", "e3", "C/000001", [10, 10, 20, 20], 0.6, 0),
        ("gsr", "1280", "C/000006", [10, 10, 20, 20], 0.4, 0),
        ("ekstraklasa", "1280", "e1", [5, 5, 15, 15], 0.7, 0),
    ]
    candidates = pd.DataFrame(
        [
            {"set": s, "row": r, "image": i}
            | dict(zip(bt.COLUMNS, b, strict=True))
            | {"conf": c, "class_id": k}
            for s, r, i, b, c, k in rows
        ]
    )
    candidates["class_name"] = np.array(bt.CLASSES)[candidates.class_id]
    for column in ("set", "row", "class_name"):
        candidates[column] = candidates[column].astype("category")
    candidates["conf"] = candidates.conf.astype(np.float32)
    candidates.to_parquet(folder / "candidates.parquet", index=False)
    images.to_parquet(folder / "images.parquet", index=False)
    record = {
        "conf": 0.001,
        "rows": {"1280": {"imgsz": 1280, "iou": 0.7, "input_size": None}, "e3": {}},
        "seconds": {"gsr/1280": 1.0, "gsr/e3": 2.0},
        "ms_per_frame": {"gsr/1280": 19.6, "gsr/e3": 40.2},
        "max_det": 300,
        "classes": dict(enumerate(bt.CLASSES)),
    }
    (folder / "dump.json").write_text(json.dumps(record))


def test_encode_makes_the_chosen_ball_the_single_ball_at_the_operating_point(tmp_path):
    reference_dump(tmp_path / "ref")
    tracks = pd.DataFrame(
        {
            "clip": "C",
            "frame": [1, 6],
            "status": ["measured", "interpolated"],
            "x1": [100.0, 50.0],
            "y1": [100.0, 50.0],
            "x2": [110.0, 60.0],
            "y2": [110.0, 60.0],
            "conf": [0.05, np.nan],
        }
    )
    bt.encode(tracks, tmp_path / "ref", tmp_path / "out", {"setting": {}}, ["2"])
    record, _, candidates = bt.bench.read_dump(tmp_path / "out")
    assert set(candidates["row"]) == {"1280"} and list(record["rows"]) == ["1280"]
    assert record["ms_per_frame"] == {"gsr/1280": 19.6}
    balls = candidates[(candidates.class_name == "ball") & (candidates["set"] == "gsr")]
    kept = balls[balls.conf >= 0.15]
    assert kept.image.tolist() == ["C/000001", "C/000006"]
    assert kept[bt.COLUMNS].to_numpy().tolist() == [
        [100, 100, 110, 110],
        [50, 50, 60, 60],
    ]
    assert (kept.conf == np.float32(0.15)).all()
    assert len(balls) == 4  # the interpolated ball is added, nothing removed
    assert balls.conf.max() < 0.16
    # the former top-1 of frame 1 is capped below the operating point, people and Ekstraklasa untouched
    assert balls[balls.x1 == 10].conf.max() == bt.CAP
    assert candidates[candidates.class_name == "player"].conf.tolist() == [
        np.float32(0.9)
    ]
    assert candidates[candidates["set"] == "ekstraklasa"].conf.tolist() == [
        np.float32(0.7)
    ]
    assert record["ball-detection 04"]["added_balls"] == 1
