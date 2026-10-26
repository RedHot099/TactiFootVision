"""ball-detection 13: the second stage on crops (scripts/ball_crops.py): the training view without game 4 and GSR
valid / test, the labels of council decision 3, the crops, the thinning of repeated mistakes, the fusion and the
rescored dump that ``ball_temporal`` tracks unchanged."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_crops as bc  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench  # noqa: E402

HOLDOUT = ["SNGS-060", "SNGS-061"]
VALID, TEST = ["SNGS-021"], ["SNGS-116"]


@pytest.fixture
def pools(tmp_path, monkeypatch):
    """Synthetic pool indexes: SoccerNet-v3 and TeamTrack images (one not in run 2c's list, one without a ball) and
    GSR frames of games 4 (holdout), 6, 9 (train) and 2 (valid)."""
    sets = tmp_path / "sets"
    pool = pd.DataFrame(
        {
            "set": ["soccernet_v3", "soccernet_v3", "teamtrack", "soccernet_v3"],
            "image": [
                str(sets / f"soccernet_v3/train/images/{n}.jpg") for n in ("a", "b")
            ]
            + [
                str(sets / "teamtrack/train/images/c.jpg"),
                str(sets / "soccernet_v3/train/images/unlisted.jpg"),
            ],
            "game": ["g1", "g1", "F_1", "g2"],
            "clip": ["x", "x", "y", "z"],
            "frame": pd.array([0, 1, 1, 0], dtype="Int64"),
            "width": [1280, 1280, 6500, 1920],
            "height": [720, 720, 1000, 1080],
            "n_ball": [1, 0, 1, 1],
        }
    )
    gsr_rows = []
    for clip, game, split in (
        (HOLDOUT[0], "4", "train"),
        ("SNGS-080", "6", "train"),
        ("SNGS-120", "9", "train"),
        (VALID[0], "2", "valid"),
    ):
        for frame in range(1, 12):
            gsr_rows.append(
                {
                    "image": str(
                        sets
                        / f"soccernet_gsr/yolo/{split}/images/{clip}_{frame:06d}.jpg"
                    ),
                    "label": str(
                        sets
                        / f"soccernet_gsr/yolo/{split}/labels/{clip}_{frame:06d}.txt"
                    ),
                    "game_id": f"gsr:{game}",
                    "clip": clip,
                    "frame": frame,
                    "gsr_split": split,
                    "width": 1920,
                    "height": 1080,
                    "n_ball": 1,
                }
            )
    gsr = pd.DataFrame(gsr_rows)
    pool.to_parquet(tmp_path / "pool.parquet")
    gsr.to_parquet(tmp_path / "pool_gsr.parquet")
    listed = list(pool.image[:3]) + list(gsr.image)
    (tmp_path / "train.txt").write_text("\n".join(listed) + "\n")
    monkeypatch.setattr(bc, "POOL", tmp_path / "pool.parquet")
    monkeypatch.setattr(bc, "POOL_GSR", tmp_path / "pool_gsr.parquet")
    monkeypatch.setattr(bc, "RUN2C_LIST", tmp_path / "train.txt")
    monkeypatch.setattr(detector_bench, "holdout_clips", lambda: list(HOLDOUT))
    monkeypatch.setattr(
        detector_bench,
        "split_label_files",
        lambda split: [
            Path(f"/gsr/{split}/{c}/Labels-GameState.json")
            for c in {"valid": VALID, "test": TEST}[split]
        ],
    )
    return tmp_path


def test_view_holds_no_game_4_and_no_gsr_valid_or_test(pools):
    view = bc.build_view()
    gsr = view[view.source == "soccernet_gsr"]
    assert set(gsr["clip"]) == {"SNGS-080", "SNGS-120"}
    assert not set(view["clip"]) & (set(HOLDOUT) | set(VALID) | set(TEST))
    assert not view.image.str.contains("SNGS-060|SNGS-021|/valid/|/test/").any()
    # roles: game 6 every frame trains, game 9 every 5th frame calibrates
    assert sorted(gsr[gsr.role == "train"].frame) == list(range(1, 12))
    assert sorted(gsr[gsr.role == "calibration"].frame) == [1, 6, 11]
    # SoccerNet-v3 / TeamTrack: only run 2c's listed images with a ball
    others = view[view.source != "soccernet_gsr"]
    assert sorted(Path(p).stem for p in others.image) == ["a", "c"]
    assert (others.role == "train").all()


@pytest.mark.parametrize(
    "change",
    [
        {"clip": HOLDOUT[1], "game": "gsr:6"},  # a holdout clip under another game id
        {"clip": "SNGS-080", "game": "gsr:4"},
        {"clip": VALID[0], "game": "gsr:6"},
        {"clip": TEST[0], "game": "gsr:9"},
        {"gsr_split": "valid"},
    ],
)
def test_check_view_raises_on_an_excluded_image(pools, change):
    view = bc.build_view()
    row = view[view.source == "soccernet_gsr"].index[0]
    for key, value in change.items():
        view.loc[row, key] = value
    if "clip" in change:
        view.loc[row, "image"] = f"/x/{change['clip']}_000001.jpg"
    with pytest.raises(SystemExit):
        bc.check_view(view)


def test_labels_positive_negative_and_ignored_band():
    truth = np.array([[100.0, 100.0, 110.0, 110.0]])  # half diagonal 7.07 px
    centres = np.array([[105.0, 105.0], [111.0, 105.0], [115.0, 105.0], [125.0, 105.0]])
    boxes = np.c_[centres - 4, centres + 4]
    m = bc.match(boxes, truth)
    np.testing.assert_allclose(m["d"], [0, 6, 10, 20])
    assert bc.label_of(m).tolist() == [1, 1, -1, 0]
    # without truth every candidate is a negative; a tiny truth ball's r is 3 px (x the image's scale)
    assert bc.label_of(bc.match(boxes, None)).tolist() == [0, 0, 0, 0]
    tiny = np.array([[0.0, 0.0, 2.0, 2.0]])
    assert bc.label_radius(tiny)[0] == 3.0
    assert bc.label_radius(tiny, bc.image_scale(3840, 2160))[0] == 6.0


def test_crop_side_and_extract():
    boxes = np.array([[0.0, 0.0, 8.0, 6.0], [0.0, 0.0, 20.0, 10.0]])
    np.testing.assert_allclose(bc.crop_side(boxes), [32.0, 60.0])
    np.testing.assert_allclose(bc.crop_side(boxes, 2.0), [64.0, 64.0])
    image = np.zeros((200, 300, 3), np.uint8)
    image[90:110, 140:160] = 255  # a white square centred at (150, 100)
    crops = bc.extract(
        image, np.array([[150.0, 100.0], [5.0, 5.0]]), np.array([40.0, 40.0])
    )
    size = bc.CROP["size"]
    assert crops.shape == (2, size, size, 3)
    middle = crops[0, size // 2 - 5 : size // 2 + 5, size // 2 - 5 : size // 2 + 5]
    assert middle.min() == 255 and crops[0, :5].max() == 0
    assert crops[1, : size // 2 - 15].max() == 0  # outside the image: black


def test_thin_chains_limits_a_repeated_mistake():
    # one static false ball on 40 consecutive frames + one jumping around (no chain)
    still = pd.DataFrame(
        {"clip": "c", "frame": np.arange(1, 41), "cx": 500.0, "cy": 300.0, "side": 32.0}
    )
    jumping = pd.DataFrame(
        {
            "clip": "c",
            "frame": np.arange(1, 41, 4),
            "cx": np.arange(10) * 200.0 + 50,
            "cy": 900.0,
            "side": 32.0,
        }
    )
    kept = bc.thin_chains(pd.concat([still, jumping], ignore_index=True))
    assert sorted(kept[kept.cy == 300].frame) == [1, 13, 25, 37]
    assert len(kept[kept.cy == 900]) == 10


def test_fusion_recovers_the_logistic_weights():
    rng = np.random.default_rng(0)
    p_crop, p_dd2c = rng.uniform(0.01, 0.99, (2, 20000))
    truth = {"a": 1.5, "b": 0.5, "c": -1.0}
    y = rng.uniform(size=20000) < bc.fused(truth, p_crop, p_dd2c)
    found = bc.fit_fusion(p_crop, p_dd2c, y.astype(int))
    for key, value in truth.items():
        assert abs(found[key] - value) < 0.1


def test_rescored_dump_is_tracked_by_ball_temporal(tmp_path, monkeypatch):
    """A rescored part keeps ``ball/04/dump25``'s columns and dtypes, other classes untouched, and
    ``ball_temporal`` reads it as a dump (``clip_candidates``, ``clip_result``)."""
    source = tmp_path / "dump25"
    (source / "parts").mkdir(parents=True)
    part = pd.DataFrame(
        {
            "clip": "SNGS-021",
            "frame": np.array([1, 1, 2], np.int16),
            "x1": np.array([100, 500, 104], np.float32),
            "y1": np.array([100, 300, 100], np.float32),
            "x2": np.array([108, 540, 112], np.float32),
            "y2": np.array([108, 400, 108], np.float32),
            "conf": np.array([0.2, 0.9, 0.05], np.float32),
            "class_id": np.array([0, 2, 0], np.int8),
        }
    )
    part.to_parquet(source / "parts/SNGS-021.parquet", index=False)
    pd.DataFrame({"clip": ["SNGS-021"] * 2, "game": "2", "frame": [1, 2]}).to_parquet(
        source / "frames.parquet", index=False
    )
    (source / "dump.json").write_text(json.dumps({"row": "1280", "conf": 0.01}))
    monkeypatch.setattr(bt, "DUMP", source)
    rescored = bc.rescore_clip(
        bt.clip_candidates("SNGS-021", source),
        lambda frame, boxes: np.full(len(boxes), 0.7 if frame == 1 else 0.3),
        lambda frame, boxes: boxes + 1,
    )
    out = tmp_path / "dump25_regress"
    bc.write_dump(out, {"SNGS-021": rescored}, {"ticket": "ball-detection 13"})
    back = bt.clip_candidates("SNGS-021", out)
    assert list(back.columns) == list(part.columns)
    assert (back.dtypes == part.dtypes).all()
    np.testing.assert_allclose(back.conf, [0.7, 0.9, 0.3])
    np.testing.assert_allclose(back.x1, [101, 500, 105])
    record = json.loads((out / "dump.json").read_text())
    assert record["ticket"] == "ball-detection 13" and record["frames"] == 2
    result = bt.clip_result(back, [np.eye(3), np.eye(3)])
    assert np.isclose(result.frames[0].ball.confidence[0], 0.7)
    assert np.isclose(result.frames[1].ball_candidates.confidence[0], 0.3)
