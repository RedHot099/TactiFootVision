"""ball-detection 10: the holdout read split (game 4 of GSR train) of scripts/detector_bench.py, ball_temporal.py and
ball_scorecard.py on a small synthetic GSR train: the clip list, the frozen homographies, the read (well placed,
phase 2's > 5 m barrier, intervals) and the card's split plumbing."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

FRAMES = [1, 2, 6, 11, 16, 21]  # 2 is not a stride-5 frame
SCALE = np.diag([0.1, 0.1, 1.0])  # 720p px -> corner metres: 10 px per metre
TRUTH = (50.0, 30.0)  # corner metres = 720p (500, 300) = 1080p (750, 450)


def gsr_labels(folder: Path, game: str, frames=FRAMES) -> None:
    """A minimal GSR ``Labels-GameState.json``: one ball per frame at ``TRUTH``, an image per frame."""
    (folder / "img1").mkdir(parents=True)
    for f in frames:
        (folder / "img1" / f"{f:06d}.jpg").write_bytes(f"{folder.name}/{f}".encode())
    labels = {
        "info": {"game_id": game},
        "categories": [{"id": 4, "name": "ball"}],
        "images": [{"image_id": f, "file_name": f"{f:06d}.jpg"} for f in frames],
        "annotations": [
            {
                "category_id": 4,
                "image_id": f,
                "bbox_pitch": {
                    "x_bottom_middle": TRUTH[0] - 52.5,
                    "y_bottom_middle": TRUTH[1] - 34.0,
                },
            }
            for f in frames
        ],
    }
    (folder / "Labels-GameState.json").write_text(json.dumps(labels))


def tracking(root: Path, sequence: str, game: str) -> None:
    folder = root / "train" / sequence
    (folder / "img1").mkdir(parents=True)
    (folder / "gameinfo.ini").write_text(
        f"[Sequence]\nname={sequence}\ngameID={game}\n"
    )


@pytest.fixture
def train(tmp_path, monkeypatch):
    """GSR train with two game-4 clips (SNGS-060, SNGS-061) and a game-6 clip, their Tracking twins, the bench and
    freeze record in tmp_path."""
    gsr = tmp_path / "gsr"
    for clip, game in (("SNGS-060", "4"), ("SNGS-061", "4"), ("SNGS-090", "6")):
        gsr_labels(gsr / "train" / clip, game)
        tracking(tmp_path / "tracking", clip.replace("SNGS", "SNMOT"), game)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"title": "frozen"}))
    monkeypatch.setattr(detector_bench, "GSR", gsr)
    monkeypatch.setattr(detector_bench, "TRACKING", tmp_path / "tracking")
    monkeypatch.setattr(detector_bench, "CHECKOUT", tmp_path)
    monkeypatch.setattr(detector_bench, "HOLDOUT_FILE", tmp_path / "holdout.json")
    monkeypatch.setattr(detector_bench, "BENCH", tmp_path / "bench")
    monkeypatch.setattr(detector_bench, "FREEZE", freeze)
    monkeypatch.setattr(bt, "FULL_FRAMES", {"train": tmp_path / "tracking" / "train"})

    class Stub:  # Dataset.from_gsr on the synthetic labels: one sample per stride-5 frame, one ball box
        @staticmethod
        def from_gsr(root, splits, stride):
            samples = []
            for labels in sorted(
                (Path(root) / splits[0]).glob("*/Labels-GameState.json")
            ):
                game = json.loads(labels.read_text())["info"]["game_id"]
                for f in FRAMES:
                    if (f - 1) % stride:
                        continue
                    annotations = tv.data.Annotations(
                        np.array([[10.0, 10.0, 16.0, 16.0]]), np.array([0])
                    )
                    meta = {
                        "sequence": labels.parent.name,
                        "game": game,
                        "frame": str(f),
                    }
                    samples.append(
                        tv.data.Sample(
                            labels.parent / "img1" / f"{f:06d}.jpg",
                            1920,
                            1080,
                            annotations,
                            meta,
                        )
                    )
            classes = ["ball", "goalkeeper", "player", "referee"]
            return tv.data.Dataset("detect", classes, {splits[0]: samples})

    monkeypatch.setattr(detector_bench, "Dataset", Stub)
    return tmp_path


def test_holdout_lists_the_game_4_clips_with_their_tracking_twins_and_never_changes(
    train,
):
    detector_bench.holdout(SimpleNamespace())
    record = json.loads(detector_bench.HOLDOUT_FILE.read_text())
    assert record["clips"] == ["SNGS-060", "SNGS-061"]
    assert record["tracking_twins"]["SNGS-061"].endswith("train/SNMOT-061")
    assert (
        record["counts"]["frames"] == 2 * 5 and record["counts"]["boxes"]["ball"] == 10
    )
    assert detector_bench.holdout_clips() == ["SNGS-060", "SNGS-061"]
    assert [
        p.parent.name for p in detector_bench.split_label_files("holdout")
    ] == record["clips"]
    assert [c.name for c in bt.clip_dirs("holdout")] == record["clips"]
    dataset = detector_bench.gsr_split("holdout")
    assert {s.meta["sequence"] for s in dataset["train"]} == {"SNGS-060", "SNGS-061"}
    detector_bench.check_manifest(dataset, "holdout")
    detector_bench.holdout(SimpleNamespace())  # the same list again: allowed
    gsr_labels(
        detector_bench.GSR / "train" / "SNGS-062", "4"
    )  # another game-4 clip appears
    tracking(train / "tracking", "SNMOT-062", "4")
    with pytest.raises(SystemExit, match="other clips"):
        detector_bench.holdout(SimpleNamespace())
    tracking(
        train / "tracking", "SNMOT-099", "4"
    )  # a Tracking sequence of game 4 without a GSR clip
    with pytest.raises(SystemExit, match="Tracking sequences"):
        detector_bench.holdout(SimpleNamespace())


def test_holdout_frames_at_25_fps_come_from_tracking_train_byte_checked(train):
    detector_bench.holdout(SimpleNamespace())
    clip = bt.clip_dirs("holdout")[0]
    assert bt.frames_dir(clip) == train / "tracking/train/SNMOT-060"
    with pytest.raises(SystemExit, match="25 fps source"):
        bt.check_frames(clip)  # the twin has no frames yet
    for path in (clip / "img1").glob("*.jpg"):
        (bt.frames_dir(clip) / "img1" / path.name).write_bytes(path.read_bytes())
    assert bt.check_frames(clip) == len(FRAMES)


def per_frame(path: Path, frames: dict[str, list[int]]) -> Path:
    """A pitch-keypoints run: per_frame.parquet (``SCALE`` on the given frames, no homography elsewhere) and its
    complete.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "clip": c,
            "frame": f,
            "method": detector_bench.HOMOGRAPHY_METHOD,
            "homography": h,
        }
        for c in ("SNGS-060", "SNGS-061")
        for f in FRAMES
        for h in [SCALE.ravel() if f in frames.get(c, []) else None]
    ]
    pd.DataFrame(rows).to_parquet(path)
    complete = {
        "files": {"per_frame.parquet": detector_bench.sha256_file(path)},
        "code": {"digest": "d"},
    } | {k: k for k in ("split", "model", "methods", "git", "sample")}
    (path.parent / "complete.json").write_text(json.dumps(complete))
    return path


def test_holdout_homographies_are_frozen_outside_the_manifest(train, monkeypatch):
    detector_bench.holdout(SimpleNamespace())
    source = per_frame(
        train / "pk/per_frame.parquet", {"SNGS-060": [1, 6, 11, 16], "SNGS-061": [1, 6]}
    )
    monkeypatch.setitem(detector_bench.HOMOGRAPHY_SOURCES, "holdout", source)
    with pytest.raises(SystemExit, match="no frozen holdout"):
        detector_bench.frozen_homographies("holdout")
    detector_bench.homographies(SimpleNamespace(split="holdout"))
    freeze = json.loads(detector_bench.FREEZE.read_text())
    record = freeze["placement_homographies_holdout"]
    assert freeze["title"] == "frozen" and "placement_homographies_test" not in freeze
    assert record["frames"] == 6 and record["clips"] == ["SNGS-060", "SNGS-061"]
    assert record["file"].endswith("gsr_holdout_homographies.parquet")
    assert record["source_run"]["code_digest"] == "d"
    found = detector_bench.frozen_homographies("holdout")
    assert sorted(found)[:2] == [("SNGS-060", 1), ("SNGS-060", 6)]
    per_frame(
        source, {"SNGS-060": [1]}
    )  # a changed source must not overwrite the record
    with pytest.raises(SystemExit, match="differs"):
        detector_bench.homographies(SimpleNamespace(split="holdout"))


def tracks_table(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"clip": c, "frame": f, "status": s}
            | dict(zip(bt.COLUMNS, [x - 5, y - 5, x + 5, y + 5], strict=True))
            for c, f, s, x, y in rows
        ]
    )


def test_read_holdout_well_placed_barrier_and_intervals(train, monkeypatch):
    """Two game-4 clips, 5 placement frames each (frame 2 is not a stride-5 frame; the game-6 clip is not read).
    SNGS-060 has homographies on 1-16, SNGS-061 on 1 and 6. 1080p (750, 450) is the truth; +150 px is 10 m off.
    Tracking: SNGS-060 well on 1, 6, 11, far on 16, 21 has no homography; SNGS-061 well on 1, no ball on 6.
    Reference (top-1): SNGS-060 well on 1, 6; SNGS-061 well on 1; no ball elsewhere."""
    detector_bench.holdout(SimpleNamespace())
    homographies = {("SNGS-060", f): SCALE for f in (1, 6, 11, 16)} | {
        ("SNGS-061", f): SCALE for f in (1, 6)
    }
    monkeypatch.setattr(
        detector_bench, "frozen_homographies", lambda split: homographies
    )
    on, off = (750, 450), (900, 450)
    tracks = tracks_table(
        [
            ("SNGS-060", 1, "measured", *on),
            ("SNGS-060", 6, "measured", *on),
            ("SNGS-060", 11, "interpolated", *on),
            ("SNGS-060", 16, "measured", *off),
            ("SNGS-060", 21, "measured", *on),
            ("SNGS-061", 1, "measured", *on),
        ]
        + [("SNGS-061", f, "none", np.nan, np.nan) for f in (6, 11, 16, 21)]
    )
    top1 = tracks_table(
        [("SNGS-060", f, "measured", *on) for f in (1, 6)]
        + [("SNGS-060", f, "none", np.nan, np.nan) for f in (11, 16, 21)]
        + [("SNGS-061", 1, "measured", *on)]
        + [("SNGS-061", f, "none", np.nan, np.nan) for f in (6, 11, 16, 21)]
    )
    found = bt.read_placement(tracks, top1, "holdout", bt.READS["holdout"])
    assert list(found) == ["holdout"]
    g = found["holdout"]
    assert g["games"] == ["4"]
    assert g["counts"] == {
        "frames": 10,
        "without_homography": 4,
        "near_goal_frames": 0,
        "clips": 2,
    }
    t, r, d = g["track"], g["top1"], g["track_vs_top1"]
    assert t["well_placed"]["value"] == pytest.approx(0.4)
    assert r["well_placed"]["value"] == pytest.approx(0.3)
    assert d["well_placed"]["difference"] == pytest.approx(0.1)
    assert t["ceiling"]["value"] == pytest.approx(0.6)
    low, high = d["well_placed"]["ci"]
    # two clips: a draw takes 060 twice (+1/5 per clip), 061 twice (0) or both (+1/10)
    assert low == pytest.approx(0.0) and high == pytest.approx(0.2)
    b = g["barrier"]
    # far of all frames: SNGS-060 1 of 5, SNGS-061 none: the draws give 0.2, 0.1 or 0
    assert b["variant"] == pytest.approx(0.1) and b["reference"] == pytest.approx(0.0)
    assert b["difference"] == pytest.approx(0.1) and b["ci"] == pytest.approx(
        [0.0, 0.2]
    )
    assert b["limit_pp"] == 1.0 and not b["met"]
    text = bt.read_markdown(found, "holdout")
    assert "| holdout | 0.00% / 10.00% | +10.00 [+0.00; +20.00] | NOT met |" in text
    same = bs.barrier(g["rows"], g["rows"], bs.Draws(g["rows"]["clip"]))
    assert same["difference"] == 0 and same["met"]


def test_card_reads_the_holdout_alone_from_holdout_dumps(monkeypatch):
    assert bs.SPLITS["holdout"] == ("4",)
    assert (
        bs.read_split("holdout") == "holdout" and bs.read_split("selection") == "valid"
    )
    bs.check_dump_split({"sets": {"gsr": {"split": "train"}}}, "train")
    bs.check_dump_split(
        {"sets": {"gsr": {}}}, "valid"
    )  # older dumps are GSR valid ones
    with pytest.raises(SystemExit, match="GSR valid images"):
        bs.check_dump_split({"sets": {"gsr": {"split": "valid"}}}, "train")
    with pytest.raises(SystemExit, match="alone"):
        bs.card(None, None, ["selection", "holdout"], Path("unused"), "1280")
    asked = []
    monkeypatch.setattr(
        detector_bench,
        "placement_rows",
        lambda top_ball, homographies, split: asked.append(split)
        or pd.DataFrame({"clip": ["SNGS-060"], "game": ["4"], "frame": [1]}),
    )
    images = pd.DataFrame(
        {
            "set": "gsr",
            "clip": ["SNGS-060"],
            "game": ["4"],
            "width": 1920,
            "height": 1080,
        }
    )
    rows = bs.placement_table({}, images, {}, ("4",), "holdout")
    assert asked == ["holdout"] and rows["clip"].tolist() == ["SNGS-060"]
    gain = {"difference": 0.02, "ci": [0.005, 0.04]}
    cards = {
        "holdout/1280": {
            "placement_vs_reference": {"well_placed": gain},
            "barrier": {"met": True},
        }
    }
    rule = bs.verdict({"cards": cards}, "1280")
    assert rule["holdout"] == gain | {"confirmed": True, "barrier": True}
    assert "primary" not in rule
