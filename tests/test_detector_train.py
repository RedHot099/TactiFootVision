"""detector-data 04: the training views of scripts/detector_train.py and the bench's set selection."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import detector_bench  # noqa: E402
import detector_train as dt  # noqa: E402

ALL = "ball,goalkeeper,player,referee"


def row(set_name, image, *, source=None, split="train", game=None, annotated=ALL):
    return {
        "set": set_name,
        "source": source or set_name,
        "split": split,
        "image": image,
        "game": game,
        "gsr_split": None,
        "near_dup_set": None,
        "annotated_classes": annotated,
        **{f"n_{c}": 1 for c in dt.CLASSES},
    }


def pool() -> pd.DataFrame:
    rows = [row(s, f"/{s}/a.jpg") for s in dt.RUN1_SETS if s != "soccernet_gsr"]
    for source in ("issia", "teamtrack", "soccer_player_detection"):
        rows = [r for r in rows if r["set"] != source]
        rows.append(row(source, f"/{source}/a.jpg", annotated="ball,player"))
    rows = [r for r in rows if r["set"] != "sportsmot_football"]
    rows.append(
        row("sportsmot_football", "/sm/a.jpg", source="sportsmot", annotated="player")
    )
    rows += [
        row("soccernet_gsr", "/gsr/train.jpg", game="gsr:4"),
        row("soccernet_gsr", "/gsr/valid.jpg", game="gsr:2", split="valid"),
        row("soccernet_v3", "/v3/dup.jpg"),
        row("soccersynth", "/synth/a.jpg", annotated="ball,player"),
        row("football_yolo", "/fy/train.jpg"),
        row("football_yolo", "/fy/valid.jpg", split="valid"),
        row("football_yolo", "/fy/dev.jpg", split="test"),
    ]
    frame = pd.DataFrame(rows)
    frame.loc[frame.image == "/v3/dup.jpg", "near_dup_set"] = "gsr_valid"
    for column in ("game", "gsr_split", "near_dup_set"):
        frame[column] = frame[column].astype("string")
    return frame


def test_run1_view_keeps_its_sets_and_drops_the_benchmark():
    images = set(dt.build_view(pool(), "run1").image)
    assert "/gsr/train.jpg" in images
    assert not images & {"/gsr/valid.jpg", "/v3/dup.jpg"}
    assert not any(i.startswith(("/synth", "/fy")) for i in images)


def test_run2_stages_read_football_yolo_without_the_dev_clips_then_soccersynth():
    stage_a = dt.build_view(pool(), "run2a")
    assert sorted(stage_a.image) == ["/fy/train.jpg", "/fy/valid.jpg"]
    stage_b = dt.build_view(pool(), "run2b")
    assert list(stage_b.image) == ["/synth/a.jpg"]
    assert dt.RUNS["run2b"]["after"] == "run2a"
    assert dt.RUNS["run2b"]["init"].parts[-4] == "run2a"
    for stage in ("run2a", "run2b"):
        assert dt.RUNS[stage]["settings"]["save_period"] == 1  # every epoch kept


def test_run2c_mixes_football_yolo_into_the_run1_data_without_the_dev_clips():
    run1 = set(dt.build_view(pool(), "run1").image)
    view = dt.build_view(pool(), "run2c")
    assert set(view.image) == run1 | {"/fy/train.jpg", "/fy/valid.jpg"}
    assert not any(i.startswith("/synth") for i in view.image)
    listed = dt.train_list(view, dt.RUNS["run2c"]["oversample"])
    assert listed.count("/fy/train.jpg") == 20
    assert listed.count("/gsr/train.jpg") == 1
    assert dt.RUNS["run2c"]["init"].parts[-4] == "run1"


def test_oversampling_repeats_paths_and_counts_images_per_epoch():
    view = dt.build_view(pool(), "run2a")
    listed = dt.train_list(view, {"football_yolo": 5})
    assert listed.count("/fy/train.jpg") == listed.count("/fy/valid.jpg") == 5
    assert listed == sorted(listed)
    stats = dt.counts(view, {"football_yolo": 5})
    assert stats["per_source"]["football_yolo"]["images"] == 2
    assert stats["images_per_epoch"] == 10
    assert dt.train_list(view, {}) == ["/fy/train.jpg", "/fy/valid.jpg"]


def test_views_check_that_any_person_sources_mask_roles():
    bad = pool()
    bad.loc[bad.set == "soccersynth", "annotated_classes"] = ALL
    dt.build_view(bad, "run1")  # soccersynth is not in run 1
    with pytest.raises(AssertionError, match="soccersynth"):
        dt.build_view(bad, "run2b")


def test_stage_init_falls_back_to_the_previous_smoke_only_in_smoke_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dt, "DATA", tmp_path)
    monkeypatch.setitem(
        dt.RUNS, "run2b", dt.RUNS["run2b"] | {"init": tmp_path / "missing/best.pt"}
    )
    with pytest.raises(FileNotFoundError, match="run run2a first"):
        dt.init_weights("run2b")
    with pytest.raises(FileNotFoundError):
        dt.init_weights("run2b", smoke=True)
    smoke = tmp_path / "runs/run2a/smoke/weights/best.pt"
    smoke.parent.mkdir(parents=True)
    smoke.write_bytes(b"x")
    assert dt.init_weights("run2b", smoke=True) == smoke
    with pytest.raises(FileNotFoundError):
        dt.init_weights("run2b")


def test_bench_epochs_lists_checkpoints_in_order_and_tabulates(tmp_path):
    for name in ("epoch10.pt", "epoch2.pt", "epoch0.pt", "best.pt", "epochx.pt"):
        (tmp_path / name).write_bytes(b"")
    assert [e for e, _ in detector_bench.epoch_checkpoints(tmp_path)] == [0, 2, 10]
    per_class = [{"class_name": c, "map50": 0.5} for c in dt.CLASSES]
    row = {"map50": 0.5, "map50_95": 0.3, "per_class": per_class}
    table = detector_bench.epoch_table(
        {2: {"gsr/1280": row}, 0: {"gsr/1280": row, "ekstraklasa/1280": row}}
    )
    assert table.epoch.tolist() == [0, 0, 2]
    assert "ap50_referee" in table.columns


def test_bench_gsr_every_matches_the_bench_file_order():
    from tactifoot_vision.data import Annotations, Dataset, Sample, Task

    empty = Annotations(np.zeros((0, 4)), np.zeros(0, int))
    samples = [
        Sample(Path(f"/{seq}/{f}.jpg"), 8, 8, empty, {"sequence": seq, "frame": str(f)})
        for seq in ("SNGS-022", "SNGS-021")
        for f in (11, 1, 6, 16)
    ]
    data = Dataset(Task.DETECT, dt.CLASSES, {"valid": samples})
    kept = detector_bench.gsr_every(data, 4)["valid"]
    assert [(s.meta["sequence"], s.meta["frame"]) for s in kept] == [
        ("SNGS-021", "1"),
        ("SNGS-022", "1"),
    ]


def test_bench_ekstraklasa_set_checks_its_labels_and_test_scores_gsr_only(
    tmp_path, monkeypatch
):
    labels = tmp_path / "dev/valid/labels"
    images = tmp_path / "dev/valid/images"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (labels / "a.txt").write_text("2 0.5 0.5 0.1 0.2\n")
    cv2.imwrite(str(images / "a.jpg"), np.zeros((72, 128, 3), np.uint8))
    (tmp_path / "dev/data.yaml").write_text(
        f"path: {tmp_path / 'dev'}\nval: valid/images\nnames: {dt.CLASSES}\n"
    )
    digest = hashlib.sha256(b"a.txt\n" + (labels / "a.txt").read_bytes()).hexdigest()
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"labels_sha256": digest}))
    monkeypatch.setattr(detector_bench, "EKSTRAKLASA_DEV", tmp_path / "dev")
    monkeypatch.setattr(detector_bench, "EKSTRAKLASA_JSON", record)
    monkeypatch.setattr(detector_bench, "gsr_split", lambda split: f"gsr-{split}")
    monkeypatch.setattr(detector_bench, "check_manifest", lambda dataset, split: None)

    sets = detector_bench.evaluation_sets("valid", ["ekstraklasa"])
    assert list(sets) == ["ekstraklasa"] and len(sets["ekstraklasa"][0]["valid"]) == 1
    assert list(detector_bench.evaluation_sets("test", None)) == ["gsr"]
    (labels / "a.txt").write_text("2 0.5 0.5 0.1 0.3\n")
    with pytest.raises(SystemExit, match="differ"):
        detector_bench.evaluation_sets("valid", ["ekstraklasa"])


def per_frame(path: Path, clip: str, frames: list[int]) -> Path:
    """A pitch-keypoints per_frame.parquet: frames of one clip, the kp03 method plus a decoy method."""
    rows = [
        {"clip": clip, "frame": f, "method": m, "homography": np.eye(3).ravel()}
        for f in frames
        for m in (detector_bench.HOMOGRAPHY_METHOD, "kp03_r10")
    ]
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_bench_placement_test_split_uses_frozen_test_homographies_and_test_clips(
    tmp_path, monkeypatch
):
    bench, freeze = tmp_path / "bench", tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"title": "frozen"}))
    source = per_frame(tmp_path / "per_frame.parquet", "SNGS-116", [1, 6])
    monkeypatch.setattr(detector_bench, "BENCH", bench)
    monkeypatch.setattr(detector_bench, "FREEZE", freeze)
    monkeypatch.setitem(detector_bench.HOMOGRAPHY_SOURCES, "test", source)
    with pytest.raises(SystemExit, match="no frozen test"):
        detector_bench.frozen_homographies("test")

    args = SimpleNamespace(split="test")
    detector_bench.homographies(args)
    record = json.loads(freeze.read_text())[detector_bench.TEST_HOMOGRAPHY_KEY]
    assert record["frames"] == 2 and record["file"].endswith(
        "gsr_test_homographies.parquet"
    )
    assert json.loads(freeze.read_text())["title"] == "frozen"
    found = detector_bench.frozen_homographies("test")
    assert sorted(found) == [("SNGS-116", 1), ("SNGS-116", 6)]
    detector_bench.homographies(args)  # same table again: allowed

    per_frame(source, "SNGS-116", [1])  # a changed source must not overwrite the record
    with pytest.raises(SystemExit, match="differs"):
        detector_bench.homographies(args)

    # placement_rows reads the split's clips: a test clip with a ball at the pitch centre, an identity homography
    clip = tmp_path / "gsr/test/SNGS-116"
    (clip / "img1").mkdir(parents=True)
    cv2.imwrite(str(clip / "img1/000001.jpg"), np.zeros((720, 1280, 3), np.uint8))
    pitch = {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}
    (clip / "Labels-GameState.json").write_text(
        json.dumps(
            {
                "info": {"game_id": 7},
                "categories": [{"id": 4, "name": "ball"}],
                "images": [{"image_id": "1", "file_name": "000001.jpg"}],
                "annotations": [
                    {"image_id": "1", "category_id": 4, "bbox_pitch": pitch}
                ],
            }
        )
    )
    monkeypatch.setattr(detector_bench, "GSR", tmp_path / "gsr")

    class Ball:
        def predict(self, frame):
            assert frame.shape == (720, 1280, 3)
            return sv.Detections(
                xyxy=np.array([[50.0, 32.0, 55.0, 36.0]]),
                confidence=np.array([0.9]),
                data={"class_name": np.array(["ball"])},
            )

    rows = detector_bench.placement_rows(
        detector_bench.model_top_ball(Ball()), found, "test"
    )
    assert rows["clip"].tolist() == ["SNGS-116"] and rows["game"].tolist() == ["7"]
    assert rows["error_m"].iloc[0] == pytest.approx(0.0)  # centre (52.5, 34) = truth
    assert detector_bench.placement_rows(
        detector_bench.model_top_ball(Ball()), found, "valid"
    ).empty
