"""Pitch keypoint training and its records (pitch-keypoints 03): run lists, the square input, sessions, manifest."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from xgpoc.pitchkp import dataset as D
from xgpoc.pitchkp import evaluator as ev
from xgpoc.pitchkp import training as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def fake_dataset(root: Path, n_train: int = 5, n_val: int = 9) -> Path:
    for split, n in (("train", n_train), ("val", n_val), ("test_gsr", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            (root / "images" / split / f"{split}_{i:03d}.jpg").write_bytes(b"")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                                    "kpt_shape": [32, 3], "flip_idx": list(D.FLIP_IDX), "nc": 1,
                                                    "names": ["pitch"]}))
    return root


def test_training_yaml_lists_train_and_a_val_subset_and_never_the_test_splits(tmp_path):
    root = fake_dataset(tmp_path / "ds")
    data_yaml = T.training_yaml(root, tmp_path / "run", val_step=4)
    config = yaml.safe_load(data_yaml.read_text())
    assert config["flip_idx"] == list(D.FLIP_IDX) and config["kpt_shape"] == [32, 3]
    train = (tmp_path / "run" / config["train"]).read_text().split()
    val = (tmp_path / "run" / config["val"]).read_text().split()
    assert [Path(p).name for p in train] == [f"train_{i:03d}.jpg" for i in range(5)]
    assert [Path(p).name for p in val] == ["val_000.jpg", "val_004.jpg", "val_008.jpg"]
    assert {Path(p).parent.name for p in train + val} == {"train", "val"}
    limited = T.training_yaml(root, tmp_path / "run2", train_limit=2, seed=0)
    assert len((limited.parent / "train.txt").read_text().split()) == 2
    with pytest.raises(ValueError, match="val_step"):
        T.training_yaml(root, tmp_path / "run3", val_step=0)


def test_augmentation_keeps_one_pitch_per_image():
    assert T.YOLO_AUGMENT["mosaic"] == 0 and T.YOLO_AUGMENT["mixup"] == 0 and T.YOLO_AUGMENT["cutmix"] == 0
    assert T.YOLO_AUGMENT["copy_paste"] == 0 and T.YOLO_AUGMENT["flipud"] == 0 and T.YOLO_AUGMENT["fliplr"] > 0


def test_option_parses_values():
    import pk_train_yolo as run

    assert run.option("scale=0") == ("scale", 0)
    assert run.option("optimizer=AdamW") == ("optimizer", "AdamW")
    with pytest.raises(Exception, match="key=value"):
        run.option("scale")


def test_square_input_pads_the_frame_to_a_centred_square_and_maps_keypoints_back():
    seen = {}

    def model(canvas):
        seen["canvas"] = canvas
        ys, xs = np.nonzero(canvas[:, :, 0] == 255)                # the frame's marked pixel, in canvas pixels
        return sv_keypoints(np.float32([[[xs[0], ys[0]]] * 32]))

    image = np.zeros((720, 1280, 3), np.uint8)
    image[100, 200] = 255
    found = ev.SquareInput(model)(image)
    canvas = seen["canvas"]
    assert canvas.shape == (1280, 1280, 3) and (canvas[:280] == ev.PAD_VALUE).all() and (canvas[1000:] == 114).all()
    np.testing.assert_allclose(found.xy[0, 0], [200, 100])
    np.testing.assert_allclose(found.data["xyxy"][0], [0, -280, 1280, 1000])


def sv_keypoints(xy):
    import supervision as sv

    return sv.KeyPoints(xy=xy, confidence=np.ones(xy.shape[:2], np.float32), class_id=np.zeros(1, int),
                        data={"xyxy": np.float32([[0, 0, 1280, 1280]]), "box_confidence": np.float32([0.9])})


def test_sessions_are_appended_one_line_per_event(tmp_path):
    import json

    import pk_train_yolo as run

    run.log_session(tmp_path / "run", {"event": "start", "at": "t0"})
    run.log_session(tmp_path / "run", {"event": "end", "seconds": 5.0})
    lines = [json.loads(line) for line in (tmp_path / "run/sessions.jsonl").read_text().splitlines()]
    assert [line["event"] for line in lines] == ["start", "end"] and lines[1]["seconds"] == 5.0


def test_manifest_records_weights_data_config_and_gpu_time(tmp_path, monkeypatch):
    import argparse
    import json

    import pk_publish_model as publish

    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text(yaml.safe_dump({"imgsz": 960, "mosaic": 0.0}))
    (run / "sessions.jsonl").write_text("\n".join(json.dumps(e) for e in (
        {"event": "start"}, {"event": "end", "seconds": 100.0}, {"event": "start"}, {"event": "end", "seconds": 20.5})))
    record = tmp_path / "complete.json"
    record.write_text(json.dumps({"dataset": {"labels_digest": "abc"}}))
    monkeypatch.setattr(publish, "DATASET_RECORD", record)
    out = tmp_path / "m.pt"
    out.write_bytes(b"weights")
    settings = tmp_path / "chosen.json"
    settings.write_text(json.dumps({"kp03": {"min_confidence": 0.3}}))
    args = argparse.Namespace(run=run, out=out, input="square", imgsz=960, settings=settings, note="n")
    got = publish.manifest(args, run / "weights/last.pt")
    from xgpoc.bundle import sha256_file

    assert got["weights"]["sha256"] == sha256_file(out) and got["data"]["labels_digest"] == "abc"
    assert got["training"]["config"]["mosaic"] == 0.0 and got["training"]["gpu_seconds"] == 120.5
    assert got["inference"] == {"backend": "yolo_pose", "input": "square", "imgsz": 960,
                                "homography": {"kp03": {"min_confidence": 0.3}}}


def test_label_free_reference_must_cover_the_same_frames(tmp_path, monkeypatch):
    import pandas as pd
    import pk_label_free_loop as loop

    monkeypatch.setattr(loop.records, "verified_file", lambda path: "")
    pd.DataFrame({"shot_id": ["a", "a", "a", "b", "b", "b"],
                  "variant": ["stretch", "stretch_refined", "letterbox"] * 2}).to_parquet(tmp_path / "per_frame.parquet")
    rows = loop.reference_rows(tmp_path, {"a", "b"})
    assert sorted(set(rows.variant)) == ["current_kp05", "current_kp05_refined"] and len(rows) == 4
    with pytest.raises(SystemExit, match="other frames"):
        loop.reference_rows(tmp_path, {"a", "c"})


def test_cache_key_names_weights_input_size_and_threshold(tmp_path):
    from xgpoc.bundle import sha256_file

    (tmp_path / "w.pt").write_bytes(b"x")
    key = ev.KeypointModel(tmp_path / "w.pt", "square", 960, 0.1).cache_key()
    assert key == f"{sha256_file(tmp_path / 'w.pt')[:12]}_square_960_i0.1"
    assert ev.KeypointModel(tmp_path / "w.pt").cache_key().endswith("_stretch_train_i0.1")   # ticket 02's names
