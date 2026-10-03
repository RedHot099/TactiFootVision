import json

import cv2
import numpy as np
import pytest
import yaml

from tactifoot_vision.data import (
    Annotations,
    Dataset,
    Sample,
    Task,
    VideoReader,
    extract_frames,
    load_dataset,
    load_statsbomb,
)

FLIP = (1, 0, 2)


def _image(path, width=64, height=48, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((height, width, 3), value, np.uint8))
    return path


def _write_yolo(root, pose=False):
    """Two train images and one valid image in Roboflow's layout."""
    lines = {
        "train/images/a.jpg": ["0 0.5 0.5 0.25 0.5", "1 0.25 0.25 0.1 0.1"],
        "train/images/b.jpg": [],  # background image without a label file
        "valid/images/c.jpg": ["1 0.75 0.5 0.2 0.4"],
    }
    for rel, rows in lines.items():
        _image(root / rel)
        if pose:
            rows = [r + " 0.5 0.5 2 0.6 0.4 1 0 0 0" for r in rows]
        if rows:
            label = root / rel.replace("images", "labels").replace(".jpg", ".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(rows) + "\n")
    config = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../valid/images",
        "nc": 2,
        "names": ["ball", "player"],
    }
    if pose:
        config |= {"kpt_shape": [3, 3], "flip_idx": list(FLIP)}
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return root / "data.yaml"


def _dataset(tmp_path, n=10):
    samples = [
        Sample(
            _image(tmp_path / f"img{i}.jpg"),
            64,
            48,
            Annotations(boxes=[[1, 2, 10, 20]], class_ids=[i % 2]),
        )
        for i in range(n)
    ]
    return Dataset(
        Task.DETECT, ["ball", "player"], {"train": samples[:8], "valid": samples[8:]}
    )


def test_annotations_validate_shapes():
    with pytest.raises(ValueError):
        Annotations(boxes=np.zeros((2, 4)), class_ids=[0])
    with pytest.raises(ValueError):
        Annotations(boxes=np.zeros((1, 4)), class_ids=[0], keypoints=np.zeros((1, 3)))
    empty = Annotations.empty(num_keypoints=4)
    assert len(empty) == 0 and empty.keypoints.shape == (0, 4, 3)


def test_annotations_convert_to_supervision():
    ann = Annotations(
        boxes=[[0, 0, 10, 10]], class_ids=[1], keypoints=[[[1, 2, 2], [3, 4, 0]]]
    )
    detections = ann.to_detections(["ball", "player"])
    assert detections.data["class_name"].tolist() == ["player"]
    keypoints = ann.to_keypoints()
    assert keypoints.confidence.tolist() == [[1.0, 0.0]]


def test_read_yolo_detection(tmp_path):
    ds = load_dataset(_write_yolo(tmp_path))
    assert ds.task is Task.DETECT
    assert ds.class_names == ["ball", "player"]
    assert ds.split_names == ["train", "valid"]  # "test" duplicates "valid"
    a = ds["train"][0].annotations
    np.testing.assert_allclose(a.boxes[0], [24, 12, 40, 36])  # 64x48 image
    assert a.class_ids.tolist() == [0, 1]
    assert len(ds["train"][1].annotations) == 0
    assert ds.summary().loc["train", "player"] == 1


def test_read_yolo_pose_and_folder_path(tmp_path):
    _write_yolo(tmp_path, pose=True)
    ds = load_dataset(tmp_path)  # folder containing data.yaml
    assert ds.task is Task.POSE and ds.num_keypoints == 3 and ds.flip_idx == FLIP
    image, ann = ds.read("train", 0)
    assert image.shape == (48, 64, 3)
    np.testing.assert_allclose(
        ann.keypoints[0], [[32, 24, 2], [38.4, 19.2, 1], [0, 0, 0]]
    )
    assert ann.flip_idx == FLIP


def test_yolo_polygon_labels_become_boxes(tmp_path):
    data_yaml = _write_yolo(tmp_path)
    label = tmp_path / "train/labels/a.txt"
    label.write_text("1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n")
    ann = load_dataset(data_yaml)["train"][0].annotations
    np.testing.assert_allclose(ann.boxes[0], [6.4, 9.6, 19.2, 28.8], rtol=1e-5)


@pytest.mark.parametrize("pose", [False, True])
@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_export_round_trip(tmp_path, fmt, pose):
    ds = load_dataset(_write_yolo(tmp_path / "src", pose=pose))
    exported = getattr(ds, f"to_{fmt}")(tmp_path / fmt)
    again = load_dataset(exported)
    assert again.task == ds.task and again.class_names == ds.class_names
    assert again.flip_idx == ds.flip_idx
    for split in ds.split_names:
        for a, b in zip(ds[split], again[split], strict=True):
            np.testing.assert_allclose(
                a.annotations.boxes, b.annotations.boxes, atol=1e-3
            )
            assert a.annotations.class_ids.tolist() == b.annotations.class_ids.tolist()
            if pose:
                labelled = a.annotations.keypoints[..., 2] > 0
                np.testing.assert_allclose(
                    a.annotations.keypoints[labelled],
                    b.annotations.keypoints[labelled],
                    atol=1e-3,
                )


def test_coco_categories_are_zero_based_and_named(tmp_path):
    root = load_dataset(_write_yolo(tmp_path / "src")).to_coco(tmp_path / "coco")
    coco = json.loads((root / "train" / "_annotations.coco.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [
        (0, "ball"),
        (1, "player"),
    ]
    assert all(c["supercategory"] != "none" for c in coco["categories"])


def test_coco_reader_drops_roboflow_placeholder_category(tmp_path):
    _image(tmp_path / "train" / "x.jpg")
    coco = {
        "categories": [
            {"id": 0, "name": "players", "supercategory": "none"},
            {"id": 1, "name": "ball", "supercategory": "players"},
            {"id": 2, "name": "player", "supercategory": "players"},
        ],
        "images": [{"id": 7, "file_name": "x.jpg", "width": 64, "height": 48}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 2, "bbox": [1, 2, 3, 4]}
        ],
    }
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
    ds = load_dataset(tmp_path)
    assert ds.class_names == ["ball", "player"]
    assert ds["train"][0].annotations.class_ids.tolist() == [1]
    np.testing.assert_allclose(ds["train"][0].annotations.boxes[0], [1, 2, 4, 6])


def test_export_refuses_to_overwrite_source(tmp_path):
    ds = load_dataset(_write_yolo(tmp_path))
    with pytest.raises(ValueError, match="Refusing"):
        ds.to_yolo(tmp_path)


def test_export_renames_duplicate_file_names(tmp_path):
    first = _dataset(tmp_path / "one", n=10)
    merged = first.merge(first)
    data_yaml = merged.to_yolo(tmp_path / "out")
    again = load_dataset(data_yaml)
    assert len(again["train"]) == 16


def test_subset_resplit_merge(tmp_path):
    ds = _dataset(tmp_path)
    small = ds.subset({"train": 3})
    assert len(small["train"]) == 3 and len(small["valid"]) == 2
    assert len(ds.subset(0.5)["train"]) == 4
    assert small.subset(3, seed=1)["train"] == small.subset(3, seed=1)["train"]
    resplit = ds.resplit(train=0.6, valid=0.2, test=0.2)
    assert [len(resplit[s]) for s in ("train", "valid", "test")] == [6, 2, 2]
    with pytest.raises(ValueError):
        ds.resplit(train=0.5, valid=0.2)
    assert len(ds.merge(ds)) == 20
    with pytest.raises(ValueError):
        ds.merge(Dataset(Task.DETECT, ["other"], {}))


def _video(path, frames=12, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for i in range(frames):
        writer.write(np.full((24, 32, 3), i * 20, np.uint8))
    writer.release()
    return path


def test_video_reader_and_frame_extraction(tmp_path):
    video = VideoReader(_video(tmp_path / "v.mp4"))
    assert (video.width, video.height, video.frame_count, video.fps) == (32, 24, 12, 10)
    assert [i for i, _ in video.frames(start=2, end=9, stride=3)] == [2, 5, 8]
    assert abs(int(video.read(5).mean()) - 100) <= 8  # neighbours differ by 20
    paths = extract_frames(video.path, tmp_path / "frames", stride=4)
    assert [p.name for p in paths] == ["v_000000.jpg", "v_000004.jpg", "v_000008.jpg"]
    paths = extract_frames(video.path, tmp_path / "range", start=2, end=7, stride=2)
    assert [p.name for p in paths] == ["v_000002.jpg", "v_000004.jpg", "v_000006.jpg"]
    with pytest.raises(IndexError):
        video.read(50)


def test_load_statsbomb_folder(tmp_path):
    events = [
        {
            "id": "e1",
            "index": 1,
            "period": 1,
            "timestamp": "00:01:02.500",
            "minute": 1,
            "second": 2,
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "A"},
        }
    ]
    frames = [
        {
            "event_uuid": "e1",
            "visible_area": [0, 0, 120, 0, 120, 80],
            "freeze_frame": [
                {
                    "teammate": True,
                    "actor": True,
                    "keeper": False,
                    "location": [10.0, 20.0],
                },
                {
                    "teammate": False,
                    "actor": False,
                    "keeper": True,
                    "location": [110.0, 40.0],
                },
            ],
        }
    ]
    (tmp_path / "m_events.json").write_text(json.dumps(events))
    (tmp_path / "m_360.json").write_text(json.dumps(frames))
    table = load_statsbomb(tmp_path)
    assert table["type"].tolist() == ["player", "goalkeeper"]
    assert table["pitch_location"].tolist() == [[10.0, 20.0], [110.0, 40.0]]
    assert table["timestamp_seconds"].iloc[0] == pytest.approx(62.5)
    assert table["type_name"].iloc[0] == "Pass"


def test_reexporting_an_export_into_itself_is_refused(tmp_path):
    exported = load_dataset(_write_yolo(tmp_path / "src")).to_yolo(tmp_path / "out")
    again = load_dataset(exported)  # images are symlinks inside tmp_path/out
    with pytest.raises(ValueError, match="Refusing"):
        again.to_yolo(tmp_path / "out")
    with pytest.raises(ValueError, match="Refusing"):
        again.to_coco(tmp_path / "out")
    assert len(load_dataset(exported)["train"]) == 2  # still intact


def test_export_does_not_clear_foreign_folders(tmp_path):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not a tactifoot export"):
        ds.to_coco(tmp_path / "busy")
    assert (tmp_path / "busy" / "notes.txt").read_text() == "keep me"
    ds.to_coco(tmp_path / "export")
    ds.to_coco(tmp_path / "export")  # re-exporting into an earlier export is fine


def test_export_keeps_labels_apart_for_images_sharing_a_stem(tmp_path):
    a = Sample(
        _image(tmp_path / "one" / "f.jpg"), 64, 48, Annotations([[1, 1, 5, 5]], [0])
    )
    b = Sample(
        _image(tmp_path / "two" / "f.png"), 64, 48, Annotations([[10, 10, 30, 30]], [1])
    )
    ds = Dataset(Task.DETECT, ["ball", "player"], {"train": [a, b], "valid": [a]})
    again = load_dataset(ds.to_yolo(tmp_path / "out"))
    classes = sorted(s.annotations.class_ids.tolist() for s in again["train"])
    assert classes == [[0], [1]]


def test_yolo_export_unlabels_keypoints_outside_the_image(tmp_path):
    keypoints = [[[70, 10, 2], [32, 24, 2], [5, 5, 1]]]  # x=70 is outside a 64 px image
    sample = Sample(
        _image(tmp_path / "src" / "a.jpg"),
        64,
        48,
        Annotations([[0, 0, 64, 48]], [0], keypoints, flip_idx=FLIP),
    )
    ds = Dataset(Task.POSE, ["pitch"], {"train": [sample], "valid": [sample]}, 3, FLIP)
    again = load_dataset(ds.to_yolo(tmp_path / "out"))["train"][0].annotations
    np.testing.assert_allclose(again.keypoints[0, :, 2], [0, 2, 1])


def test_coco_ids_start_at_one(tmp_path):
    root = load_dataset(_write_yolo(tmp_path / "src")).to_coco(tmp_path / "coco")
    coco = json.loads((root / "train" / "_annotations.coco.json").read_text())
    assert min(i["id"] for i in coco["images"]) == 1
    assert min(a["id"] for a in coco["annotations"]) == 1


@pytest.mark.parametrize("link", ["symlink", "hardlink", "copy"])
@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_reexport_into_the_same_folder_works_for_every_link_mode(tmp_path, fmt, link):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    export = getattr(ds, f"to_{fmt}")
    export(tmp_path / "out", link=link)
    export(tmp_path / "out", link=link)
    smaller = getattr(ds.subset({"train": 1}), f"to_{fmt}")
    assert len(load_dataset(smaller(tmp_path / "out", link=link))["train"]) == 1


def test_export_refuses_to_clear_images_it_did_not_write(tmp_path):
    """Images that land in an export folder by other means survive a re-export."""
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    ds.to_yolo(tmp_path / "out", link="copy")
    foreign = _image(tmp_path / "out" / "train" / "images" / "extra_aug0.jpg")
    with pytest.raises(ValueError, match="extra_aug0.jpg"):
        ds.subset({"train": 1}).to_yolo(tmp_path / "out")
    assert foreign.is_file()


def test_export_keeps_augmented_images_written_inside_its_folders(tmp_path):
    # The scenario of a later export deleting augment_dataset's output: the
    # augmented images sit inside a folder the COCO export manages.
    from tactifoot_vision.augment import ColorJitter, augment_dataset

    ds = load_dataset(_write_yolo(tmp_path / "src"))
    ds.to_coco(tmp_path / "out")
    augmented = augment_dataset(
        ds, ColorJitter(p=1.0), tmp_path / "out" / "train", progress=False
    )
    written = [s.image_path for s in augmented["train"] if "_aug" in s.image_path.name]
    assert written
    with pytest.raises(ValueError, match="did not write"):
        ds.to_coco(tmp_path / "out")
    assert all(path.is_file() for path in written)
