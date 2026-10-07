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


# ------------------------------------------------------- review round 2
@pytest.mark.parametrize(
    ("start", "end", "stride", "match"),
    [
        (-2, 3, 1, "start must be >= 0"),
        (5, 3, 1, "end .* must be >= start"),
        (0, None, 0, "stride must be >= 1"),
    ],
)
def test_video_frame_ranges_are_checked_when_called(
    tmp_path, start, end, stride, match
):
    video = VideoReader(_video(tmp_path / "v.mp4"))
    with pytest.raises(ValueError, match=match):
        video.frames(start=start, end=end, stride=stride)  # no next() needed
    with pytest.raises(ValueError, match=match):
        extract_frames(
            video.path, tmp_path / "out", start=start, end=end, stride=stride
        )
    assert list(video.frames(start=3, end=3)) == []  # an empty range is not an error


def test_video_read_rejects_negative_indices(tmp_path):
    video = VideoReader(_video(tmp_path / "v.mp4"))
    with pytest.raises(IndexError, match="-1"):
        video.read(-1)


@pytest.mark.parametrize(
    ("n", "fractions", "sizes"),
    [
        (5, {"train": 0.5, "valid": 0.5}, [2, 3, 0]),
        (3, {"train": 0.5, "valid": 0.5}, [2, 1, 0]),
        (7, {"train": 0.9, "valid": 0.1}, [6, 1, 0]),
        (5, {"train": 0.0, "valid": 0.6, "test": 0.4}, [0, 3, 2]),
        (10, {"train": 0.6, "valid": 0.2, "test": 0.2}, [6, 2, 2]),
    ],
)
def test_resplit_leaves_zero_fraction_splits_empty(tmp_path, n, fractions, sizes):
    ds = _dataset(tmp_path, n=n)
    resplit = ds.resplit(**{"train": 0.0, "valid": 0.0, "test": 0.0} | fractions)
    assert [len(resplit[s]) for s in ("train", "valid", "test")] == sizes
    assert len(resplit) == n


def test_resplit_rejects_negative_fractions(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        _dataset(tmp_path).resplit(train=1.2, valid=-0.2)


def test_subset_normalises_numbers(tmp_path):
    ds = _dataset(tmp_path)  # 8 train, 2 valid
    assert len(ds.subset(np.float32(0.5))["train"]) == 4  # a fraction
    assert len(ds.subset(np.int64(3))["train"]) == 3  # a count
    assert len(ds.subset(1.0)["train"]) == 8
    assert len(ds.subset({"train": np.float64(0.25)})["train"]) == 2
    with pytest.raises(ValueError, match="fraction"):
        ds.subset(1.5)
    with pytest.raises(ValueError, match="count"):
        ds.subset(-1)
    with pytest.raises(ValueError, match="number"):
        ds.subset("3")


def test_yolo_reader_checks_class_ids(tmp_path):
    data_yaml = _write_yolo(tmp_path)
    (tmp_path / "train/labels/a.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match=r"a\.txt.*class id 2.*2 classes"):
        load_dataset(data_yaml)
    (tmp_path / "train/labels/a.txt").write_text("-1 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match=r"a\.txt.*class id -1"):
        load_dataset(data_yaml)


def test_coco_reader_checks_category_ids(tmp_path):
    _image(tmp_path / "train" / "x.jpg")
    coco = {
        "categories": [{"id": 0, "name": "ball", "supercategory": "objects"}],
        "images": [{"id": 1, "file_name": "x.jpg", "width": 64, "height": 48}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [1, 2, 3, 4]}
        ],
    }
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match=r"_annotations\.coco\.json.*category_id 3"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_reexport_refuses_to_delete_files_it_did_not_write(tmp_path, fmt):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    export = getattr(ds, f"to_{fmt}")
    export(tmp_path / "out", link="copy")
    folder = tmp_path / "out" / "train" / ("images" if fmt == "yolo" else "")
    note = folder / "notes.txt"
    note.write_text("keep")
    with pytest.raises(ValueError, match="notes.txt"):
        export(tmp_path / "out", link="copy")
    assert note.read_text() == "keep"


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_reexport_with_fewer_splits_removes_the_old_split(tmp_path, fmt):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    train_only = ds.with_split("valid", [])
    out = tmp_path / "out"
    getattr(ds, f"to_{fmt}")(out, link="copy")
    reloaded = load_dataset(getattr(train_only, f"to_{fmt}")(out, link="copy"))
    assert reloaded.split_names == ["train"]
    assert not (out / "valid").exists()
    # The ownership record survives: the full dataset exports there again.
    again = load_dataset(getattr(ds, f"to_{fmt}")(out, link="copy"))
    assert again.split_names == ["train", "valid"]


def test_switching_the_export_format_removes_the_old_format(tmp_path):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    ds.to_yolo(tmp_path / "out")
    root = ds.to_coco(tmp_path / "out")
    assert not (root / "data.yaml").exists()
    assert load_dataset(root).split_names == ["train", "valid"]


def test_export_checks_its_arguments_before_deleting_anything(tmp_path):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    ds.to_yolo(tmp_path / "out", link="copy")
    with pytest.raises(ValueError, match="link must be"):
        ds.to_yolo(tmp_path / "out", link="copi")
    assert (tmp_path / "out" / "train" / "images" / "a.jpg").is_file()
    missing = Sample(tmp_path / "gone.jpg", 64, 48, Annotations.empty())
    with pytest.raises(FileNotFoundError, match="gone.jpg"):
        ds.with_split("test", [missing]).to_coco(tmp_path / "out")
    assert (tmp_path / "out" / "train" / "images" / "a.jpg").is_file()


# ------------------------------------------------------- review round 3
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_format_switch_refuses_a_root_file_the_export_did_not_write(tmp_path, kind):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    out = tmp_path / "out"
    ds.to_coco(out)
    user = tmp_path / "user.yaml"
    user.write_text("user_note: preserve me\n")
    if kind == "file":
        (out / "data.yaml").write_text(user.read_text())
    else:
        (out / "data.yaml").symlink_to(user)
    marker = (out / ".tactifoot-export").read_text()
    with pytest.raises(ValueError, match=r"data\.yaml.*did not write"):
        ds.to_yolo(out)
    assert (
        (out / "data.yaml").read_text()
        == user.read_text()
        == "user_note: preserve me\n"
    )
    assert (out / "data.yaml").is_symlink() == (kind == "symlink")
    assert (out / ".tactifoot-export").read_text() == marker
    assert Dataset.from_coco(out).split_names == ["train", "valid"]  # still intact


def _pose(path, flip_idx, num_keypoints=2):
    keypoints = [[[2, 5, 2], [30, 5, 2], [20, 10, 2]][:num_keypoints]]
    flip_idx = flip_idx[:num_keypoints]
    annotations = Annotations([[0, 0, 40, 20]], [0], keypoints, flip_idx)
    sample = Sample(_image(path, width=40, height=20), 40, 20, annotations)
    return Dataset(Task.POSE, ["pitch"], {"train": [sample]}, num_keypoints, flip_idx)


def test_merge_rejects_pose_datasets_with_another_keypoint_layout(tmp_path):
    swap = _pose(tmp_path / "a.jpg", (1, 0))
    with pytest.raises(ValueError, match=r"flip_idx.*\(1, 0\).*\(0, 1\)"):
        swap.merge(_pose(tmp_path / "b.jpg", (0, 1)))
    with pytest.raises(ValueError, match="2 and 3 keypoints"):
        swap.merge(_pose(tmp_path / "c.jpg", (1, 0, 2), num_keypoints=3))


def test_merged_pose_labels_keep_their_meaning_through_an_export(tmp_path):
    from tactifoot_vision.augment import HorizontalFlip

    first, second = (
        _pose(tmp_path / "src" / name, (1, 0)) for name in ("a.jpg", "b.jpg")
    )
    merged = first.merge(second)
    reloaded = load_dataset(merged.to_yolo(tmp_path / "out"))
    assert reloaded.flip_idx == merged.flip_idx == (1, 0)
    flip = HorizontalFlip(p=1)
    for before, after in zip(merged["train"], reloaded["train"], strict=True):
        assert after.annotations.flip_idx == before.annotations.flip_idx == (1, 0)
        _, expected = flip(before.read_image(), before.annotations)
        _, flipped = flip(after.read_image(), after.annotations)
        np.testing.assert_allclose(flipped.keypoints, expected.keypoints, atol=1e-3)


def test_read_takes_one_sample_without_copying_the_split(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    monkeypatch.setattr(Dataset, "__getitem__", lambda *_: pytest.fail("copied"))
    image, annotations = ds.read("val", -1)
    assert image.shape == (48, 64, 3)
    original = ds.splits["valid"][-1].annotations
    assert annotations is not original and annotations.boxes is not original.boxes
    np.testing.assert_array_equal(annotations.boxes, original.boxes)


# ------------------------------------------------------- review round 3 (Fable)
@pytest.mark.parametrize(
    "entry", ["../victim/", "../victim/notes.txt", "train/../../victim/", "ABSOLUTE"]
)
def test_a_marker_entry_outside_the_export_is_refused(tmp_path, entry):
    from tactifoot_vision.data._files import EXPORT_MARKER

    ds = load_dataset(_write_yolo(tmp_path / "src"))
    out = tmp_path / "export"
    ds.to_yolo(out)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep me")
    absolute = tmp_path / "absolute.txt"
    absolute.write_text("keep me too")
    entry = entry.replace("ABSOLUTE", str(absolute))
    marker = out / EXPORT_MARKER
    marker.write_text(marker.read_text() + entry + "\n")
    with pytest.raises(ValueError, match="lies outside"):
        ds.to_yolo(out)
    assert (victim / "notes.txt").read_text() == "keep me"
    assert absolute.read_text() == "keep me too"
    assert (out / "train" / "images" / "a.jpg").exists()  # the export is untouched


def test_a_marker_folder_reached_through_a_symlink_is_refused(tmp_path):
    from tactifoot_vision.data._files import EXPORT_MARKER

    ds = load_dataset(_write_yolo(tmp_path / "src"))
    out = tmp_path / "export"
    ds.to_yolo(out)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep me")
    (out / "elsewhere").symlink_to(victim)
    marker = out / EXPORT_MARKER
    marker.write_text(marker.read_text() + "elsewhere/\nelsewhere/notes.txt\n")
    with pytest.raises(ValueError, match="lies outside"):
        ds.to_yolo(out)
    assert (victim / "notes.txt").read_text() == "keep me"


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_each_link_mode_places_its_kind_of_file_and_spares_the_source(tmp_path, fmt):
    ds = load_dataset(_write_yolo(tmp_path / "src"))
    source = ds["train"][0].image_path
    original = source.read_bytes()
    folder = {"yolo": "train/images", "coco": "train"}[fmt]
    for link in ("symlink", "hardlink", "copy", "hardlink"):  # re-exports included
        getattr(ds, f"to_{fmt}")(tmp_path / "out", link=link)
        image = tmp_path / "out" / folder / source.name
        assert image.is_symlink() == (link == "symlink")
        assert image.samefile(source) == (link != "copy")
        if link == "hardlink":
            assert image.stat().st_nlink >= 2
        assert image.read_bytes() == original
    assert source.read_bytes() == original and source.stat().st_nlink == 2


def test_a_flip_idx_without_one_entry_per_keypoint_fails_at_load(tmp_path):
    data_yaml = _write_yolo(tmp_path, pose=True)
    config = yaml.safe_load(data_yaml.read_text())
    data_yaml.write_text(yaml.safe_dump(config | {"flip_idx": [1, 0]}))
    with pytest.raises(
        ValueError, match=r"data\.yaml.*flip_idx has 2 entries.*3 keypoints"
    ):
        load_dataset(data_yaml)
    with pytest.raises(ValueError, match="flip_idx has 2 entries"):
        Dataset(Task.POSE, ["pitch"], {}, num_keypoints=3, flip_idx=(1, 0))


def test_an_odd_label_row_names_its_file_and_line(tmp_path):
    data_yaml = _write_yolo(tmp_path)
    (tmp_path / "train/labels/a.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n\n0 0.1 0.1 0.5 0.5 0.9\n"
    )
    with pytest.raises(ValueError, match=r"a\.txt, line 3: .*got 5"):
        load_dataset(data_yaml)


def test_two_value_keypoints_are_visible_when_either_coordinate_is_set(tmp_path):
    data_yaml = _write_yolo(tmp_path)
    config = yaml.safe_load(data_yaml.read_text()) | {"kpt_shape": [3, 2]}
    data_yaml.write_text(yaml.safe_dump(config))
    for label in (tmp_path / "train/labels/a.txt", tmp_path / "valid/labels/c.txt"):
        label.write_text("1 0.5 0.5 0.2 0.2 0.5 0.5 0.25 0 0 0\n")
    keypoints = load_dataset(data_yaml)["train"][0].annotations.keypoints
    np.testing.assert_allclose(keypoints[0], [[32, 24, 2], [16, 0, 2], [0, 0, 0]])


def test_coco_reader_keeps_a_real_class_whose_supercategory_is_none(tmp_path):
    _image(tmp_path / "train" / "x.jpg")
    coco = {
        "categories": [
            {"id": 0, "name": "players", "supercategory": "none"},
            {"id": 1, "name": "ball", "supercategory": "players"},
            {"id": 2, "name": "player", "supercategory": "players"},
            {"id": 3, "name": "referee", "supercategory": "none"},
        ],
        "images": [{"id": 7, "file_name": "x.jpg", "width": 64, "height": 48}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 2, 3, 4]}
        ],
    }
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
    ds = load_dataset(tmp_path)
    assert ds.class_names == ["ball", "player", "referee"]
    assert ds["train"][0].annotations.class_ids.tolist() == [2]
