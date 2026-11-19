from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import (
    Dataset,
    LabelledClip,
    LabelledFrame,
    Task,
    Truth,
    broadcast_homographies,
    load_dataset,
    load_labelled_clip,
    synthetic_pitch_clip,
)
from tactifoot_vision.pitch import SoccerPitch, pitch_to_frame

CLASSES = ["player", "goalkeeper", "referee", "ball"]
SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=25
seqLength={length}
imWidth=96
imHeight=54
imExt=.jpg
"""
GAMEINFO = """[Sequence]
trackletID_1= player team left;7
trackletID_2= goalkeepers team right;1
trackletID_3= referee;main
trackletID_4= ball;1
trackletID_5= other;1
"""
# Frame 2 has no box; track 5 is an "other" tracklet, left out of the clip.
GT = """1,1,10,12,8,16,1,-1,-1,-1
1,4,50,30,3,3,1,-1,-1,-1
1,5,0,0,4,4,1,-1,-1,-1
3,2,60,10,9,18,1,-1,-1,-1
3,3,30,20,7,15,1,-1,-1,-1
3,1,11,13,8,16,1,-1,-1,-1
"""


def _clip(root, name, length=3, gt=GT):
    """A SoccerNet Tracking sequence folder with its images, loaded as a labelled clip."""
    folder = root / name
    (folder / "gt").mkdir(parents=True)
    (folder / "img1").mkdir()
    (folder / "seqinfo.ini").write_text(SEQINFO.format(name=name, length=length))
    (folder / "gameinfo.ini").write_text(GAMEINFO)
    (folder / "gt" / "gt.txt").write_text(gt)
    for n in range(1, length + 1):
        cv2.imwrite(str(folder / "img1" / f"{n:06d}.jpg"), np.full((54, 96, 3), n))
    return load_labelled_clip(folder)


def _clip_of(sample, clips):
    return next(c for c in clips if sample.image_path.parent == c.video_path.parent)


def _files(root):
    return sorted(p for p in Path(root).rglob("*") if not p.is_dir())


def test_a_clip_becomes_detection_samples_of_its_frames_in_place(tmp_path):
    clip = _clip(tmp_path, "SNMOT-001")

    dataset = Dataset.from_clips(clip, train=1, valid=0)

    assert dataset.task is Task.DETECT
    assert dataset.class_names == CLASSES
    first, last = dataset["train"]
    images = tmp_path / "SNMOT-001" / "img1"
    assert (first.image_path, last.image_path) == (
        images / "000001.jpg",
        images / "000003.jpg",
    )
    assert (first.width, first.height) == (96, 54)
    np.testing.assert_allclose(
        first.annotations.boxes, [[10, 12, 18, 28], [50, 30, 53, 33]]
    )
    assert [CLASSES[i] for i in first.annotations.class_ids] == ["player", "ball"]
    assert [CLASSES[i] for i in last.annotations.class_ids] == [
        "goalkeeper",
        "referee",
        "player",
    ]


def test_a_frame_without_boxes_is_left_out(tmp_path):
    # A clip cannot tell an empty frame from an unlabelled one.
    clip = _clip(tmp_path, "SNMOT-001")

    dataset = Dataset.from_clips([clip], train=1, valid=0)

    assert [s.image_path.name for s in dataset] == ["000001.jpg", "000003.jpg"]


def test_every_frame_of_a_clip_lands_in_one_split(tmp_path):
    clips = [_clip(tmp_path, f"SNMOT-{i:03d}", length=4) for i in range(10)]
    for clip in clips[:5]:  # the first five also have a box in frames 2 and 4
        (clip.video_path.parent.parent / "gt" / "gt.txt").write_text(
            GT + "2,1,10,12,8,16,1,-1,-1,-1\n4,1,10,12,8,16,1,-1,-1,-1\n"
        )
    clips = [load_labelled_clip(c.video_path.parent.parent) for c in clips]

    for seed in range(5):
        dataset = Dataset.from_clips(clips, train=0.6, valid=0.3, test=0.1, seed=seed)

        split_of = {}
        for split in dataset.split_names:
            for sample in dataset[split]:
                split_of.setdefault(_clip_of(sample, clips).name, set()).add(split)
        assert all(len(splits) == 1 for splits in split_of.values())
        # Fractions count clips, not frames: 6, 3 and 1 of the 10 clips.
        clips_per_split = {
            split: {_clip_of(s, clips).name for s in dataset[split]}
            for split in dataset.split_names
        }
        assert {s: len(c) for s, c in clips_per_split.items()} == {
            "train": 6,
            "valid": 3,
            "test": 1,
        }
        frames = {s: len(dataset[s]) for s in dataset.split_names}
        assert sum(frames.values()) == 5 * 4 + 5 * 2


def test_the_split_of_the_clips_follows_the_seed(tmp_path):
    clips = [_clip(tmp_path, f"SNMOT-{i:03d}") for i in range(6)]

    def valid_clips(seed):
        dataset = Dataset.from_clips(clips, train=0.5, valid=0.5, seed=seed)
        return {_clip_of(s, clips).name for s in dataset["valid"]}

    assert valid_clips(0) == valid_clips(0)
    assert len({frozenset(valid_clips(seed)) for seed in range(10)}) > 1


def test_a_zero_fraction_leaves_its_split_empty(tmp_path):
    clips = [_clip(tmp_path, f"SNMOT-{i:03d}") for i in range(3)]

    dataset = Dataset.from_clips(clips, train=0.5, valid=0.5, test=0)

    assert dataset.split_names == ["train", "valid"]


@pytest.mark.parametrize(
    "fractions", [{"train": 0.5, "valid": 0.2}, {"train": 1.2, "valid": -0.2}]
)
def test_fractions_must_be_valid(tmp_path, fractions):
    with pytest.raises(ValueError, match="fractions"):
        Dataset.from_clips(_clip(tmp_path, "SNMOT-001"), **fractions)


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_the_export_round_trips(tmp_path, fmt):
    clips = [_clip(tmp_path / "clips", f"SNMOT-{i:03d}") for i in range(4)]
    dataset = Dataset.from_clips(clips, train=0.5, valid=0.25, test=0.25)

    exported = getattr(dataset, f"to_{fmt}")(tmp_path / "out")
    again = load_dataset(exported)

    assert again.class_names == CLASSES
    assert again.split_names == dataset.split_names
    for split in dataset.split_names:
        before = sorted(dataset[split], key=lambda s: s.image_path.resolve())
        after = sorted(again[split], key=lambda s: s.image_path.resolve())
        assert [s.image_path.resolve() for s in after] == [s.image_path for s in before]
        for a, b in zip(before, after, strict=True):
            assert (b.width, b.height) == (a.width, a.height)
            np.testing.assert_allclose(
                b.annotations.boxes, a.annotations.boxes, atol=1e-3
            )
            np.testing.assert_array_equal(
                b.annotations.class_ids, a.annotations.class_ids
            )


def test_building_the_dataset_copies_and_writes_no_frame(tmp_path):
    clips = [_clip(tmp_path, f"SNMOT-{i:03d}") for i in range(2)]
    before = [(p, p.stat().st_mtime_ns) for p in _files(tmp_path)]

    dataset = Dataset.from_clips(clips)

    assert [(p, p.stat().st_mtime_ns) for p in _files(tmp_path)] == before
    assert all(
        s.image_path.parent == tmp_path / s.image_path.parts[-3] / "img1"
        for s in dataset
    )


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
@pytest.mark.parametrize("link", ["symlink", "hardlink", "copy"])
def test_only_the_copy_link_mode_copies_frames(tmp_path, fmt, link):
    clip = _clip(tmp_path / "clips", "SNMOT-001")
    dataset = Dataset.from_clips(clip, train=1, valid=0)

    root = Path(getattr(dataset, f"to_{fmt}")(tmp_path / "out", link=link)).parent
    images = [p for p in _files(tmp_path / "out") if p.suffix == ".jpg"]

    assert root.is_dir() and len(images) == 2
    sources = {s.image_path.stat().st_ino for s in dataset}
    for image in images:
        assert image.is_symlink() == (link == "symlink")
        assert (image.stat().st_ino in sources) == (link != "copy")


def test_the_classes_are_the_package_vocabulary_whatever_a_clip_holds(tmp_path):
    clip = _clip(tmp_path, "SNMOT-001")
    referee_only = _clip(tmp_path, "SNMOT-002", gt="1,3,30,20,7,15,1,-1,-1,-1\n")

    dataset = Dataset.from_clips([clip, referee_only], train=1, valid=0)

    assert dataset.class_names == CLASSES
    assert dataset.summary().loc["train"].to_dict() == {
        "images": 3,
        "objects": 6,
        "player": 2,
        "goalkeeper": 1,
        "referee": 2,
        "ball": 1,
    }


def _people(names):
    return sv.Detections(
        xyxy=np.tile([1.0, 2.0, 5.0, 9.0], (len(names), 1)),
        tracker_id=np.arange(len(names)),
        data={"class_name": np.array(names)},
    )


def test_a_clip_without_box_truth_is_refused(tmp_path):
    clip = LabelledClip(
        name="calibration-only",
        video_path=tmp_path / "img1" / "%06d.jpg",
        fps=25,
        width=96,
        height=54,
        frames=(LabelledFrame(0, sv.Detections.empty(), sv.Detections.empty()),),
        truth=frozenset({Truth.CALIBRATION}),
    )

    with pytest.raises(ValueError, match="calibration-only.*boxes"):
        Dataset.from_clips(clip)


def test_a_clip_read_from_a_video_file_is_refused(tmp_path):
    # Its frames have no image file to reference in place.
    video = tmp_path / "match.mp4"
    video.write_bytes(b"")
    clip = LabelledClip(
        name="match",
        video_path=video,
        fps=25,
        width=96,
        height=54,
        frames=(LabelledFrame(0, _people(["player"]), sv.Detections.empty()),),
        truth=frozenset({Truth.BOXES}),
    )

    with pytest.raises(ValueError, match=r"match\.mp4.*image sequence"):
        Dataset.from_clips(clip)


@pytest.mark.parametrize(
    "change",
    [
        "remove 000001.jpg",  # would move every label one frame back
        "remove 000002.jpg",  # a gap: VideoReader stops before it
        "remove 000003.jpg",
        "add 000000.jpg",  # would move every label one frame on
        "add 000004.jpg",
    ],
)
def test_images_that_are_not_one_per_frame_are_refused(tmp_path, change):
    clip = _clip(tmp_path, "SNMOT-001")
    action, name = change.split()
    image = tmp_path / "SNMOT-001" / "img1" / name
    if action == "remove":
        image.unlink()
    else:
        cv2.imwrite(str(image), np.zeros((54, 96, 3)))

    with pytest.raises(ValueError, match=r"SNMOT-001 has 3 frames .* \d images"):
        Dataset.from_clips(clip)


def test_boxes_are_clipped_to_the_image_and_round_trip_alike(tmp_path):
    gt = (
        "1,1,-4,40,10,20,1,-1,-1,-1\n"  # crosses the left and bottom edges
        "1,3,100,10,5,5,1,-1,-1,-1\n"  # wholly outside: dropped
        "2,3,200,10,5,5,1,-1,-1,-1\n"  # the frame's only box, outside: no sample
    )
    clip = _clip(tmp_path / "clips", "SNMOT-001", gt=gt)

    dataset = Dataset.from_clips(clip, train=1, valid=0)

    (sample,) = dataset["train"]
    np.testing.assert_allclose(sample.annotations.boxes, [[0, 40, 6, 54]])
    for fmt in ("yolo", "coco"):
        again = load_dataset(getattr(dataset, f"to_{fmt}")(tmp_path / fmt))
        (reloaded,) = again["train"]
        np.testing.assert_allclose(
            reloaded.annotations.boxes, [[0, 40, 6, 54]], atol=1e-3
        )


@pytest.mark.parametrize("class_name", ["other", None])
def test_a_box_outside_the_class_vocabulary_names_its_clip_and_frame(
    tmp_path, class_name
):
    cv2.imwrite(str(tmp_path / "000001.jpg"), np.zeros((54, 96, 3)))
    people = _people(["player"])
    if class_name is None:
        del people.data["class_name"]
    else:
        people.data["class_name"] = np.array([class_name])
    clip = LabelledClip(
        name="odd",
        video_path=tmp_path / "%06d.jpg",
        fps=25,
        width=96,
        height=54,
        frames=(LabelledFrame(0, people, sv.Detections.empty()),),
        truth=frozenset({Truth.BOXES}),
    )

    with pytest.raises(
        ValueError, match=f"Frame 0 of clip odd has class '{class_name or ''}'"
    ):
        Dataset.from_clips(clip)


def test_the_same_clip_twice_is_refused(tmp_path):
    # Its frames could land in two splits.
    clip = _clip(tmp_path, "SNMOT-001")

    with pytest.raises(ValueError, match="SNMOT-001.*twice"):
        Dataset.from_clips([clip, clip])


def test_the_lowest_numbered_image_of_the_sequence_is_frame_0(tmp_path):
    # As VideoReader and the pipeline read it; SoccerNet folders start at 1.
    for n in range(2):
        cv2.imwrite(str(tmp_path / f"frame{n:04d}.png"), np.zeros((54, 96, 3)))
    clip = LabelledClip(
        name="from-zero",
        video_path=tmp_path / "frame%04d.png",
        fps=25,
        width=96,
        height=54,
        frames=tuple(
            LabelledFrame(i, _people(["player"]), sv.Detections.empty())
            for i in range(2)
        ),
        truth=frozenset({Truth.BOXES}),
    )

    dataset = Dataset.from_clips(clip, train=1, valid=0)

    assert [s.image_path.name for s in dataset] == ["frame0000.png", "frame0001.png"]


def test_the_same_clip_reached_by_another_path_is_refused(tmp_path):
    clip = _clip(tmp_path, "SNMOT-001")
    (tmp_path / "link").symlink_to(tmp_path / "SNMOT-001")

    with pytest.raises(ValueError, match="twice"):
        Dataset.from_clips([clip, load_labelled_clip(tmp_path / "link")])


def test_a_sequence_opencv_cannot_open_is_refused(tmp_path):
    # OpenCV only finds a first image numbered 0 to 4.
    for n in range(10, 12):
        cv2.imwrite(str(tmp_path / f"{n:06d}.jpg"), np.zeros((54, 96, 3)))
    clip = LabelledClip(
        name="late",
        video_path=tmp_path / "%06d.jpg",
        fps=25,
        width=96,
        height=54,
        frames=tuple(
            LabelledFrame(i, _people(["player"]), sv.Detections.empty())
            for i in range(2)
        ),
        truth=frozenset({Truth.BOXES}),
    )

    with pytest.raises(ValueError, match=r"late.*000010\.jpg.*0 to 4"):
        Dataset.from_clips(clip)


def test_no_clip_is_refused():
    with pytest.raises(ValueError, match="at least one clip"):
        Dataset.from_clips([])


def test_a_clip_without_a_boxed_frame_warns_and_keeps_its_split_slot(tmp_path, caplog):
    clips = [_clip(tmp_path, "SNMOT-001"), _clip(tmp_path, "SNMOT-002", gt="")]

    with caplog.at_level("WARNING"):
        dataset = Dataset.from_clips(clips, train=0.5, valid=0.5)

    assert "SNMOT-002 has no frame with a box" in caplog.text
    # The empty clip took the other split's one slot, which stays empty.
    assert len(dataset.split_names) == 1
    assert len(dataset) == 2


# ------------------------------------------------- pitch keypoints (Task.POSE)
FRAME = (320, 180)


def _calibrated(root, name, homographies):
    """A synthetic clip of these frame -> pitch homographies, as an image sequence."""
    return synthetic_pitch_clip(root / name / "%06d.png", homographies, FRAME)


def _wide_view():
    """Frame -> pitch homography of the centre: the far touchline on row 15, the near one on the bottom row."""
    src = np.float32([[27.5, 68], [77.5, 68], [87.5, 0], [17.5, 0]])
    dst = np.float32([[0, 180], [320, 180], [290, 15], [30, 15]])
    return cv2.getPerspectiveTransform(dst, src)


def test_a_calibrated_clip_becomes_pitch_keypoint_samples(tmp_path):
    homographies = broadcast_homographies(4, FRAME, seed=0)
    clip = _calibrated(tmp_path, "views", homographies)

    dataset = Dataset.from_clips(clip, train=1, valid=0, task=Task.POSE)

    pitch = SoccerPitch()
    assert dataset.task is Task.POSE
    assert dataset.class_names == ["pitch"]
    assert dataset.num_keypoints == 32
    assert dataset.flip_idx == pitch.flip_idx
    samples = dataset["train"]
    assert [s.image_path for s in samples] == [
        tmp_path / "views" / f"{n:06d}.png" for n in range(4)
    ]
    shown = 0
    for sample, homography in zip(samples, homographies, strict=True):
        annotations = sample.annotations
        assert (sample.width, sample.height) == FRAME
        assert list(annotations.class_ids) == [0]
        assert annotations.flip_idx == pitch.flip_idx
        (keypoints,) = annotations.keypoints
        expected = pitch_to_frame(pitch.vertices, homography)
        inside = ((expected >= 0) & (expected <= FRAME)).all(axis=1)
        np.testing.assert_array_equal(keypoints[:, 2], np.where(inside, 2, 0))
        np.testing.assert_allclose(keypoints[inside, :2], expected[inside], atol=1)
        np.testing.assert_array_equal(keypoints[~inside], 0)
        shown += inside.sum()
    assert 0 < shown < 4 * 32


def test_the_pitch_box_holds_the_part_of_the_frame_that_shows_the_pitch(tmp_path):
    # Stands above the far touchline (row 15); the pitch fills the rest.
    clip = _calibrated(tmp_path, "wide", [_wide_view()])

    (sample,) = Dataset.from_clips(clip, train=1, valid=0, task="pose")["train"]

    np.testing.assert_allclose(sample.annotations.boxes, [[0, 15, 320, 180]], atol=1)


def test_a_landmark_beyond_the_horizon_is_invisible(tmp_path):
    # A camera standing at x = 50 on the pitch looks towards x = 105: the
    # left half lies behind it, and through the homography alone its
    # landmarks would land in the frame, mirrored.
    to_frame = np.array([[160, 100, -11400], [40, 0, -1700], [1, 0, -50.0]])
    clip = _calibrated(tmp_path, "horizon", [np.linalg.inv(to_frame)])
    vertices = SoccerPitch().vertices
    projected = pitch_to_frame(vertices, np.linalg.inv(to_frame))
    behind = vertices[:, 0] < 50
    mirrored = behind & ((projected >= 0) & (projected <= FRAME)).all(axis=1)
    assert mirrored.any()

    (sample,) = Dataset.from_clips(clip, train=1, valid=0, task="pose")["train"]

    visibility = sample.annotations.keypoints[0, :, 2]
    assert (visibility[behind] == 0).all()
    assert (visibility[~behind] == 2).any()


def test_a_frame_without_calibration_is_left_out(tmp_path):
    clip = _calibrated(tmp_path, "gap", broadcast_homographies(3, FRAME, seed=0))
    frames = list(clip.frames)
    frames[1] = replace(frames[1], homography=None)
    clip = replace(clip, frames=tuple(frames))

    dataset = Dataset.from_clips(clip, train=1, valid=0, task="pose")

    assert [s.image_path.name for s in dataset] == ["000000.png", "000002.png"]


def test_every_frame_of_a_calibrated_clip_lands_in_one_split(tmp_path):
    clips = [
        _calibrated(tmp_path, f"views-{i}", broadcast_homographies(2, FRAME, seed=i))
        for i in range(5)
    ]

    dataset = Dataset.from_clips(clips, train=0.6, valid=0.2, test=0.2, task="pose")

    clips_per_split = {
        split: {_clip_of(s, clips).name for s in dataset[split]}
        for split in dataset.split_names
    }
    assert {s: len(c) for s, c in clips_per_split.items()} == {
        "train": 3,
        "valid": 1,
        "test": 1,
    }
    assert {s: len(dataset[s]) for s in dataset.split_names} == {
        "train": 6,
        "valid": 2,
        "test": 2,
    }


def test_a_clip_without_calibration_truth_is_refused_by_name(tmp_path):
    clip = _clip(tmp_path, "SNMOT-001")  # boxes, no calibration

    with pytest.raises(ValueError, match="SNMOT-001.*calibration"):
        Dataset.from_clips(clip, task="pose")


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
def test_the_pitch_keypoint_export_round_trips(tmp_path, fmt):
    clips = [
        _calibrated(
            tmp_path / "clips", f"views-{i}", broadcast_homographies(3, FRAME, seed=i)
        )
        for i in range(2)
    ]
    dataset = Dataset.from_clips(clips, train=0.5, valid=0.5, task=Task.POSE)

    again = load_dataset(getattr(dataset, f"to_{fmt}")(tmp_path / "out"))

    assert again.task is Task.POSE
    assert again.class_names == ["pitch"]
    assert (again.num_keypoints, again.flip_idx) == (32, SoccerPitch().flip_idx)
    assert again.split_names == dataset.split_names
    for split in dataset.split_names:
        before = sorted(dataset[split], key=lambda s: s.image_path.resolve())
        after = sorted(again[split], key=lambda s: s.image_path.resolve())
        assert [s.image_path.resolve() for s in after] == [s.image_path for s in before]
        for a, b in zip(before, after, strict=True):
            np.testing.assert_allclose(
                b.annotations.boxes, a.annotations.boxes, atol=1e-2
            )
            np.testing.assert_allclose(
                b.annotations.keypoints, a.annotations.keypoints, atol=1e-2
            )
