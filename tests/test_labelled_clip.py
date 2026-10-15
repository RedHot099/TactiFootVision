import cv2
import numpy as np
import pytest

from tactifoot_vision.data import LabelledClip, Truth, load_labelled_clip

SEQINFO = """[Sequence]
name=SNMOT-060
imDir=img1
frameRate=25
seqLength=3
imWidth=1920
imHeight=1080
imExt=.jpg
"""
GAMEINFO = """[Sequence]
name=SNMOT-060
gameID=1
num_tracklets=5
trackletID_1= player team left;7
trackletID_2= goalkeepers team right;1
trackletID_3= referee;main
trackletID_4= ball;1
trackletID_5= other;1
"""
GT = """1,1,100,200,30,60,1,-1,-1,-1
1,2,500.5,300,40,80,1,-1,-1,-1
1,4,910,540,8,9,1,-1,-1,-1
1,5,0,0,10,10,1,-1,-1,-1
3,3,700,400,25,70,1,-1,-1,-1
3,1,104,201,30,60,1,-1,-1,-1
"""


def _sequence(root, name="SNMOT-060", gt=GT, gameinfo=GAMEINFO, seqinfo=SEQINFO):
    """A SoccerNet Tracking sequence folder as the dataset ships it (no images)."""
    folder = root / name
    (folder / "gt").mkdir(parents=True)
    (folder / "seqinfo.ini").write_text(seqinfo)
    (folder / "gameinfo.ini").write_text(gameinfo)
    (folder / "gt" / "gt.txt").write_text(gt)
    return folder


def test_soccernet_tracking_loads_boxes_ids_and_classes(tmp_path):
    clip = load_labelled_clip(_sequence(tmp_path))

    assert isinstance(clip, LabelledClip)
    assert clip.name == "SNMOT-060"
    assert clip.fps == 25
    assert (clip.width, clip.height) == (1920, 1080)
    first = clip.frame(0)
    np.testing.assert_array_equal(
        first.detections.xyxy, [[100, 200, 130, 260], [500.5, 300, 540.5, 380]]
    )
    np.testing.assert_array_equal(first.detections.tracker_id, [1, 2])
    assert list(first.detections.data["class_name"]) == ["player", "goalkeeper"]
    np.testing.assert_array_equal(first.ball.xyxy, [[910, 540, 918, 549]])
    np.testing.assert_array_equal(first.ball.tracker_id, [4])
    assert list(first.ball.data["class_name"]) == ["ball"]
    last = clip.frame(2)
    np.testing.assert_array_equal(last.detections.tracker_id, [3, 1])
    assert list(last.detections.data["class_name"]) == ["referee", "player"]
    assert len(last.ball) == 0


def test_soccernet_tracking_frames_align_with_the_video_frame_index(tmp_path):
    """MOT frame 1 is ``img1/000001.jpg``, which the pipeline reads as frame 0."""
    folder = _sequence(tmp_path)
    (folder / "img1").mkdir()
    for number in (1, 2, 3):
        image = np.full((1080, 1920, 3), 60 * number, np.uint8)
        cv2.imwrite(str(folder / "img1" / f"{number:06d}.jpg"), image)

    clip = load_labelled_clip(folder)

    # Every frame of the sequence, the unlabelled middle one included.
    assert [frame.index for frame in clip] == [0, 1, 2]
    assert len(clip) == 3
    assert len(clip.frame(1).detections) == len(clip.frame(1).ball) == 0
    capture = cv2.VideoCapture(str(clip.video_path))
    for frame in clip:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame.index)
        ok, image = capture.read()
        assert ok
        assert abs(image.mean() - 60 * (frame.index + 1)) < 2
    capture.release()


def test_reading_a_frame_that_is_not_in_the_clip_fails(tmp_path):
    clip = load_labelled_clip(_sequence(tmp_path))

    with pytest.raises(IndexError, match="Frame 3 .*SNMOT-060"):
        clip.frame(3)


def test_soccernet_tracking_has_identities_boxes_and_ball_but_no_pitch_truth(tmp_path):
    clip = load_labelled_clip(_sequence(tmp_path))

    assert clip.truth == {Truth.IDENTITIES, Truth.BOXES, Truth.BALL}
    assert Truth.PITCH_POSITIONS not in clip.truth
    assert Truth.CALIBRATION not in clip.truth
    assert all(frame.homography is None for frame in clip)


def test_loading_copies_and_writes_no_frame(tmp_path):
    folder = _sequence(tmp_path)
    (folder / "img1").mkdir()
    cv2.imwrite(str(folder / "img1" / "000001.jpg"), np.zeros((8, 8, 3), np.uint8))
    before = sorted(tmp_path.rglob("*"))

    clip = load_labelled_clip(folder)

    assert sorted(tmp_path.rglob("*")) == before
    # The clip points at the sequence's own images, in place.
    assert clip.video_path == folder / "img1" / "%06d.jpg"


@pytest.mark.parametrize("missing", ["gt/gt.txt", "gameinfo.ini", "seqinfo.ini"])
def test_a_missing_soccernet_tracking_file_names_it_and_the_layout(tmp_path, missing):
    folder = _sequence(tmp_path)
    (folder / missing).unlink()

    with pytest.raises(ValueError) as error:
        load_labelled_clip(folder)

    message = str(error.value)
    assert str(folder / missing) in message
    assert "gt/gt.txt, gameinfo.ini, seqinfo.ini" in message


@pytest.mark.parametrize("make", [lambda p: p.mkdir(), lambda p: p.write_text("x")])
def test_an_unknown_layout_names_the_path_and_the_expected_layout(tmp_path, make):
    path = tmp_path / "clip"
    make(path)

    with pytest.raises(ValueError) as error:
        load_labelled_clip(path)

    message = str(error.value)
    assert str(path) in message
    assert "SoccerNet Tracking" in message
    assert "gt/gt.txt, gameinfo.ini, seqinfo.ini" in message


def test_a_box_of_a_tracklet_missing_from_gameinfo_fails(tmp_path):
    folder = _sequence(tmp_path, gt=GT + "2,9,1,1,5,5,1,-1,-1,-1\n")

    with pytest.raises(ValueError, match=r"track 9\b.*gameinfo\.ini"):
        load_labelled_clip(folder)


def test_an_unknown_tracklet_class_fails(tmp_path):
    folder = _sequence(tmp_path, gameinfo=GAMEINFO.replace("other;1", "coach;1"))

    with pytest.raises(ValueError, match="'coach'"):
        load_labelled_clip(folder)


def test_a_box_outside_the_sequence_fails(tmp_path):
    folder = _sequence(tmp_path, gt=GT + "4,1,1,1,5,5,1,-1,-1,-1\n")

    with pytest.raises(ValueError, match=r"frame 4\b.*seqLength=3"):
        load_labelled_clip(folder)


def test_a_seqinfo_without_its_sequence_values_fails(tmp_path):
    folder = _sequence(tmp_path, seqinfo="[Sequence]\nname=SNMOT-060\n")

    with pytest.raises(ValueError, match=r"seqinfo\.ini.*frameRate"):
        load_labelled_clip(folder)
