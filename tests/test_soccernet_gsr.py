"""The SoccerNet Game State Reconstruction reader, on synthetic ``Labels-GameState.json`` files."""

import json
import math

import numpy as np
import pytest

from tactifoot_vision.data import (
    UNKNOWN_TEAM,
    LabelledClip,
    Truth,
    load_labelled_clip,
)
from tactifoot_vision.pipeline import NO_TEAM
from tactifoot_vision.pitch import SoccerPitch
from tactifoot_vision.pitch.homography import frame_to_pitch

# A camera on the halfway line, 55 m behind the near touchline and 12 m up,
# looking at the centre spot: its principal point sees the centre spot.
TILT = math.degrees(math.atan2(55, 12))
CAMERA = {
    "pan_degrees": 0.0,
    "tilt_degrees": TILT,
    "roll_degrees": 0.0,
    "position_meters": [0.0, 55.0, -12.0],
    "x_focal_length": 2000.0,
    "y_focal_length": 2000.0,
    "principal_point": [960.0, 540.0],
    "radial_distortion": [0.0] * 6,
    "tangential_distortion": [0.0] * 2,
    "thin_prism_distortion": [0.0] * 4,
}


def _person(image, track, role, team, jersey, box, foot):
    """An object annotation as GSR ships it; ``foot`` is centre-origin metres or None."""
    x, y, w, h = box
    return {
        "id": f"{image}{track:02d}",
        "image_id": image,
        "track_id": track,
        "supercategory": "object",
        "category_id": {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4}.get(
            role, 1
        ),
        "attributes": {"role": role, "jersey": jersey, "team": team},
        "bbox_image": {
            "x": x,
            "y": y,
            "x_center": x + w / 2,
            "y_center": y + h / 2,
            "w": w,
            "h": h,
        },
        "bbox_pitch": None
        if foot is None
        else {"x_bottom_middle": foot[0], "y_bottom_middle": foot[1]},
    }


def _camera(image, parameters=CAMERA):
    return {
        "id": f"{image}01",
        "image_id": image,
        "supercategory": "camera",
        "category_id": 6,
        "parameters": parameters,
    }


ANNOTATIONS = [
    _person("1060000001", 1, "player", "left", "7", (100, 200, 30, 60), (0.0, 0.0)),
    _person("1060000001", 2, "goalkeeper", "right", "1", (500, 300, 40, 80), None),
    _person("1060000001", 3, "referee", None, None, (700, 400, 25, 70), (-52.5, -34)),
    _person("1060000001", 4, "player", None, None, (800, 400, 25, 70), (52.5, 34)),
    _person("1060000001", 5, "other", None, None, (0, 0, 10, 10), (1.0, 1.0)),
    _person("1060000001", 6, "ball", None, None, (910, 540, 8, 9), (2.0, 3.0)),
    _camera("1060000001"),
    {
        "id": "106000000100",
        "image_id": "1060000001",
        "supercategory": "pitch",
        "category_id": 5,
        "lines": {"Middle line": [{"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.9}]},
    },
    _person("1060000003", 1, "player", "left", "7", (104, 201, 30, 60), (0.5, 0.0)),
]


def _labels(annotations=ANNOTATIONS, seq_length=3):
    return {
        "info": {
            "version": "1.3",
            "name": "SNGS-060",
            "im_dir": "img1",
            "frame_rate": 25,
            "seq_length": seq_length,
            "im_ext": ".jpg",
        },
        "images": [
            {
                "is_labeled": True,
                "image_id": f"10600000{n:02d}",
                "file_name": f"{n:06d}.jpg",
                "height": 1080,
                "width": 1920,
            }
            for n in range(1, seq_length + 1)
        ],
        "annotations": annotations,
        "categories": [
            {"supercategory": "object", "id": 1, "name": "player"},
            {"supercategory": "object", "id": 2, "name": "goalkeeper"},
            {"supercategory": "object", "id": 3, "name": "referee"},
            {"supercategory": "object", "id": 4, "name": "ball"},
            {"supercategory": "pitch", "id": 5, "name": "pitch"},
            {"supercategory": "camera", "id": 6, "name": "camera"},
        ],
    }


def _sequence(root, labels=None, name="SNGS-060"):
    """A SoccerNet GSR sequence folder as the dataset ships it (no images)."""
    folder = root / name
    folder.mkdir()
    text = labels if isinstance(labels, str) else json.dumps(labels or _labels())
    (folder / "Labels-GameState.json").write_text(text)
    return folder


def test_soccernet_gsr_loads_boxes_roles_teams_jerseys_and_pitch_positions(tmp_path):
    folder = _sequence(tmp_path)

    clip = load_labelled_clip(folder)

    assert isinstance(clip, LabelledClip)
    assert clip.name == "SNGS-060"
    assert clip.fps == 25
    assert (clip.width, clip.height) == (1920, 1080)
    assert clip.video_path == folder / "img1" / "%06d.jpg"
    # Image 000001.jpg is the pipeline's frame 0; the unlabelled frame is kept.
    assert [frame.index for frame in clip] == [0, 1, 2]
    first = clip.frame(0).detections
    np.testing.assert_array_equal(
        first.xyxy,
        [[100, 200, 130, 260], [500, 300, 540, 380], [700, 400, 725, 470],
         [800, 400, 825, 470]],
    )  # fmt: skip
    np.testing.assert_array_equal(first.tracker_id, [1, 2, 3, 4])
    assert list(first.data["class_name"]) == [
        "player",
        "goalkeeper",
        "referee",
        "player",
    ]
    # left -> 0, right -> 1; a referee has no team, a player with a null team
    # an unknown one.
    np.testing.assert_array_equal(first.data["team_id"], [0, 1, NO_TEAM, UNKNOWN_TEAM])
    assert list(first.data["jersey"]) == ["7", "1", "", ""]
    np.testing.assert_array_equal(
        first.data["pitch_xy"], [[52.5, 34], [np.nan, np.nan], [0, 0], [105, 68]]
    )
    ball = clip.frame(0).ball
    np.testing.assert_array_equal(ball.xyxy, [[910, 540, 918, 549]])
    np.testing.assert_array_equal(ball.tracker_id, [6])
    assert list(ball.data["class_name"]) == ["ball"]
    assert len(clip.frame(1).detections) == len(clip.frame(1).ball) == 0
    np.testing.assert_array_equal(clip.frame(2).detections.tracker_id, [1])
    np.testing.assert_allclose(clip.frame(2).detections.data["pitch_xy"], [[53, 34]])


def test_a_gsr_centre_spot_is_the_soccer_pitch_centre_spot(tmp_path):
    """GSR is centre-origin metres, y towards the near touchline; SoccerPitch has its origin in a corner."""
    clip = load_labelled_clip(_sequence(tmp_path))
    pitch = SoccerPitch()

    pitch_xy = clip.frame(0).detections.data["pitch_xy"]

    np.testing.assert_array_equal(pitch_xy[0], [pitch.length / 2, pitch.width / 2])
    np.testing.assert_array_equal(pitch_xy[2], [0, 0])  # far left corner
    np.testing.assert_array_equal(pitch_xy[3], [pitch.length, pitch.width])


def test_a_gsr_camera_becomes_a_frame_to_pitch_homography(tmp_path):
    clip = load_labelled_clip(_sequence(tmp_path))
    homography = clip.frame(0).homography

    # The principal point sees the centre spot.
    np.testing.assert_allclose(
        frame_to_pitch([[960, 540]], homography), [[52.5, 34]], atol=1e-3
    )
    # The pinhole image of a pitch point maps back onto it: the near corner of
    # the right penalty box, centre-origin (36, 20.16), is (88.5, 54.16).
    tilt = math.radians(TILT)
    rotation = np.array(
        [
            [1, 0, 0],
            [0, math.cos(tilt), math.sin(tilt)],
            [0, -math.sin(tilt), math.cos(tilt)],
        ]
    )  # world -> camera
    camera = rotation @ (np.array([36.0, 20.16, 0.0]) - [0.0, 55.0, -12.0])
    pixel = 2000 * camera[:2] / camera[2] + [960, 540]
    np.testing.assert_allclose(
        frame_to_pitch([pixel], homography), [[88.5, 54.16]], atol=1e-3
    )
    # Frames without a camera annotation have no calibration.
    assert clip.frame(1).homography is None
    assert clip.frame(2).homography is None


def test_soccernet_gsr_reports_its_truth_kinds(tmp_path):
    clip = load_labelled_clip(_sequence(tmp_path))

    assert clip.truth == {
        Truth.IDENTITIES,
        Truth.BOXES,
        Truth.BALL,
        Truth.PITCH_POSITIONS,
        Truth.CALIBRATION,
        Truth.TEAMS,
    }


def test_a_gsr_sequence_without_cameras_has_no_calibration(tmp_path):
    people = [a for a in ANNOTATIONS if a["supercategory"] != "camera"]

    clip = load_labelled_clip(_sequence(tmp_path, _labels(people)))

    assert Truth.CALIBRATION not in clip.truth
    assert all(frame.homography is None for frame in clip)


def test_loading_a_gsr_sequence_writes_nothing(tmp_path):
    _sequence(tmp_path)
    before = sorted(tmp_path.rglob("*"))

    load_labelled_clip(tmp_path / "SNGS-060")

    assert sorted(tmp_path.rglob("*")) == before


def test_an_unknown_layout_names_the_gsr_layout_too(tmp_path):
    with pytest.raises(ValueError) as error:
        load_labelled_clip(tmp_path)

    assert "SoccerNet GSR sequence folder (Labels-GameState.json)" in str(error.value)


def test_a_gsr_file_that_is_not_json_names_it(tmp_path):
    folder = _sequence(tmp_path, "{not json")

    with pytest.raises(ValueError, match=r"Labels-GameState\.json is not"):
        load_labelled_clip(folder)


def test_a_gsr_file_without_its_sequence_values_names_them(tmp_path):
    labels = _labels()
    del labels["info"]["seq_length"]

    with pytest.raises(ValueError, match=r"Labels-GameState\.json.*seq_length"):
        load_labelled_clip(_sequence(tmp_path, labels))


def test_an_unknown_gsr_role_fails(tmp_path):
    coach = _person("1060000002", 9, "coach", None, None, (1, 1, 5, 5), None)

    with pytest.raises(ValueError, match=r"'coach'.*known: player"):
        load_labelled_clip(_sequence(tmp_path, _labels([*ANNOTATIONS, coach])))


def test_an_unknown_gsr_team_fails(tmp_path):
    middle = _person("1060000002", 9, "player", "middle", "3", (1, 1, 5, 5), None)

    with pytest.raises(ValueError, match=r"team 'middle'"):
        load_labelled_clip(_sequence(tmp_path, _labels([*ANNOTATIONS, middle])))


def test_a_gsr_annotation_of_an_unlisted_image_fails(tmp_path):
    stray = _person("1060000009", 9, "player", "left", "3", (1, 1, 5, 5), None)

    with pytest.raises(ValueError, match=r"image 1060000009.*not in its images"):
        load_labelled_clip(_sequence(tmp_path, _labels([*ANNOTATIONS, stray])))


def test_a_gsr_image_outside_the_sequence_fails(tmp_path):
    labels = _labels()
    labels["images"][-1]["file_name"] = "000004.jpg"

    with pytest.raises(ValueError, match=r"000004\.jpg.*seq_length=3"):
        load_labelled_clip(_sequence(tmp_path, labels))
