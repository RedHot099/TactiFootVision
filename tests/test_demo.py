"""download_demo lays one SoccerTrack v2 match out for the package readers, from a stubbed source."""

import json

import cv2
import numpy as np
import pytest
import requests
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

import tactifoot_vision as tv
from tactifoot_vision.data import demo

FPS, FRAMES, SIZE = 10, 50, (64, 48)
K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
D = np.zeros((4, 1))
KNEW = np.array([[25.0, 0.0, 32.0], [0.0, 25.0, 24.0], [0.0, 0.0, 1.0]])
# Pitch metres -> frame pixels of the stub: x scaled, y flipped.
PITCH_TO_FRAME = np.array([[0.5, 0.0, 2.0], [0.0, -0.5, 40.0], [0.0, 0.0, 1.0]])


def _source(match: str, into, token=None):
    """A stub for demo._download: a tiny clip with two tracks in every frame, and the raw files."""
    clip = into / "mot" / "clips" / f"{match}.mp4"
    clip.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), (i * 5) % 255, dtype=np.uint8))
    writer.release()
    # Frames count from 0, as the dataset's MOT files do.
    rows = [
        f"{f},{t},{10 * t + f % 3},{5},{8},{12},-1,-1,-1,-1"
        for f in range(FRAMES)
        for t in (1, 2)
    ]
    (into / "mot" / f"{match}.txt").write_text("\n".join(rows) + "\n")
    raw = into / "raw" / match
    raw.mkdir(parents=True)
    (raw / f"{match}_keypoints.json").write_text("{}")
    np.savez(raw / f"{match}_camera_intrinsics.npz", K=K, D=D, Knew=KNEW)
    np.save(raw / f"{match}_homography.npy", PITCH_TO_FRAME)
    bas = into / "bas" / match
    bas.mkdir(parents=True)
    (bas / f"{match}_12_class_events.json").write_text("[]")


@pytest.fixture
def stub_source(monkeypatch):
    calls = []

    def download(match, into, token):
        calls.append((match, token))
        _source(match, into, token)

    monkeypatch.setattr(demo, "_download", download)
    return calls


def test_the_demo_lays_out_a_clip_and_a_detection_set(tmp_path, stub_source, capsys):
    data = tv.data.download_demo(
        tmp_path / "demo", match="M", clip_seconds=2.0, train_stride=5
    )

    assert stub_source == [("M", None)]
    assert data.root == tmp_path / "demo" and data.match == "M"
    clip = tv.data.load_labelled_clip(data.clip_dir)
    assert (
        len(clip),
        clip.fps,
        clip.size if hasattr(clip, "size") else (clip.width, clip.height),
    ) == (20, FPS, SIZE)
    assert {int(t) for f in clip for t in f.detections.tracker_id} == {1, 2}
    assert set(clip.frame(0).detections.data["class_name"]) == {"player"}
    # The clip is the last 2 s: MOT frame 30 became its first frame, with its
    # boxes in the clip's own pixels.
    assert clip.frame(0).detections.xyxy[0].tolist() == [
        10 + 30 % 3,
        5,
        18 + 30 % 3,
        17,
    ]
    calibration = np.load(data.calibration)
    assert set(calibration.files) == {"K", "D", "Knew", "frame_to_pitch"}
    assert np.allclose(calibration["K"], K) and np.allclose(calibration["D"], D)
    assert np.allclose(calibration["Knew"], KNEW)
    assert np.allclose(calibration["frame_to_pitch"], np.linalg.inv(PITCH_TO_FRAME))

    dataset = tv.data.Dataset.from_yolo(data.detect_yaml)
    assert "player" in dataset.class_names
    assert (len(dataset["train"]), len(dataset["valid"])) == (4, 2)
    assert all(len(s.annotations) == 2 for s in dataset)

    assert data.calibration_dir.is_dir() and data.events_json.is_file()
    assert (data.root / "M" / "ATTRIBUTION.md").read_text().strip() == demo.ATTRIBUTION
    assert data.attribution == demo.ATTRIBUTION
    assert "CC BY 4.0" in capsys.readouterr().out


def test_a_second_call_touches_no_network_and_returns_the_same_paths(
    tmp_path, stub_source, capsys
):
    first = tv.data.download_demo(tmp_path, match="M", clip_seconds=2.0, train_stride=5)
    again = tv.data.download_demo(tmp_path, match="M", clip_seconds=2.0, train_stride=5)

    assert again == first
    assert len(stub_source) == 1
    assert "CC BY 4.0" in capsys.readouterr().out


def test_other_arguments_on_the_same_root_are_refused(tmp_path, stub_source):
    tv.data.download_demo(tmp_path, match="M", clip_seconds=2.0, train_stride=5)

    with pytest.raises(ValueError, match="already holds a demo"):
        tv.data.download_demo(tmp_path, match="M", clip_seconds=1.0, train_stride=5)


def test_a_marker_records_what_was_made(tmp_path, stub_source):
    tv.data.download_demo(tmp_path, match="M", clip_seconds=2.0, train_stride=5)

    assert json.loads((tmp_path / "M" / demo.MARKER).read_text()) == {
        "dataset": demo.DATASET,
        "revision": demo.REVISION,
        "layout": demo.LAYOUT,
        "match": "M",
        "clip_seconds": 2.0,
        "train_stride": 5,
    }


def test_a_clip_as_long_as_the_video_is_refused(tmp_path, stub_source):
    with pytest.raises(ValueError, match="less than the whole clip"):
        tv.data.download_demo(
            tmp_path, match="M", clip_seconds=FRAMES / FPS, train_stride=5
        )
    assert not (tmp_path / "M" / demo.MARKER).exists()


def test_a_clip_shorter_than_a_frame_is_refused(tmp_path, stub_source):
    with pytest.raises(ValueError, match="at least a frame"):
        tv.data.download_demo(tmp_path, match="M", clip_seconds=0.01, train_stride=5)


def _hub_download_raising(error):
    def hf_hub_download(*args, **kwargs):
        raise error

    return hf_hub_download


def test_a_gated_download_says_how_to_get_access(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 403
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _hub_download_raising(
            GatedRepoError("Cannot access gated repo", response=response)
        ),
    )

    with pytest.raises(PermissionError, match="HF_TOKEN") as error:
        tv.data.download_demo(tmp_path)
    assert demo.DATASET_URL in str(error.value)


def test_a_refused_download_says_so(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 401
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _hub_download_raising(HfHubHTTPError("Unauthorized", response=response)),
    )

    with pytest.raises(PermissionError, match="HTTP 401"):
        tv.data.download_demo(tmp_path)


def test_a_failed_download_is_a_connection_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _hub_download_raising(OSError("Temporary failure in name resolution")),
    )

    with pytest.raises(
        ConnectionError, match="Could not download mot/clips/118577.mp4"
    ):
        tv.data.download_demo(tmp_path)


@pytest.mark.model
def test_the_real_match_downloads_and_loads_through_the_readers():
    """Needs the Hugging Face gate accepted and a token (saved login on omarchy); 300 MB once."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "demo"
    data = tv.data.download_demo(root)

    clip = tv.data.load_labelled_clip(data.clip_dir)
    assert (len(clip), clip.fps, clip.width, clip.height) == (750, 25, 4096, 1080)
    assert {int(t) for t in clip.frame(0).detections.tracker_id} == set(range(1, 23))
    assert all(0 <= x <= 4096 for x in clip.frame(0).detections.xyxy[:, [0, 2]].ravel())
    calibration = np.load(data.calibration)
    # Every player's feet land on the pitch, give or take the calibration's error.
    xyxy = clip.frame(0).detections.xyxy
    feet = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]].reshape(-1, 1, 2)
    moved = cv2.fisheye.undistortPoints(
        feet, calibration["K"], calibration["D"], P=calibration["Knew"]
    ).reshape(-1, 2)
    pitch = (calibration["frame_to_pitch"] @ np.c_[moved, np.ones(len(moved))].T).T
    pitch = pitch[:, :2] / pitch[:, 2:3]
    assert ((pitch[:, 0] > -5) & (pitch[:, 0] < 110)).all()
    assert ((pitch[:, 1] > -5) & (pitch[:, 1] < 73)).all()

    dataset = tv.data.Dataset.from_yolo(data.detect_yaml)
    assert len(dataset["train"]) == 350 and len(dataset["valid"]) == 175
    assert data.events_json.is_file() and data.calibration_dir.is_dir()
