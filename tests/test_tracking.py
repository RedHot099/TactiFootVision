from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.tracking import (
    ByteTrackTracker,
    SAM2Tracker,
    available_trackers,
    clean_ball_path,
    create_tracker,
)
from tactifoot_vision.tracking import sam2 as sam2_module

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "external" / "segment-anything-2-real-time"
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def _detections(boxes, names, confidence=0.9) -> sv.Detections:
    names = np.array(names, dtype=str)
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.full(len(names), confidence, dtype=np.float32),
        class_id=np.array([{"player": 2, "referee": 3}[n] for n in names], dtype=int),
        data={"class_name": names},
    )


# ------------------------------------------------------------------ bytetrack
def test_registry_has_both_backends():
    assert {"bytetrack", "sam2"} <= set(available_trackers())
    assert isinstance(create_tracker("bytetrack"), ByteTrackTracker)


def test_bytetrack_keeps_ids_and_class_names():
    tracker = ByteTrackTracker()
    tracker.reset(fps=25)
    ids = []
    for step in range(6):
        boxes = [[10 + step, 10, 30 + step, 50], [200 - step, 100, 220 - step, 140]]
        detections = _detections(boxes, ["player", "referee"])
        tracked = tracker.update(detections, FRAME)
        assert detections.tracker_id is None  # the input is not modified
        assert len(tracked) == 2
        by_name = dict(zip(tracked.data["class_name"], tracked.tracker_id, strict=True))
        ids.append((by_name["player"], by_name["referee"]))
        # class names stay aligned with their boxes
        player = tracked.data["class_name"] == "player"
        assert tracked.xyxy[player][0][0] == pytest.approx(10 + step)
    assert len(set(ids)) == 1
    assert ids[0][0] != ids[0][1]


def test_bytetrack_empty_output_keeps_data_keys():
    tracker = ByteTrackTracker()
    tracked = tracker.update(_detections(np.zeros((0, 4)), []), FRAME)
    assert len(tracked) == 0
    assert "class_name" in tracked.data
    assert tracked.tracker_id is not None and len(tracked.tracker_id) == 0
    # low-confidence detections do not start tracks but keep their data keys
    tracked = tracker.update(_detections([[0, 0, 10, 10]], ["player"], 0.05), FRAME)
    assert len(tracked) == 0 and "class_name" in tracked.data


def test_bytetrack_reset_uses_video_fps_unless_fixed():
    tracker = ByteTrackTracker(lost_track_buffer=30)
    tracker.reset(fps=60)
    assert tracker._tracker.max_time_lost == 60
    fixed = ByteTrackTracker(lost_track_buffer=30, frame_rate=30)
    fixed.reset(fps=60)
    assert fixed._tracker.max_time_lost == 30


# ----------------------------------------------------------------------- ball
def test_clean_ball_path_drops_jumps():
    path = np.array(
        [[0, 0], [1, 0], [np.nan, np.nan], [2, 0], [50, 50], [3, 0], [3.5, 0.5]],
        dtype=float,
    )
    cleaned = clean_ball_path(path, max_jump=2.0)
    assert np.isnan(cleaned[2]).all()
    assert np.isnan(cleaned[4]).all()  # outlier
    np.testing.assert_allclose(cleaned[[0, 1, 3, 5, 6]], path[[0, 1, 3, 5, 6]])
    assert np.isfinite(path[4]).all()  # input untouched


def test_clean_ball_path_allows_longer_moves_after_gaps():
    # undetected for 9 frames while travelling 15 m: plausible at 2 m per frame
    path = np.array([[0, 0]] + [[np.nan, np.nan]] * 9 + [[15, 0], [16, 0]], dtype=float)
    np.testing.assert_allclose(clean_ball_path(path, 2.0), path)
    # the same move within one frame is not
    cleaned = clean_ball_path(path[[0, 10, 11]], 2.0)
    assert np.isnan(cleaned[1:]).all()
    # a rejected outlier does not lock out the path: the allowance keeps growing
    path = np.array([[0, 0], [30, 0], [30, 0], [30, 0], [30, 0]], dtype=float)
    assert np.isnan(clean_ball_path(path, 8.0)[1:3]).all()
    assert np.isfinite(clean_ball_path(path, 8.0)[4]).all()


def test_clean_ball_path_validates_input():
    with pytest.raises(ValueError):
        clean_ball_path(np.zeros((3, 2)), max_jump=0)
    with pytest.raises(ValueError):
        clean_ball_path(np.zeros((3, 3)), max_jump=1)
    assert clean_ball_path(np.zeros((0, 2)), 1.0).shape == (0, 2)


# ----------------------------------------------------------------------- SAM2
class _FakePredictor:
    """Stands in for SAM2's camera predictor: masks are the prompted boxes."""

    def __init__(self) -> None:
        self.prompts: dict[int, np.ndarray] = {}
        self.hidden: set[int] = set()
        self.loads = 0
        self.frame_idx = 0

    def load_first_frame(self, frame):
        self.prompts = {}
        self.shape = frame.shape[:2]
        self.loads += 1

    def add_new_prompt(self, frame_idx, obj_id, bbox):
        assert frame_idx == 0 and bbox.shape == (1, 4)
        self.prompts[obj_id] = bbox[0].astype(int)

    def track(self, frame):
        import torch

        self.frame_idx += 1
        ids = list(self.prompts)
        logits = torch.full((len(ids), 1, *self.shape), -1.0)
        for i, obj_id in enumerate(ids):
            if obj_id not in self.hidden:
                x1, y1, x2, y2 = self.prompts[obj_id]
                logits[i, 0, y1:y2, x1:x2] = 1.0
        return ids, logits


@pytest.fixture
def fake_sam2(monkeypatch):
    predictor = _FakePredictor()
    monkeypatch.setattr(sam2_module, "_build_predictor", lambda *args: predictor)

    def make(**kwargs) -> tuple[SAM2Tracker, _FakePredictor]:
        return SAM2Tracker("sam.pt", "sam.yaml", device="cpu", **kwargs), predictor

    return make


def test_sam2_seeds_from_first_detections(fake_sam2):
    tracker, predictor = fake_sam2()
    detections = _detections(
        [[10, 10, 30, 50], [100, 20, 120, 60]], ["player", "referee"]
    )
    tracked = tracker.update(detections, FRAME)
    assert list(tracked.tracker_id) == [1, 2]
    assert list(tracked.data["class_name"]) == ["player", "referee"]
    assert list(tracked.class_id) == [2, 3]
    assert tracked.mask is not None and tracked.mask.shape == (2, 240, 320)
    np.testing.assert_allclose(tracked.xyxy[0], [10, 10, 29, 49])  # from the mask
    predictor.hidden.add(1)  # an object whose mask vanishes is left out
    tracked = tracker.update(detections, FRAME)
    assert list(tracked.tracker_id) == [2]


def test_sam2_promotes_new_objects_after_min_hits(fake_sam2):
    tracker, predictor = fake_sam2(candidate_min_hits=3, reseed_interval=0)
    known = [[10, 10, 30, 50]]
    tracker.update(_detections(known, ["player"]), FRAME)
    boxes = [*known, [200, 100, 220, 140]]
    counts = []
    for _ in range(4):
        tracked = tracker.update(_detections(boxes, ["player", "referee"]), FRAME)
        counts.append(len(tracked))
    assert counts == [1, 1, 2, 2]  # promoted on the third sighting
    assert predictor.loads == 2  # initial prompt + one re-seed
    by_id = dict(zip(tracked.tracker_id, tracked.data["class_name"], strict=True))
    assert by_id == {1: "player", 2: "referee"}


def test_sam2_reseed_respects_cooldown(fake_sam2):
    tracker, predictor = fake_sam2(candidate_min_hits=1, reseed_interval=5)
    tracker.update(_detections([[10, 10, 30, 50]], ["player"]), FRAME)
    boxes = [[10, 10, 30, 50], [200, 100, 220, 140]]
    tracker.update(_detections(boxes, ["player", "player"]), FRAME)  # frame 1: re-seed
    assert predictor.loads == 2
    boxes.append([100, 150, 120, 190])
    lengths = [
        len(tracker.update(_detections(boxes, ["player"] * 3), FRAME)) for _ in range(5)
    ]
    assert lengths == [2, 2, 2, 2, 3]  # frames 2-5 cool down, frame 6 re-seeds
    tracker.reset()
    assert len(tracker.update(_detections(boxes[:1], ["player"]), FRAME)) == 1


def test_sam2_config_resolution(tmp_path):
    repo = tmp_path / "sam2-checkout"
    (repo / "sam2" / "configs" / "sam2.1").mkdir(parents=True)
    (repo / "sam2" / "build_sam.py").touch()
    config = repo / "sam2" / "configs" / "sam2.1" / "tiny.yaml"
    config.touch()
    assert sam2_module._resolve_config(config, None) == (
        repo,
        "configs/sam2.1/tiny.yaml",
    )
    assert sam2_module._resolve_config(Path("configs/sam2.1/tiny.yaml"), repo) == (
        repo,
        "configs/sam2.1/tiny.yaml",
    )
    outside = tmp_path / "other.yaml"
    outside.touch()
    with pytest.raises(ValueError, match="inside"):
        sam2_module._resolve_config(outside, repo)
    with pytest.raises(ValueError, match="repo_dir"):
        sam2_module._resolve_config(outside, None)
    with pytest.raises(FileNotFoundError):
        sam2_module._resolve_config(Path("configs/missing.yaml"), repo)


@pytest.mark.model
def test_sam2_real_checkpoint_tracks_boxes():
    pytest.importorskip("hydra", reason="SAM2 needs hydra-core")
    checkpoint = SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"
    if not checkpoint.is_file():
        pytest.skip("SAM2 checkout/checkpoint not available")
    from tactifoot_vision.data import VideoReader

    tracker = SAM2Tracker(
        checkpoint, SAM2_REPO / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
    )
    reader = VideoReader(ROOT / "data" / "videos" / "broadcast_60s.mp4")
    frame = reader.read(0)
    # two players from frame 0 (1080p)
    boxes = [[356, 564, 382, 618], [925, 446, 949, 494]]
    tracked = tracker.update(_detections(boxes, ["player", "player"]), frame)
    assert set(tracked.tracker_id) <= {1, 2}
    for _, frame in reader.frames(start=1, end=6):
        tracked = tracker.update(_detections(boxes, ["player", "player"]), frame)
    assert len(tracked) >= 1
    assert tracked.mask is not None and tracked.mask.any(axis=(1, 2)).all()
