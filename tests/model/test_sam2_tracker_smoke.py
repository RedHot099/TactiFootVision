from pathlib import Path

import pytest

from tactifoot_vision.config import SAM2Config
from tactifoot_vision.domain import BBox, Detection, DetectionSet, TrackSet
from tactifoot_vision.io import read_frames
from tactifoot_vision.tracking import SAM2Tracker


@pytest.mark.model
@pytest.mark.sam2
def test_sam2_tracker_smoke() -> None:
    checkpoint = Path(
        "external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"
    )
    config_path = Path(
        "external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    if not checkpoint.is_file():
        pytest.skip(f"SAM2 checkpoint not found: {checkpoint}")
    if not config_path.is_file():
        pytest.skip(f"SAM2 config not found: {config_path}")
    frames = list(read_frames("data/soccernet_dummy/img1"))[:2]
    if len(frames) < 2:
        pytest.skip("SAM2 smoke test requires at least two dummy frames.")

    try:
        tracker = SAM2Tracker(
            SAM2Config(
                checkpoint=checkpoint,
                model_config_path=config_path,
                device="auto",
                max_side=512,
                max_objects=1,
                min_mask_area=1.0,
            )
        )
    except Exception as exc:
        pytest.skip(f"SAM2 runtime is unavailable: {exc}")

    seed = DetectionSet(
        (
            Detection(
                bbox=BBox(10.0, 10.0, 80.0, 120.0),
                class_id=2,
                class_name="player",
                confidence=0.9,
            ),
        )
    )
    first = tracker.update(frames[0], seed)
    second = tracker.update(frames[1], DetectionSet.empty())

    assert isinstance(first, TrackSet)
    assert isinstance(second, TrackSet)
    for tracks in (first, second):
        for track in tracks:
            assert track.track_id > 0
            assert track.bbox.width >= 0
            assert track.bbox.height >= 0
            assert track.class_name
            assert track.confidence is None or 0.0 <= track.confidence <= 1.0
