from pathlib import Path

import numpy as np
import pytest

from tactifoot_vision.config import SAM2Config
from tactifoot_vision.domain import BBox, Detection, DetectionSet, Frame, PipelineError
from tactifoot_vision.enums import Sam2OutputBoxMode
from tactifoot_vision.tracking import SAM2Tracker


class FakePredictor:
    def __init__(self) -> None:
        self.prompts: list[tuple[int, np.ndarray]] = []
        self.loaded_frames = 0
        self.reset_calls = 0
        self.fail_track = False

    def reset_state(self) -> None:
        self.reset_calls += 1

    def load_first_frame(self, frame: np.ndarray) -> None:
        self.loaded_frames += 1
        self.frame_shape = frame.shape

    def add_new_prompt(self, *, frame_idx: int, obj_id: int, bbox: np.ndarray) -> None:
        _ = frame_idx
        self.prompts.append((obj_id, bbox.copy()))

    def track(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _ = frame
        if self.fail_track:
            raise RuntimeError("predictor failed")
        ids = np.array([prompt[0] for prompt in self.prompts], dtype=int)
        logits = np.zeros((len(ids), 12, 12), dtype=np.float32)
        for index, (_obj_id, bbox) in enumerate(self.prompts):
            x1, y1, x2, y2 = bbox[0].astype(int)
            logits[index, max(0, y1) : min(12, y2), max(0, x1) : min(12, x2)] = 3.0
        return ids, logits


@pytest.fixture()
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "sam2.pt"
    config = tmp_path / "sam2.yaml"
    checkpoint.write_text("checkpoint")
    config.write_text("config")
    return checkpoint, config


@pytest.fixture()
def frame() -> Frame:
    return Frame(index=0, image=np.zeros((12, 12, 3), dtype=np.uint8))


def _config(
    artifacts: tuple[Path, Path],
    **kwargs: object,
) -> SAM2Config:
    checkpoint, config = artifacts
    return SAM2Config(
        checkpoint=checkpoint,
        model_config_path=config,
        min_mask_area=1.0,
        **kwargs,
    )


def _detections(count: int = 1) -> DetectionSet:
    detections = []
    for index in range(count):
        detections.append(
            Detection(
                bbox=BBox(1.0 + index, 1.0 + index, 5.0 + index, 5.0 + index),
                class_id=2,
                class_name="player",
                confidence=0.9,
            )
        )
    return DetectionSet(tuple(detections))


def test_first_update_initializes_with_detections(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts), predictor_factory=lambda *_args: predictor
    )

    result = tracker.update(frame, _detections())

    assert len(result) == 1
    assert predictor.loaded_frames == 1
    assert predictor.prompts[0][0] == 1
    assert result.tracks[0].class_name == "player"


def test_second_update_tracks_without_detections(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts), predictor_factory=lambda *_args: predictor
    )
    tracker.update(frame, _detections())

    result = tracker.update(
        Frame(index=1, image=np.zeros((12, 12, 3), dtype=np.uint8)),
        DetectionSet.empty(),
    )

    assert len(result) == 1
    assert result.tracks[0].track_id == 1


def test_empty_first_detections_return_empty_tracks(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts), predictor_factory=lambda *_args: predictor
    )

    result = tracker.update(frame, DetectionSet.empty())

    assert len(result) == 0
    assert predictor.loaded_frames == 1


def test_max_objects_limits_prompts(artifacts: tuple[Path, Path], frame: Frame) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts, max_objects=1), predictor_factory=lambda *_args: predictor
    )

    tracker.update(frame, _detections(3))

    assert len(predictor.prompts) == 1


def test_reseed_interval_refreshes_prompts(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts, reseed_interval=1),
        predictor_factory=lambda *_args: predictor,
    )
    tracker.update(frame, _detections())

    tracker.update(
        Frame(index=1, image=np.zeros((12, 12, 3), dtype=np.uint8)),
        _detections(),
    )

    assert predictor.loaded_frames == 2


def test_reseed_iou_reuses_matching_track_id(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts, reseed_interval=1, reseed_iou=0.2),
        predictor_factory=lambda *_args: predictor,
    )
    tracker.update(frame, _detections())

    result = tracker.update(
        Frame(index=1, image=np.zeros((12, 12, 3), dtype=np.uint8)),
        _detections(),
    )

    assert result.tracks[0].track_id == 1


@pytest.mark.parametrize(
    "mode",
    [
        Sam2OutputBoxMode.MASK,
        Sam2OutputBoxMode.DETECTOR,
        Sam2OutputBoxMode.DETECTOR_STRICT,
        Sam2OutputBoxMode.DETECTOR_BLEND,
    ],
)
def test_output_box_modes_do_not_crash(
    artifacts: tuple[Path, Path],
    frame: Frame,
    mode: Sam2OutputBoxMode,
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts, reseed_interval=1, output_box_mode=mode),
        predictor_factory=lambda *_args: predictor,
    )
    tracker.update(frame, _detections())

    result = tracker.update(
        Frame(index=1, image=np.zeros((12, 12, 3), dtype=np.uint8)),
        _detections(),
    )

    assert len(result) == 1


def test_predictor_failure_becomes_pipeline_error(
    artifacts: tuple[Path, Path], frame: Frame
) -> None:
    predictor = FakePredictor()
    tracker = SAM2Tracker(
        _config(artifacts), predictor_factory=lambda *_args: predictor
    )
    tracker.update(frame, _detections())
    predictor.fail_track = True

    with pytest.raises(PipelineError, match="frame 1"):
        tracker.update(
            Frame(index=1, image=np.zeros((12, 12, 3), dtype=np.uint8)),
            DetectionSet.empty(),
        )
