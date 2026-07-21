import numpy as np

from tactifoot_vision.detection import FakeDetector
from tactifoot_vision.domain import Frame
from tactifoot_vision.pipeline import InferencePipeline
from tactifoot_vision.tracking import FakeTracker


def test_fake_pipeline_processes_three_frames_without_duplicates() -> None:
    frames = [
        Frame(
            index=i,
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            timestamp_seconds=float(i),
        )
        for i in range(3)
    ]

    result = InferencePipeline(detector=FakeDetector(), tracker=FakeTracker()).run(
        frames
    )

    assert [frame.frame_index for frame in result.frames] == [0, 1, 2]
    assert [len(frame.tracks) for frame in result.frames] == [2, 2, 2]
