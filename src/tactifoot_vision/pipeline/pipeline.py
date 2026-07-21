from collections.abc import Iterable
from pathlib import Path

from tactifoot_vision.detection import Detector
from tactifoot_vision.domain import Frame, FrameResult, PipelineResult
from tactifoot_vision.io import read_frames
from tactifoot_vision.pipeline.frame_processor import FrameProcessor
from tactifoot_vision.tracking import Tracker


class InferencePipeline:
    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        projector: object | None = None,
        team_assigner: object | None = None,
    ) -> None:
        self.processor = FrameProcessor(
            detector=detector,
            tracker=tracker,
            projector=projector,
            team_assigner=team_assigner,
        )

    def run(
        self, frames: Iterable[Frame], *, max_frames: int | None = None
    ) -> PipelineResult:
        results: list[FrameResult] = []
        for count, frame in enumerate(frames):
            if max_frames is not None and count >= max_frames:
                break
            results.append(self.processor.process(frame))
        return PipelineResult(tuple(results))

    def run_video(
        self, path: str | Path, *, max_frames: int | None = None
    ) -> PipelineResult:
        return self.run(read_frames(path), max_frames=max_frames)
