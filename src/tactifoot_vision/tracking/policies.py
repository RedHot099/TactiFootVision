from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReseedPolicy:
    interval: int | None = None
    iou_threshold: float = 0.3

    def should_reseed(self, frame_index: int) -> bool:
        return (
            self.interval is not None
            and frame_index > 0
            and frame_index % self.interval == 0
        )


@dataclass(frozen=True, slots=True)
class TrackLifecyclePolicy:
    drop_after_missing_frames: int = 30
