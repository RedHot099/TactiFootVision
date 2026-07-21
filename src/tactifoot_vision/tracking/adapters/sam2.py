from collections.abc import Callable
from typing import Any

from tactifoot_vision.config.schemas import SAM2Config
from tactifoot_vision.domain import (
    DetectionSet,
    Frame,
    PipelineError,
    TrackSet,
)
from tactifoot_vision.enums import Sam2OutputBoxMode

PredictorFactory = Callable[[str, str, str], object]


class SAM2Tracker:
    def __init__(
        self,
        config: SAM2Config,
        *,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.config = config
        from tactifoot_vision.tracking.sam2_runtime import SAM2Runtime

        self._runtime: Any = SAM2Runtime(config, predictor_factory=predictor_factory)

    def update(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        try:
            if not self._runtime.initialized:
                return self._runtime.initialize(frame, detections)
            tracks = self._runtime.track(frame)
            if self._should_reseed(frame, detections):
                refreshed = self._runtime.refresh_prompts(frame, detections)
                return self._apply_output_box_mode(tracks, refreshed)
            return tracks
        except Exception as exc:
            if isinstance(exc, PipelineError):
                raise
            raise PipelineError(f"SAM2 tracking failed on frame {frame.index}") from exc

    def reset(self) -> None:
        self._runtime.reset()

    def _should_reseed(self, frame: Frame, detections: DetectionSet) -> bool:
        if self.config.reseed_interval is None or len(detections) == 0:
            return False
        return frame.index > 0 and frame.index % self.config.reseed_interval == 0

    def _apply_output_box_mode(
        self, mask_tracks: TrackSet, refreshed_tracks: TrackSet
    ) -> TrackSet:
        if self.config.output_box_mode == Sam2OutputBoxMode.MASK:
            return mask_tracks
        if self.config.output_box_mode == Sam2OutputBoxMode.DETECTOR:
            return refreshed_tracks
        if self.config.output_box_mode == Sam2OutputBoxMode.DETECTOR_STRICT:
            return refreshed_tracks if len(refreshed_tracks) else mask_tracks
        if self.config.output_box_mode == Sam2OutputBoxMode.DETECTOR_BLEND:
            return _blend_tracks(mask_tracks, refreshed_tracks)
        return mask_tracks


def _blend_tracks(mask_tracks: TrackSet, detector_tracks: TrackSet) -> TrackSet:
    from tactifoot_vision.domain import BBox, Track

    detector_by_id = {track.track_id: track for track in detector_tracks}
    blended = []
    for mask_track in mask_tracks:
        detector_track = detector_by_id.get(mask_track.track_id)
        if detector_track is None:
            blended.append(mask_track)
            continue
        blended.append(
            Track(
                track_id=mask_track.track_id,
                bbox=BBox(
                    x1=0.5 * mask_track.bbox.x1 + 0.5 * detector_track.bbox.x1,
                    y1=0.5 * mask_track.bbox.y1 + 0.5 * detector_track.bbox.y1,
                    x2=0.5 * mask_track.bbox.x2 + 0.5 * detector_track.bbox.x2,
                    y2=0.5 * mask_track.bbox.y2 + 0.5 * detector_track.bbox.y2,
                ),
                class_id=mask_track.class_id,
                class_name=mask_track.class_name,
                confidence=mask_track.confidence,
                team_id=mask_track.team_id,
                data=mask_track.data,
            )
        )
    return TrackSet(tuple(blended))
