from tactifoot_vision.domain import DetectionSet, Frame, Track, TrackSet


class FakeTracker:
    def update(self, frame: Frame, detections: DetectionSet) -> TrackSet:
        tracks: list[Track] = []
        for index, detection in enumerate(detections):
            tracks.append(
                Track(
                    track_id=index + 1,
                    bbox=detection.bbox,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    data=detection.data,
                )
            )
        return TrackSet(tuple(tracks))

    def reset(self) -> None:
        return None
