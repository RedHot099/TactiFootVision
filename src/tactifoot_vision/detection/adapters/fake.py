from tactifoot_vision.domain import BBox, Detection, DetectionSet, Frame


class FakeDetector:
    def predict(self, frame: Frame) -> DetectionSet:
        offset = float(frame.index)
        return DetectionSet(
            (
                Detection(
                    bbox=BBox(10.0 + offset, 20.0, 50.0 + offset, 100.0),
                    class_id=2,
                    class_name="player",
                    confidence=0.95,
                ),
                Detection(
                    bbox=BBox(80.0, 40.0 + offset, 95.0, 55.0 + offset),
                    class_id=0,
                    class_name="ball",
                    confidence=0.80,
                ),
            )
        )
