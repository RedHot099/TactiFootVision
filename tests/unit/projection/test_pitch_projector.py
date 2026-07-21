import numpy as np

from tactifoot_vision.domain import BBox, Frame, Track, TrackSet
from tactifoot_vision.keypoints import Keypoint, KeypointSet
from tactifoot_vision.projection import HomographyEstimator, PitchProjector


class FakeKeypointDetector:
    def predict(self, frame: Frame) -> KeypointSet:
        _ = frame
        return KeypointSet(
            (
                Keypoint(0, 0.0, 0.0, 1.0),
                Keypoint(2, 100.0, 0.0, 1.0),
                Keypoint(3, 0.0, 100.0, 1.0),
                Keypoint(5, 100.0, 100.0, 1.0),
            )
        )


def test_pitch_projector_projects_track_bottom_center() -> None:
    projector = PitchProjector(
        keypoint_detector=FakeKeypointDetector(),
        estimator=HomographyEstimator(min_keypoints=4),
    )
    frame = Frame(index=0, image=np.zeros((100, 100, 3), dtype=np.uint8))
    tracks = TrackSet((Track(1, BBox(40.0, 40.0, 60.0, 80.0), 2, "player"),))

    result = projector.project(frame=frame, keypoints=None, tracks=tracks)

    assert result.status == "available"
    assert result.points_by_track_id[1].x == np.float32(52.5)
    assert result.points_by_track_id[1].y == np.float32(54.4)


def test_pitch_projector_empty_keypoints_unavailable() -> None:
    projector = PitchProjector()

    result = projector.project(
        frame=Frame(index=0, image=np.zeros((10, 10, 3), dtype=np.uint8)),
        keypoints=KeypointSet.empty(),
        tracks=TrackSet.empty(),
    )

    assert result.status == "unavailable"


def test_pitch_projector_skips_ball_when_ball_projection_disabled() -> None:
    projector = PitchProjector(
        keypoint_detector=FakeKeypointDetector(),
        estimator=HomographyEstimator(min_keypoints=4),
        project_ball=False,
    )
    frame = Frame(index=0, image=np.zeros((100, 100, 3), dtype=np.uint8))
    tracks = TrackSet((Track(7, BBox(40.0, 40.0, 50.0, 50.0), 0, "ball"),))

    result = projector.project(frame=frame, keypoints=None, tracks=tracks)

    assert result.status == "available"
    assert result.ball is None
    assert 7 not in result.points_by_track_id
