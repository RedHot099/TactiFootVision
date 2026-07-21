import numpy as np

from tactifoot_vision.ball import LinearBallTrajectoryReconstructor
from tactifoot_vision.domain import (
    BBox,
    DetectionSet,
    FrameResult,
    PipelineResult,
    PitchPoint,
    PitchProjection,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import BallTrajectorySource


def test_linear_reconstructor_fills_middle_and_edge_gaps() -> None:
    result = PipelineResult(
        (
            _frame(1),
            _frame(2, image_center=(20.0, 10.0), pitch_center=(2.0, 1.0)),
            _frame(3),
            _frame(4, image_center=(40.0, 30.0), pitch_center=(4.0, 3.0)),
            _frame(5),
        )
    )

    trajectory = LinearBallTrajectoryReconstructor().reconstruct(result)
    by_frame = trajectory.by_frame()

    assert by_frame[1].source == BallTrajectorySource.EXTRAPOLATED
    assert by_frame[1].image_x == 20.0
    assert by_frame[3].source == BallTrajectorySource.INTERPOLATED
    assert by_frame[3].image_x == 30.0
    assert by_frame[3].pitch_y == 2.0
    assert by_frame[5].source == BallTrajectorySource.EXTRAPOLATED
    assert trajectory.observed_count == 2


def test_linear_reconstructor_returns_missing_points_without_observations() -> None:
    result = PipelineResult((_frame(1), _frame(2)))

    trajectory = LinearBallTrajectoryReconstructor().reconstruct(result)

    assert len(trajectory) == 2
    assert all(point.source == BallTrajectorySource.MISSING for point in trajectory)


def test_linear_reconstructor_drops_single_frame_outlier() -> None:
    result = PipelineResult(
        (
            _frame(1, image_center=(0.0, 0.0)),
            _frame(2, image_center=(1000.0, 0.0)),
            _frame(3, image_center=(20.0, 0.0)),
        )
    )

    trajectory = LinearBallTrajectoryReconstructor(
        max_speed_pixels_per_frame=30.0
    ).reconstruct(result)

    point = trajectory.point_at(2)
    assert point is not None
    assert point.source == BallTrajectorySource.INTERPOLATED
    assert point.image_x == 10.0


def _frame(
    frame_index: int,
    *,
    image_center: tuple[float, float] | None = None,
    pitch_center: tuple[float, float] | None = None,
) -> FrameResult:
    tracks = TrackSet.empty()
    if image_center is not None:
        image_x, image_y = image_center
        tracks = TrackSet(
            (
                Track(
                    track_id=1,
                    bbox=BBox(
                        image_x - 1.0, image_y - 1.0, image_x + 1.0, image_y + 1.0
                    ),
                    class_id=0,
                    class_name="ball",
                    confidence=1.0,
                ),
            )
        )
    projection = None
    if pitch_center is not None:
        projection = PitchProjection(
            status="available",
            ball=PitchPoint(pitch_center[0], pitch_center[1]),
            homography=np.eye(3),
        )
    return FrameResult(
        frame_index=frame_index,
        timestamp_seconds=None,
        detections=DetectionSet.empty(),
        tracks=tracks,
        projection=projection,
    )
