import numpy as np

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
from tactifoot_vision.shots import MetadataShotDetector
from tactifoot_vision.xg import VideoXgEstimator


def test_video_xg_estimator_sums_two_detected_shots() -> None:
    result = PipelineResult(tuple(_frame(frame) for frame in range(1, 7)))
    estimator = VideoXgEstimator(
        shot_detector=MetadataShotDetector(
            action_frames=(2, 5),
            action_class="Goal",
            window_before=1,
            window_after=1,
        )
    )

    summary = estimator.run(result, group_id="match-1")

    assert summary.group_id == "match-1"
    assert summary.shot_count == 2
    assert summary.total_xg == sum(prediction.xg for prediction in summary.predictions)
    assert all(0.0 < prediction.xg < 1.0 for prediction in summary.predictions)


def _frame(frame_index: int) -> FrameResult:
    ball_x = 90.0 + frame_index
    tracks = TrackSet(
        (
            Track(
                track_id=1,
                bbox=BBox(ball_x - 1.0, 33.0, ball_x + 1.0, 35.0),
                class_id=0,
                class_name="ball",
                confidence=1.0,
            ),
            Track(
                track_id=2,
                bbox=BBox(0.0, 0.0, 1.0, 1.0),
                class_id=2,
                class_name="player",
                confidence=1.0,
            ),
        )
    )
    projection = PitchProjection(
        status="available",
        points_by_track_id={2: PitchPoint(96.0, 34.0)},
        ball=PitchPoint(ball_x, 34.0),
        homography=np.eye(3),
    )
    return FrameResult(
        frame_index=frame_index,
        timestamp_seconds=None,
        detections=DetectionSet.empty(),
        tracks=tracks,
        projection=projection,
    )
