import pytest

from tactifoot_vision.domain import DetectionSet, FrameResult, PipelineResult, TrackSet


def test_pipeline_result_rejects_duplicate_frame_indexes() -> None:
    frame = FrameResult(
        frame_index=0,
        timestamp_seconds=0.0,
        detections=DetectionSet.empty(),
        tracks=TrackSet.empty(),
    )

    with pytest.raises(ValueError, match="duplicate frame indexes"):
        PipelineResult((frame, frame))


def test_empty_pipeline_result_can_be_created() -> None:
    result = PipelineResult(())

    assert result.frames == ()
    assert result.artifacts == ()
