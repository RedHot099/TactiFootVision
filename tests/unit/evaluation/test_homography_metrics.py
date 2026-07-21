import math

import pytest

from tactifoot_vision.datasets.soccernet_gsr import (
    GsrAthleteAnnotation,
    GsrFrame,
    GsrImageBBox,
    GsrPitchBBox,
    SoccerNetGsrLabels,
)
from tactifoot_vision.evaluation.homography import (
    HomographyRecord,
    project_gsr_athletes,
    summarize_homography_metrics,
)


def make_labels(pitch_x: float = 10.0, pitch_y: float = 20.0) -> SoccerNetGsrLabels:
    return SoccerNetGsrLabels(
        sequence="SNGS-001",
        version="1.3",
        frames=(
            GsrFrame(
                image_id="1001000001",
                frame=1,
                has_labeled_pitch=True,
                has_labeled_camera=True,
                has_labeled_person=True,
            ),
        ),
        athletes=(
            GsrAthleteAnnotation(
                annotation_id="1",
                image_id="1001000001",
                frame=1,
                track_id=7,
                role="player",
                jersey="9",
                team="left",
                bbox_image=GsrImageBBox(x=5.0, y=10.0, w=10.0, h=10.0),
                bbox_pitch=GsrPitchBBox(
                    x_bottom_middle=pitch_x,
                    y_bottom_middle=pitch_y,
                ),
            ),
        ),
        lines=(),
    )


def make_two_frame_labels() -> SoccerNetGsrLabels:
    first = make_labels()
    return SoccerNetGsrLabels(
        sequence=first.sequence,
        version=first.version,
        frames=(
            first.frames[0],
            GsrFrame(
                image_id="1001000002",
                frame=2,
                has_labeled_pitch=True,
                has_labeled_camera=True,
                has_labeled_person=True,
            ),
        ),
        athletes=(
            first.athletes[0],
            GsrAthleteAnnotation(
                annotation_id="2",
                image_id="1001000002",
                frame=2,
                track_id=7,
                role="player",
                jersey="9",
                team="left",
                bbox_image=GsrImageBBox(x=8.0, y=10.0, w=10.0, h=10.0),
                bbox_pitch=GsrPitchBBox(x_bottom_middle=13.0, y_bottom_middle=20.0),
            ),
        ),
        lines=(),
    )


def test_projection_metrics_are_zero_for_perfect_identity_homography() -> None:
    labels = make_labels()
    homography = HomographyRecord.available(
        sequence="SNGS-001",
        frame=1,
        method="identity",
        homography_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    projections = project_gsr_athletes(labels, [homography])
    metrics = summarize_homography_metrics(
        projections,
        [homography],
        expected_frames={"SNGS-001": {1}},
    )

    assert projections[0].error_m == 0.0
    assert metrics["identity"]["median_error_m"] == 0.0
    assert metrics["identity"]["success@1m"] == 1.0
    assert metrics["identity"]["availability"] == 1.0
    assert metrics["identity"]["locsim_tau5"] == 1.0


def test_projection_metrics_report_temporal_jitter_for_consecutive_track_points() -> (
    None
):
    labels = make_two_frame_labels()
    homographies = [
        HomographyRecord.available(
            sequence="SNGS-001",
            frame=1,
            method="identity",
            homography_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        HomographyRecord.available(
            sequence="SNGS-001",
            frame=2,
            method="identity",
            homography_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
    ]

    projections = project_gsr_athletes(labels, homographies)
    metrics = summarize_homography_metrics(projections, homographies)

    assert metrics["identity"]["temporal_jitter"] == 3.0


def test_projection_metrics_capture_known_translation_error() -> None:
    labels = make_labels()
    homography = HomographyRecord.available(
        sequence="SNGS-001",
        frame=1,
        method="shift",
        homography_3x3=[[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    )

    projections = project_gsr_athletes(labels, [homography])
    metrics = summarize_homography_metrics(
        projections,
        [homography],
        expected_frames={"SNGS-001": {1}},
    )

    assert projections[0].error_m == pytest.approx(math.sqrt(5.0))
    assert metrics["shift"]["median_error_m"] == pytest.approx(math.sqrt(5.0))
    assert metrics["shift"]["success@2m"] == 0.0
    assert metrics["shift"]["success@5m"] == 1.0


def test_projection_metrics_are_zero_for_matching_scale_homography() -> None:
    labels = make_labels(pitch_x=20.0, pitch_y=40.0)
    homography = HomographyRecord.available(
        sequence="SNGS-001",
        frame=1,
        method="scale",
        homography_3x3=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
    )

    projections = project_gsr_athletes(labels, [homography])

    assert projections[0].pitch_x_pred == 20.0
    assert projections[0].pitch_y_pred == 40.0
    assert projections[0].error_m == 0.0
