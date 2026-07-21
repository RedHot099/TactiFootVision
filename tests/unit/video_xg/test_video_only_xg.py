from pathlib import Path

import pytest

from tactifoot_vision.ball import BallTrajectory, BallTrajectoryPoint
from tactifoot_vision.domain import (
    BBox,
    DetectionSet,
    FrameResult,
    PitchPoint,
    PitchProjection,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import ShotDetectorKind
from tactifoot_vision.shots import ShotCandidate, ShotWindow
from tactifoot_vision.video_xg import (
    ForbiddenVideoXgInputError,
    VideoFreezeContextXgEstimator,
    VideoGeometryXgEstimator,
    VideoKinematicContextXgEstimator,
    VideoOnlyXgRunner,
    VideoShotFeatures,
    assert_video_only_columns,
    build_video_shot_features,
    run_video_only_xg_experiment,
    write_video_features_csv,
)
from tactifoot_vision.video_xg.projection_features import (
    HomographyArtifactProvider,
    ProjectionQualityAnnotator,
)
from tactifoot_vision.video_xg.xg_calibration import (
    DataBallPySimpleXgBaseline,
    NeuralVideoXgCalibrator,
    QualityAwareXgEnsemble,
)


def test_video_only_protocol_rejects_statsbomb_inputs() -> None:
    with pytest.raises(ForbiddenVideoXgInputError):
        assert_video_only_columns(["shot_id", "shot_x", "shot_statsbomb_xg"])


def test_video_geometry_xg_is_higher_for_closer_central_shot() -> None:
    estimator = VideoGeometryXgEstimator()
    close = VideoShotFeatures(
        shot_id="close",
        frame_index=10,
        shot_x=99.0,
        shot_y=34.0,
    )
    far = VideoShotFeatures(
        shot_id="far",
        frame_index=20,
        shot_x=70.0,
        shot_y=5.0,
    )

    assert estimator.predict(close).xg > estimator.predict(far).xg


def test_video_context_xg_penalizes_blocking_defenders() -> None:
    estimator = VideoFreezeContextXgEstimator()
    open_shot = VideoShotFeatures(
        shot_id="open",
        frame_index=10,
        shot_x=99.0,
        shot_y=34.0,
        nearest_player_distance=8.0,
        goalkeeper_distance=12.0,
        defender_count_in_cone=0,
    )
    blocked = VideoShotFeatures(
        shot_id="blocked",
        frame_index=10,
        shot_x=99.0,
        shot_y=34.0,
        nearest_player_distance=1.0,
        goalkeeper_distance=3.0,
        defender_count_in_cone=3,
    )

    assert estimator.predict(open_shot).xg > estimator.predict(blocked).xg


def test_video_kinematic_context_rewards_goal_direction() -> None:
    estimator = VideoKinematicContextXgEstimator()
    toward_goal = VideoShotFeatures(
        shot_id="toward",
        frame_index=10,
        shot_x=92.0,
        shot_y=34.0,
        ball_speed=25.0,
        ball_direction_to_goal=1.0,
    )
    away_from_goal = VideoShotFeatures(
        shot_id="away",
        frame_index=10,
        shot_x=92.0,
        shot_y=34.0,
        ball_speed=25.0,
        ball_direction_to_goal=-1.0,
    )

    assert estimator.predict(toward_goal).xg > estimator.predict(away_from_goal).xg


def test_video_only_runner_writes_predictions_and_reference_metrics(
    tmp_path: Path,
) -> None:
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "\n".join(
            [
                "shot_id,frame_index,shot_x,shot_y,goalkeeper_distance,defender_count_in_cone",
                "s1,10,99,34,12,0",
                "s2,20,80,20,5,2",
            ]
        ),
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "\n".join(
            [
                "shot_id,reference_xg,is_goal",
                "s1,0.4,1",
                "s2,0.05,0",
            ]
        ),
        encoding="utf-8",
    )

    summary, metrics, artifacts = VideoOnlyXgRunner().run(
        features_path,
        output_dir=tmp_path / "out",
        reference_path=reference_path,
        group_id="match-1",
    )

    assert summary.group_id == "match-1"
    assert summary.shot_count == 2
    assert metrics["matched_shots"] == 2.0
    assert metrics["mae_vs_reference_xg"] > 0.0
    assert {artifact.format for artifact in artifacts} == {
        "video_only_xg_shots_csv",
        "video_only_xg_summary_json",
    }
    assert (tmp_path / "out" / "video_only_shots.csv").exists()
    assert (tmp_path / "out" / "video_only_summary.json").exists()


def test_build_video_shot_features_uses_pipeline_projection() -> None:
    candidate = ShotCandidate(
        frame_index=2,
        window=ShotWindow(1, 3),
        confidence=0.8,
        detector_kind=ShotDetectorKind.KINEMATIC,
    )
    trajectory = BallTrajectory(
        (
            BallTrajectoryPoint(frame_index=1, image_x=90.0, image_y=34.0),
            BallTrajectoryPoint(
                frame_index=2,
                image_x=100.0,
                image_y=34.0,
                pitch_x=100.0,
                pitch_y=34.0,
            ),
        )
    )

    features = build_video_shot_features(
        shot_id="shot-1",
        candidate=candidate,
        trajectory=trajectory,
        frame_result=_projected_frame(),
    )

    assert features.shot_id == "shot-1"
    assert features.shot_x == 100.0
    assert features.goalkeeper_distance == 5.0
    assert features.shot_confidence == 0.8


def test_video_only_experiment_compares_default_methods(tmp_path: Path) -> None:
    features_path = write_video_features_csv(
        (
            VideoShotFeatures(
                shot_id="s1",
                frame_index=10,
                shot_x=99.0,
                shot_y=34.0,
                goalkeeper_distance=12.0,
            ),
            VideoShotFeatures(
                shot_id="s2",
                frame_index=20,
                shot_x=80.0,
                shot_y=20.0,
                goalkeeper_distance=5.0,
                defender_count_in_cone=2,
            ),
        ),
        tmp_path / "video_features.csv",
    )
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "\n".join(["shot_id,reference_xg,is_goal", "s1,0.3,1", "s2,0.05,0"]),
        encoding="utf-8",
    )

    summary, artifacts = run_video_only_xg_experiment(
        features_path=features_path,
        output_dir=tmp_path / "experiment",
        reference_path=reference_path,
        group_id="match-1",
    )

    assert summary["group_id"] == "match-1"
    assert len(summary["methods"]) == 3
    assert (tmp_path / "experiment" / "method_metrics.csv").exists()
    assert (tmp_path / "experiment" / "comparison_report.md").exists()
    assert {artifact.format for artifact in artifacts} >= {
        "video_only_xg_method_metrics_csv",
        "video_only_xg_report_markdown",
    }


def test_homography_provider_uses_last_stable_frame_within_window(
    tmp_path: Path,
) -> None:
    sampled = _sampled_frames()
    homographies = tmp_path / "homographies.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "global_frame_index": 0,
                "status": "available",
                "projection_confidence": 0.8,
                "homography": "H",
            }
        ]
    ).to_parquet(homographies, index=False)

    projected = HomographyArtifactProvider(homographies, max_age_seconds=1.5).project(
        sampled
    )

    assert projected.iloc[1]["status"] == "last_stable_homography"
    assert projected.iloc[2]["status"] == "homography_unavailable"


def test_quality_aware_ensemble_preserves_xg_monotonicity_signal() -> None:
    pd = pytest.importorskip("pandas")
    predictions = pd.DataFrame(
        [
            {
                "shot_id": "close",
                "frame_index": 1,
                "method": "video_geometry",
                "xg": 0.2,
            },
            {
                "shot_id": "close",
                "frame_index": 1,
                "method": "video_freeze_context",
                "xg": 0.3,
            },
            {
                "shot_id": "close",
                "frame_index": 1,
                "method": "video_kinematic_context",
                "xg": 0.4,
            },
            {
                "shot_id": "far",
                "frame_index": 2,
                "method": "video_geometry",
                "xg": 0.02,
            },
            {
                "shot_id": "far",
                "frame_index": 2,
                "method": "video_freeze_context",
                "xg": 0.03,
            },
            {
                "shot_id": "far",
                "frame_index": 2,
                "method": "video_kinematic_context",
                "xg": 0.04,
            },
        ]
    )
    features = pd.DataFrame(
        [
            {
                "shot_id": "close",
                "frame_index": 1,
                "feature_source": "observed",
                "shot_confidence": 0.9,
                "projection_confidence": 0.8,
            },
            {
                "shot_id": "far",
                "frame_index": 2,
                "feature_source": "observed",
                "shot_confidence": 0.9,
                "projection_confidence": 0.8,
            },
        ]
    )

    ensemble = QualityAwareXgEnsemble().predict(predictions, features)

    values = dict(zip(ensemble["shot_id"], ensemble["xg"], strict=True))
    assert values["close"] > values["far"]


def test_neural_video_xg_calibrator_learns_reference_ordering() -> None:
    pd = pytest.importorskip("pandas")
    features = pd.DataFrame(
        [
            _feature_row("close", 1, 101.0, 34.0),
            _feature_row("central", 2, 96.0, 34.0),
            _feature_row("wide", 3, 94.0, 18.0),
            _feature_row("far", 4, 70.0, 8.0),
            _feature_row("very_far", 5, 55.0, 4.0),
        ]
    )
    reference = pd.DataFrame(
        [
            {"shot_id": "close", "reference_xg": 0.55},
            {"shot_id": "central", "reference_xg": 0.30},
            {"shot_id": "wide", "reference_xg": 0.12},
            {"shot_id": "far", "reference_xg": 0.04},
            {"shot_id": "very_far", "reference_xg": 0.02},
        ]
    )

    predictions = NeuralVideoXgCalibrator().predict(features, reference)
    values = dict(zip(predictions["shot_id"], predictions["xg"], strict=True))

    assert set(predictions["method"]) == {"neural_video_xg"}
    assert all(0.0 < value < 1.0 for value in values.values())
    assert values["close"] > values["far"]


def test_databallpy_simple_xg_baseline_is_location_sensitive() -> None:
    pd = pytest.importorskip("pandas")
    features = pd.DataFrame(
        [
            _feature_row("close", 1, 101.0, 34.0),
            _feature_row("far", 2, 70.0, 8.0),
        ]
    )

    predictions = DataBallPySimpleXgBaseline().predict(features)
    values = dict(zip(predictions["shot_id"], predictions["xg"], strict=True))

    assert set(predictions["method"]) == {"databallpy_simple_xg"}
    assert values["close"] > values["far"]


def test_projection_annotator_adds_quality_columns() -> None:
    pd = pytest.importorskip("pandas")
    features = pd.DataFrame([{"shot_id": "s1", "frame_index": 1}])
    homographies = pd.DataFrame(
        [
            {
                "global_frame_index": 1,
                "status": "line_box_heuristic",
                "projection_confidence": 0.35,
            }
        ]
    )

    annotated = ProjectionQualityAnnotator().annotate(features, homographies)

    assert annotated.iloc[0]["projection_status"] == "line_box_heuristic"
    assert annotated.iloc[0]["projection_confidence"] == pytest.approx(0.35)


def _feature_row(shot_id: str, frame_index: int, shot_x: float, shot_y: float) -> dict:
    return {
        "shot_id": shot_id,
        "frame_index": frame_index,
        "shot_x": shot_x,
        "shot_y": shot_y,
        "goal_x": 105.0,
        "goal_y": 34.0,
        "nearest_player_distance": 5.0,
        "goalkeeper_distance": 10.0,
        "defender_count_in_cone": 0,
        "ball_speed": 18.0,
        "ball_direction_to_goal": 0.9,
        "shot_confidence": 0.8,
        "projection_confidence": 0.5,
        "feature_source": "observed",
    }


def _projected_frame() -> FrameResult:
    return FrameResult(
        frame_index=2,
        timestamp_seconds=None,
        detections=DetectionSet.empty(),
        tracks=TrackSet(
            (
                Track(
                    track_id=1,
                    bbox=BBox(0.0, 0.0, 1.0, 1.0),
                    class_id=1,
                    class_name="goalkeeper",
                    confidence=1.0,
                ),
            )
        ),
        projection=PitchProjection(
            status="available",
            points_by_track_id={1: PitchPoint(105.0, 34.0)},
            ball=PitchPoint(100.0, 34.0),
            homography=None,
        ),
    )


def _sampled_frames():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        [
            {
                "global_frame_index": 0,
                "global_seconds": 0.0,
                "part_index": 0,
                "part_frame_index": 0,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 50,
                "fps": 1.0,
            },
            {
                "global_frame_index": 1,
                "global_seconds": 1.0,
                "part_index": 0,
                "part_frame_index": 1,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 50,
                "fps": 1.0,
            },
            {
                "global_frame_index": 2,
                "global_seconds": 3.0,
                "part_index": 0,
                "part_frame_index": 2,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 50,
                "fps": 1.0,
            },
        ]
    )
