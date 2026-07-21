from pathlib import Path

import pandas as pd

from tactifoot_vision.cli import main
from tactifoot_vision.enums import (
    BallReconstructionMethod,
    VideoShotDetectorKind,
    VideoXgCalibrationVariant,
    XgModelKind,
)
from tactifoot_vision.video_xg import (
    ContactKinematicShotDetector,
    TemporalShotRanker,
    load_video_only_xg_end_to_end_config,
)
from tactifoot_vision.video_xg.ablation import VideoXgWeaknessAblationRunner
from tactifoot_vision.video_xg.ball_reconstruction import (
    KalmanRtsBallReconstructorV2,
    ViterbiBallPathReconstructor,
)
from tactifoot_vision.video_xg.end_to_end import (
    _calibrated_xg_predictions,
    _extract_features,
    _reconstruct_ball,
    _reconstruct_ball_stage,
    _refine_candidates_stage,
    _run_homography_stage,
)
from tactifoot_vision.video_xg.shot_quality import (
    AdaptiveShotNms,
    ShotPatternScorer,
    ShotWindowFeatureExtractor,
    SoftCompositeThresholdSelector,
)
from tactifoot_vision.video_xg.shot_ranking import (
    DenseContactRefiner,
    HardNegativeCalibratedShotRanker,
    HighRecallCascadeShotRanker,
    LearnedTemporalShotRanker,
    RuleSweepShotRanker,
    ShotCandidateFeatureExtractor,
    WindowedTemporalShotRanker,
)
from tactifoot_vision.video_xg.stages import ChunkedDetectionRunner


def test_video_only_end_to_end_config_loads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke",
                "video_parts:",
                "  - part1.mp4",
                "  - part2.mp4",
                "output_dir: out",
                "detection:",
                "  variant: chunked_batched",
                "  chunk_size: 2",
                "ball:",
                "  method: kalman_rts",
                "shots:",
                "  kind: contact_kinematic",
                "xg:",
                "  models:",
                "    - video_geometry",
                "    - video_kinematic_context",
                "  calibration_variant: neural_video_xg",
            ]
        ),
        encoding="utf-8",
    )

    config = load_video_only_xg_end_to_end_config(config_path)

    assert config.name == "smoke"
    assert config.detection.variant.value == "chunked_batched"
    assert config.detection.chunk_size == 2
    assert config.ball.method is BallReconstructionMethod.KALMAN_RTS
    assert config.shots.kind is VideoShotDetectorKind.CONTACT_KINEMATIC
    assert config.xg.models == (
        XgModelKind.VIDEO_GEOMETRY,
        XgModelKind.VIDEO_KINEMATIC_CONTEXT,
    )
    assert config.xg.calibration_variant is VideoXgCalibrationVariant.NEURAL_VIDEO_XG


def test_contact_kinematic_detector_finds_synthetic_kick() -> None:
    ball = pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "pitch_x": 40.0 + index,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
            }
            for index in range(5)
        ]
        + [
            {
                "global_frame_index": 5 + index,
                "global_seconds": (5 + index) * 0.1,
                "part_index": 0,
                "part_frame_index": 5 + index,
                "pitch_x": 45.0 + 6.0 * index,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
            }
            for index in range(1, 6)
        ]
    )
    tracks = pd.DataFrame(
        [
            {
                "global_frame_index": 4,
                "class_name": "player",
                "pitch_x": 44.0,
                "pitch_y": 34.0,
            }
        ]
    )

    candidates = ContactKinematicShotDetector(
        min_candidate_confidence=0.1,
        temporal_nms_seconds=1.0,
        contact_distance_m=3.0,
    ).generate(ball, tracks)

    assert not candidates.empty
    assert candidates.iloc[0]["source"] == "contact_kinematic"


def test_temporal_ranker_suppresses_nearby_candidates() -> None:
    candidates = pd.DataFrame(
        [
            {"global_seconds": 10.0, "score": 0.9},
            {"global_seconds": 11.0, "score": 0.8},
            {"global_seconds": 25.0, "score": 0.7},
        ]
    )

    ranked = TemporalShotRanker(temporal_nms_seconds=8.0).rank(candidates)

    assert ranked["global_seconds"].tolist() == [10.0, 25.0]


def test_ball_reconstruction_interpolates_missing_middle_frame() -> None:
    sampled = _sampled_frames()
    detections = pd.DataFrame(
        [
            _ball_detection(0, 10.0),
            _ball_detection(2, 30.0),
        ]
    )

    trajectory = _reconstruct_ball(
        sampled,
        detections,
        max_gap_seconds=1.0,
        max_speed_mps=200.0,
    )

    middle = trajectory[trajectory["global_frame_index"].eq(1)].iloc[0]
    assert middle["source"] == "kalman_rts_interpolated"
    assert 10.0 < middle["image_x"] < 30.0


def test_ball_reconstruction_rejects_physical_outlier() -> None:
    sampled = _sampled_frames()
    detections = pd.DataFrame(
        [
            _ball_detection(0, 10.0),
            _ball_detection(1, 90.0),
            _ball_detection(2, 12.0),
        ]
    )

    trajectory = _reconstruct_ball(
        sampled,
        detections,
        max_gap_seconds=1.0,
        max_speed_mps=20.0,
    )

    outlier = trajectory[trajectory["global_frame_index"].eq(1)].iloc[0]
    assert outlier["source"] == "kalman_rts_interpolated"
    assert outlier["image_x"] < 30.0


def test_chunked_detector_writes_chunks_and_resumes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: chunked",
                "video_parts:",
                "  - part1.mp4",
                f"output_dir: {output_dir}",
                "detection:",
                "  variant: chunked_batched",
                "  chunk_size: 2",
                "detector:",
                "  backend: fake",
            ]
        ),
        encoding="utf-8",
    )
    config = load_video_only_xg_end_to_end_config(config_path)

    first = ChunkedDetectionRunner().run(config, _sampled_frames(), output_dir)
    second = ChunkedDetectionRunner().run(config, _sampled_frames(), output_dir)

    assert len(first) == len(second) == 6
    assert (output_dir / "02_detections.parquet").exists()
    assert (output_dir / "02_detections" / "chunk_00000.parquet").exists()
    assert (output_dir / "02_detections" / "chunk_00001.parquet").exists()
    assert (output_dir / "02_detections" / "manifest.json").exists()


def test_viterbi_ball_path_prefers_physically_plausible_detection() -> None:
    sampled = _sampled_frames()
    detections = pd.DataFrame(
        [
            _ball_detection(0, 10.0),
            {**_ball_detection(1, 95.0), "confidence": 0.99},
            {**_ball_detection(1, 12.0), "confidence": 0.55},
            _ball_detection(2, 14.0),
        ]
    )

    trajectory = ViterbiBallPathReconstructor(max_speed_mps=60.0).reconstruct(
        sampled, detections
    )

    frame_one = trajectory[trajectory["global_frame_index"].eq(1)].iloc[0]
    assert frame_one["image_x"] < 30.0
    assert frame_one["source"] == "viterbi_observed"


def test_kalman_v2_marks_long_gap_as_missing_not_center_fallback() -> None:
    trajectory = pd.DataFrame(
        [
            {
                "global_frame_index": 0,
                "global_seconds": 0.0,
                "part_index": 0,
                "part_frame_index": 0,
                "image_x": 10.0,
                "image_y": 50.0,
                "pitch_x": 10.5,
                "pitch_y": 34.0,
                "confidence": 0.9,
                "source": "observed",
                "uncertainty": 0.1,
            },
            {
                "global_frame_index": 1,
                "global_seconds": 5.0,
                "part_index": 0,
                "part_frame_index": 1,
                "image_x": 50.0,
                "image_y": 50.0,
                "pitch_x": 52.5,
                "pitch_y": 34.0,
                "confidence": 0.1,
                "source": "missing_center_fallback",
                "uncertainty": 1.0,
            },
            {
                "global_frame_index": 2,
                "global_seconds": 10.0,
                "part_index": 0,
                "part_frame_index": 2,
                "image_x": 90.0,
                "image_y": 50.0,
                "pitch_x": 94.5,
                "pitch_y": 34.0,
                "confidence": 0.9,
                "source": "observed",
                "uncertainty": 0.1,
            },
        ]
    )

    reconstructed = KalmanRtsBallReconstructorV2(max_gap_seconds=1.0).reconstruct(
        trajectory
    )

    middle = reconstructed[reconstructed["global_frame_index"].eq(1)].iloc[0]
    assert middle["source"] == "missing"
    assert pd.isna(middle["pitch_x"])


def test_rule_sweep_ranker_is_deterministic_for_reference() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _synthetic_shot_tracks()
    )
    reference = pd.DataFrame([{"reference_seconds": 0.5}])

    first = RuleSweepShotRanker(temporal_nms_seconds=1.0).rank(features, reference)
    second = RuleSweepShotRanker(temporal_nms_seconds=1.0).rank(features, reference)

    assert first["global_seconds"].tolist() == second["global_seconds"].tolist()
    assert abs(first.iloc[0]["global_seconds"] - 0.5) <= 0.2


def test_learned_ranker_uses_reference_only_for_training_labels() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _synthetic_shot_tracks()
    )
    reference = pd.DataFrame([{"reference_seconds": 0.5}])

    ranked = LearnedTemporalShotRanker(temporal_nms_seconds=1.0).rank(
        features, reference
    )

    assert not ranked.empty
    assert "reference_seconds" not in ranked.columns
    assert "is_goal" not in ranked.columns


def test_direction_resolver_downranks_ball_moving_away_from_goal() -> None:
    scored = ShotPatternScorer().score(pd.DataFrame([_manual_window_feature()]))

    assert scored.iloc[0]["direction_to_resolved_goal"] < 0.0
    assert scored.iloc[0]["veto_multiplier"] < 1.0


def test_contact_before_acceleration_raises_pattern_score() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _synthetic_shot_tracks()
    )
    scored = ShotPatternScorer().score(ShotWindowFeatureExtractor().transform(features))

    before_kick = scored[scored["global_seconds"].eq(0.2)].iloc[0]
    contact_kick = scored[scored["global_seconds"].eq(0.4)].iloc[0]
    assert contact_kick["contact_pattern_score"] > before_kick["contact_pattern_score"]
    assert contact_kick["pattern_score"] > before_kick["pattern_score"]


def test_acceleration_without_contact_fails_precision_gate_when_tracks_exist() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _far_tracks()
    )
    scored = ShotPatternScorer().score(ShotWindowFeatureExtractor().transform(features))
    kick = scored[scored["global_seconds"].eq(0.5)].iloc[0]

    assert bool(kick["tracks_observed_in_window"])
    assert not bool(kick["precision_gate"])


def test_acceleration_without_contact_can_pass_when_tracks_are_missing() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), pd.DataFrame()
    )
    scored = ShotPatternScorer().score(ShotWindowFeatureExtractor().transform(features))
    kick = scored[scored["global_seconds"].eq(0.5)].iloc[0]

    assert not bool(kick["tracks_observed_in_window"])
    assert bool(kick["precision_gate"])


def test_recontact_after_candidate_penalizes_dribble_like_sequence() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_dribble_ball(), _synthetic_dribble_tracks()
    )
    scored = ShotPatternScorer().score(ShotWindowFeatureExtractor().transform(features))
    candidate = scored[scored["global_seconds"].eq(0.4)].iloc[0]

    assert candidate["post_recontact_score"] > 0.0
    assert candidate["recontact_penalty"] > 0.0


def test_long_shot_with_high_speed_keeps_zone_exception() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_long_range_ball(), _synthetic_long_range_tracks()
    )
    scored = ShotPatternScorer(long_shot_distance_m=35.0).score(
        ShotWindowFeatureExtractor(long_shot_distance_m=35.0).transform(features)
    )
    candidate = scored[scored["global_seconds"].eq(0.4)].iloc[0]

    assert bool(candidate["long_shot_exception"])
    assert bool(candidate["precision_gate"])


def test_adaptive_nms_keeps_rebound_with_separate_contact() -> None:
    scored = pd.DataFrame(
        [
            _scored_candidate(10.0, 0.9, contact=0.8, progress=5.0),
            _scored_candidate(12.0, 0.8, contact=0.7, progress=4.0),
            _scored_candidate(12.4, 0.7, contact=0.7, progress=4.0),
        ]
    )

    selected = AdaptiveShotNms(temporal_nms_seconds=8.0).select(scored)

    assert selected["global_seconds"].tolist() == [10.0, 12.0]


def test_threshold_selector_preserves_soft_recall_floor() -> None:
    scored = pd.DataFrame(
        [
            _scored_candidate(10.0, 0.9),
            _scored_candidate(20.0, 0.8),
            _scored_candidate(30.0, 0.7),
            _scored_candidate(40.0, 0.2),
        ]
    )
    reference = pd.DataFrame(
        [
            {"reference_seconds": 10.2},
            {"reference_seconds": 20.1},
            {"reference_seconds": 30.1},
        ]
    )
    selector = SoftCompositeThresholdSelector(recall_floor_hit2=0.78)

    selected = selector.select(scored, reference, lambda frame: frame.copy())

    assert selected.metrics["hit@2s"] >= 0.78
    assert selected.threshold > 0.2


def test_new_rankers_do_not_emit_reference_columns() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _synthetic_shot_tracks()
    )
    reference = pd.DataFrame([{"reference_seconds": 0.5}])
    rankers = [
        HighRecallCascadeShotRanker(temporal_nms_seconds=1.0),
        HardNegativeCalibratedShotRanker(temporal_nms_seconds=1.0),
        WindowedTemporalShotRanker(temporal_nms_seconds=1.0),
    ]

    ranked = [
        rankers[0].rank(features),
        rankers[1].rank(features, reference),
        rankers[2].rank(features, reference),
    ]

    assert all("reference_seconds" not in frame.columns for frame in ranked)
    assert all("is_goal" not in frame.columns for frame in ranked)


def test_dense_refiner_moves_candidate_to_local_contact_acceleration_peak() -> None:
    features = ShotCandidateFeatureExtractor().transform(
        _synthetic_shot_ball(), _synthetic_shot_tracks()
    )
    candidates = pd.DataFrame(
        [
            {
                "shot_id": "video-shot-0001",
                "global_frame_index": 2,
                "global_seconds": 0.2,
                "part_index": 0,
                "part_frame_index": 2,
                "score": 0.4,
                "confidence": 0.4,
                "source": "test",
                "nearest_player_distance": 1.0,
                "ball_speed": 1.0,
                "ball_direction_to_goal": 1.0,
            }
        ]
    )

    refined = DenseContactRefiner(
        window_before_seconds=0.3, window_after_seconds=0.5
    ).refine(candidates, features)

    assert refined.iloc[0]["source"] == "dense_local_refinement"
    assert refined.iloc[0]["global_seconds"] >= 0.4


def test_ablation_runner_reads_existing_run_and_writes_rankings(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    reference_events = tmp_path / "events.parquet"
    output_dir = tmp_path / "improvement"
    _write_fake_baseline_run(baseline_dir, reference_events)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: improvement",
                "video_parts:",
                "  - part1.mp4",
                f"output_dir: {tmp_path / 'unused'}",
                "detector:",
                "  backend: fake",
                "ablation:",
                f"  baseline_run_dir: {baseline_dir}",
                f"  output_dir: {output_dir}",
                "shots:",
                "  max_candidates: 4",
                "evaluation:",
                f"  reference_events: {reference_events}",
            ]
        ),
        encoding="utf-8",
    )
    config = load_video_only_xg_end_to_end_config(config_path)

    result = VideoXgWeaknessAblationRunner().run(config)

    assert result.output_dir == output_dir
    assert (output_dir / "00_baseline_summary.json").exists()
    assert (output_dir / "02_ball_ablation.csv").exists()
    assert (output_dir / "03_shot_ablation.csv").exists()
    assert (output_dir / "11_shot_fp_diagnostics.csv").exists()
    assert (output_dir / "12_candidate_window_features.parquet").exists()
    assert (output_dir / "13_shot_fp_ablation.csv").exists()
    assert (output_dir / "14_selected_refined_shots.parquet").exists()
    assert (output_dir / "15_predictions_after_fp_reduction.csv").exists()
    assert (output_dir / "05_xg_ablation.csv").exists()
    assert (output_dir / "final_shot_detection_report.md").exists()
    assert (output_dir / "final_variant_ranking.csv").exists()
    ranking = pd.read_csv(output_dir / "final_variant_ranking.csv")
    assert not ranking.empty
    assert "composite_score" in ranking.columns


def test_video_xg_benchmark_detection_cli_writes_artifact(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    reference_events = tmp_path / "events.parquet"
    output_dir = tmp_path / "benchmark"
    _write_fake_baseline_run(baseline_dir, reference_events)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: benchmark",
                "video_parts:",
                "  - part1.mp4",
                f"output_dir: {tmp_path / 'unused'}",
                "detector:",
                "  backend: fake",
                "detection:",
                "  variant: chunked_batched",
                "  chunk_size: 4",
                "  benchmark_max_frames: 3",
                "ablation:",
                f"  baseline_run_dir: {baseline_dir}",
                f"  output_dir: {output_dir}",
            ]
        ),
        encoding="utf-8",
    )

    status = main(["video-xg", "benchmark-detection", "--config", str(config_path)])

    assert status == 0
    assert (output_dir / "01_detection_benchmark.csv").exists()
    assert (output_dir / "02_detections" / "manifest.json").exists()


def test_end_to_end_runner_applies_winning_variants(tmp_path: Path) -> None:
    reference_events = tmp_path / "events.parquet"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: winners",
                "video_parts:",
                "  - part1.mp4",
                f"output_dir: {tmp_path / 'run'}",
                "detector:",
                "  backend: fake",
                "ball:",
                "  variant: optical_flow_template",
                "calibration:",
                "  variant: line_box_heuristic",
                "shots:",
                "  variant: learned_temporal",
                "  max_candidates: 8",
                "xg:",
                "  calibration_variant: coefficient_fit",
                "evaluation:",
                f"  reference_events: {reference_events}",
            ]
        ),
        encoding="utf-8",
    )
    config = load_video_only_xg_end_to_end_config(config_path)
    sampled = _long_sampled_frames()
    detections = pd.DataFrame(
        [_ball_detection(index, 55.0 + index * 0.6) for index in range(60)]
    )
    timeline = pd.DataFrame(
        [
            {
                "part_index": 0,
                "path": "part1.mp4",
                "start_seconds": 0.0,
                "duration_seconds": 6.0,
                "fps": 10.0,
                "frame_count": 60,
                "width": 100,
                "height": 100,
            }
        ]
    )
    _write_reference_events(reference_events, timestamp="00:00:02.500", xg=0.2)
    ball = _reconstruct_ball_stage(config, sampled, detections)
    homographies = _run_homography_stage(config, sampled)
    refined = _refine_candidates_stage(
        config,
        pd.DataFrame(),
        _synthetic_long_shot_ball(),
        _synthetic_long_shot_tracks(),
        timeline,
    )
    features = _extract_features(
        refined,
        _synthetic_long_shot_ball(),
        _synthetic_long_shot_tracks(),
        homographies,
    )
    predictions = pd.DataFrame(
        [
            {
                "shot_id": row.shot_id,
                "frame_index": int(row.frame_index),
                "method": "video_kinematic_context",
                "xg": 0.05,
            }
            for row in features.itertuples(index=False)
        ]
    )
    calibrated = _calibrated_xg_predictions(
        config,
        features,
        refined,
        timeline,
        predictions,
    )

    assert "missing_center_fallback" not in set(ball["source"])
    assert set(homographies["status"]) == {"line_box_heuristic"}
    assert not refined.empty
    assert set(refined["refinement_status"]) == {"learned_temporal"}
    assert set(features["projection_status"]) == {"line_box_heuristic"}
    assert set(calibrated["method"]) == {"coefficient_fit"}


def test_precision_shot_variant_runs_without_reference_events(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: runtime_precision",
                "video_parts:",
                "  - part1.mp4",
                f"output_dir: {tmp_path / 'run'}",
                "detector:",
                "  backend: fake",
                "shots:",
                "  variant: hard_negative_calibrated",
                "  max_candidates: 8",
            ]
        ),
        encoding="utf-8",
    )
    config = load_video_only_xg_end_to_end_config(config_path)
    timeline = pd.DataFrame(
        [
            {
                "part_index": 0,
                "path": "part1.mp4",
                "start_seconds": 0.0,
                "duration_seconds": 6.0,
                "fps": 10.0,
                "frame_count": 60,
                "width": 100,
                "height": 100,
            }
        ]
    )

    refined = _refine_candidates_stage(
        config,
        pd.DataFrame(),
        _synthetic_long_shot_ball(),
        _synthetic_long_shot_tracks(),
        timeline,
    )

    assert not refined.empty
    assert set(refined["refinement_status"]) == {"hard_negative_calibrated"}
    assert "reference_seconds" not in refined.columns


def _sampled_frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 100,
                "fps": 10.0,
            }
            for index in range(3)
        ]
    )


def _long_sampled_frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 100,
                "fps": 10.0,
            }
            for index in range(60)
        ]
    )


def _ball_detection(frame_index: int, center_x: float) -> dict[str, object]:
    return {
        "global_frame_index": frame_index,
        "global_seconds": frame_index * 0.1,
        "part_index": 0,
        "part_frame_index": frame_index,
        "detection_index": 0,
        "class_id": 0,
        "class_name": "ball",
        "confidence": 0.9,
        "x1": center_x - 1.0,
        "y1": 49.0,
        "x2": center_x + 1.0,
        "y2": 51.0,
        "width": 100,
        "height": 100,
    }


def _synthetic_shot_ball() -> pd.DataFrame:
    rows = []
    for index in range(10):
        if index < 5:
            pitch_x = 60.0 + index * 0.5
        else:
            pitch_x = 62.0 + (index - 4) * 8.0
        rows.append(
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "image_x": pitch_x,
                "image_y": 34.0,
                "pitch_x": pitch_x,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
                "uncertainty": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_shot_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": 4,
                "global_seconds": 0.4,
                "part_index": 0,
                "part_frame_index": 4,
                "track_id": 1,
                "class_name": "player",
                "confidence": 1.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "pitch_x": 62.0,
                "pitch_y": 34.0,
            }
        ]
    )


def _far_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": 4,
                "global_seconds": 0.4,
                "part_index": 0,
                "part_frame_index": 4,
                "track_id": 1,
                "class_name": "player",
                "confidence": 1.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "pitch_x": 20.0,
                "pitch_y": 34.0,
            }
        ]
    )


def _synthetic_away_ball() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "image_x": 70.0 - index * 3.0,
                "image_y": 34.0,
                "pitch_x": 70.0 - index * 3.0,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
                "uncertainty": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_dribble_ball() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "image_x": 60.0 + index * 1.0,
                "image_y": 34.0,
                "pitch_x": 60.0 + index * 1.0,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
                "uncertainty": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_dribble_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "track_id": 1,
                "class_name": "player",
                "confidence": 1.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "pitch_x": 60.0 + index * 1.0,
                "pitch_y": 34.0,
            }
            for index in range(3, 8)
        ]
    )


def _synthetic_long_range_ball() -> pd.DataFrame:
    rows = []
    for index in range(10):
        pitch_x = 45.0 + index * 6.0
        rows.append(
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "image_x": pitch_x,
                "image_y": 34.0,
                "pitch_x": pitch_x,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
                "uncertainty": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_long_range_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": 4,
                "global_seconds": 0.4,
                "part_index": 0,
                "part_frame_index": 4,
                "track_id": 1,
                "class_name": "player",
                "confidence": 1.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "pitch_x": 69.0,
                "pitch_y": 34.0,
            }
        ]
    )


def _scored_candidate(
    seconds: float,
    score: float,
    *,
    contact: float = 0.5,
    progress: float = 3.0,
) -> dict[str, object]:
    frame_index = int(seconds * 10)
    return {
        "shot_id": f"shot-{frame_index}",
        "global_frame_index": frame_index,
        "global_seconds": seconds,
        "part_index": 0,
        "part_frame_index": frame_index,
        "score": score,
        "confidence": score,
        "source": "test",
        "nearest_player_distance": 1.0,
        "ball_speed": 20.0,
        "ball_direction_to_goal": 1.0,
        "pre_contact_score": contact,
        "post_goal_progress_m": progress,
        "precision_gate": True,
    }


def _manual_window_feature() -> dict[str, object]:
    return {
        "global_frame_index": 1,
        "global_seconds": 0.1,
        "part_index": 0,
        "part_frame_index": 1,
        "pitch_x": 90.0,
        "pitch_y": 34.0,
        "ball_speed": 12.0,
        "ball_acceleration": 8.0,
        "contact_score": 0.5,
        "ball_direction_to_goal": -1.0,
        "distance_to_goal": 15.0,
        "distance_score": 0.5,
        "resolved_goal_x": 105.0,
        "resolved_goal_y": 34.0,
        "direction_to_resolved_goal": -0.8,
        "goal_progress_m": -2.0,
        "distance_to_resolved_goal": 15.0,
        "pre_contact_score": 0.5,
        "pre_contact_min_distance": 1.0,
        "post_recontact_score": 0.0,
        "post_goal_progress_m": -2.0,
        "post_direction_consistency": 0.0,
        "post_speed_mean": 10.0,
        "post_speed_max": 12.0,
        "tracks_observed_in_window": True,
        "contact_acceleration_order_score": 0.5,
        "shot_zone_score": 0.6,
        "long_shot_exception": False,
        "recontact_penalty": 0.0,
    }


def _synthetic_long_shot_ball() -> pd.DataFrame:
    rows = []
    for index in range(60):
        if index < 25:
            pitch_x = 60.0 + index * 0.2
        else:
            pitch_x = 65.0 + (index - 24) * 1.8
        rows.append(
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "image_x": pitch_x,
                "image_y": 34.0,
                "pitch_x": pitch_x,
                "pitch_y": 34.0,
                "confidence": 1.0,
                "source": "observed",
                "uncertainty": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_long_shot_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": 24,
                "global_seconds": 2.4,
                "part_index": 0,
                "part_frame_index": 24,
                "track_id": 1,
                "class_name": "player",
                "confidence": 1.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "pitch_x": 65.0,
                "pitch_y": 34.0,
            }
        ]
    )


def _write_reference_events(
    reference_events: Path,
    *,
    timestamp: str,
    xg: float,
) -> None:
    pd.DataFrame(
        [
            {
                "id": "ref-1",
                "type": "Shot",
                "period": 1,
                "timestamp": timestamp,
                "shot_statsbomb_xg": xg,
                "shot_outcome": "Goal",
                "team": "A",
            }
        ]
    ).to_parquet(reference_events, index=False)


def _write_fake_baseline_run(baseline_dir: Path, reference_events: Path) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    timeline = pd.DataFrame(
        [
            {
                "part_index": 0,
                "path": "part1.mp4",
                "start_seconds": 0.0,
                "duration_seconds": 1.0,
                "fps": 10.0,
                "frame_count": 10,
                "width": 100,
                "height": 100,
            }
        ]
    )
    timeline.to_json(baseline_dir / "00_video_timeline.json", orient="records")
    sampled = pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "global_seconds": index * 0.1,
                "part_index": 0,
                "part_frame_index": index,
                "video_path": "part1.mp4",
                "width": 100,
                "height": 100,
                "fps": 10.0,
            }
            for index in range(10)
        ]
    )
    sampled.to_parquet(baseline_dir / "01_sampled_frames.parquet", index=False)
    detections = pd.DataFrame(
        [_ball_detection(index, 60.0 + index * 4.0) for index in range(10)]
    )
    detections.to_parquet(baseline_dir / "02_detections.parquet", index=False)
    tracks = _synthetic_shot_tracks()
    tracks.to_parquet(baseline_dir / "03_tracks.parquet", index=False)
    pd.DataFrame(
        [
            {
                "global_frame_index": index,
                "status": "degraded_image_normalized",
                "projection_confidence": 0.25,
                "homography": "",
            }
            for index in range(10)
        ]
    ).to_parquet(baseline_dir / "04_homographies.parquet", index=False)
    ball = _synthetic_shot_ball()
    ball.to_parquet(baseline_dir / "05_ball_trajectory.parquet", index=False)
    candidates = pd.DataFrame(
        [
            {
                "shot_id": "video-shot-0001",
                "global_frame_index": 5,
                "global_seconds": 0.5,
                "part_index": 0,
                "part_frame_index": 5,
                "score": 0.8,
                "confidence": 0.8,
                "source": "contact_kinematic",
                "nearest_player_distance": 1.0,
                "ball_speed": 40.0,
                "ball_direction_to_goal": 1.0,
            }
        ]
    )
    candidates.to_parquet(baseline_dir / "06_shot_candidates.parquet", index=False)
    candidates.assign(refinement_status="scan_fps_candidate").to_parquet(
        baseline_dir / "07_refined_shots.parquet", index=False
    )
    features = pd.DataFrame(
        [
            {
                "shot_id": "video-shot-0001",
                "frame_index": 5,
                "global_seconds": 0.5,
                "part_index": 0,
                "part_frame_index": 5,
                "shot_x": 70.0,
                "shot_y": 34.0,
                "goal_x": 105.0,
                "goal_y": 34.0,
                "nearest_player_distance": 1.0,
                "goalkeeper_distance": 12.0,
                "defender_count_in_cone": 0,
                "ball_speed": 40.0,
                "ball_direction_to_goal": 1.0,
                "shot_confidence": 0.8,
                "feature_source": "observed",
            }
        ]
    )
    features.to_csv(baseline_dir / "08_video_features.csv", index=False)
    pd.DataFrame(
        [
            {
                "shot_id": "video-shot-0001",
                "frame_index": 5,
                "method": "video_kinematic_context",
                "xg": 0.1,
            }
        ]
    ).to_csv(baseline_dir / "09_predictions.csv", index=False)
    (baseline_dir / "10_metrics.json").write_text(
        '{"predicted_shots": 1, "reference_shots": 1, "hit@0.5s": 1, "hit@1.0s": 1, "hit@2.0s": 1, "temporal_mae_seconds": 0}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "id": "ref-1",
                "type": "Shot",
                "period": 1,
                "timestamp": "00:00:00.500",
                "shot_statsbomb_xg": 0.2,
                "shot_outcome": "Goal",
                "team": "A",
            }
        ]
    ).to_parquet(reference_events, index=False)
