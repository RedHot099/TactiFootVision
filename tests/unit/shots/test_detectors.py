from tactifoot_vision.ball import BallTrajectory, BallTrajectoryPoint
from tactifoot_vision.enums import ShotDetectorKind, ShotOutcome
from tactifoot_vision.shots import KinematicShotDetector, MetadataShotDetector


def test_kinematic_detector_selects_fastest_ball_movement_and_clamps_window() -> None:
    trajectory = BallTrajectory(
        (
            BallTrajectoryPoint(1, 0.0, 0.0),
            BallTrajectoryPoint(2, 1.0, 0.0),
            BallTrajectoryPoint(3, 50.0, 0.0),
            BallTrajectoryPoint(4, 55.0, 0.0),
        )
    )

    candidates = KinematicShotDetector(window_before=5, window_after=5).detect(
        trajectory
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.frame_index == 3
    assert candidate.window.start_frame == 1
    assert candidate.window.end_frame == 4
    assert candidate.detector_kind == ShotDetectorKind.KINEMATIC


def test_metadata_detector_supports_multiple_known_shot_frames() -> None:
    trajectory = BallTrajectory(
        tuple(BallTrajectoryPoint(frame, float(frame), 0.0) for frame in range(1, 8))
    )

    candidates = MetadataShotDetector(
        action_frames=(2, 6),
        action_class="Shots on target",
        window_before=1,
        window_after=2,
    ).detect(trajectory)

    assert [candidate.frame_index for candidate in candidates] == [2, 6]
    assert candidates[0].outcome == ShotOutcome.ON_TARGET
    assert candidates[1].window.start_frame == 5
    assert candidates[1].window.end_frame == 7
