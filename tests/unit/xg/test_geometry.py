from tactifoot_vision.enums import ShotDetectorKind, ShotOutcome
from tactifoot_vision.shots import ShotCandidate, ShotWindow
from tactifoot_vision.xg import GeometryXgEstimator, XgShotFeatures


def test_geometry_xg_is_higher_for_closer_wider_central_shot() -> None:
    estimator = GeometryXgEstimator()
    close = XgShotFeatures(
        shot_x=100.0,
        shot_y=34.0,
        distance_to_goal=8.0,
        angle_to_goal=0.6,
        centrality=1.0,
    )
    far = XgShotFeatures(
        shot_x=70.0,
        shot_y=10.0,
        distance_to_goal=35.0,
        angle_to_goal=0.1,
        centrality=0.3,
    )

    close_xg = estimator.predict(close, _candidate()).xg
    far_xg = estimator.predict(far, _candidate()).xg

    assert close_xg > far_xg


def test_geometry_xg_uses_penalty_prior() -> None:
    estimator = GeometryXgEstimator(penalty_xg=0.77)
    features = XgShotFeatures(
        shot_x=94.0,
        shot_y=34.0,
        distance_to_goal=11.0,
        angle_to_goal=0.5,
        centrality=1.0,
        is_penalty=True,
    )

    prediction = estimator.predict(features, _candidate(outcome=ShotOutcome.PENALTY))

    assert prediction.xg == 0.77


def _candidate(outcome: ShotOutcome = ShotOutcome.UNKNOWN) -> ShotCandidate:
    return ShotCandidate(
        frame_index=1,
        window=ShotWindow(1, 1),
        confidence=1.0,
        detector_kind=ShotDetectorKind.METADATA,
        outcome=outcome,
    )
