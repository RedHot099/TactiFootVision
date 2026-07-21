import numpy as np

from tactifoot_vision.keypoints import Keypoint, KeypointSet
from tactifoot_vision.projection import HomographyEstimator, apply_homography


def test_four_points_project_known_points() -> None:
    keypoints = KeypointSet(
        (
            Keypoint(0, 0.0, 0.0, 1.0),
            Keypoint(2, 100.0, 0.0, 1.0),
            Keypoint(3, 0.0, 100.0, 1.0),
            Keypoint(5, 100.0, 100.0, 1.0),
        )
    )
    estimator = HomographyEstimator(min_keypoints=4)

    matrix = estimator.update(keypoints)

    assert matrix is not None
    projected = apply_homography(np.array([[100.0, 100.0]], dtype=np.float32), matrix)
    assert projected[0, 0] == np.float32(105.0)
    assert projected[0, 1] == np.float32(68.0)


def test_three_points_return_none() -> None:
    estimator = HomographyEstimator(min_keypoints=4)

    assert (
        estimator.update(
            KeypointSet((Keypoint(0, 0, 0), Keypoint(1, 1, 0), Keypoint(2, 2, 0)))
        )
        is None
    )


def test_low_confidence_points_are_ignored() -> None:
    estimator = HomographyEstimator(min_keypoints=4, min_confidence=0.5)

    matrix = estimator.update(
        KeypointSet(
            (
                Keypoint(0, 0, 0, 1.0),
                Keypoint(2, 100, 0, 1.0),
                Keypoint(3, 0, 100, 1.0),
                Keypoint(5, 100, 100, 0.1),
            )
        )
    )

    assert matrix is None


def test_smoothing_returns_mean_matrix() -> None:
    estimator = HomographyEstimator(min_keypoints=4, smoothing_window=2)
    first = KeypointSet(
        (
            Keypoint(0, 0, 0),
            Keypoint(2, 100, 0),
            Keypoint(3, 0, 100),
            Keypoint(5, 100, 100),
        )
    )
    second = KeypointSet(
        (
            Keypoint(0, 0, 0),
            Keypoint(2, 200, 0),
            Keypoint(3, 0, 200),
            Keypoint(5, 200, 200),
        )
    )

    first_matrix = estimator.update(first)
    second_matrix = estimator.update(second)

    assert first_matrix is not None
    assert second_matrix is not None
    assert not np.allclose(first_matrix, second_matrix)
