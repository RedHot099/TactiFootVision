"""FisheyeUndistortion moves frame points as OpenCV's fisheye model does."""

import numpy as np
import pytest

from tactifoot_vision.pitch import FisheyeUndistortion

CAMERA = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])


def test_the_principal_point_stays_and_other_points_move_outwards():
    undistortion = FisheyeUndistortion(CAMERA, np.zeros(4), CAMERA)

    moved = undistortion.points([[320, 240], [520, 240], [320, 40]])
    np.testing.assert_allclose(moved[0], [320, 240], atol=1e-6)
    # An equidistant fisheye squeezes the edges: undistorting pushes them out.
    assert moved[1, 0] > 520 and moved[1, 1] == pytest.approx(240, abs=1e-6)
    assert moved[2, 1] < 40 and moved[2, 0] == pytest.approx(320, abs=1e-6)
    assert undistortion.points(np.zeros((0, 2))).shape == (0, 2)


def test_a_new_camera_matrix_rescales_the_result():
    half = CAMERA.copy()
    half[:2] /= 2
    undistortion = FisheyeUndistortion(CAMERA, np.zeros(4), half)

    np.testing.assert_allclose(
        undistortion.points([[320, 240]]), [[160, 120]], atol=1e-6
    )


def test_loading_needs_the_three_arrays(tmp_path):
    np.savez(tmp_path / "cal.npz", K=CAMERA, D=np.zeros((4, 1)), Knew=CAMERA)
    loaded = FisheyeUndistortion.load(tmp_path / "cal.npz")
    np.testing.assert_allclose(loaded.camera, CAMERA)
    assert loaded.distortion.shape == (4,) and (loaded.distortion == 0).all()
    np.testing.assert_allclose(loaded.new_camera, CAMERA)

    np.savez(tmp_path / "bad.npz", K=CAMERA)
    with pytest.raises(ValueError, match="needs K, D and Knew"):
        FisheyeUndistortion.load(tmp_path / "bad.npz")


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="3x3"):
        FisheyeUndistortion(np.eye(2), np.zeros(4), CAMERA)
    with pytest.raises(ValueError, match="4 fisheye coefficients"):
        FisheyeUndistortion(CAMERA, np.zeros(5), CAMERA)
