"""Line refinement of the homography (tv.pitch.lines) on frames rendered through a known homography."""

import cv2
import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.pitch import (
    HomographyEstimator,
    SoccerPitch,
    frame_to_pitch,
    lines,
    pitch_markings,
)
from tactifoot_vision.pitch.lines import line_mask, refine_homography

SIZE = (1280, 720)


def _camera(pitch: SoccerPitch) -> np.ndarray:
    """Pitch -> frame: a broadcast-like view of the right half (far touchline at the top)."""
    src = np.float32(
        [
            [pitch.length / 2, 0],
            [pitch.length, 0],
            [pitch.length, pitch.width],
            [pitch.length / 2, pitch.width],
        ]
    )
    dst = np.float32([[200, 80], [1180, 90], [1270, 700], [10, 690]])
    return cv2.getPerspectiveTransform(src, dst).astype(float)


def _render(pitch: SoccerPitch, to_frame: np.ndarray) -> np.ndarray:
    """Grass with the painted markings drawn through ``to_frame`` (3 px white lines)."""
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    frame[:] = (40, 130, 50)
    lines, _ = pitch_markings(pitch, samples=256)
    for line in lines:
        dense = np.vstack(
            [np.linspace(a, b, 50) for a, b in zip(line[:-1], line[1:], strict=True)]
        )
        xy = cv2.perspectiveTransform(dense.reshape(-1, 1, 2), to_frame).reshape(-1, 2)
        cv2.polylines(frame, [np.round(xy).astype(np.int32)], False, (235, 235, 235), 3)
    return frame


def _metres_off(pitch, matrix, to_frame) -> float:
    """Largest pitch error of ``matrix`` over a grid of the visible half."""
    xs, ys = np.meshgrid(
        np.linspace(pitch.length / 2 + 5, pitch.length - 2, 6),
        np.linspace(5, pitch.width - 5, 5),
    )
    truth = np.column_stack([xs.ravel(), ys.ravel()])
    frame_points = cv2.perspectiveTransform(truth.reshape(-1, 1, 2), to_frame)
    return float(
        np.linalg.norm(frame_to_pitch(frame_points, matrix) - truth, axis=1).max()
    )


def _shifted(to_frame: np.ndarray, dx: float, dy: float, zoom: float) -> np.ndarray:
    """A frame -> pitch matrix that places everything ``(dx, dy)`` off and ``zoom`` x too large."""
    off = np.array([[zoom, 0, dx], [0, zoom, dy], [0, 0, 1.0]])
    matrix = off @ np.linalg.inv(to_frame)
    return matrix / matrix[2, 2]


@pytest.mark.parametrize("size", [(105, 68), (120, 80)])
def test_refinement_recovers_a_homography_metres_off(size):
    pitch = SoccerPitch(*size)
    to_frame = _camera(pitch)
    frame = _render(pitch, to_frame)
    start = _shifted(to_frame, 1.5, -1.0, 1.02)
    assert _metres_off(pitch, start, to_frame) > 1.5
    fit = refine_homography(frame, start, pitch)
    assert fit.refined and fit.inliers > 0.95 > fit.start_inliers
    assert _metres_off(pitch, fit.matrix, to_frame) < 0.3  # pitch units


def test_the_true_homography_is_kept():
    pitch = SoccerPitch()
    to_frame = _camera(pitch)
    truth = np.linalg.inv(to_frame)
    fit = refine_homography(_render(pitch, to_frame), truth / truth[2, 2], pitch)
    assert fit.start_inliers > 0.98
    assert _metres_off(pitch, fit.matrix, to_frame) < 0.05


def test_without_visible_lines_the_start_is_kept():
    pitch = SoccerPitch()
    to_frame = _camera(pitch)
    start = _shifted(to_frame, 1.5, -1.0, 1.02)
    stands = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)  # no grass anywhere
    fit = refine_homography(stands, start, pitch)
    assert not fit.refined and fit.points == 0 and np.isnan(fit.inliers)
    np.testing.assert_array_equal(fit.matrix, start)


def test_a_fit_that_ends_off_the_lines_keeps_the_start():
    # Dashed lines (every other 40-pixel column of the frame painted over): at
    # best about half of the visible markings can land on a line.
    pitch = SoccerPitch()
    to_frame = _camera(pitch)
    frame = _render(pitch, to_frame)
    for x in range(0, SIZE[0], 80):
        frame[:, x : x + 40] = (40, 130, 50)
    start = _shifted(to_frame, 1.5, -1.0, 1.02)
    strict = refine_homography(frame, start, pitch, min_inliers=0.9)
    assert not strict.refined and strict.inliers == strict.start_inliers
    np.testing.assert_array_equal(strict.matrix, start)
    lenient = refine_homography(frame, start, pitch, min_inliers=0.2)
    assert lenient.refined and 0.2 <= lenient.inliers < 0.9
    assert _metres_off(pitch, lenient.matrix, to_frame) < 0.3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_inliers": 1.5}, "min_inliers must be in"),
        ({"inlier_px": 0}, "inlier_px must be > 0"),
        ({"search_px": -1}, "search_px must be > 0"),
        ({"step": 0}, "step must be > 0"),
    ],
)
def test_refinement_settings_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        refine_homography(np.zeros((72, 128, 3), np.uint8), np.eye(3), **kwargs)


def test_people_boxes_are_not_lines():
    pitch = SoccerPitch()
    frame = _render(pitch, _camera(pitch))
    box = np.array([[60, 330, 160, 440]])  # over the halfway line
    assert line_mask(frame)[330:440, 60:160].any()
    assert not line_mask(frame, box)[330:440, 60:160].any()


# ------------------------------------------------- in the homography estimator
def _landmarks(pitch, to_frame, seen=(17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28)):
    """Keypoints of the landmarks ``seen``, each a few pixels off (as a model places them)."""
    xy = cv2.perspectiveTransform(pitch.vertices.reshape(-1, 1, 2), to_frame)
    noise = np.random.default_rng(1).normal(0, 6, xy.shape)
    confidence = np.zeros((1, len(xy)), np.float32)
    confidence[0, list(seen)] = 0.9
    return sv.KeyPoints(
        xy=(xy + noise).reshape(1, -1, 2).astype(np.float32), confidence=confidence
    )


def test_the_estimator_refines_its_keypoint_fit_on_the_frame():
    pitch = SoccerPitch()
    to_frame = _camera(pitch)
    frame = _render(pitch, to_frame)
    keypoints = _landmarks(pitch, to_frame)
    raw = HomographyEstimator(pitch).fit(keypoints)
    estimator = HomographyEstimator(pitch, refine_with_lines=True)
    refined = estimator.fit(keypoints, frame)
    assert (
        _metres_off(pitch, refined, to_frame) < 0.2 < _metres_off(pitch, raw, to_frame)
    )
    smoothed = estimator.update(keypoints, frame, boxes=np.zeros((0, 4)))
    np.testing.assert_allclose(smoothed, refined)
    with pytest.raises(ValueError, match="needs the frame"):
        estimator.fit(keypoints)


def test_without_lines_the_estimator_keeps_its_keypoint_fit():
    pitch = SoccerPitch()
    keypoints = _landmarks(pitch, _camera(pitch))
    stands = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    refined = HomographyEstimator(pitch, refine_with_lines=True).fit(keypoints, stands)
    np.testing.assert_array_equal(refined, HomographyEstimator(pitch).fit(keypoints))


def test_a_refinement_that_ends_in_a_mirror_image_keeps_the_start(monkeypatch):
    # The markings are symmetric about the long axis, so a mirror image of the
    # true view (near and far touchline swapped) puts every marking on a line;
    # no camera above the pitch sees one. On SoccerNet frames 1% of refined fits
    # ended there from starts that were not mirror images.
    pitch = SoccerPitch()
    to_frame = _camera(pitch)
    frame = _render(pitch, to_frame)
    start = _shifted(to_frame, 1.5, -1.0, 1.02)
    flip = np.array([[1, 0, 0], [0, -1, pitch.width], [0, 0, 1.0]])
    control_px = np.array(lines._CONTROL) * SIZE
    control = cv2.perspectiveTransform(control_px.reshape(-1, 1, 2), start)
    mirror_px = cv2.perspectiveTransform(control, to_frame @ flip).reshape(-1, 2)
    monkeypatch.setattr(
        lines, "_least_squares", lambda *args: (mirror_px - control_px).ravel()
    )
    fit = refine_homography(frame, start, pitch)
    assert not fit.refined
    np.testing.assert_array_equal(fit.matrix, start)
