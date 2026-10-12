"""Coordinate conventions of the detector features: metres, the attacked goal, alignment, StatsBomb -> metres."""

import cv2
import numpy as np
import pytest
import supervision as sv
from xgd_fixtures import frame_with
from xgpoc.detector.conventions import (
    GOAL,
    LENGTH,
    PITCH,
    POSTS,
    WIDTH,
    align,
    attacks_right,
    statsbomb_to_metres,
)
from xgpoc.detector.features import shot_features

from tactifoot_vision.pitch import HomographyEstimator, frame_to_pitch

# frame <- pitch: a broadcast-like view of the right half (pitch metres -> 1280 x 720 pixels)
CAMERA = cv2.getPerspectiveTransform(np.float32([[52.5, 0], [105, 0], [105, 68], [52.5, 68]]),
                                     np.float32([[200, 80], [1180, 90], [1270, 700], [10, 690]])).astype(float)


# ------------------------------------------------------------ StatsBomb -> metres
@pytest.mark.parametrize(("statsbomb", "vertex"), [
    ((120, 0), 24), ((120, 80), 29),            # corners: off by the 120 x 80 vs 105 x 68 mismatch, see below
    ((102, 18), 17), ((102, 62), 20),           # penalty box corners
    ((114, 30), 22), ((114, 50), 23),           # goal box corners
    ((108, 40), 21),                            # penalty spot
    ((120, 18), 25), ((120, 62), 28), ((120, 30), 26), ((120, 50), 27)])
def test_statsbomb_markings_land_on_the_package_landmarks(statsbomb, vertex):
    metres = statsbomb_to_metres(statsbomb)
    error = np.hypot(*(metres - PITCH.vertices[vertex]))
    # Yards around the goal: every marking of the attacked end within 10 cm (the laws give yards; 18 yd = 16.46 m, not
    # 16.5 m); the corner flags are the exception, as a
    # StatsBomb pitch is 80 yards (73.2 m) wide and the package's 68 m.
    assert error < (2.7 if vertex in (24, 29) else 0.1)


def test_statsbomb_goal_and_posts_are_the_package_goal():
    np.testing.assert_allclose(statsbomb_to_metres([120, 40]), GOAL)
    np.testing.assert_allclose(statsbomb_to_metres([[120, 36], [120, 44]]), np.vstack(POSTS), atol=3e-3)
    np.testing.assert_allclose(statsbomb_to_metres([108, 40]), [LENGTH - 12 * 0.9144, 34])  # 12 yards is 10.97 m
    # A yard is a yard: distances scale by 0.9144 exactly.
    a, b = np.array([100.0, 30.0]), np.array([110.0, 47.0])
    assert np.hypot(*(statsbomb_to_metres(a) - statsbomb_to_metres(b))) == pytest.approx(0.9144 * np.hypot(*(a - b)))


# ------------------------------------------------------- attacked goal and alignment
def test_the_attacked_goal_is_the_nearer_one():
    assert attacks_right([94, 10]) and attacks_right([60, 34]) and attacks_right([52.5, 34])
    assert not attacks_right([11, 34]) and not attacks_right([52.4, 60])


def test_alignment_is_a_rotation_not_a_mirror():
    points = np.array([[10, 20], [3, 60]], float)
    np.testing.assert_array_equal(align(points, True), points)
    rotated = align(points, False)
    np.testing.assert_allclose(rotated, [[95, 48], [102, 8]])
    np.testing.assert_allclose(align(rotated, False), points)        # an involution
    # The attacker's left: facing x = 0 it is the near touchline (large y); aligned, small y, as in StatsBomb.
    assert align([20, 60], False)[1] < WIDTH / 2


def _mirrored(players, ball):
    """The same situation with the teams attacking the other goal (the pitch rotated by 180 degrees)."""
    return [(LENGTH - x, WIDTH - y, team, cls) for x, y, team, cls in players], (LENGTH - ball[0], WIDTH - ball[1])


def test_mirrored_fixtures_give_identical_aligned_features():
    players = [(93.5, 30, 0, "player"), (100, 33, 1, "player"), (99, 50, 1, "player"), (97, 31, 0, "player"),
               (103.5, 35, 1, "goalkeeper"), (60, 10, 0, "player")]
    ball = (94.0, 30.5)
    right = shot_features(frame_with(players, ball))
    left = shot_features(frame_with(*_mirrored(players, ball)))
    assert right["attack_right"] == 1.0 and left["attack_right"] == 0.0
    for name, value in right.items():
        if name != "attack_right":
            assert left[name] == pytest.approx(value, nan_ok=True), name


def test_statsbomb_and_detector_agree_on_one_situation_after_alignment():
    # One situation: StatsBomb draws it attacking x = 120; on the broadcast frame the team attacks x = 0.
    statsbomb_shot = np.array([108.0, 34.0])
    shot = statsbomb_to_metres(statsbomb_shot)
    detected = np.array([LENGTH, WIDTH]) - shot                       # where the frame puts the ball
    np.testing.assert_allclose(align(detected, attacks_right(detected)), shot)


# ----------------------------------------------- homography exactness in metres
def _keypoints(to_frame, shift=(0.0, 0.0)):
    shifted = (PITCH.vertices + np.float32(shift)).reshape(-1, 1, 2)
    xy = cv2.perspectiveTransform(shifted, to_frame).reshape(1, -1, 2).astype(np.float32)
    return sv.KeyPoints(xy=xy, confidence=np.full((1, 32), 0.9, np.float32))


def _shot_location(to_frame, keypoints, truth):
    """Shot location in metres of a ball truly at `truth`, through the package's stateless fit and the features."""
    matrix = HomographyEstimator(PITCH).fit(keypoints)
    ball_px = cv2.perspectiveTransform(np.float32([[truth]]), to_frame).reshape(1, 2)
    ball = tuple(frame_to_pitch(ball_px, matrix)[0])
    features = shot_features(frame_with([(90, 30, 0, "player")], ball, homography=matrix))
    return np.array([features["shot_x"], features["shot_y"]])


def test_a_perfect_homography_gives_zero_metres():
    truth = np.array([94.0, 30.0])
    assert np.hypot(*(_shot_location(CAMERA, _keypoints(CAMERA), truth) - truth)) < 1e-3


@pytest.mark.parametrize("shift", [(1.0, 0.0), (0.0, 2.0), (-3.0, 4.0)])
def test_a_known_landmark_shift_gives_exactly_that_error(shift):
    truth = np.array([94.0, 30.0])
    found = _shot_location(CAMERA, _keypoints(CAMERA, shift), truth)
    np.testing.assert_allclose(found - truth, -np.asarray(shift), atol=1e-3)
    assert np.hypot(*(found - truth)) == pytest.approx(np.hypot(*shift), abs=1e-3)
