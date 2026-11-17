"""Drawing the line model's points and a lines sample's pitch lines, beside the keypoints."""

import dataclasses

import numpy as np

import tactifoot_vision as tv
import tactifoot_vision.viz as viz
from line_points import line_points
from tactifoot_vision.data.annotations import Annotations
from tactifoot_vision.pipeline.result import FrameResult
from test_viz import HOMOGRAPHY, PITCH, H, W, _frame_result


def _with_line_points(frame_result: FrameResult) -> FrameResult:
    points = line_points(HOMOGRAPHY, (W, H), pitch=PITCH, per_class=6)
    return dataclasses.replace(frame_result, keypoints=None, line_points=points)


def test_line_points_on_their_marking_are_orange_and_others_red() -> None:
    frame_result = _with_line_points(_frame_result(0))
    points = frame_result.line_points
    xy = points.xy.copy()
    xy[0] = (160, 120)  # moved off its marking
    xy[1] = (np.nan, np.inf)  # not drawable: skipped
    frame_result.line_points = tv.pitch.LinePoints(points.marking, xy, points.prob)
    annotator = viz.FrameAnnotator(
        draw_boxes=False, draw_labels=False, draw_pitch_lines=False
    )
    image = annotator.annotate(np.zeros((H, W, 3), np.uint8), frame_result)
    assert tuple(image[120, 160]) == (0, 0, 255)  # BGR red
    x, y = np.round(xy[2]).astype(int)
    assert tuple(image[y, x]) == (0, 165, 255)  # orange
    faint = viz.FrameAnnotator(
        draw_boxes=False,
        draw_labels=False,
        draw_pitch_lines=False,
        line_point_threshold=0.95,
    )
    assert not faint.annotate(np.zeros((H, W, 3), np.uint8), frame_result).any()
    off = viz.FrameAnnotator(
        draw_boxes=False,
        draw_labels=False,
        draw_pitch_lines=False,
        draw_line_points=False,
    )
    assert not off.annotate(np.zeros((H, W, 3), np.uint8), frame_result).any()


def test_a_keypoint_frame_is_drawn_as_before() -> None:
    """Without line points the new options draw nothing: the keypoint frame comes out the same."""
    frame_result = _frame_result(0)
    image = np.zeros((H, W, 3), np.uint8)
    default = viz.FrameAnnotator().annotate(image, frame_result)
    without = viz.FrameAnnotator(draw_line_points=False).annotate(image, frame_result)
    np.testing.assert_array_equal(default, without)


def test_draw_annotations_draws_the_pitch_lines_of_a_lines_sample() -> None:
    annotations = Annotations(
        np.zeros((0, 4)), np.zeros(0), lines={"Middle line": [[50, 20], [50, 220]]}
    )
    frame = np.zeros((H, W, 3), np.uint8)
    image = viz.draw_annotations(frame, annotations)
    b, g, r = image[120, 50].tolist()  # orange on the (anti-aliased) line
    assert b == 0 and r > 200 and 100 < g < 200
    assert not image[120, 150].any()
