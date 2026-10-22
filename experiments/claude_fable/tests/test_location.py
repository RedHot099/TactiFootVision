"""Shot-location rules on planted frames: the feet rule, the copies of the frozen code and the stack tables
(`xgpoc.detector.location`, `xgpoc.stack_location.inputs`; rule research/feet_location_rule.md)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.detector import location as loc
from xgpoc.detector.features import COLUMNS, shot_features, situation
from xgpoc.stack import inputs as si
from xgpoc.stack_location import inputs as sl

from tactifoot_vision.pipeline import NO_TEAM

ATT, DEF = 0, 1
# The ball (94, 34) projects nearest player 0, but in the image it flies above player 1 (a header): the feet rule
# picks player 1 and places the shot at his feet (90, 30).
HEADER = [
    (93.6, 34.2, ATT, "player"),
    (90.0, 30.0, ATT, "player"),
    (100.0, 34.0, DEF, "player"),
    (104.0, 34.5, DEF, "goalkeeper"),
    (50.0, 30.0, NO_TEAM, "referee"),
]
BOXES = [[0, 0, 10, 30], [500, 100, 520, 160], [300, 0, 310, 30], [400, 0, 410, 30], [200, 0, 210, 30]]
BALL_BOX = [500, 100, 510, 110]                     # centre (505, 105), inside player 1's box


def imaged(frame, boxes, ball_box=None):
    """`frame` with the people's image boxes `boxes` and, given, the ball's `ball_box`."""
    frame.detections.xyxy[:] = np.asarray(boxes, np.float32)
    if ball_box is not None:
        frame.ball.xyxy[:] = np.asarray([ball_box], np.float32)
    return frame


def header_frame(ball=(94.0, 34.0)):
    return imaged(frame_with(HEADER, ball), BOXES, BALL_BOX if ball is not None else None)


def test_the_feet_rule_takes_the_player_nearest_the_ball_in_the_image_and_his_feet():
    frame = header_frame()
    pitch, feet = situation(frame), loc.feet_situation(frame)
    assert pitch.shooter == 0 and np.allclose(pitch.shot, (94.0, 34.0))
    assert feet.shooter == 1 and np.allclose(feet.shot, (90.0, 30.0))
    assert np.allclose(feet.ball, pitch.ball) and feet.ball_source == pitch.ball_source and not feet.shooter_by_goal
    assert feet.goalkeeper == 3 and not feet.proxy and feet.attack_right
    f = loc.situation_features(frame, feet)
    assert f["dist"] == pytest.approx(np.hypot(15.0, 4.0))
    assert (f["shot_x"], f["shot_y"], f["shot_from_ball"], f["ball_ok"]) == (90.0, 30.0, 1.0, 1.0)
    assert f["shooter_ball_dist"] == pytest.approx(np.hypot(4.0, 4.0))      # feet to the ball's projection


def test_a_box_containing_the_ball_centre_is_at_distance_zero_and_ties_go_to_the_first_player():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [30, 30, 40, 40]], float)
    assert loc.box_distance(boxes, np.array([7.0, 7.0])).tolist() == [0.0, 0.0, pytest.approx(np.hypot(23, 23))]
    frame = imaged(frame_with(HEADER, (94.0, 34.0)), [[500, 100, 520, 160], *BOXES[1:]], BALL_BOX)
    assert loc.image_shooter(frame, situation(frame)) == 0                   # players 0 and 1 both contain it


def test_the_attacked_goal_and_the_goalkeeper_follow_the_feet_not_the_airborne_ball():
    scene = [(92.0, 30.0, ATT, "player"), (60.0, 34.0, ATT, "player"), (103.0, 34.0, DEF, "player"),
             (2.0, 34.0, DEF, "player")]
    boxes = [[500, 100, 520, 160], [0, 0, 10, 30], [300, 0, 310, 30], [400, 0, 410, 30]]
    frame = imaged(frame_with(scene, (40.0, 34.0)), boxes, BALL_BOX)          # the ball projects in the left half
    pitch, feet = situation(frame), loc.feet_situation(frame)
    assert pitch.shooter == 1 and not pitch.attack_right and pitch.goalkeeper == 3 and pitch.proxy
    assert feet.shooter == 0 and feet.attack_right and feet.goalkeeper == 2 and feet.proxy


def test_without_a_ball_or_a_player_the_feet_rule_is_the_pitch_rule():
    for frame in (header_frame(ball=None), frame_with([(50.0, 30.0, NO_TEAM, "referee")], (94.0, 34.0))):
        assert _same(loc.feet_situation(frame), situation(frame))


def _same(a, b) -> bool:
    return all(np.array_equal(x, y, equal_nan=True) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(b), strict=True))


@pytest.mark.parametrize("ball, fallback", [((94.0, 34.0), None), ((94.0, 34.0), 1), (None, None), ((140.0, 34.0), None)])
def test_situation_features_with_the_pitch_situation_is_shot_features(ball, fallback):
    for scene in (HEADER, [(p[0], p[1], NO_TEAM, p[3]) for p in HEADER]):
        frame = frame_with(scene, ball, fallback=fallback, keypoint_conf=np.linspace(0, 1, 32))
        a, b = loc.situation_features(frame, situation(frame), 0.3), shot_features(frame, 0.3)
        assert list(a) == list(b) == COLUMNS
        assert np.array_equal(np.array(list(a.values())), np.array(list(b.values())), equal_nan=True)


def test_rule_features_of_an_unreadable_frame_are_missing():
    f = loc.rule_features(None, loc.FEET_T0)
    assert f["shot_ok"] == 0.0 and np.isnan(f["dist"])


def test_the_named_rules():
    assert loc.RULES["E3"].source == x3.CONFIGS["E3"] and loc.E3.situation is situation and loc.E3.e3_flags_at == ()
    assert loc.FEET_T0.source.source == (0, -400, -800) and loc.FEET_T0.situation is loc.feet_situation
    assert loc.FEET_T0.e3_flags_at == (0,) and loc.FEET_T0.source.order(-400) == x3.CONFIGS["E3"].order(-400)


# ------------------------------------------------------------------ cached frames and stack tables
SCENE = [(93.6, 34.2, ATT, "player"), (88.0, 30.0, ATT, "player"), (100.0, 34.0, DEF, "player"),
         (104.0, 34.5, DEF, "goalkeeper"), (50.0, 30.0, NO_TEAM, "referee")]
SCENE_BOXES = [[0, 0, 10, 30], [500, 100, 520, 160], [300, 0, 310, 30], [400, 0, 410, 30], [200, 0, 210, 30]]


def planted_frames() -> pd.DataFrame:
    """Frame rows of three shots. a: balls at t0 (in the image above player 1) and -400 ms (above player 0); b: only
    a t0 ball (E3's last fallback, so -800 ms was analysed); c: no t0 ball, a -400 ms ball off the pitch."""
    above = {1: BALL_BOX, 0: [0, 0, 10, 10]}
    rows = []
    for shot_id, balls in (("a", {0: ((94.0, 34.0), 1), -400: ((92.0, 33.0), 0)}),
                           ("b", {0: ((94.0, 34.0), 1), -400: None, -800: None}),
                           ("c", {0: None, -400: ((95.0, -6.0), 1), -800: None})):
        for offset, ball in balls.items():
            frame = frame_with(SCENE, ball[0] if ball else None, keypoint_conf=np.full(32, 0.9))
            frame = imaged(frame, SCENE_BOXES, above[ball[1]] if ball else None)
            frame.keypoints.data["box_confidence"] = np.array([0.95])
            rows.append(ex.frame_row(shot_id, offset, f"{shot_id}{offset}.jpg", frame))
    return pd.DataFrame(rows)


def test_feature_rows_under_e3_are_the_extractor_s_rows():
    frames = planted_frames()
    mine = loc.feature_rows(frames, loc.E3)
    frozen = x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS)[mine.columns]
    assert list(mine.columns) == ["shot_id", "offset_ms", *COLUMNS]
    pd.testing.assert_frame_equal(mine, frozen)


def test_feet_t0_takes_the_t0_ball_first_and_the_feet():
    rows = loc.feature_rows(planted_frames(), loc.FEET_T0).set_index(["shot_id", "offset_ms"])
    a0 = rows.loc[("a", 0)]
    assert a0.ball_source == 0.0 and (a0.shot_x, a0.shot_y) == (88.0, 30.0)               # t0 ball, player 1's feet
    assert (rows.loc[("a", -400)].shot_x, rows.loc[("a", -400)].shot_y) == (93.6, 34.2)    # -400 ms ball, player 0
    assert rows.loc[("c", 0)].ball_source == 1.0                                          # falls back to -400 ms


def test_the_e3_table_under_e3_is_the_frozen_table():
    frames = planted_frames()
    cached = x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS)
    frozen, wrapped = si.e3_table(frames, cached)
    copy, wrapped_copy = sl.e3_table(frames, cached, loc.E3)
    pd.testing.assert_frame_equal(copy, frozen)
    assert wrapped_copy == wrapped


def test_the_feet_t0_table_keeps_e3_s_t0_ball_source_columns_only():
    frames = planted_frames()
    e3, _ = si.e3_table(frames, x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS))
    feet, _ = sl.e3_table(frames, loc.feature_rows(frames, loc.FEET_T0), loc.FEET_T0)
    t0_flags = [si.col(f"ball_{level}", 0) for level in si.ball_levels(0)] + ["shot_from_ball@0"]
    pd.testing.assert_frame_equal(feet[t0_flags], e3[t0_flags])
    assert feet.loc["a", "ball_m400@0"] == 1.0                     # E3's: the -400 ms ball, though feet_t0 used t0's
    assert feet.loc["a", "dist@0"] == pytest.approx(np.hypot(105 - 88.0, 34 - 30.0))
    assert e3.loc["a", "dist@0"] == pytest.approx(np.hypot(105 - 92.0, 34 - 33.0))
    assert feet.loc["c", "shot_missing@-400"] == 0.0 and e3.loc["c", "shot_missing@-400"] == 1.0   # feet on the pitch


def test_the_feet_t0_table_refuses_a_cache_built_under_another_rule():
    frames = planted_frames()
    with pytest.raises(ValueError, match="differ from the cache"):
        sl.e3_table(frames, x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS), loc.FEET_T0)


def test_a_rule_whose_ball_levels_are_not_e3_s_columns_is_refused():
    rule = loc.LocationRule("t0_only", x3.SourceConfig("t0_only", (0, -400)), loc.feet_situation)
    frames = planted_frames()
    with pytest.raises(ValueError, match="not E3's columns"):
        sl.e3_table(frames, loc.feature_rows(frames, rule), rule)
