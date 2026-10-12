"""H3 geometry in metres and the shot situation on planted frames (xg-detector 01)."""

import json

import numpy as np
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector.conventions import statsbomb_to_metres
from xgpoc.detector.features import (
    COLUMNS,
    FLAGS,
    GEOMETRY,
    geometry,
    in_cone,
    missing_features,
    shot_features,
)
from xgpoc.vendor.probe_targets import freeze_frame_targets

from tactifoot_vision.pipeline import NO_TEAM

ATT, DEF = 0, 1
# Attacking x = 105. Shooter at the ball (94, 34): 11 m out, central.
SCENE = [
    (93.6, 34.2, ATT, "player"),      # shooter (nearest the ball)
    (100.0, 34.0, DEF, "player"),     # in the cone
    (99.0, 35.0, DEF, "player"),      # in the cone, 5.10 m from the ball
    (96.0, 36.0, DEF, "player"),      # outside the cone, 2.83 m from the ball
    (98.0, 33.0, ATT, "player"),      # teammate in the cone: never counted
    (101.0, 50.0, DEF, "player"),     # far outside
    (104.0, 34.5, DEF, "goalkeeper"),
    (50.0, 30.0, NO_TEAM, "referee"),
]
BALL = (94.0, 34.0)


def test_planted_geometry():
    f = shot_features(frame_with(SCENE, BALL))
    assert f["dist"] == pytest.approx(11.0)
    assert f["angle"] == pytest.approx(2 * np.arctan(3.66 / 11.0))
    assert (f["ff_def_cone"], f["def_cone_ge1"], f["def_cone_ge2"]) == (2.0, 1.0, 1.0)
    assert f["ff_near_opp"] == pytest.approx(np.hypot(2, 2))
    assert f["ff_opp_3m"] == 1.0
    assert f["gk_depth"] == pytest.approx(1.0)
    assert f["gk_dist_goal"] == pytest.approx(np.hypot(1.0, 0.5))
    assert f["gk_off_line_abs"] == pytest.approx(0.5)
    assert {k: f[k] for k in FLAGS} == {"homography_ok": 1.0, "ball_ok": 1.0, "ball_fallback": 0.0, "shot_ok": 1.0,
                                        "shot_from_ball": 1.0, "shooter_ok": 1.0, "shooter_by_goal": 0.0, "teams_ok": 1.0,
                                        "gk_ok": 1.0, "gk_detected": 1.0, "gk_proxy": 0.0}
    assert (f["n_players"], f["n_goalkeepers"], f["n_defenders"], f["n_attackers"], f["n_unknown_team"]) == (6, 1, 4, 1, 0)
    assert f["shooter_ball_dist"] == pytest.approx(np.hypot(0.4, 0.2))
    assert (f["shot_x"], f["shot_y"], f["attack_right"], f["ball_source"]) == (94.0, 34.0, 1.0, 0.0)
    assert list(f) == COLUMNS


def test_cone_edges_count_and_the_goalkeeper_is_not_an_outfield_defender():
    shot = np.array([94.0, 34.0])
    edge = shot + 0.5 * (np.array([105.0, 34 - 3.66]) - shot)       # on the line shot -> post
    assert in_cone([edge, [105, 34], [104, 37.7], [93.9, 34]], shot).tolist() == [True, True, False, False]
    keeper_only = [SCENE[0], SCENE[6]]
    f = shot_features(frame_with(keeper_only, BALL))
    assert (f["ff_def_cone"], f["def_cone_ge1"], f["ff_opp_3m"]) == (0.0, 0.0, 0.0)
    assert np.isnan(f["ff_near_opp"])                                # no outfield opponent visible
    assert f["gk_depth"] == pytest.approx(1.0)


def test_an_unknown_shooter_team_is_missing_never_everyone_a_defender():
    scene = [(x, y, NO_TEAM if i == 0 else t, c) for i, (x, y, t, c) in enumerate(SCENE)]
    f = shot_features(frame_with(scene, BALL))
    assert f["teams_ok"] == 0.0
    for name in ("ff_def_cone", "def_cone_ge1", "def_cone_ge2", "ff_near_opp", "ff_opp_3m", "n_defenders"):
        assert np.isnan(f[name]), name
    # The v1.x bug counted every outfield player in the cone as a defender (here 3, the teammate included).
    assert f["dist"] == pytest.approx(11.0) and f["gk_depth"] == pytest.approx(1.0)
    # No teams at all in the frame: the same.
    nobody = [(x, y, NO_TEAM, c) for x, y, _, c in SCENE]
    assert np.isnan(shot_features(frame_with(nobody, BALL))["ff_def_cone"])


def test_players_without_a_team_are_neither_defenders_nor_attackers():
    scene = [*SCENE[:4], (100.5, 34.5, NO_TEAM, "player"), *SCENE[4:]]
    f = shot_features(frame_with(scene, BALL))
    assert (f["ff_def_cone"], f["n_unknown_team"]) == (2.0, 1.0)


def test_without_a_ball_the_shooter_is_the_player_nearest_a_goal():
    scene = [(70.0, 20.0, ATT, "player"), (95.0, 30.0, ATT, "player"), (90.0, 40.0, DEF, "player"),
             (104.0, 34.0, DEF, "goalkeeper")]
    f = shot_features(frame_with(scene, None))
    assert (f["ball_ok"], f["shooter_by_goal"], f["shot_from_ball"], f["shot_ok"]) == (0.0, 1.0, 0.0, 1.0)
    assert (f["shot_x"], f["shot_y"]) == (95.0, 30.0)
    assert f["dist"] == pytest.approx(np.hypot(10, 4)) and np.isnan(f["shooter_ball_dist"]) and np.isnan(f["ball_source"])
    assert (f["ff_def_cone"], f["n_defenders"]) == (0.0, 1.0)
    # ... and that goal decides the direction: the same players near the other goal attack x = 0.
    mirrored = [(105 - x, 68 - y, t, c) for x, y, t, c in scene]
    g = shot_features(frame_with(mirrored, None))
    assert g["attack_right"] == 0.0 and g["dist"] == pytest.approx(f["dist"]) and g["shot_y"] == 30.0
    # The rule's known weakness, flagged by shooter_by_goal: a defender nearer the goal becomes the "shooter".
    blocker = shot_features(frame_with([*scene, (100.0, 33.0, DEF, "player")], None))
    assert (blocker["shot_x"], blocker["shooter_by_goal"]) == (100.0, 1.0)


def test_goalkeeper_proxy_plays_the_goalkeepers_role():
    scene = [SCENE[0], SCENE[1], SCENE[3], (103.5, 34.0, DEF, "player"), (40.0, 30.0, DEF, "goalkeeper")]
    f = shot_features(frame_with(scene, BALL))
    assert (f["gk_ok"], f["gk_detected"], f["gk_proxy"]) == (1.0, 0.0, 1.0)  # the far goalkeeper is the other team's
    assert f["gk_depth"] == pytest.approx(1.5)
    assert f["ff_def_cone"] == 1.0                                   # the proxy is not an outfield defender ...
    assert f["n_defenders"] == 2.0
    near = [SCENE[0], (95.0, 35.0, DEF, "player")]
    g = shot_features(frame_with(near, BALL))                        # ... but counts as an opponent within 3 m
    assert (g["gk_proxy"], g["ff_opp_3m"], g["ff_def_cone"]) == (1.0, 1.0, 0.0)
    assert (f["gk_proxy_same_team"], g["gk_proxy_same_team"]) == (0.0, 0.0)
    teammate = shot_features(frame_with([SCENE[0], (103.5, 34.0, ATT, "player")], BALL))
    assert (teammate["gk_proxy"], teammate["gk_proxy_same_team"]) == (1.0, 1.0)   # no team filter, but recorded
    assert np.isnan(shot_features(frame_with(SCENE, BALL))["gk_proxy_same_team"])   # a detected goalkeeper
    alone = shot_features(frame_with([SCENE[0], (40.0, 30.0, DEF, "player")], BALL))
    assert alone["gk_ok"] == 0.0 and np.isnan(alone["gk_depth"])     # nobody in the attacked half


def test_fallback_ball_and_missing_inputs_are_flagged():
    f = shot_features(frame_with(SCENE, BALL, fallback=2))
    assert (f["ball_ok"], f["ball_fallback"], f["ball_source"]) == (1.0, 1.0, 2.0)
    assert f["dist"] == pytest.approx(11.0)
    nothing = shot_features(frame_with([], None))
    assert nothing["homography_ok"] == 1.0 and nothing["shot_ok"] == 0.0
    assert all(np.isnan(nothing[name]) for name in GEOMETRY)
    off = frame_with(SCENE, (np.nan, np.nan))                         # a ball without a pitch position is no ball
    assert shot_features(off)["ball_ok"] == 0.0 and shot_features(off)["shooter_by_goal"] == 1.0
    empty = missing_features()
    assert all(empty[flag] == 0.0 for flag in FLAGS) and all(np.isnan(empty[name]) for name in GEOMETRY)
    counted = shot_features(frame_with(SCENE, BALL, keypoint_conf=[0.9] * 10 + [0.2] * 22))
    assert counted["n_keypoints"] == 10.0


def test_geometry_matches_the_vendor_probe_targets_in_metres():
    # A StatsBomb freeze frame (yards, attacking x = 120) through the vendor's probe targets, and the same positions in
    # metres through `geometry`: distances scale by a yard, counts agree (no point near a cone edge, where the 3.66 m
    # posts and StatsBomb's 4 yards = 3.658 m could disagree).
    shot = (106.0, 37.0)
    frame = [{"teammate": False, "location": [116.0, 39.0], "position": {"name": "Center Back"}},
             {"teammate": False, "location": [112.0, 42.5], "position": {"name": "Left Back"}},
             {"teammate": False, "location": [107.0, 38.5], "position": {"name": "Right Back"}},
             {"teammate": False, "location": [100.0, 20.0], "position": {"name": "Center Midfield"}},
             {"teammate": False, "location": [118.5, 40.5], "position": {"name": "Goalkeeper"}},
             {"teammate": True, "location": [113.0, 39.0], "position": {"name": "Center Forward"}}]
    vendor = freeze_frame_targets(*shot, json.loads(json.dumps(frame)))
    opponents = [p for p in frame if not p["teammate"]]
    outfield = statsbomb_to_metres([p["location"] for p in opponents if p["position"]["name"] != "Goalkeeper"])
    keeper = statsbomb_to_metres([118.5, 40.5])
    ours = geometry(statsbomb_to_metres(shot), defenders_xy=outfield, opponents_xy=np.vstack([outfield, keeper]),
                    goalkeeper_xy=keeper)
    assert ours["ff_def_cone"] == vendor["ff_def_cone"] == 1.0
    for name in ("gk_depth", "gk_dist_goal", "gk_off_line_abs"):
        assert ours[name] == pytest.approx(0.9144 * vendor[name], abs=1e-9), name
    assert ours["dist"] == pytest.approx(0.9144 * vendor_dist(shot))
    # Within 3 m (ours) vs within 3 yards (the probe target): the same here, one opponent 1.65 m away.
    assert ours["ff_opp_3m"] == vendor["ff_opp_3m"] == 1.0


def vendor_dist(shot) -> float:
    return float(np.hypot(120 - shot[0], 40 - shot[1]))


def test_geometry_without_teams_or_goalkeeper_is_missing():
    out = geometry(np.array([94.0, 34.0]))
    assert out["dist"] == pytest.approx(11.0)
    assert all(np.isnan(out[k]) for k in GEOMETRY if k not in ("dist", "angle"))
    assert all(np.isnan(v) for v in geometry(None).values())
