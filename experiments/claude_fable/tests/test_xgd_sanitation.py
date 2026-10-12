"""The label-free sanitation rule and the count rules of approach 1 on planted frames (xg-detector 03, council C1)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector import sanitation as sn
from xgpoc.detector.features import FLAGS, GEOMETRY, shot_features, situation

from tactifoot_vision.pipeline import NO_TEAM

ATT, DEF = 0, 1
SCENE = [                                     # attacking x = 105, the ball 11 m out (as test_xgd_features)
    (93.6, 34.2, ATT, "player"),              # shooter
    (100.0, 34.0, DEF, "player"),             # in the cone
    (99.0, 35.0, DEF, "player"),              # in the cone
    (96.0, 36.0, DEF, "player"),              # 2.83 m from the ball
    (98.0, 33.0, ATT, "player"),              # teammate
    (101.0, 50.0, DEF, "player"),
    (104.0, 34.5, DEF, "goalkeeper"),
    (50.0, 30.0, NO_TEAM, "referee"),
]
BALL = (94.0, 34.0)


def sanitised(players, ball=BALL, *, homography_ok=True, **kwargs):
    return sn.sanitised_features(situation(frame_with(players, ball, **kwargs)), homography_ok)


def test_on_the_pitch_everything_equals_the_extracted_features():
    raw, clean = shot_features(frame_with(SCENE, BALL)), sanitised(SCENE)
    assert {k: clean[k] for k in [*GEOMETRY, *FLAGS]} == pytest.approx({k: raw[k] for k in [*GEOMETRY, *FLAGS]})
    assert (clean["n_valid"], clean["n_masked"], clean["shot_masked"], clean["gk_masked"]) == (7.0, 0.0, 0.0, 0.0)
    assert clean["ball_source"] == 0.0 and list(clean) == sn.SANITISED


def test_an_off_pitch_point_becomes_missing_and_is_not_counted():
    near_line = [(104.0, 70.0, ATT, "player"), (104.0, 72.5, DEF, "player"), (104.0, 66.0, DEF, "player"),
                 (104.0, 34.0, DEF, "goalkeeper")]
    raw = shot_features(frame_with(near_line, (104.0, 70.5)))
    clean = sanitised(near_line, (104.0, 70.5))
    assert raw["ff_opp_3m"] == 1.0 and raw["n_defenders"] == 2.0          # the extraction counted the 72.5 m point
    assert (clean["ff_opp_3m"], clean["n_valid"], clean["n_masked"]) == (0.0, 3.0, 1.0)
    assert clean["ff_near_opp"] == pytest.approx(np.hypot(0, 4.5))       # nearest valid opponent: the 66 m one


def test_the_band_is_three_metres_inclusive_on_every_side():
    inside = [(-3.0, 34.0), (108.0, 34.0), (50.0, -3.0), (50.0, 71.0)]
    outside = [(-3.01, 34.0), (108.01, 34.0), (50.0, -3.01), (50.0, 71.01), (np.nan, 30.0), (3466.0, 20.0)]
    assert sn.on_pitch(inside).tolist() == [True] * 4
    assert sn.on_pitch(outside).tolist() == [False] * 6


def test_a_masked_shot_location_masks_every_geometry_feature_and_the_shot_is_kept():
    clean = sanitised(SCENE, (60.0, -10.0))
    assert (clean["shot_masked"], clean["shot_ok"], clean["ball_ok"], clean["teams_ok"], clean["gk_ok"]) == (1, 0, 0, 0, 0)
    assert all(np.isnan(clean[name]) for name in GEOMETRY)
    assert clean["n_valid"] == 7.0                                         # the players are still counted as valid
    assert np.isnan(clean["ball_source"])
    nothing = sanitised([], None)
    assert (nothing["shot_ok"], nothing["shot_masked"], nothing["n_valid"]) == (0.0, 0.0, 0.0)


def test_no_reselection_an_off_pitch_ball_never_falls_back_to_the_shooters_feet():
    # The extraction took the ball, so the shot location is the ball's; masking it does not make the shooter's feet
    # (an on-pitch player) the shot location.
    clean = sanitised([(100.0, 69.0, ATT, "player"), (95.0, 40.0, DEF, "player")], (104.0, 75.0))
    assert (clean["shot_masked"], clean["shot_ok"], clean["shooter_by_goal"]) == (1.0, 0.0, 0.0)


def test_an_unknown_shooter_team_gives_missing_counts_not_zero():
    unknown = [(x, y, NO_TEAM if i == 0 else t, c) for i, (x, y, t, c) in enumerate(SCENE)]
    clean = sanitised(unknown)
    assert clean["teams_ok"] == 0.0
    for name in sn.TEAM_COUNTS:
        assert np.isnan(clean[name]), name
    assert clean["dist"] == pytest.approx(11.0) and clean["gk_depth"] == pytest.approx(1.0)
    # a known team with no opponent in view is a real 0, and the valid-detection count tells it from missing
    alone = sanitised([SCENE[0], SCENE[6]])
    assert (alone["ff_def_cone"], alone["ff_opp_3m"], alone["n_valid"]) == (0.0, 0.0, 2.0)


def test_an_off_pitch_shooter_makes_the_team_unknown():
    # the ball is on the pitch, the nearest player (the shooter) projects 4 m beyond the touchline
    clean = sanitised([(100.0, 72.5, ATT, "player"), (101.0, 66.0, DEF, "player")], (100.0, 70.5))
    assert (clean["shot_ok"], clean["shooter_ok"], clean["teams_ok"]) == (1.0, 0.0, 0.0)
    assert np.isnan(clean["ff_def_cone"]) and np.isfinite(clean["dist"])


def test_without_a_homography_the_counts_are_missing():
    clean = sanitised(SCENE, homography_ok=False)
    assert clean["homography_ok"] == 0.0 and clean["teams_ok"] == 0.0
    assert all(np.isnan(clean[name]) for name in sn.TEAM_COUNTS)
    assert clean["dist"] == pytest.approx(11.0)
    # the real case: the t0 frame has no fit (people without positions) and the ball comes from an earlier frame
    frame = frame_with(SCENE, BALL, fallback=1)
    frame.detections.data["pitch_xy"] = np.full((len(SCENE), 2), np.nan)
    real = sn.sanitised_features(situation(dataclasses.replace(frame, homography=None)), False)
    assert (real["shot_ok"], real["ball_fallback"], real["n_valid"], real["teams_ok"]) == (1.0, 1.0, 0.0, 0.0)
    assert np.isnan(real["ff_def_cone"]) and np.isnan(real["ff_opp_3m"])


@pytest.mark.parametrize(("keeper", "kept"), [
    ((104.0, 34.5), True),            # in the box
    ((86.0, 34.0), True),             # 2.5 m outside the box front
    ((85.0, 34.0), False),            # 3.5 m outside it
    ((100.0, 57.0), True),            # 2.84 m beside the box
    ((100.0, 57.5), False),           # 3.34 m beside it
    ((107.5, 34.0), True),            # 2.5 m behind the goal line
])
def test_the_goalkeeper_zone_is_the_penalty_area_plus_three_metres(keeper, kept):
    scene = [*SCENE[:6], (*keeper, DEF, "goalkeeper")]
    clean = sanitised(scene)
    assert (clean["gk_ok"], clean["gk_masked"]) == (float(kept), float(not kept))
    assert np.isfinite(clean["gk_depth"]) == kept and np.isfinite(clean["gk_dist_goal"]) == kept
    assert clean["ff_def_cone"] == 2.0                                     # never an outfield defender either way


def test_a_masked_goalkeeper_keeps_its_role_and_a_proxy_is_masked_the_same_way():
    edge = [(83.6, 34.2, ATT, "player"), (82.0, 35.0, DEF, "player")]    # shot from 21 m; the proxy 6.5 m outside the box
    raw, clean = shot_features(frame_with(edge, (84.0, 34.0))), sanitised(edge, (84.0, 34.0))
    assert raw["gk_proxy"] == 1.0 and clean["gk_masked"] == 1.0
    assert np.isnan(clean["gk_off_line_abs"]) and clean["ff_opp_3m"] == 1.0   # still an opponent within 3 m
    assert clean["ff_def_cone"] == 0.0                                    # and still out of the outfield counts
    off = [SCENE[0], (110.0, 36.0, DEF, "player"), (100.0, 45.0, DEF, "player")]   # the proxy 5 m behind the goal line
    masked = sanitised(off)
    assert (masked["gk_masked"], masked["gk_ok"]) == (1.0, 0.0) and masked["n_masked"] == 1.0


def test_raster_points_put_each_person_in_one_channel():
    scene_xy, scene_channel = sn.raster_points(situation(frame_with(SCENE, BALL)))
    xy, channel = scene_xy, scene_channel
    assert channel.tolist() == [sn.ATTACKER, sn.DEFENDER, sn.DEFENDER, sn.DEFENDER, sn.ATTACKER, sn.DEFENDER, sn.KEEPER,
                                sn.BALL]                                  # the referee is never drawn
    assert xy[-1].tolist() == [94.0, 34.0]
    unknown = [(x, y, NO_TEAM if i == 0 else t, c) for i, (x, y, t, c) in enumerate(SCENE)]
    _, channel = sn.raster_points(situation(frame_with(unknown, BALL)))
    assert channel.tolist() == [sn.ATTACKER, *[sn.UNKNOWN] * 5, sn.KEEPER, sn.BALL]   # nobody becomes a defender
    far_keeper = [*SCENE[:6], (80.0, 34.0, DEF, "goalkeeper")]
    _, channel = sn.raster_points(situation(frame_with(far_keeper, BALL)))
    assert sn.KEEPER not in channel.tolist() and channel.tolist().count(sn.DEFENDER) == 4 + 1
    mirrored = [(105 - x, 68 - y, t, c) for x, y, t, c in SCENE]           # attacking x = 0: aligned by rotation
    xy_m, channel_m = sn.raster_points(situation(frame_with(mirrored, (105 - BALL[0], 68 - BALL[1]))))
    assert np.allclose(xy_m, scene_xy) and channel_m.tolist() == scene_channel.tolist()
    xy_off, _ = sn.raster_points(situation(frame_with(SCENE, (60.0, -10.0))))
    assert len(xy_off) == 0                                                # no valid shot location: no raster


def _cached_frames(tmp_path, frames: dict[tuple[str, int], object]) -> pd.DataFrame:
    rows = [ex.frame_row(shot_id, offset, f"/f/{shot_id}{offset}.jpg", frame) for (shot_id, offset), frame in frames.items()]
    ex.frames_table(rows).to_parquet(tmp_path / "frames.parquet")
    return pd.read_parquet(tmp_path / "frames.parquet")


def test_the_same_rule_applies_at_t0_and_400_through_the_cached_frames(tmp_path):
    off_pitch = [*SCENE, (100.0, 75.0, DEF, "player")]
    frames = _cached_frames(tmp_path, {("a", 0): frame_with(off_pitch, BALL), ("a", -400): frame_with(off_pitch, BALL)})
    table = sn.sanitised_table(frames, ex.CONFIGS["D3"]).set_index("offset_ms")
    for offset in (0, -400):
        row = table.loc[offset]
        assert (row.n_masked, row.n_valid, row.ff_def_cone, row.aligned) == (1.0, 7.0, 2.0, True)
        assert len(row.pts_x) == len(row.pts_channel) == 8
    replayed = ex.replay_features(frames, ex.CONFIGS["D3"], ex.STUDY_OFFSETS)
    assert replayed.n_defenders.tolist() == [5.0, 5.0]                    # the extraction itself counted the point


def test_the_ball_fallback_is_the_extractions_and_an_unreadable_frame_is_a_missing_row(tmp_path):
    no_ball = frame_with(SCENE, None)
    frames = _cached_frames(tmp_path, {("a", 0): no_ball, ("a", -400): frame_with(SCENE, (60.0, -9.0)), ("a", -800): None})
    table = sn.sanitised_table(frames, ex.CONFIGS["D3"]).set_index("offset_ms")
    # t0 took the -400 ms ball, as extracted; it is off the pitch, so t0's shot location is masked (no new choice)
    assert (table.loc[0, "shot_masked"], table.loc[0, "ball_ok"]) == (1.0, 0.0)
    assert (table.loc[-400, "shot_masked"], table.loc[-400, "aligned"]) == (1.0, False)
    unreadable = _cached_frames(tmp_path, {("b", 0): None, ("b", -400): None, ("b", -800): None})
    row = sn.sanitised_table(unreadable, ex.CONFIGS["D3"]).iloc[0]
    assert (row.shot_ok, row.homography_ok, row.n_valid, row.aligned) == (0.0, 0.0, 0.0, False)
    assert all(np.isnan(row[name]) for name in GEOMETRY)
