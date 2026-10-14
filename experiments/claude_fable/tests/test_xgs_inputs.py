"""xg-stack inputs (ticket 01, council C5): the SoccerNet flag windows on hand-made annotation lists, the E3 replay and
sanitation on planted frames, the ball one-hot, the missing groups, the population checks and the outcome object."""

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.stack import inputs as si

from tactifoot_vision.pipeline import NO_TEAM

SHOT_AT = 600_000                              # the shot annotation, first half, 10:00


def annotation(label: str, dt_ms: int, team: str = "home", half: int = 1) -> dict:
    position = SHOT_AT + dt_ms
    return {"gameTime": f"{half} - {position // 60000:02d}:{position // 1000 % 60:02d}", "label": label,
            "position": str(position), "team": team, "visibility": "visible"}


def flags(*raw: dict, own: str = "Shots on target", own_team: str = "home") -> dict:
    annotations = [annotation(own, 0, own_team), *raw]
    return si.shot_flags(si.parse_annotations(annotations), 1, SHOT_AT)


NONE = dict.fromkeys(si.WINDOWS, 0.0) | {"sn_team_unknown": 0.0}


@pytest.mark.parametrize(("label", "flag", "inside", "outside"), [
    ("Direct free-kick", "sn_direct_fk", (-3000, 0, 300), (-3001, 301, 2000)),
    ("Indirect free-kick", "sn_indirect_fk_3s", (-3000, -1, 0), (-3001, 1, 1500)),
    ("Corner", "sn_corner_20s", (-20000, -5000, 0), (-20001, 1, 8000)),
    ("Shots off target", "sn_prev_shot_5s", (-5000, -2000, -300), (-5001, -299, 0, 300, 3000)),
    ("Shots on target", "sn_prev_shot_5s", (-5000, -300), (-5001, -299, 0, 2500)),
])
def test_each_flag_counts_same_team_annotations_inside_its_window_only(label, flag, inside, outside):
    for dt in inside:
        assert flags(annotation(label, dt)) == NONE | {flag: 1.0}, dt
    for dt in outside:                          # positive dt: after the shot (post-shot), never counted
        assert flags(annotation(label, dt)) == NONE, dt


def test_the_other_team_the_other_half_and_goal_annotations_never_count():
    assert flags(annotation("Direct free-kick", -1000, "away"), annotation("Corner", -5000, "away"),
                 annotation("Shots on target", -2000, "away")) == NONE
    assert flags(annotation("Corner", -5000, half=2), annotation("Direct free-kick", 0, half=2)) == NONE
    assert flags(*[annotation("Goal", dt, team) for dt in (-20000, -3000, -1000, 0, 200) for team in ("home", "away")]) == NONE


def test_the_shot_s_own_label_type_and_goals_do_not_change_any_flag():
    context = [annotation("Corner", -12000), annotation("Shots off target", -4000), annotation("Direct free-kick", 200)]
    expected = NONE | {"sn_corner_20s": 1.0, "sn_prev_shot_5s": 1.0, "sn_direct_fk": 1.0}
    assert flags(*context, own="Shots on target") == expected
    assert flags(*context, own="Shots off target") == expected
    assert flags(*context, annotation("Goal", 0), annotation("Goal", -4000, "away")) == expected


@pytest.mark.parametrize("raw", [
    [annotation("Shots on target", 0, "not applicable"), annotation("Corner", -5000)],               # unknown team
    [annotation("Goal", 0), annotation("Corner", -5000)],                                            # no shot annotation
    [annotation("Shots on target", 0), annotation("Shots off target", 0, "away"), annotation("Corner", -5000)],  # ambiguous
    [annotation("Shots on target", 50), annotation("Corner", -5000)],                                # not at the position
])
def test_an_unknown_shooting_team_sets_every_flag_to_zero_never_every_team(raw):
    assert si.shot_flags(si.parse_annotations(raw), 1, SHOT_AT) == dict.fromkeys(si.WINDOWS, 0.0) | {"sn_team_unknown": 1.0}


def test_two_annotations_of_the_same_team_at_the_shot_are_one_team():
    raw = [annotation("Shots on target", 0), annotation("Shots off target", 0), annotation("Corner", -5000)]
    assert si.shot_flags(si.parse_annotations(raw), 1, SHOT_AT) == NONE | {"sn_corner_20s": 1.0}


def test_parse_drops_goal_and_erases_the_shot_type():
    parsed = si.parse_annotations([annotation("Goal", 0), annotation("Shots on target", 0), annotation("Shots off target", -10, "away"),
                                   annotation("Throw-in", -10, "not applicable")])
    assert [(a.kind, a.team) for a in parsed] == [("shot", "home"), ("shot", "away"), ("Throw-in", None)]


def test_flags_table_follows_the_shots_rows():
    shots = pd.DataFrame({"game": ["a", "a", "b"], "half": [1, 1, 1], "position_ms": [SHOT_AT, SHOT_AT + 4000, SHOT_AT]},
                         index=["s1", "s2", "s3"])
    annotations = {"a": [annotation("Shots on target", 0), annotation("Shots off target", 4000)],
                   "b": [annotation("Shots on target", 0, "away"), annotation("Corner", -1000, "away")]}
    table = si.flags_table(shots, annotations)
    assert list(table.columns) == si.S_FLAGS and list(table.index) == ["s1", "s2", "s3"]
    assert table.sn_prev_shot_5s.tolist() == [0.0, 1.0, 0.0] and table.sn_corner_20s.tolist() == [0.0, 0.0, 1.0]


# ------------------------------------------------------------------ E3 columns
def test_ball_levels_follow_e3_s_source_order():
    assert si.ball_levels(0) == ("none", "own", "m400", "m800")
    assert si.ball_levels(-400) == ("none", "own", "m800")


@pytest.mark.parametrize(("source", "level"), [(np.nan, "none"), (0.0, "own"), (1.0, "m400"), (2.0, "m800")])
def test_offset_row_one_hot_of_the_sanitised_ball_source(source, level):
    clean = si.sn.missing_row() | {"ball_source": source, "shot_ok": 1.0, "dist": 10.0, "angle": 0.5}
    raw = si.missing_features() | {"shot_x": 95.0, "n_keypoints": 9.0, "n_players": 12.0}
    row = si.offset_row(0, raw, clean, 0.95)
    assert {k: row[si.col(f"ball_{k}", 0)] for k in si.ball_levels(0)} == {k: float(k == level) for k in si.ball_levels(0)}
    assert (row["shot_x@0"], row["shot_missing@0"], row["kp_conf_missing@0"]) == (95.0, 0.0, 0.0)


def test_offset_row_masks_shot_x_with_the_geometry():
    clean = si.sn.missing_row() | {"shot_masked": 1.0}
    raw = si.missing_features() | {"shot_x": 112.0, "n_keypoints": 9.0, "n_players": 12.0}
    row = si.offset_row(0, raw, clean, np.nan)
    assert np.isnan(row["shot_x@0"]) and np.isnan(row["dist@0"]) and row["shot_missing@0"] == 1.0
    assert row["kp_conf_missing@0"] == 1.0 and row["shot_masked@0"] == 1.0 and row["ball_none@0"] == 1.0


ATT, DEF = 0, 1
SCENE = [(93.6, 34.2, ATT, "player"), (100.0, 34.0, DEF, "player"), (98.0, 33.0, ATT, "player"),
         (104.0, 34.5, DEF, "goalkeeper"), (50.0, 30.0, NO_TEAM, "referee")]


def planted_frames() -> pd.DataFrame:
    """Frame rows of three shots: a t0 and a -400 ms ball (E3 takes -400); only a t0 ball (its last fallback, -800
    analysed first); a -400 ms ball 6 m off the pitch (masked by the sanitation)."""
    rows = []
    for shot_id, balls in (("a", {0: (94.0, 34.0), -400: (92.0, 33.0)}), ("b", {0: (94.0, 34.0), -400: None, -800: None}),
                           ("c", {0: None, -400: (95.0, -6.0), -800: None})):
        for offset, ball in balls.items():
            frame = frame_with(SCENE, ball, keypoint_conf=np.full(32, 0.9))
            frame.keypoints.data["box_confidence"] = np.array([0.95])
            rows.append(ex.frame_row(shot_id, offset, f"{shot_id}{offset}.jpg", frame))
    return pd.DataFrame(rows)


def test_e3_table_replays_the_cached_features_and_applies_the_sanitation():
    frames = planted_frames()
    cached = x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS)
    table = si.e3_table(frames, cached)
    assert list(table.index) == ["a", "b", "c"] and list(table.columns) == si.E3_COLUMNS
    assert table.loc["a", "ball_m400@0"] == 1.0 and table.loc["a", "ball_own@-400"] == 1.0
    assert table.loc["a", "dist@0"] == pytest.approx(np.hypot(105 - 92.0, 34 - 33.0))       # the -400 ms ball
    assert table.loc["b", "ball_own@0"] == 1.0 and table.loc["b", "ball_none@-400"] == 1.0
    assert table.loc["c", "shot_missing@-400"] == 1.0 and np.isnan(table.loc["c", "shot_x@-400"])
    assert table.loc["c", "shot_masked@-400"] == 1.0 and table.loc["c", "kp_instance_conf@0"] == pytest.approx(0.95)


def test_e3_table_refuses_cached_features_that_the_replay_does_not_reproduce():
    frames = planted_frames()
    cached = x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS)
    cached.loc[0, "dist"] += 0.01
    with pytest.raises(ValueError, match="differ from the cache"):
        si.e3_table(frames, cached)
    with pytest.raises(ValueError, match="not the cached rows"):
        si.e3_table(frames, cached.iloc[1:])


def test_missing_groups_are_checked():
    frames = planted_frames()
    table = si.e3_table(frames, x3.replay_features(frames, si.E3_CONFIG, si.OFFSETS))
    table.loc["a", "shot_missing@0"] = 1.0
    with pytest.raises(ValueError, match="not missing exactly"):
        si.check_missing_groups(table)


# ------------------------------------------------------------------ population and outcomes
def write_sources(tmp_path, fold_of_b: int = 1, penalty_in_oof: bool = False):
    index = pd.DataFrame({"shot_id": ["a1", "a2", "b1", "h1"], "game": ["A", "A", "B", "H"], "half": 1,
                          "position_ms": [1, 2, 3, 4], "is_penalty": [False, True, False, False],
                          "goal": [True, False, False, True], "label": "Shots on target"})
    index.to_parquet(tmp_path / "index.parquet")
    oof = pd.DataFrame({"shot_id": ["a1", "a2", "b1"], "fold": [0, 0, fold_of_b], "z_g_combo": [-1.0, 0.5, -2.0],
                        "is_penalty": [False, not penalty_in_oof, False]})
    oof.to_csv(tmp_path / "oof.csv", index=False)
    plan = pd.DataFrame({"game": ["A", "B", "H"], "role": ["training", "training", "evaluation"], "fold": [0, 1, np.nan]})
    return tmp_path / "index.parquet", tmp_path / "oof.csv", plan


def test_training_shots_keeps_the_non_penalty_shots_with_their_host_fold(tmp_path):
    index, oof, plan = write_sources(tmp_path)
    shots = si.training_shots(index, oof, plan)
    assert list(shots.index) == ["a1", "b1"] and shots.fold.tolist() == [0, 1]
    assert "goal" not in shots.columns and "label" not in shots.columns


def test_training_shots_refuses_a_wrong_fold_or_penalty_flag(tmp_path):
    index, oof, plan = write_sources(tmp_path, fold_of_b=3)
    with pytest.raises(ValueError, match="host fold"):
        si.training_shots(index, oof, plan)
    index, oof, plan = write_sources(tmp_path, penalty_in_oof=True)
    with pytest.raises(ValueError, match="penalty flag"):
        si.training_shots(index, oof, plan)


def test_outcomes_are_returned_only_in_the_population_s_order():
    table = pd.DataFrame({"game": ["A", "B"]}, index=pd.Index(["s1", "s2"], name="shot_id"))
    population = si.Population(table, {}, {})
    assert si.Outcomes(np.array(["s1", "s2"]), np.array([1, 0])).aligned(population).tolist() == [1, 0]
    with pytest.raises(ValueError, match="order"):
        si.Outcomes(np.array(["s2", "s1"]), np.array([0, 1])).aligned(population)
