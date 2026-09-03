import json
from pathlib import Path

from xgpoc.data.soccernet_labels import parse_labels


def _write(tmp_path: Path, anns):
    p = tmp_path / "Labels-v2.json"
    p.write_text(json.dumps({"annotations": anns}))
    return p


def _ann(game_time, label, position, team="home", visibility="visible"):
    return {"gameTime": game_time, "label": label, "position": str(position), "team": team, "visibility": visibility}


def test_shot_followed_by_goal_is_goal(tmp_path):
    anns = [
        _ann("1 - 10:00", "Shots on target", 600000),
        _ann("1 - 10:03", "Goal", 603000),
        _ann("2 - 05:00", "Shots off target", 300000, team="away"),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [e.goal for e in ev] == [True, False]
    assert all(e.source == "shot" for e in ev)


def test_goal_without_shot_becomes_event(tmp_path):
    anns = [_ann("1 - 10:00", "Goal", 600000, team="away", visibility="not shown")]
    ev = parse_labels(_write(tmp_path, anns))
    assert len(ev) == 1 and ev[0].goal and ev[0].source == "goal_only" and ev[0].position_ms == 599000


def test_goal_of_other_team_not_matched(tmp_path):
    anns = [
        _ann("1 - 10:00", "Shots on target", 600000),
        _ann("1 - 10:02", "Goal", 602000, team="away"),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [e.goal for e in ev] == [False, True]


def test_goal_credited_to_nearest_preceding_shot(tmp_path):
    # save at 10:00.0, rebound at 10:03.0, ball over the line at 10:03.5
    anns = [
        _ann("1 - 10:00", "Shots on target", 600000),
        _ann("1 - 10:03", "Shots on target", 603000),
        _ann("1 - 10:03", "Goal", 603500),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [(e.position_ms, e.goal) for e in ev] == [(600000, False), (603000, True)]


def test_rebound_after_post_hit(tmp_path):
    # off-target (post) followed by the rebound that scores: the off-target shot stays a miss
    anns = [
        _ann("1 - 10:00", "Shots off target", 600000),
        _ann("1 - 10:02", "Shots on target", 602500),
        _ann("1 - 10:03", "Goal", 603000),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [(e.label, e.goal) for e in ev] == [("Shots off target", False), ("Shots on target", True)]


def test_penalty_flag_goes_to_nearest_shot_only(tmp_path):
    anns = [
        _ann("1 - 10:00", "Penalty", 600000),
        _ann("1 - 10:00", "Shots on target", 600000),
        _ann("1 - 10:00", "Goal", 600800),
        _ann("1 - 10:25", "Shots on target", 625000),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [(e.is_penalty, e.goal) for e in ev] == [(True, True), (False, False)]


def test_penalty_shot_annotated_slightly_before_label(tmp_path):
    anns = [
        _ann("1 - 09:59", "Shots on target", 599600),
        _ann("1 - 10:00", "Penalty", 600000),
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert len(ev) == 1 and ev[0].is_penalty
