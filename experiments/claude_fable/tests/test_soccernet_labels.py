import json
from pathlib import Path

from xgpoc.data.soccernet_labels import parse_labels


def _write(tmp_path: Path, anns):
    p = tmp_path / "Labels-v2.json"
    p.write_text(json.dumps({"annotations": anns}))
    return p


def test_shot_followed_by_goal_is_goal(tmp_path):
    anns = [
        {"gameTime": "1 - 10:00", "label": "Shots on target", "position": "600000", "team": "home", "visibility": "visible"},
        {"gameTime": "1 - 10:03", "label": "Goal", "position": "603000", "team": "home", "visibility": "visible"},
        {"gameTime": "2 - 05:00", "label": "Shots off target", "position": "300000", "team": "away", "visibility": "visible"},
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [e.goal for e in ev] == [True, False]
    assert all(e.source == "shot" for e in ev)


def test_goal_without_shot_becomes_event(tmp_path):
    anns = [{"gameTime": "1 - 10:00", "label": "Goal", "position": "600000", "team": "away", "visibility": "not shown"}]
    ev = parse_labels(_write(tmp_path, anns))
    assert len(ev) == 1 and ev[0].goal and ev[0].source == "goal_only" and ev[0].position_ms == 599000


def test_goal_of_other_team_not_matched(tmp_path):
    anns = [
        {"gameTime": "1 - 10:00", "label": "Shots on target", "position": "600000", "team": "home", "visibility": "visible"},
        {"gameTime": "1 - 10:02", "label": "Goal", "position": "602000", "team": "away", "visibility": "visible"},
    ]
    ev = parse_labels(_write(tmp_path, anns))
    assert [e.goal for e in ev] == [False, True]
