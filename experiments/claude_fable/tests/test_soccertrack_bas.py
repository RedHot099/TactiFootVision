import json
from pathlib import Path

from xgpoc.data.soccertrack_bas import HALF_OFFSET_MS, parse_bas


def _write(tmp_path: Path, actions):
    p = tmp_path / "bas.json"
    p.write_text(json.dumps({"match_id": 1, "fps": 25.0, "actions": actions}))
    return p


def test_shot_then_goal_same_team(tmp_path):
    acts = [
        {"gameTime": "1 - 10:00", "label": "SHOT", "position": "600000", "team": "left"},
        {"gameTime": "1 - 10:02", "label": "GOAL", "position": "602000", "team": "left"},
        {"gameTime": "1 - 20:00", "label": "SHOT", "position": "1200000", "team": "right"},
    ]
    a = parse_bas(_write(tmp_path, acts))
    assert [(x.label, x.goal, x.trusted) for x in a] == [("SHOT", True, True), ("SHOT", False, True)]


def test_goal_without_candidate_is_weak(tmp_path):
    acts = [{"gameTime": "2 - 60:00", "label": "GOAL", "position": str(60 * 60000), "team": "left"}]
    a = parse_bas(_write(tmp_path, acts))
    assert len(a) == 1 and a[0].goal and not a[0].trusted and a[0].reason == "goal_without_candidate"
    assert a[0].video_ms == 60 * 60000 - HALF_OFFSET_MS


def test_goal_credited_to_nearest_candidate(tmp_path):
    acts = [
        {"gameTime": "1 - 10:00", "label": "FREE KICK", "position": "600000", "team": "left"},
        {"gameTime": "1 - 10:04", "label": "SHOT", "position": "604000", "team": "left"},
        {"gameTime": "1 - 10:06", "label": "GOAL", "position": "606000", "team": "left"},
    ]
    a = parse_bas(_write(tmp_path, acts))
    by = {x.label: x for x in a}
    assert by["SHOT"].goal and not by["FREE KICK"].goal


def test_other_team_goal_not_credited(tmp_path):
    acts = [
        {"gameTime": "1 - 10:00", "label": "SHOT", "position": "600000", "team": "left"},
        {"gameTime": "1 - 10:02", "label": "GOAL", "position": "602000", "team": "right"},
    ]
    a = parse_bas(_write(tmp_path, acts))
    shots = [x for x in a if x.label == "SHOT"]
    assert not shots[0].goal and any(x.reason == "goal_without_candidate" for x in a)


def test_unflagged_headers_and_free_kicks_are_not_trusted_shots(tmp_path):
    acts = [{"gameTime": "1 - 10:00", "label": label, "position": str(600000 + i * 60000), "team": "left"}
            for i, label in enumerate(("HEADER", "FREE KICK"))]
    attempts = parse_bas(_write(tmp_path, acts))
    assert len(attempts) == 2
    assert all(not a.trusted and a.reason == "shot_intent_unverified" for a in attempts)


def test_explicit_shot_flag_makes_header_a_candidate(tmp_path):
    acts = [{"gameTime": "1 - 10:00", "label": "HEADER", "position": "600000", "team": "left", "is_shot": True}]
    assert parse_bas(_write(tmp_path, acts))[0].trusted
