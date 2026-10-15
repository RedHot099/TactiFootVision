"""Match analysis over a pipeline result (``tv.analysis``)."""

import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.pipeline import NO_TEAM, FrameResult, PipelineResult

Person = tuple[str, int, float]  # class name, team id, pitch x (NaN: no position)


def _result(frames: list[list[Person]]) -> PipelineResult:
    """One frame per list of people; every pitch y is mid-pitch."""
    results = []
    for index, people in enumerate(frames):
        n = len(people)
        detections = sv.Detections(
            xyxy=np.tile(np.float32([0, 0, 10, 20]), (n, 1)),
            tracker_id=np.arange(n),
            data={
                "class_name": np.array([name for name, _, _ in people]),
                "team_id": np.array([team for _, team, _ in people]),
                "pitch_xy": np.float32([[x, 34.0] for _, _, x in people]),
            },
        )
        results.append(
            FrameResult(
                index,
                index / 25,
                detections,
                sv.Detections.empty(),
                ball_candidates=sv.Detections.empty(),
            )
        )
    return PipelineResult(results, 25.0, (100, 50), ["goalkeeper", "player"])


def _period(home_x: float, away_x: float, home_keeper_x: float) -> PipelineResult:
    """Team 0 around ``home_x`` with its keeper, team 1 around ``away_x``, a referee."""
    return _result([
        [
            ("goalkeeper", 0, home_keeper_x),
            ("player", 0, home_x - 4),
            ("player", 0, home_x + 2),
            ("player", 1, away_x),
            ("player", 1, np.nan),  # no pitch position: ignored
            ("referee", NO_TEAM, 104.0),  # no team: ignored
        ],
        [("player", 0, home_x), ("player", 1, away_x + 3)],
    ])  # fmt: skip


def test_attack_direction_flips_between_the_periods():
    # Period 1: team 0 defends the goal at x = 0 and attacks towards x = length.
    first = _period(home_x=35, away_x=70, home_keeper_x=3)
    # Period 2: the teams have changed ends.
    second = _period(home_x=70, away_x=35, home_keeper_x=102)
    assert tv.analysis.attack_direction(first) == {0: 1, 1: -1}
    assert tv.analysis.attack_direction(second) == {0: -1, 1: 1}


def test_attack_direction_compares_the_teams_not_the_halfway_line():
    # The camera stays in one half: both teams stand beyond the halfway line,
    # team 1 deeper towards the goal at x = length, which it defends.
    result = _period(home_x=60, away_x=85, home_keeper_x=58)
    assert tv.analysis.attack_direction(result) == {0: 1, 1: -1}


def test_attack_direction_needs_two_teams_with_positions():
    one_team = _result([[("player", 0, 30.0), ("player", 1, np.nan)]])
    with pytest.raises(ValueError, match="two teams.*team 0"):
        tv.analysis.attack_direction(one_team)
    three_teams = _result(
        [[("player", 0, 30.0), ("player", 1, 50.0), ("player", 2, 70.0)]]
    )
    with pytest.raises(ValueError, match="two teams.*teams 0, 1, 2"):
        tv.analysis.attack_direction(three_teams)
