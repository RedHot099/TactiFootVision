"""Match analysis over a pipeline result (``tv.analysis``)."""

import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.pipeline import NO_TEAM, FrameResult, PipelineResult
from tactifoot_vision.pitch import SoccerPitch

Person = tuple[str, int, float]  # class name, team id, pitch x (NaN: no position)


def _result(frames: list[list[Person]]) -> PipelineResult:
    """One frame per list of people; every pitch y is mid-pitch."""
    return _build(
        [[(name, team, x, 34.0) for name, team, x in people] for people in frames]
    )


def _placed(
    frames: list[list[tuple[int, float, float]]], pitch: SoccerPitch | None = None
) -> PipelineResult:
    """One frame per list of ``(team id, pitch x, pitch y)`` players."""
    return _build([[("player", *p) for p in people] for people in frames], pitch)


def _build(
    frames: list[list[tuple[str, int, float, float]]], pitch: SoccerPitch | None = None
) -> PipelineResult:
    """One frame per list of ``(class name, team id, pitch x, pitch y)`` people."""
    results = []
    for index, people in enumerate(frames):
        n = len(people)
        detections = sv.Detections(
            xyxy=np.tile(np.float32([0, 0, 10, 20]), (n, 1)),
            tracker_id=np.arange(n),
            data={
                "class_name": np.array([name for name, _, _, _ in people]),
                "team_id": np.array([team for _, team, _, _ in people], dtype=int),
                "pitch_xy": np.array([[x, y] for *_, x, y in people]).reshape(n, 2),
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
    return PipelineResult(
        results, 25.0, (100, 50), ["goalkeeper", "player"], pitch or SoccerPitch()
    )


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


def test_attack_direction_refuses_teams_too_close_to_call():
    level = _result([[("player", 0, 40.0), ("player", 1, 40.0)]])
    with pytest.raises(ValueError, match="0.00 m apart"):
        tv.analysis.attack_direction(level, min_separation=0)
    close = _result([[("player", 0, 40.0), ("player", 1, 40.8)]])
    with pytest.raises(ValueError, match="0.80 m apart.*min_separation=1.0"):
        tv.analysis.attack_direction(close)  # the 1 m default
    assert tv.analysis.attack_direction(close, min_separation=0.5) == {0: 1, 1: -1}
    # Metres, not pitch units: 0.8 units of a 120-unit pitch are 0.7 m.
    close.pitch = SoccerPitch(120, 80)
    with pytest.raises(ValueError, match="0.70 m apart"):
        tv.analysis.attack_direction(close, min_separation=0.75)
    with pytest.raises(ValueError, match="min_separation must be >= 0"):
        tv.analysis.attack_direction(close, min_separation=-1)


# ------------------------------------------------------------- space control
@pytest.mark.parametrize("smoothing", [1.2, 0.0])
def test_space_control_splits_two_symmetric_players_on_their_bisector(smoothing):
    # Symmetric about the line x + y = 70; every cell centre sits on a
    # whole-metre sum, so a diagonal of cells lies exactly on the line.
    result = _placed([[(0, 30.0, 20.0), (1, 50.0, 40.0)]])
    grid = tv.analysis.space_control(result, smoothing=smoothing)
    assert grid.control.shape == (68, 105) and grid.pitch == result.pitch
    side = np.add.outer(grid.y, grid.x) - 70  # rows are y, columns x
    assert (grid.control[side < 0] > 0.5).all()
    assert (grid.control[side == 0] == 0.5).all() and (side == 0).sum() == 68
    assert (grid.control[side > 0] < 0.5).all()


def test_space_control_measures_distances_in_metres():
    # On a 120 x 80 pitch a unit across is longer than a unit along, so
    # players symmetric about x = 60 split there, and the grid still has
    # one cell per metre.
    pitch = SoccerPitch(120, 80)
    result = _placed([[(1, 40.0, 10.0), (0, 80.0, 10.0)]], pitch)
    grid = tv.analysis.space_control(result)
    assert grid.control.shape == (68, 105)
    np.testing.assert_allclose(grid.x[[0, -1]], [60 / 105, 120 - 60 / 105])
    on_line = grid.x == 60
    assert on_line.sum() == 1 and (grid.control[:, on_line] == 0.5).all()
    assert (grid.control[:, grid.x < 60] < 0.5).all()
    assert (grid.control[:, grid.x > 60] > 0.5).all()


def test_space_control_values_lie_in_0_1():
    rng = np.random.default_rng(0)
    frames = [
        [(team, *xy) for team in (0, 1) for xy in rng.uniform(0, [105, 68], (11, 2))]
        for _ in range(4)
    ]
    result = _placed(frames)
    for smoothing in (0.0, 0.3, 1.2, 50.0):
        control = tv.analysis.space_control(result, smoothing=smoothing).control
        assert np.isfinite(control).all()
        assert control.min() >= 0 and control.max() <= 1
        assert 0 < control.mean() < 1  # both teams control something
    hard = tv.analysis.space_control(result, frames=0, smoothing=0).control
    assert set(np.unique(hard)) <= {0.0, 0.5, 1.0}


def test_space_control_averages_a_frame_range():
    # The teams swap places between frames 1 and 2: over both, each point is
    # shared evenly; frame 0 is outside the range.
    result = _placed([
        [(0, 10.0, 10.0), (1, 20.0, 10.0)],
        [(0, 30.0, 34.0), (1, 75.0, 34.0)],
        [(0, 75.0, 34.0), (1, 30.0, 34.0)],
    ])  # fmt: skip
    one = tv.analysis.space_control(result, frames=1).control
    assert (one[:, :52] > 0.5).all()
    both = tv.analysis.space_control(result, frames=slice(1, 3)).control
    np.testing.assert_allclose(both, 0.5)
    every = tv.analysis.space_control(result).control
    frames = [tv.analysis.space_control(result, frames=i).control for i in range(3)]
    np.testing.assert_allclose(every, np.mean(frames, axis=0))


def test_space_control_counts_positioned_team_members_only():
    result = _placed([
        [
            (0, 30.0, 34.0),
            (1, 75.0, 34.0),
            (NO_TEAM, 52.0, 34.0),  # a referee
            (1, np.nan, np.nan),  # no pitch position
        ],
        [(0, 50.0, 34.0), (1, np.nan, np.nan)],  # team 1 not positioned: skipped
    ])  # fmt: skip
    grid = tv.analysis.space_control(result)
    np.testing.assert_array_equal(
        grid.control, tv.analysis.space_control(result, frames=0).control
    )
    np.testing.assert_allclose(grid.control[:, grid.x == 52.5], 0.5)
    with pytest.raises(ValueError, match="both teams 0 and 1.* 1 frame selected"):
        tv.analysis.space_control(result, frames=1)
    with pytest.raises(ValueError, match=" 0 frames selected"):
        tv.analysis.space_control(result, frames=slice(5, 9))


def test_space_control_cell_size_and_smoothing_options():
    result = _placed([[(0, 30.0, 34.0), (1, 75.0, 34.0)]])
    coarse = tv.analysis.space_control(result, cell_size=5)
    assert coarse.control.shape == (14, 21)
    np.testing.assert_allclose(np.diff(coarse.x), 5)
    np.testing.assert_allclose(np.diff(coarse.y), 68 / 14)
    sharp = tv.analysis.space_control(result, smoothing=0.5).control
    soft = tv.analysis.space_control(result, smoothing=5).control
    assert sharp[34, 0] > soft[34, 0] > 0.5  # deep in team 0's space
    with pytest.raises(ValueError, match="cell_size must be > 0"):
        tv.analysis.space_control(result, cell_size=0)
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        tv.analysis.space_control(result, smoothing=-1)


def test_space_control_grid_checks_its_values():
    with pytest.raises(ValueError, match="NumPy array.*np.asarray"):
        tv.analysis.SpaceControl([[0.5, 0.5]], SoccerPitch())
    with pytest.raises(ValueError, match=r"non-empty \(rows, columns\).*\(105,\)"):
        tv.analysis.SpaceControl(np.full(105, 0.5), SoccerPitch())
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        tv.analysis.SpaceControl(np.full((68, 105), 1.5), SoccerPitch())
