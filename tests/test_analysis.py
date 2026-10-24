"""Match analysis over a pipeline result (``tv.analysis``)."""

import numpy as np
import pandas as pd
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.enums import SpaceModel
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


# ---------------------------------------------------------------- velocities
def _moving(
    xy: np.ndarray,
    indices: np.ndarray,
    teams: list[int] | None = None,
    ids: np.ndarray | None = None,
    pitch: SoccerPitch | None = None,
) -> PipelineResult:
    """``xy[frame, person]`` pitch positions at video frame ``indices`` (25 fps)."""
    frames = []
    for index, people, frame_ids in zip(
        indices, xy, ids if ids is not None else [None] * len(xy), strict=True
    ):
        n = len(people)
        detections = sv.Detections(
            xyxy=np.tile(np.float32([0, 0, 10, 20]), (n, 1)),
            tracker_id=np.arange(n) if frame_ids is None else np.asarray(frame_ids),
            data={
                "class_name": np.full(n, "player"),
                "team_id": np.array(teams if teams is not None else [0] * n),
                "pitch_xy": np.asarray(people, dtype=float),
            },
        )
        frames.append(
            FrameResult(
                int(index),
                index / 25,
                detections,
                sv.Detections.empty(),
                ball_candidates=sv.Detections.empty(),
            )
        )
    return PipelineResult(frames, 25.0, (100, 50), ["player"], pitch or SoccerPitch())


@pytest.mark.parametrize("stride", [1, 4, 10])
def test_velocities_are_metres_per_second_at_any_stride(stride):
    # A player runs at (3, -1) m/s; on a 120 x 80 pitch that is
    # (3 * 120 / 105, -80 / 68) units a second.
    pitch = SoccerPitch(120, 80)
    indices = np.arange(0, 50, stride)
    seconds = indices / 25
    units = np.stack([20 + 3 * seconds * 120 / 105, 40 - seconds * 80 / 68], axis=-1)
    result = _moving(units[:, None], indices, pitch=pitch)
    speeds = tv.analysis.velocities(result)
    assert len(speeds) == len(result.frames)
    for frame, velocity in zip(result.frames, speeds, strict=True):
        assert velocity.shape == (len(frame.detections), 2)
    # At stride 10 (0.4 s) the 0.5 s window widens to the frames either side.
    # From stride 4 (0.16 s) the track's ends hold two frames in their window
    # and have none; inside, every window holds three or more.
    ends = stride >= 4
    inner = np.concatenate(speeds[1:-1] if ends else speeds)
    np.testing.assert_allclose(inner, np.tile([3.0, -1.0], (len(inner), 1)), atol=1e-9)
    assert np.isnan(np.concatenate([speeds[0], speeds[-1]])).all() == ends


def test_velocities_need_three_frames_and_a_plausible_speed():
    # Three frames 0.04 s apart: only the middle one's window holds three.
    indices = np.arange(3)
    walk = np.stack([10 + 2 * indices / 25, np.full(3, 34.0)], -1)
    few = tv.analysis.velocities(_moving(walk[:, None], indices), window=0.08)
    assert np.isnan(few[0]).all() and np.isnan(few[2]).all()
    np.testing.assert_allclose(few[1], [[2, 0]], atol=1e-9)
    # A 15 m/s "run" (a homography jump) is above the 12 m/s cap.
    dash = np.stack([10 + 15 * indices / 25, np.full(3, 34.0)], -1)
    result = _moving(np.stack([walk, dash], 1), indices)
    capped = tv.analysis.velocities(result)
    np.testing.assert_allclose(capped[1][0], [2, 0], atol=1e-9)
    assert np.isnan(capped[1][1]).all()
    np.testing.assert_allclose(
        tv.analysis.velocities(result, max_speed=20)[1][1], [15, 0], atol=1e-9
    )
    with pytest.raises(ValueError, match="max_speed must be > 0"):
        tv.analysis.velocities(result, max_speed=0)


def test_velocities_smooth_the_jitter_of_positions():
    # 2 m/s along x, with 0.3 m of noise on each position.
    indices = np.arange(100)
    noise = np.random.default_rng(0).normal(0, 0.3, (100, 2))
    xy = np.stack([10 + 2 * indices / 25, np.full(100, 34.0)], -1) + noise
    result = _moving(xy[:, None], indices)
    raw = np.diff(xy, axis=0) * 25  # frame-to-frame velocities
    assert np.abs(raw - [2, 0]).max() > 10

    def error(window: float) -> float:
        speeds = np.concatenate(tv.analysis.velocities(result, window=window))
        return np.abs(speeds[13:-13] - [2, 0]).max()  # whole windows only

    assert error(1.0) < error(0.5) < 3 and error(1.0) < 1.5
    with pytest.raises(ValueError, match="window must be a finite number > 0"):
        tv.analysis.velocities(result, window=0)


def test_velocities_follow_track_ids_not_rows():
    # Two players swap rows between frames; a third shows up once (one
    # position: no velocity), a fourth has no tracker id or no position.
    indices = np.arange(4)
    t = indices / 25
    a = np.stack([10 + 4 * t, np.full(4, 30.0)], -1)  # 4 m/s along x
    b = np.stack([np.full(4, 60.0), 20 - 2 * t], -1)  # -2 m/s along y
    xy = [
        [a[0], b[0], [50.0, 50.0]],
        [b[1], a[1], [np.nan, np.nan]],
        [a[2], b[2], [70.0, 10.0]],
        [b[3], a[3], [80.0, 10.0]],
    ]
    ids = [[7, 9, 3], [9, 7, 5], [7, 9, -1], [9, 7, 11]]
    speeds = tv.analysis.velocities(_moving(np.array(xy), indices, ids=ids))
    for frame_ids, velocity in zip(ids, speeds, strict=True):
        np.testing.assert_allclose(velocity[frame_ids.index(7)], [4, 0], atol=1e-9)
        np.testing.assert_allclose(velocity[frame_ids.index(9)], [0, -2], atol=1e-9)
        assert np.isnan(velocity[2]).all()  # one sample, no position, untracked


def test_velocities_refuse_a_tracker_id_twice_in_a_frame():
    result = _moving(np.zeros((2, 2, 2)), np.arange(2), ids=[[4, 5], [4, 4]])
    with pytest.raises(ValueError, match="Frame 1 has a tracker id twice"):
        tv.analysis.velocities(result)


def test_velocities_need_a_tracker():
    result = _moving(np.zeros((2, 1, 2)), np.arange(2))
    for frame in result.frames:
        frame.detections.tracker_id = None
    assert all(np.isnan(v).all() for v in tv.analysis.velocities(result))


# ------------------------------------------------------------- pitch control
def test_pitch_control_gives_the_same_grid_as_nearest_player():
    rng = np.random.default_rng(1)
    indices = np.arange(6)
    start = rng.uniform(0, [105, 68], (22, 2))
    run = rng.uniform(-6, 6, (22, 2))  # m/s
    xy = start + run * (indices / 25)[:, None, None]
    result = _moving(xy, indices, teams=[0] * 11 + [1] * 11)
    for cell_size in (1.0, 4.0):
        nearest = tv.analysis.space_control(result, cell_size=cell_size)
        pitch = tv.analysis.space_control(
            result, cell_size=cell_size, model="pitch_control"
        )
        assert type(pitch) is tv.analysis.SpaceControl and pitch.pitch == result.pitch
        assert pitch.control.shape == nearest.control.shape
        np.testing.assert_array_equal(pitch.x, nearest.x)
        np.testing.assert_array_equal(pitch.y, nearest.y)
        assert np.isfinite(pitch.control).all()
        assert pitch.control.min() >= 0 and pitch.control.max() <= 1
        assert pitch.control.min() < 0.1 and pitch.control.max() > 0.9
    np.testing.assert_array_equal(
        tv.analysis.space_control(result, model="nearest_player").control,
        tv.analysis.space_control(result).control,
    )


def test_pitch_control_the_closer_player_wins_a_contested_point():
    # Standing players at x = 40 and x = 70 on one line: the halfway point
    # x = 55 is shared, and a point at x = 50 is nearer team 0's player.
    result = _placed([[(0, 40.0, 34.0), (1, 70.0, 34.0)]])
    grid = tv.analysis.space_control(result, model="pitch_control")
    row = grid.control[np.searchsorted(grid.y, 34.0)]
    np.testing.assert_allclose(row[grid.x == 55.5], 1 - row[grid.x == 54.5])
    assert row[grid.x == 50.5] > 0.6 and row[grid.x == 59.5] < 0.4
    assert (row[grid.x < 55] > 0.5).all() and (row[grid.x > 55] < 0.5).all()


def test_pitch_control_the_faster_player_wins_a_contested_point():
    # Both players are 20 m from x = 60; team 0's runs towards it at 5 m/s,
    # team 1's stands. With a standing team 0, the point is shared.
    indices = np.arange(25)
    t = indices / 25
    running = np.stack([40 - 5 * (t[-1] - t), np.full(25, 34.0)], -1)
    standing = np.tile([80.0, 34.0], (25, 1))
    result = _moving(np.stack([running, standing], 1), indices, teams=[0, 1])
    grid = tv.analysis.space_control(result, frames=-1, model="pitch_control")
    column, row = np.searchsorted(grid.x, 60.0), np.searchsorted(grid.y, 34.0)
    assert grid.x[column] == 60.5 and grid.control[row, column] > 0.7
    still = _moving(
        np.stack([np.tile([40.0, 34.0], (25, 1)), standing], 1), indices, [0, 1]
    )
    shared = tv.analysis.space_control(still, frames=-1, model="pitch_control")
    shared = shared.control[row]
    np.testing.assert_allclose(shared[column - 1], 1 - shared[column])
    # A faster top speed lets the same lead count for less time.
    quick = tv.analysis.space_control(
        result, frames=-1, model="pitch_control", max_speed=8.0
    )
    assert 0.5 < quick.control[row, column] < grid.control[row, column]


def test_space_control_model_options():
    result = _placed([[(0, 30.0, 34.0), (1, 75.0, 34.0)]])
    with pytest.raises(
        ValueError, match="Unknown model 'voronoi'.*nearest_player, pitch_control"
    ):
        tv.analysis.space_control(result, model="voronoi")
    np.testing.assert_array_equal(
        tv.analysis.space_control(result, model=SpaceModel.PITCH_CONTROL).control,
        tv.analysis.space_control(result, model="pitch_control").control,
    )
    with pytest.raises(
        ValueError, match="'pitch_control' takes reaction_time, max_speed.*smoothing"
    ):
        tv.analysis.space_control(result, model="pitch_control", smoothing=1)
    with pytest.raises(
        ValueError, match="'nearest_player' takes smoothing; got max_speed"
    ):
        tv.analysis.space_control(result, max_speed=5)
    bounds = [(option, ">= 0", -1) for option in ("reaction_time", "intercept_sigma")]
    bounds += [
        (option, "> 0", 0)
        for option in ("max_speed", "control_rate", "velocity_window")
    ]
    for option, bound, low in bounds:
        for value in (low, np.inf, np.nan):
            with pytest.raises(
                ValueError, match=f"{option} must be a finite number {bound}"
            ):
                tv.analysis.space_control(
                    result, model="pitch_control", **{option: value}
                )


def test_pitch_control_warns_when_cells_are_still_free_after_20_s(caplog):
    result = _placed([[(0, 30.0, 34.0), (1, 75.0, 34.0)]] * 3)
    slow = tv.analysis.space_control(result, model="pitch_control", control_rate=0.01)
    assert caplog.text.count("stopped after 20 s") == 1  # once a call
    assert "control_rate=0.01" in caplog.text
    assert slow.control[34, 0] > 0.5 and slow.control[34, -1] < 0.5
    caplog.clear()
    tv.analysis.space_control(result, model="pitch_control")
    assert not caplog.text


def test_pitch_control_reads_velocities_near_the_selected_frames_only():
    # Team 0's player runs at 5 m/s towards x = 60 in frames 20 to 24; long
    # before, the tracks are elsewhere. Frame 24 alone gives the same grid as
    # the whole run's velocities would.
    indices = np.arange(25)
    t = indices / 25
    running = np.stack([40 - 5 * (t[-1] - t), np.full(25, 34.0)], -1)
    running[:10] = [5.0, 5.0]
    standing = np.tile([80.0, 34.0], (25, 1))
    result = _moving(np.stack([running, standing], 1), indices, teams=[0, 1])
    grid = tv.analysis.space_control(result, frames=24, model="pitch_control")
    velocity = tv.analysis.velocities(result)[24]
    np.testing.assert_allclose(velocity, [[5, 0], [0, 0]], atol=1e-9)
    late = _moving(np.stack([running, standing], 1)[15:], indices[15:], [0, 1])
    np.testing.assert_allclose(
        grid.control,
        tv.analysis.space_control(late, frames=-1, model="pitch_control").control,
    )


def test_space_control_documents_each_model_option_and_default():
    doc = tv.analysis.space_control.__doc__
    for option, default in [
        ("smoothing", "1.2"),
        ("reaction_time", "0.7"),
        ("max_speed", "5.0"),
        ("intercept_sigma", "0.45"),
        ("control_rate", "4.3"),
        ("velocity_window", "0.5"),
    ]:
        assert f"``{option}`` ({default}" in doc


# ------------------------------------------------- carrier, possession, passes
# People of a scripted play: tracker id -> (class name, team id, foot x in
# pixels). Every box is 20 x 50 px with its foot (bottom centre) at y = 200.
PLAY = {
    1: ("player", 0, 100.0),  # A
    2: ("player", 0, 300.0),  # B
    3: ("goalkeeper", 1, 500.0),  # C
    9: ("referee", NO_TEAM, 104.0),  # nearer the ball than A, never the carrier
}
NEAR = {"A": (110.0, 195.0), "B": (310.0, 195.0), "C": (510.0, 195.0)}
LOOSE = (700.0, 50.0)  # visible, nobody close


def _play(
    script: str,
    people: dict[int, tuple[str, int, float]] = PLAY,
    stride: int = 1,
    tracked: bool = True,
) -> PipelineResult:
    """One frame per letter: the ball near ``A``, ``B`` or ``C``, loose (``.``)
    or missing (``-``); lower case is a ball filled in by ``track_ball``.

    No pitch positions or homography: the carrier rule works in the image.
    """
    frames = []
    for position, letter in enumerate(script):
        index = position * stride
        ids = np.array(list(people))
        feet = np.array([x for _, _, x in people.values()])
        detections = sv.Detections(
            xyxy=np.stack([feet - 10, np.full(len(feet), 150), feet + 10,
                           np.full(len(feet), 200)], axis=1).astype(np.float32),
            tracker_id=ids if tracked else None,
            data={
                "class_name": np.array([name for name, _, _ in people.values()]),
                "team_id": np.array([team for _, team, _ in people.values()]),
                "pitch_xy": np.full((len(ids), 2), np.nan),
            },
        )  # fmt: skip
        ball = sv.Detections.empty()
        if letter != "-":
            x, y = NEAR.get(letter.upper(), LOOSE)
            ball = sv.Detections(
                xyxy=np.float32([[x - 5, y - 5, x + 5, y + 5]]),
                data={
                    "class_name": np.array(["ball"]),
                    "interpolated": np.array([letter.islower()]),
                },
            )
        frames.append(
            FrameResult(index, index / 25, detections, ball, ball_candidates=ball)
        )
    return PipelineResult(frames, 25.0, (800, 400), ["ball", "player"])


def _letters(carriers: list[int | None]) -> str:
    return "".join({None: ".", 1: "A", 2: "B", 3: "C"}[c] for c in carriers)


#        frame  0         1         2
#               0123456789012345678901
SCRIPT = "AAAA--BBB...CCCAAAbbbB"
CARRIERS = "..AAAAAABBB...CCCAAABB"


def test_a_scripted_play_gives_the_exact_carrier_sequence():
    # A new carrier, or none, is confirmed once seen for 0.12 s: the third
    # frame in a row at 25 fps; a missing ball holds the carrier, a filled-in
    # one counts like any other.
    carriers = tv.analysis.ball_carriers(_play(SCRIPT))
    assert _letters(carriers) == CARRIERS
    assert all(c is None or type(c) is int for c in carriers)


def test_a_scripted_play_gives_the_exact_passes_and_possession():
    result = _play(SCRIPT)
    carriers = tv.analysis.ball_carriers(result)
    passes = tv.analysis.passes(result, carriers)
    # A to B, then (C of the other team in between) A to B again; frames are
    # video indices: the last frame of the passer and the first of the receiver.
    assert passes.to_dict("records") == [
        {"team_id": 0, "from_track_id": 1, "to_track_id": 2, "from_frame": 7, "to_frame": 8},
        {"team_id": 0, "from_track_id": 1, "to_track_id": 2, "from_frame": 19, "to_frame": 20},
    ]  # fmt: skip
    assert tv.analysis.pass_counts(result, carriers) == {0: 2, 1: 0}
    # 17 frames have a carrier: 14 of team 0, 3 of team 1.
    shares = tv.analysis.possession(result, carriers)
    assert shares == pytest.approx({0: 14 / 17, 1: 3 / 17})
    assert tv.analysis.possession(result, carriers, slice(11, 17)) == {0: 0.0, 1: 1.0}
    assert tv.analysis.possession(result, carriers, 7) == {0: 1.0, 1: 0.0}


def test_flicker_shorter_than_the_hysteresis_creates_no_pass():
    # B is nearest for two frames, the ball looks loose for two: A keeps it.
    result = _play("AAAABBAA..AABAB-BA")
    carriers = tv.analysis.ball_carriers(result)
    assert _letters(carriers) == "..AAAAAAAAAAAAAAAA"
    assert tv.analysis.passes(result, carriers).empty
    assert tv.analysis.pass_counts(result, carriers) == {0: 0, 1: 0}
    # A loose frame between B's frames starts B's count again.
    assert _letters(tv.analysis.ball_carriers(_play("AAAB.B.B"))) == "..AAAAAA"


def test_the_hysteresis_and_distance_are_options():
    result = _play("AABBA..")
    carriers = tv.analysis.ball_carriers(result, hysteresis=0)  # one frame
    assert _letters(carriers) == "AABBA.."
    assert len(tv.analysis.passes(result, carriers)) == 2  # A to B, B to A
    # The ball is 0.22 box heights from the nearest foot.
    assert set(tv.analysis.ball_carriers(result, max_distance=0.2)) == {None}
    assert _letters(tv.analysis.ball_carriers(_play("AAA"), max_distance=0.3)) == "..A"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_distance": 0}, "max_distance"),
        ({"max_distance": np.inf}, "max_distance"),
        ({"hysteresis": -0.1}, "hysteresis"),
        ({"hysteresis": np.inf}, "hysteresis"),
        ({"hysteresis": np.nan}, "hysteresis"),
        ({"max_gap": 0}, "max_gap"),
        ({"max_gap": np.nan}, "max_gap"),
    ],
)
def test_ball_carriers_check_their_options(options, message):
    with pytest.raises(ValueError, match=message):
        tv.analysis.ball_carriers(_play("AAA"), **options)


def test_the_carrier_is_the_nearest_close_player_relative_to_box_height():
    # A tall box (100 px) 40 px from the ball is nearer, in box heights, than
    # a 50 px one 30 px away; a person without a team, nearer still, is not.
    tall = {
        1: ("player", 0, 100.0),
        2: ("player", 1, 170.0),
        7: ("player", NO_TEAM, 130.0),
    }
    result = _play("AAA", people=tall)
    for frame in result:
        frame.detections.xyxy[0, 1] = 100  # A is 100 px tall
        frame.ball.xyxy[:] = [[135, 195, 145, 205]]  # 40 px from A, 30 from B
    assert tv.analysis.ball_carriers(result) == [None, None, 1]


def test_glossary_definitions():
    # Ball carrier: the nearest player close enough to the ball (above).
    # Possession: a team's share of the frames with one of its players as carrier.
    # Pass: a change of carrier between two players of the same team, so not
    # a change to the other team, and not a carrier keeping the ball after a
    # loose spell, while a teammate taking a loose ball is one.
    result = _play("AAA...AAA...BBBCCC")
    carriers = tv.analysis.ball_carriers(result)
    assert _letters(carriers) == "..AAA...AAA...BBBC"
    passes = tv.analysis.passes(result, carriers)
    assert passes[["from_track_id", "to_track_id"]].values.tolist() == [[1, 2]]
    assert tv.analysis.possession(result, carriers) == pytest.approx({0: 0.9, 1: 0.1})


def test_carriers_need_tracker_ids():
    result = _play("AAA", tracked=False)
    assert tv.analysis.ball_carriers(result) == [None, None, None]


def test_possession_and_passes_check_the_carriers():
    result = _play("AAA")
    with pytest.raises(ValueError, match="one carrier per frame"):
        tv.analysis.possession(result, [1, 1])
    with pytest.raises(ValueError, match="one carrier per frame"):
        tv.analysis.passes(result, [1])
    with pytest.raises(ValueError, match="no ball carrier"):
        tv.analysis.possession(result, [None, None, None])
    with pytest.raises(ValueError, match="no ball carrier"):
        tv.analysis.possession(result, [None, None, 1], slice(0, 2))


def test_the_carrier_table_gives_track_and_team_per_frame():
    result = _play("AAABBB.CCC", stride=2)
    table = tv.analysis.carrier_table(result, tv.analysis.ball_carriers(result))
    assert table.columns.tolist() == ["frame", "time", "track_id", "team_id"]
    assert table["frame"].tolist() == list(range(0, 20, 2))
    assert table["time"].iloc[1] == pytest.approx(0.08)
    # At 0.08 s a frame, two frames in a row confirm a carrier.
    assert table["track_id"].tolist() == [pd.NA, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    assert table["team_id"].tolist() == [pd.NA, 0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_export_writes_the_carrier_and_pass_tables(tmp_path):
    result = _play("AAABBB.CCC", stride=2)
    out = result.export(tmp_path / "run", team_names=["Home", "Away"])
    carriers = pd.read_csv(out / "carriers.csv", keep_default_na=False)
    assert carriers.columns.tolist() == [
        "frame", "time", "track_id", "team_id", "team_name"
    ]  # fmt: skip
    assert carriers["track_id"].astype(str).tolist() == [
        "", "1", "1", "1", "2", "2", "2", "2", "3", "3"
    ]  # fmt: skip
    assert carriers["team_name"].tolist()[0:2] + carriers["team_name"].tolist()[-1:] == [
        "", "Home", "Away"
    ]  # fmt: skip
    assert pd.read_csv(out / "passes.csv").to_dict("records") == [{
        "team_id": 0, "team_name": "Home", "from_track_id": 1, "to_track_id": 2,
        "from_frame": 6, "to_frame": 8,
    }]  # fmt: skip
    # Without names the column is empty; a result without a carrier still
    # writes both tables with their headers.
    unnamed = _play("...", stride=2).export(tmp_path / "unnamed")
    assert pd.read_csv(unnamed / "carriers.csv")["team_name"].isna().all()
    assert pd.read_csv(unnamed / "passes.csv").columns.tolist() == [
        "team_id", "team_name", "from_track_id", "to_track_id", "from_frame", "to_frame"
    ]  # fmt: skip


def test_a_person_without_a_box_never_takes_the_ball():
    result = _play("AAAA")
    for frame in result:
        frame.detections.xyxy[1] = np.nan  # B
    assert _letters(tv.analysis.ball_carriers(result)) == "..AA"
    loose = _play("....")
    for frame in loose:
        frame.detections.xyxy[1] = np.nan
    assert set(tv.analysis.ball_carriers(loose)) == {None}


@pytest.mark.parametrize("carrier", [42, 9])  # not in the result; the referee
def test_carriers_must_be_people_who_can_carry_the_ball(carrier):
    with pytest.raises(ValueError, match=f"{carrier}.*ball_carriers"):
        tv.analysis.passes(_play("AAA"), [None, 1, carrier])


@pytest.mark.parametrize(
    ("stride", "carriers"),
    [(1, "..AAAAAA"), (2, ".AAABB.."), (10, "AAABB..A")],
)
def test_the_hysteresis_is_in_seconds_at_any_stride(stride, carriers):
    # 0.12 s is three frames at 25 fps, two at 12.5 and one at 2.5.
    assert (
        _letters(tv.analysis.ball_carriers(_play("AAABB..A", stride=stride)))
        == carriers
    )


def test_the_carrier_is_held_while_the_ball_is_unseen_for_up_to_max_gap():
    # The ball is last seen at 0.08 s; 1 s later the carrier is gone.
    result = _play("AAA" + "-" * 30)
    assert _letters(tv.analysis.ball_carriers(result)) == "..A" + "A" * 24 + "." * 6
    assert _letters(tv.analysis.ball_carriers(result, max_gap=np.inf))[-1] == "A"
    # The gap also ends a challenge: B, seen twice before it, needs three
    # frames again after it.
    result = _play("AAABB" + "-" * 30 + "BB")
    assert _letters(tv.analysis.ball_carriers(result))[-3:] == "..."
    assert _letters(tv.analysis.ball_carriers(result, max_gap=2.0))[-3:] == "ABB"


def test_export_checks_the_carrier_options_before_writing(tmp_path):
    with pytest.raises(ValueError, match="hysteresis"):
        _play("AAA").export(tmp_path / "run", carrier_hysteresis=-1.0)
    assert not (tmp_path / "run").exists()
    out = _play("AAA").export(tmp_path / "set", carrier_hysteresis=0.0)
    assert pd.read_csv(out / "carriers.csv")["track_id"].tolist() == [1, 1, 1]
