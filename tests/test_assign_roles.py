"""The ``assign_roles`` post-process: goalkeepers, referees and off-pitch people."""

import copy
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.pipeline import (
    NO_TEAM,
    POSTPROCESSES,
    AppliedPostProcess,
    FrameResult,
    ObjectMasks,
    PipelineResult,
    assign_roles,
    join_tracks,
    reproject,
)
from tactifoot_vision.pitch import SoccerPitch

FPS = 25.0
FRAMES = 20
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

Path = Callable[[int], tuple[float, float]]  # frame -> metres on 105 x 68
KITS = {  # embedding centre of each kit
    "red": [1.0, 0.0, 0.0, 0.0],
    "blue": [0.0, 1.0, 0.0, 0.0],
    "yellow": [0.0, 0.0, 1.0, 0.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "green": [0.0, 0.0, 0.7, 0.7],
}


def _at(x: float, y: float) -> Path:
    return lambda index: (x + 0.1 * np.sin(index), y)


def _homography(pitch: SoccerPitch) -> np.ndarray:
    """frame -> pitch: the pitch fills 1000 x 600 pixels."""
    return np.diag([pitch.length / 1000, pitch.width / 600, 1.0])


def _feet_box(metres: tuple[float, float]) -> list[float]:
    """A person box whose bottom centre stands on ``metres``."""
    x, y = np.array(metres) * [1000 / 105, 600 / 68]
    return [x - 5, y - 20, x + 5, y]


# (tracker_id, class_name, team_id, kit or None, where)
Track = tuple[int, str, int, str | None, Path]

# Red (team 0) stands nearer x = 0 than blue (team 1), so red defends that goal.
PLAYERS: list[Track] = [
    *[(1 + i, "player", 0, "red", _at(25 + 4 * i, 10 + 12 * i)) for i in range(5)],
    *[(11 + i, "player", 1, "blue", _at(60 + 4 * i, 8 + 12 * i)) for i in range(5)],
]
# Kit outliers the vote put into a team: one in each penalty area, one in midfield.
KEEPER_LEFT: Track = (21, "player", 1, "yellow", _at(3, 34))
KEEPER_RIGHT: Track = (22, "player", 0, "green", _at(101, 30))
REFEREE: Track = (23, "player", 0, "black", _at(50, 40))
SCENE = [*PLAYERS, KEEPER_LEFT, KEEPER_RIGHT, REFEREE]


def _result(
    tracks: list[Track],
    pitch: SoccerPitch | None = None,
    scale: float = 1.0,
    spread: float = 0.02,
    masks: bool = False,
) -> PipelineResult:
    """The tracks on every frame; kits are embeddings ``scale`` apart with ``spread`` noise."""
    pitch = pitch or SoccerPitch()
    rng = np.random.default_rng(0)
    frames = []
    for index in range(FRAMES):
        rows = [
            (tracker_id, name, team, _feet_box(path(index)))
            for tracker_id, name, team, _, path in tracks
        ]
        people = sv.Detections(
            xyxy=np.array([box for *_, box in rows], np.float32).reshape(-1, 4),
            confidence=np.full(len(rows), 0.9, np.float32),
            class_id=np.array([CLASS_NAMES.index(n) for _, n, *_ in rows], int),
            tracker_id=np.array([tracker_id for tracker_id, *_ in rows], int),
            data={
                "class_name": np.array([name for _, name, *_ in rows]),
                "team_id": np.array([team for _, _, team, _ in rows], int),
            },
        )
        object_masks = None
        if masks:
            object_masks = ObjectMasks(
                origins=np.array([[i, 0] for i in range(len(rows))]).reshape(-1, 2),
                crops=[np.full((1, 1), True) for _ in rows],
            )
        frames.append(
            FrameResult(
                index,
                index / FPS,
                people,
                sv.Detections.empty(),
                ball_candidates=sv.Detections.empty(),
                masks=object_masks,
            )
        )
    embeddings = {
        tracker_id: scale * (np.array(KITS[kit]) + rng.normal(0, spread, 4))
        for tracker_id, _, _, kit, _ in tracks
        if kit is not None
    }
    result = PipelineResult(
        frames,
        FPS,
        (1000, 600),
        CLASS_NAMES,
        pitch=pitch,
        track_embeddings=embeddings,
    )
    return reproject(result, [_homography(pitch)] * len(frames))


def _roles(result: PipelineResult) -> dict[int, set[tuple[str, int]]]:
    """``(class_name, team_id)`` of every tracked row, by track."""
    roles: dict[int, set[tuple[str, int]]] = {}
    for frame in result:
        people = frame.detections
        if people.tracker_id is None:
            continue
        for track, name, team in zip(
            people.tracker_id, people.data["class_name"], frame.team_ids, strict=True
        ):
            roles.setdefault(int(track), set()).add((str(name), int(team)))
    return roles


@pytest.mark.parametrize(
    "pitch", [SoccerPitch(), SoccerPitch(120, 80)], ids=["105x68", "120x80"]
)
def test_kit_outliers_become_goalkeepers_of_the_defending_team_and_a_referee(pitch):
    roles = _roles(assign_roles(_result(SCENE, pitch)))

    assert roles[21] == {("goalkeeper", 0)}  # red defends the goal at x = 0
    assert roles[22] == {("goalkeeper", 1)}
    assert roles[23] == {("referee", NO_TEAM)}
    for tracker_id, _, team, *_ in PLAYERS:
        assert roles[tracker_id] == {("player", team)}


def test_the_sides_follow_where_the_teams_stand():
    # blue now stands nearer x = 0: the keepers swap teams, not places
    mirrored = [
        (tracker_id, name, 1 - team, kit, path)
        for tracker_id, name, team, kit, path in PLAYERS
    ]
    roles = _roles(assign_roles(_result([*mirrored, KEEPER_LEFT, KEEPER_RIGHT])))

    assert roles[21] == {("goalkeeper", 1)}
    assert roles[22] == {("goalkeeper", 0)}


@pytest.mark.parametrize(
    ("scale", "spread"), [(1.0, 0.02), (300.0, 0.02), (0.01, 0.05)]
)
def test_the_outlier_rule_needs_no_threshold_per_match(scale, spread):
    # kits as far apart as 300 or 0.01 embedding units, teams tight or loose:
    # the default outlier rule finds the same roles without any tuning
    roles = _roles(assign_roles(_result(SCENE, scale=scale, spread=spread)))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[22] == {("goalkeeper", 1)}
    assert roles[23] == {("referee", NO_TEAM)}
    assert all(roles[t] == {("player", team)} for t, _, team, *_ in PLAYERS)


def test_the_outlier_option_is_relative_to_the_team_spread():
    # with a limit beyond the kits' distance nobody is an outlier
    roles = _roles(assign_roles(_result(SCENE), outlier_spread=1000.0))

    assert roles[21] == {("player", 1)}
    assert roles[23] == {("player", 0)}


def test_only_the_outlier_nearest_each_goal_keeps_it():
    second = (24, "player", 1, "yellow", _at(10, 30))  # also in the box, further out
    roles = _roles(assign_roles(_result([*SCENE, second])))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[24] == {("referee", NO_TEAM)}


def test_an_outlier_mostly_outside_the_penalty_area_is_no_goalkeeper():
    # in the box for a quarter of the frames, at the edge of the area the rest
    def visits(index: int) -> tuple[float, float]:
        return (3.0, 34.0) if index % 4 == 0 else (20.0, 34.0)

    roles = _roles(
        assign_roles(_result([*PLAYERS, (21, "player", 1, "yellow", visits)]))
    )

    assert roles[21] == {("referee", NO_TEAM)}


@pytest.mark.parametrize(
    "pitch", [SoccerPitch(), SoccerPitch(120, 80)], ids=["105x68", "120x80"]
)
def test_people_off_the_pitch_are_left_out(pitch):
    steward = (31, "player", 0, "red", _at(50, -6))  # red kit, behind the boards
    cameraman = (32, "player", 1, "black", _at(110, 30))  # behind the goal
    linesman = (33, "player", 1, "black", _at(70, -0.8))  # on the touchline
    detector_referee = (34, "referee", NO_TEAM, None, _at(52, 75))
    result = _result([*SCENE, steward, cameraman, linesman, detector_referee], pitch)

    roles = _roles(assign_roles(result))

    assert {31, 32, 34}.isdisjoint(roles)
    assert roles[33] == {("referee", NO_TEAM)}
    assert roles[21] == {("goalkeeper", 0)}


def test_leaving_people_out_keeps_their_masks_in_step():
    steward = (31, "player", 0, "red", _at(50, -6))
    result = _result([*SCENE, steward], masks=True)

    roles_result = assign_roles(result)

    for before, after in zip(result, roles_result, strict=True):
        assert after.masks is not None and len(after.masks) == len(after.detections)
        kept = before.detections.tracker_id != 31
        np.testing.assert_array_equal(after.masks.origins, before.masks.origins[kept])


def test_detector_roles_are_kept():
    # judged by kit and place, the referee would keep the goal at x = 0 and the
    # goalkeeper would be a referee; the detector's classes and teams stay
    detector_referee = (41, "referee", NO_TEAM, "black", _at(1, 34))
    detector_keeper = (42, "goalkeeper", 1, "black", _at(50, 20))
    result = _result([*SCENE, detector_referee, detector_keeper])

    roles = _roles(assign_roles(result))

    assert roles[41] == {("referee", NO_TEAM)}
    assert roles[42] == {("goalkeeper", 1)}
    assert roles[21] == {("goalkeeper", 0)}


def test_a_goal_with_a_detector_goalkeeper_gets_no_second_one():
    detector_keeper = (41, "goalkeeper", 0, "yellow", _at(2, 30))
    roles = _roles(assign_roles(_result([*SCENE, detector_keeper])))

    assert roles[41] == {("goalkeeper", 0)}
    assert roles[21] == {("referee", NO_TEAM)}
    assert roles[22] == {("goalkeeper", 1)}  # the other goal still gets one


def test_the_class_id_follows_the_new_class_name():
    result = _result(SCENE)

    relabelled = assign_roles(result)

    for frame, old in zip(relabelled, result, strict=True):
        names = frame.detections.data["class_name"]
        expected = [CLASS_NAMES.index(n) for n in names]
        np.testing.assert_array_equal(frame.detections.class_id, expected)
        assert (old.detections.class_id == CLASS_NAMES.index("player")).all()


def test_frames_without_a_tracker_are_left_alone():
    result = _result(SCENE)
    untracked = result.frames[3]
    untracked.detections.tracker_id = None

    relabelled = assign_roles(result)

    assert relabelled.frames[3] is untracked
    assert _roles(relabelled)[21] == {("goalkeeper", 0)}


def test_a_team_of_too_few_player_tracks_is_warned_about(caplog):
    thin_blue = [track for track in SCENE if track[0] not in (12, 13, 14, 15)]

    assign_roles(_result(thin_blue))

    assert "Team 1 has 2 player tracks" in caplog.text


def test_a_player_track_joined_from_tracks_uses_their_embeddings():
    # the yellow keeper is lost after frame 9 and comes back as track 25;
    # join_tracks gives it id 21 and the embeddings stay by the tracker's ids
    keeper_end = (21, "player", 1, "yellow", _at(3, 34))
    keeper_start = (25, "player", 1, "yellow", _at(3, 34))
    result = _result([*PLAYERS, keeper_end, keeper_start])
    for frame in result:
        keep = ((frame.detections.tracker_id != 21) | (frame.index < 10)) & (
            (frame.detections.tracker_id != 25) | (frame.index >= 12)
        )
        frame.detections = frame.detections[keep]
    joined = join_tracks(result)

    roles = _roles(assign_roles(joined))

    assert 25 not in roles
    assert roles[21] == {("goalkeeper", 0)}


def test_tracks_without_an_embedding_or_a_position_keep_their_role():
    unembedded = (51, "player", 1, None, _at(3, 30))
    result = _result([*SCENE, unembedded])
    for frame in result:  # the referee has no homography-given position
        frame.detections.data["pitch_xy"][frame.detections.tracker_id == 23] = np.nan

    roles = _roles(assign_roles(result))

    assert roles[51] == {("player", 1)}
    assert roles[23] == {("referee", NO_TEAM)}  # still a kit outlier
    assert roles[21] == {("goalkeeper", 0)}


def test_the_input_is_left_alone_and_the_call_is_recorded():
    result = _result(SCENE)
    before = copy.deepcopy(result)

    by_name = POSTPROCESSES.get("assign_roles")(result)

    assert _roles(result) == _roles(before)
    for frame, old in zip(result, before, strict=True):
        np.testing.assert_array_equal(frame.team_ids, old.team_ids)
    assert by_name.postprocesses == [
        AppliedPostProcess(
            "assign_roles",
            {"outlier_spread": 3.0, "min_box_share": 0.5, "off_pitch_margin": 1.5},
        )
    ]
    assert by_name.track_embeddings == result.track_embeddings


def test_a_result_without_team_embeddings_is_refused():
    result = _result(SCENE)
    result.track_embeddings = {}
    with pytest.raises(ValueError, match="team embeddings"):
        assign_roles(result)

    del result.track_embeddings  # as a result saved before they were kept
    with pytest.raises(ValueError, match="team embeddings"):
        assign_roles(result)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"outlier_spread": 0}, "outlier_spread"),
        ({"min_box_share": 1.5}, "min_box_share"),
        ({"off_pitch_margin": -1}, "off_pitch_margin"),
    ],
)
def test_options_are_checked(options, match):
    with pytest.raises(ValueError, match=match):
        assign_roles(_result(SCENE), **options)


def test_the_analysis_package_imports_first():
    # assign_roles uses tv.analysis, which imports the pipeline
    code = "import tactifoot_vision.analysis.direction"
    subprocess.run([sys.executable, "-c", code], check=True)
