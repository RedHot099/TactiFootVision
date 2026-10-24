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
    "grey": [0.7, 0.0, 0.0, 0.3],  # nearer red than black
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
# Ten tracks a team, as a match shows: kits are told apart by the teams' spreads.
PLAYERS: list[Track] = [
    *[(1 + i, "player", 0, "red", _at(20 + 3 * i, 4 + 6.5 * i)) for i in range(10)],
    *[(11 + i, "player", 1, "blue", _at(58 + 3 * i, 2 + 6.5 * i)) for i in range(10)],
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
    samples: dict[int, int] | None = None,
) -> PipelineResult:
    """The tracks on every frame; kits are embeddings ``scale`` apart with ``spread`` noise.

    Each embedding is a mean over ``samples[tracker_id]`` crops (20 when missing).
    """
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
        tracker_id: (scale * (np.array(KITS[kit]) + rng.normal(0, spread, 4))).astype(
            np.float16
        )
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
        track_embedding_samples={t: (samples or {}).get(t, 20) for t in embeddings},
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


def test_officials_the_vote_put_into_a_team_are_found():
    # a few blue players in view and every official voted blue: a centre and
    # spread of the voted team would be the officials' as much as blue's
    few_blue = PLAYERS[10:16]
    linesmen = [
        (31 + i, "player", 1, "black", _at(20 + 25 * i, -0.5)) for i in range(4)
    ]
    officials = [
        (21, "player", 1, "yellow", _at(3, 34)),
        (22, "player", 1, "green", _at(101, 30)),
        (23, "player", 1, "black", _at(50, 40)),
        *linesmen,
    ]

    roles = _roles(assign_roles(_result([*PLAYERS[:10], *few_blue, *officials])))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[22] == {("goalkeeper", 1)}
    for tracker_id, *_ in [officials[2], *linesmen]:
        assert roles[tracker_id] == {("referee", NO_TEAM)}
    for tracker_id, _, team, *_ in [*PLAYERS[:10], *few_blue]:
        assert roles[tracker_id] == {("player", team)}


@pytest.mark.parametrize("n_kits", [5, 6, 7])
def test_a_team_split_into_kits_is_put_back_together(n_kits):
    # only two kits on the pitch: the spare kits split the teams, not officials
    roles = _roles(assign_roles(_result(PLAYERS, spread=0.1), n_kits=n_kits))

    assert all(roles[t] == {("player", team)} for t, _, team, *_ in PLAYERS)


def test_the_outlier_option_is_relative_to_the_team_spread():
    # with a limit beyond the kits' distance nobody is an outlier
    roles = _roles(assign_roles(_result(SCENE), outlier_spread=1000.0))

    assert roles[21] == {("player", 1)}
    assert roles[23] == {("player", 0)}


def test_every_track_of_the_keepers_kit_in_the_area_is_the_goalkeeper():
    # the keeper tracked in two pieces the joining left apart
    second = (24, "player", 1, "yellow", _at(10, 30))
    roles = _roles(assign_roles(_result([*SCENE, second])))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[24] == {("goalkeeper", 0)}


def test_an_outlier_of_another_kit_in_the_area_stays_a_referee():
    # the referee in the box three frames in four: fewer rows there than the keeper
    def visits(index: int) -> tuple[float, float]:
        return (8.0, 40.0) if index % 4 else (30.0, 40.0)

    referee = (24, "player", 0, "black", visits)
    roles = _roles(assign_roles(_result([*SCENE, referee])))

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


def test_a_detector_goalkeeper_gives_its_goal_its_kit():
    # the keeper's other pieces join it; the referee in the box stays a referee
    detector_keeper = (41, "goalkeeper", 0, "yellow", _at(2, 30))
    referee = (24, "player", 0, "black", _at(8, 40))
    roles = _roles(assign_roles(_result([*SCENE, detector_keeper, referee])))

    assert roles[41] == {("goalkeeper", 0)}
    assert roles[21] == {("goalkeeper", 0)}
    assert roles[24] == {("referee", NO_TEAM)}
    assert roles[22] == {("goalkeeper", 1)}  # the other goal still gets one


def test_a_goal_whose_detector_goalkeeper_has_no_outlier_kit_gets_none():
    detector_keeper = (41, "goalkeeper", 0, "red", _at(2, 30))
    referee = (24, "player", 0, "black", _at(8, 40))
    roles = _roles(assign_roles(_result([*PLAYERS, detector_keeper, referee])))

    assert roles[41] == {("goalkeeper", 0)}
    assert roles[24] == {("referee", NO_TEAM)}


def test_the_class_id_follows_the_new_class_name():
    result = _result(SCENE)

    relabelled = assign_roles(result)

    for frame, old in zip(relabelled, result, strict=True):
        names = frame.detections.data["class_name"]
        expected = [CLASS_NAMES.index(n) for n in names]
        np.testing.assert_array_equal(frame.detections.class_id, expected)
        assert (old.detections.class_id == CLASS_NAMES.index("player")).all()


def test_a_class_the_result_does_not_name_keeps_the_players_class_id(caplog):
    result = _result(SCENE)
    result.class_names = ["ball", "goalkeeper", "player"]

    with caplog.at_level("INFO"):
        relabelled = assign_roles(result)

    for frame in relabelled:
        is_referee = frame.detections.data["class_name"] == "referee"
        assert is_referee.any()
        assert (frame.detections.class_id[is_referee] == 2).all()
    assert "class names have no 'referee'" in caplog.text


def test_frames_without_a_tracker_are_left_alone():
    result = _result(SCENE)
    untracked = result.frames[3]
    untracked.detections.tracker_id = None

    relabelled = assign_roles(result)

    assert relabelled.frames[3] is untracked
    assert _roles(relabelled)[21] == {("goalkeeper", 0)}


def test_a_team_of_too_few_player_tracks_is_warned_about(caplog):
    thin_blue = [track for track in SCENE if track[0] not in range(12, 21)]

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


def test_a_joined_track_weighs_its_tracks_by_their_samples():
    # 99 red crops and one yellow: the joined track is a red player, where
    # an unweighted mean would put it halfway to yellow, an outlier
    red_part = (21, "player", 0, "red", _at(3, 34))
    yellow_part = (25, "player", 0, "yellow", _at(3, 34))
    result = _result([*PLAYERS, red_part, yellow_part], samples={21: 99, 25: 1})
    for frame in result:
        keep = ((frame.detections.tracker_id != 21) | (frame.index < 10)) & (
            (frame.detections.tracker_id != 25) | (frame.index >= 12)
        )
        frame.detections = frame.detections[keep]

    roles = _roles(assign_roles(join_tracks(result)))

    assert roles[21] == {("player", 0)}


def test_one_track_keeps_one_goal_at_most():
    # in each penalty area half the time: the goal at x = 0 takes it first
    def both_boxes(index: int) -> tuple[float, float]:
        return (3.0, 34.0) if index % 2 else (101.0, 34.0)

    keeper = (21, "player", 1, "yellow", both_boxes)
    roles = _roles(assign_roles(_result([*PLAYERS, keeper]), min_box_share=0.4))

    assert roles[21] == {("goalkeeper", 0)}


def test_undecided_sides_leave_the_goalkeeper_candidates_players(caplog):
    # both teams stand around the halfway line: no side to give a goalkeeper
    level = [
        (tracker_id, name, team, kit, _at(52 + 0.1 * team, 6 + 6 * (tracker_id % 10)))
        for tracker_id, name, team, kit, _ in PLAYERS
    ]
    steward = (31, "player", 0, "red", _at(50, -6))

    roles = _roles(
        assign_roles(_result([*level, KEEPER_LEFT, KEEPER_RIGHT, REFEREE, steward]))
    )

    assert roles[21] == {("player", 1)}
    assert roles[22] == {("player", 0)}
    assert roles[23] == {("referee", NO_TEAM)}
    assert 31 not in roles
    assert "cannot tell which goal each team defends" in caplog.text


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
            {
                "n_kits": 6,
                "outlier_spread": 3.0,
                "min_box_share": 0.5,
                "off_pitch_margin": 1.5,
            },
        )
    ]
    assert by_name.track_embeddings == result.track_embeddings


def test_a_result_without_team_embeddings_is_refused():
    result = _result(SCENE)
    result.track_embeddings = {}
    with pytest.raises(ValueError, match="team embeddings"):
        assign_roles(result)


def test_a_result_saved_before_team_embeddings_were_kept_is_refused(tmp_path):
    result = _result(SCENE)
    del result.track_embeddings, result.track_embedding_samples
    path = result.save(tmp_path / "result.pkl")

    with pytest.raises(ValueError, match="before results kept the team embeddings"):
        PipelineResult.load(path)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"n_kits": 1}, "n_kits"),
        ({"outlier_spread": 0}, "outlier_spread"),
        ({"min_box_share": 1.5}, "min_box_share"),
        ({"min_box_share": 0}, "min_box_share"),
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


def test_teams_in_one_kit_leave_no_kit_of_neither_team(caplog):
    # both teams in red: no kit tells them apart, so nobody's role changes
    red = [(t, name, team, "red", path) for t, name, team, _, path in PLAYERS]
    officials = [(31 + i, "player", 0, "black", _at(20 + 15 * i, 40)) for i in range(5)]

    with caplog.at_level("WARNING", logger="tactifoot_vision.pipeline.roles"):
        roles = _roles(assign_roles(_result([*red, *officials]), n_kits=2))

    assert "one kit for every team" in caplog.text
    assert all(roles[t] == {("player", team)} for t, _, team, *_ in red + officials)


def test_five_tracks_a_team_are_enough():
    # a short clip: k-means splits a team of five, and the part it splits off
    # is no kit of neither team
    scene = [*PLAYERS[:5], *PLAYERS[10:15], KEEPER_LEFT, KEEPER_RIGHT, REFEREE]
    roles = _roles(assign_roles(_result(scene)))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[22] == {("goalkeeper", 1)}
    assert roles[23] == {("referee", NO_TEAM)}
    for tracker_id, _, team, *_ in [*PLAYERS[:5], *PLAYERS[10:15]]:
        assert roles[tracker_id] == {("player", team)}


@pytest.mark.parametrize("n_kits", [8, 30])
def test_more_kits_than_pairs_of_tracks_split_no_team_into_officials(n_kits):
    # a kit of one track has no spread to measure its team by
    roles = _roles(assign_roles(_result(SCENE), n_kits=n_kits))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[22] == {("goalkeeper", 1)}
    assert roles[23] == {("referee", NO_TEAM)}
    assert all(roles[t] == {("player", team)} for t, _, team, *_ in PLAYERS)


def test_a_keeper_near_a_teams_kit_in_a_short_shot_keeps_a_kit_of_its_own():
    # six tracks, four kits: a kit each, or the grey keeper joins red's kit
    scene = [
        *PLAYERS[:2],
        *PLAYERS[10:12],
        (21, "player", 1, "grey", _at(3, 34)),
        REFEREE,
    ]
    roles = _roles(assign_roles(_result(scene)))

    assert roles[21] == {("goalkeeper", 0)}
    assert roles[23] == {("referee", NO_TEAM)}
    for tracker_id, _, team, *_ in [*PLAYERS[:2], *PLAYERS[10:12]]:
        assert roles[tracker_id] == {("player", team)}
