"""The ``join_tracks`` post-process: tracks joined into player tracks."""

import copy
from collections.abc import Callable

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.pipeline import (
    NO_TEAM,
    POSTPROCESSES,
    AppliedPostProcess,
    FrameResult,
    PipelineResult,
    join_tracks,
    reproject,
)
from tactifoot_vision.pitch import SoccerPitch

FPS = 25.0

Path = Callable[[float], tuple[float, float]]  # seconds -> metres on 105 x 68


def _homography(pitch: SoccerPitch) -> np.ndarray:
    """frame -> pitch: the pitch fills 1000 x 600 pixels."""
    return np.diag([pitch.length / 1000, pitch.width / 600, 1.0])


def _feet_box(pitch: SoccerPitch, metres: tuple[float, float]) -> list[float]:
    """A person box whose bottom centre stands on ``metres``."""
    x, y = np.array(metres) * [1000 / 105, 600 / 68]
    return [x - 5, y - 20, x + 5, y]


def _result(
    tracks: list[tuple[int, int, range, Path]],
    pitch: SoccerPitch | None = None,
    stride: int = 1,
) -> PipelineResult:
    """People given as ``(tracker_id, team_id, frames, path)``, every ``stride``-th frame."""
    pitch = pitch or SoccerPitch()
    end = max(frames.stop for _, _, frames, _ in tracks)
    frames = []
    for index in range(0, end, stride):
        rows = [
            (tracker_id, team, _feet_box(pitch, path(index / FPS)))
            for tracker_id, team, span, path in tracks
            if index in span
        ]
        people = sv.Detections(
            xyxy=np.array([box for *_, box in rows], np.float32).reshape(-1, 4),
            confidence=np.full(len(rows), 0.9, np.float32),
            class_id=np.ones(len(rows), int),
            tracker_id=np.array([tracker_id for tracker_id, *_ in rows], int),
            data={
                "class_name": np.array(["player"] * len(rows)),
                "team_id": np.array([team for _, team, _ in rows], int),
            },
        )
        frames.append(
            FrameResult(
                index,
                index / FPS,
                people,
                sv.Detections.empty(),
                ball_candidates=sv.Detections.empty(),
            )
        )
    result = PipelineResult(frames, FPS, (1000, 600), ["ball", "player"], pitch=pitch)
    return reproject(result, [_homography(pitch)] * len(frames))


def _walk(start: tuple[float, float], velocity: tuple[float, float]) -> Path:
    return lambda t: (start[0] + velocity[0] * t, start[1] + velocity[1] * t)


def _ids_by_original(result: PipelineResult) -> dict[int, set[int]]:
    """The player-track ids each original track id ended up with."""
    ids: dict[int, set[int]] = {}
    for frame in result:
        people = frame.detections
        if people.tracker_id is None:
            continue
        for original, new in zip(
            people.data["original_tracker_id"], people.tracker_id, strict=True
        ):
            ids.setdefault(int(original), set()).add(int(new))
    return ids


# Two players of one team cross: A runs right along y = 30, B left along
# y = 33, and they meet at x = 50 after 2 s. The tracker loses both while they
# pass and picks them up again under new ids, A's new id below B's.
A, B = _walk((40, 30), (5, 0)), _walk((60, 33), (-5, 0))
CROSSING = [
    (1, 0, range(0, 48), A),
    (2, 0, range(0, 48), B),
    (4, 0, range(54, 100), A),
    (3, 0, range(54, 100), B),
]


@pytest.mark.parametrize(
    "pitch", [SoccerPitch(), SoccerPitch(120, 80)], ids=["105x68", "120x80"]
)
@pytest.mark.parametrize("stride", [1, 3])
def test_join_tracks_joins_the_fragments_of_two_crossing_players(pitch, stride):
    result = _result(CROSSING, pitch, stride)

    joined = join_tracks(result)

    assert _ids_by_original(joined) == {1: {1}, 2: {2}, 4: {1}, 3: {2}}
    assert joined.track_ids == [1, 2]


def test_join_tracks_joins_a_chain_of_fragments_into_one_player_track():
    walk = _walk((20, 10), (3, 1))
    result = _result([
        (5, 1, range(0, 20), walk),
        (8, 1, range(25, 40), walk),
        (2, 1, range(45, 60), walk),
    ])  # fmt: skip

    joined = join_tracks(result)

    assert _ids_by_original(joined) == {5: {5}, 8: {5}, 2: {5}}


def test_join_tracks_never_joins_across_teams():
    """An opponent appears where the player was lost; the player reappears 3 m on."""
    player = _walk((50, 30), (2, 0))
    result = _result([
        (1, 0, range(0, 25), player),
        (2, 1, range(30, 60), _walk((52, 30), (0, 1))),
        (3, 0, range(35, 60), lambda t: (player(t)[0], 33)),
    ])  # fmt: skip

    joined = join_tracks(result)
    any_team = join_tracks(result, same_team=False)

    assert _ids_by_original(joined) == {1: {1}, 2: {2}, 3: {1}}
    assert _ids_by_original(any_team) == {1: {1}, 2: {1}, 3: {3}}


def test_join_tracks_joins_tracks_without_teams_only_to_tracks_without_teams():
    walk = _walk((50, 30), (2, 0))
    result = _result([
        (1, NO_TEAM, range(0, 25), walk),
        (2, 0, range(30, 60), walk),
        (3, NO_TEAM, range(30, 60), lambda t: (walk(t)[0], 31)),
    ])  # fmt: skip

    assert _ids_by_original(join_tracks(result)) == {1: {1}, 2: {2}, 3: {1}}


@pytest.mark.parametrize("stride", [1, 2])
def test_join_tracks_leaves_gaps_beyond_max_gap_apart(stride):
    """The track ends at frame 18 and the next starts at 58: a 1.6 s gap."""
    still = _walk((50, 30), (0, 0))
    result = _result(
        [(1, 0, range(0, 19), still), (2, 0, range(58, 80), still)], stride=stride
    )

    assert _ids_by_original(join_tracks(result, max_gap=1.6)) == {1: {1}, 2: {1}}
    assert _ids_by_original(join_tracks(result, max_gap=1.5)) == {1: {1}, 2: {2}}


def test_join_tracks_leaves_tracks_too_far_apart_for_max_speed():
    """The next track starts 10 m away 1 s later: 1.5 m of slack plus 8.5 m/s."""
    result = _result([
        (1, 0, range(0, 25), _walk((50, 30), (0, 0))),
        (2, 0, range(49, 70), _walk((60, 30), (0, 0))),
    ])  # fmt: skip

    assert _ids_by_original(join_tracks(result, max_speed=8.6)) == {1: {1}, 2: {1}}
    assert _ids_by_original(join_tracks(result, max_speed=8.4)) == {1: {1}, 2: {2}}


def test_join_tracks_keeps_the_original_ids_and_adds_no_rows():
    result = _result(CROSSING)

    joined = join_tracks(result)

    for before, after in zip(result, joined, strict=True):
        assert len(after.detections) == len(before.detections)
        np.testing.assert_array_equal(
            after.detections.data["original_tracker_id"], before.detections.tracker_id
        )
        np.testing.assert_array_equal(after.detections.xyxy, before.detections.xyxy)
    assert "original_tracker_id" not in result[0].detections.data  # input untouched
    assert result.track_ids == [1, 2, 3, 4]


def test_join_tracks_is_registered_and_recorded_with_its_options():
    result = _result(CROSSING)

    joined = POSTPROCESSES.get("join_tracks")(result, max_gap=1.0)

    assert joined.postprocesses == [
        AppliedPostProcess(
            "join_tracks",
            {"max_gap": 1.0, "max_speed": 9.0, "slack": 1.5, "same_team": True},
        )
    ]


def test_join_tracks_leaves_untracked_people_and_untracked_frames_alone():
    result = _result(CROSSING)
    frames = list(result.frames)
    untracked = copy.deepcopy(frames[0].detections)
    untracked.tracker_id = None
    frames[0] = FrameResult(
        0, 0.0, untracked, frames[0].ball, ball_candidates=frames[0].ball_candidates
    )
    frames[1].detections.tracker_id[:] = -1  # UNTRACKED
    result = PipelineResult(frames, FPS, (1000, 600), ["ball", "player"])

    joined = join_tracks(result)

    assert joined[0].detections.tracker_id is None
    assert joined[1].detections.tracker_id.tolist() == [-1, -1]
    assert _ids_by_original(joined)[4] == {1}


@pytest.mark.parametrize(
    "options",
    [{"max_gap": 0}, {"max_speed": 0}, {"slack": -1}],
    ids=["max_gap", "max_speed", "slack"],
)
def test_join_tracks_refuses_invalid_options(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        join_tracks(_result(CROSSING), **options)


def test_join_tracks_refuses_unordered_timestamps():
    result = _result(CROSSING)
    result.frames.reverse()

    with pytest.raises(ValueError, match="increasing timestamps"):
        join_tracks(result)
