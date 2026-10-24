"""The ``join_tracks`` post-process: join tracks into player tracks.

The tracker loses a player behind another player, at the edge of the frame or
in a fast camera move, and picks them up again under a new id. Across the
whole video, :func:`join_tracks` links each track end to a later track start
of the same team that the player could have run to, and gives the linked
tracks one id: the player track's.

Distances are in metres and times in seconds whatever the pitch units, as in
:func:`~tactifoot_vision.pipeline.track_ball`.
"""

import dataclasses
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from tactifoot_vision.enums import DetectionKey, PostProcessName
from tactifoot_vision.pipeline.postprocess import (
    POSTPROCESSES,
    metres_per_unit,
    remap_track_ids,
)
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    UNTRACKED,
    FrameResult,
    PipelineResult,
)

logger = logging.getLogger(__name__)

# metres a second of gap adds to the cost of a link (the prototype's 0.05 per frame at 25 fps)
_GAP_COST = 1.25
_EPS = 1e-6  # seconds or metres: timestamps are index / fps


@dataclasses.dataclass(frozen=True)
class _Tracks:
    """One entry per track: where and when it starts and ends, its team and class."""

    ids: np.ndarray
    teams: np.ndarray
    classes: np.ndarray  # the most frequent class_name ("" without one)
    first_seen: np.ndarray  # seconds of the first row
    last_seen: np.ndarray  # seconds of the last row
    # the first and last rows with a pitch position (NaN for none)
    start_time: np.ndarray
    start_xy: np.ndarray
    end_time: np.ndarray
    end_xy: np.ndarray


@POSTPROCESSES.register(PostProcessName.JOIN_TRACKS)
def join_tracks(
    result: PipelineResult,
    max_gap: float = 1.6,
    max_speed: float = 9.0,
    slack: float = 1.5,
    same_team: bool = True,
) -> PipelineResult:
    """Copy of ``result`` whose people's ``tracker_id`` is their player track's id.

    A track may be followed by a track that starts after it ends, at most
    ``max_gap`` s later, within ``slack + max_speed * dt`` m of where it ended
    (``dt`` being the time between its last position on the pitch and the
    other's first), and with the same ``team_id`` (``NO_TEAM`` joins only
    ``NO_TEAM`` of the same ``class_name``, so a referee never takes a
    player's id when there are no teams). A track's team and class are the
    most frequent among its rows. Of these links the cheapest are taken first, the cost being
    the distance plus 1.25 m per second of gap, and each track gets at most
    one successor and one predecessor. Tracks linked one after another form
    one player track, with the id of its earliest track. A track without a
    pitch position (no homography in any of its frames) is never linked.

    The tracker's ids are kept in ``detections.data["original_tracker_id"]``
    (see :func:`~tactifoot_vision.pipeline.remap_track_ids`). No rows are
    added: a player's position stays unknown between the tracks. People
    without a tracker id (``UNTRACKED``) and frames without a tracker are
    left alone.

    Args:
        result: the result whose tracks to join.
        max_gap: longest time in seconds between a track's end and the next one's start.
        max_speed: metres per second a player may have run in the gap.
        slack: metres of position error allowed on top of the running distance.
        same_team: whether linked tracks need the same ``team_id`` (and,
            without a team, the same ``class_name``).
    """
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")
    if max_speed <= 0:
        raise ValueError("max_speed must be positive")
    if slack < 0:
        raise ValueError("slack must be >= 0")
    times = np.array([f.timestamp for f in result.frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("join_tracks needs frames in order of increasing timestamps")
    tracks = _tracks(result.frames, metres_per_unit(result.pitch))
    successors = _link(tracks, max_gap, max_speed, slack, same_team)
    mapping = _player_tracks(tracks.ids, successors)
    logger.info(
        "Joined %d tracks into %d player tracks",
        len(tracks.ids),
        len(set(mapping.values())),
    )
    return remap_track_ids(result, mapping)


def _tracks(frames: list[FrameResult], scale: np.ndarray) -> _Tracks:
    rows: defaultdict[int, list[tuple[float, np.ndarray, int, str]]]
    rows = defaultdict(list)
    for frame in frames:
        people = frame.detections
        if people.tracker_id is None:
            continue
        xy = frame.pitch_xy * scale
        names = people.data.get(DetectionKey.CLASS_NAME, np.full(len(people), ""))
        for tracker_id, position, team, name in zip(
            people.tracker_id, xy, frame.team_ids, names, strict=True
        ):
            if tracker_id != UNTRACKED:
                row = (frame.timestamp, position, int(team), str(name))
                rows[int(tracker_id)].append(row)
    ids = sorted(rows)
    teams, classes, first_seen, last_seen, starts, ends = [], [], [], [], [], []
    for tracker_id in ids:
        track = rows[tracker_id]
        teams.append(_most_frequent(team for *_, team, _ in track))
        classes.append(_most_frequent(name for *_, name in track))
        first_seen.append(track[0][0])
        last_seen.append(track[-1][0])
        located = [(time, xy) for time, xy, *_ in track if np.isfinite(xy).all()]
        none = (np.nan, np.full(2, np.nan))
        starts.append(located[0] if located else none)
        ends.append(located[-1] if located else none)
    return _Tracks(
        ids=np.array(ids, dtype=int),
        teams=np.array(teams, dtype=int),
        classes=np.array(classes, dtype=str),
        first_seen=np.array(first_seen, dtype=float),
        last_seen=np.array(last_seen, dtype=float),
        start_time=np.array([time for time, _ in starts], dtype=float),
        start_xy=np.array([xy for _, xy in starts], dtype=float).reshape(-1, 2),
        end_time=np.array([time for time, _ in ends], dtype=float),
        end_xy=np.array([xy for _, xy in ends], dtype=float).reshape(-1, 2),
    )


def _link(
    tracks: _Tracks, max_gap: float, max_speed: float, slack: float, same_team: bool
) -> dict[int, int]:
    """Index of each linked track's successor: the cheapest allowed links first."""
    links = []
    for a in range(len(tracks.ids)):
        gap = tracks.first_seen - tracks.last_seen[a]
        distance = np.linalg.norm(tracks.start_xy - tracks.end_xy[a], axis=1)
        reach = slack + max_speed * (tracks.start_time - tracks.end_time[a])
        # NaN positions fail the distance test
        allowed = (gap > 0) & (gap <= max_gap + _EPS) & (distance <= reach + _EPS)
        if same_team:
            allowed &= tracks.teams == tracks.teams[a]
            if tracks.teams[a] == NO_TEAM:
                allowed &= tracks.classes == tracks.classes[a]
        for b in np.flatnonzero(allowed):
            links.append((distance[b] + _GAP_COST * gap[b], a, int(b)))
    successors: dict[int, int] = {}
    predecessors: set[int] = set()
    for _, a, b in sorted(links):
        if a not in successors and b not in predecessors:
            successors[a] = b
            predecessors.add(b)
    return successors


def _most_frequent[T](values: Iterable[T]) -> T:
    """The most frequent value, the first seen of those tied."""
    return Counter(values).most_common(1)[0][0]


def _player_tracks(ids: np.ndarray, successors: dict[int, int]) -> dict[int, int]:
    """Player-track id of every track: the id of the earliest track linked to it."""
    heads = set(range(len(ids))) - set(successors.values())
    mapping = {}
    for head in sorted(heads):
        track: int | None = head
        while track is not None:
            mapping[int(ids[track])] = int(ids[head])
            track = successors.get(track)
    return mapping
