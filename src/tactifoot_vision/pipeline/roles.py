"""The ``assign_roles`` post-process: goalkeepers, referees and people off the pitch.

On wide footage the detector seldom predicts ``goalkeeper`` or ``referee``, so
the team vote puts goalkeepers, referees and linesmen into a team, and people
beside the pitch count as players. :func:`assign_roles` finds them from the
team embeddings the pass kept (``PipelineResult.track_embeddings``) and from
where they stand, with no colour threshold of any match.

Distances on the pitch are in metres whatever the pitch units, as in
:func:`~tactifoot_vision.pipeline.join_tracks`.
"""

import dataclasses
import logging
from collections import Counter, defaultdict

import numpy as np

from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, metres_per_unit
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    UNTRACKED,
    ObjectMasks,
    PipelineResult,
)

logger = logging.getLogger(__name__)

PLAYER, GOALKEEPER, REFEREE = "player", "goalkeeper", "referee"


@dataclasses.dataclass(frozen=True)
class _Track:
    """One player track: its most frequent class and team, positions and embedding."""

    id: int
    name: str
    team: int
    xy: np.ndarray  # (K, 2) metres of its rows with a pitch position
    embedding: np.ndarray | None  # mean of its tracks' team embeddings


@POSTPROCESSES.register("assign_roles")
def assign_roles(
    result: PipelineResult,
    outlier_spread: float = 3.0,
    min_box_share: float = 0.5,
    off_pitch_margin: float = 1.5,
) -> PipelineResult:
    """Copy of ``result`` with goalkeepers and referees found and people off the pitch left out.

    A track is judged as a whole (a player track after
    :func:`~tactifoot_vision.pipeline.join_tracks`), by its most frequent
    ``class_name`` and ``team_id``; its embedding is the mean of the team
    embeddings of the tracks it joins (``result.track_embeddings``).

    1. A track whose median pitch position lies more than
       ``off_pitch_margin`` m outside the pitch (stewards, cameramen,
       substitutes behind the boards) is left out: its rows are dropped.
    2. Each team's centre is the median embedding of its ``player`` tracks,
       and its spread their median distance to that centre. A ``player``
       track farther than ``outlier_spread`` times the spread from every
       team's centre has a kit of neither team: it is "other". Measuring
       against each team's own spread keeps the rule free of colour
       thresholds that would hold for one match only.
    3. Of the others in at least ``min_box_share`` of their positioned rows
       inside a penalty area, the one whose median position is nearest that
       goal becomes the ``goalkeeper`` of the team defending it, which
       :func:`~tactifoot_vision.analysis.attack_direction` tells from where
       the teams stand. The other others become ``referee`` with ``NO_TEAM``
       (linesmen and staff on the pitch included).

    Tracks the detector calls ``goalkeeper`` or ``referee`` keep their
    class and team, and tracks without an embedding or a pitch position are
    not judged by what they lack. ``class_id`` follows a new class name when
    ``result.class_names`` holds it. People without a tracker id are left
    alone.

    Args:
        result: a result of a pipeline run with a team classifier and a tracker.
        outlier_spread: how many team spreads from both team centres a kit
            is an outlier.
        min_box_share: least share of a goalkeeper's positioned rows inside
            the penalty area.
        off_pitch_margin: metres outside the pitch a median position may lie
            and still count as on it (linesmen walk the touchline).

    Raises:
        ValueError: on an option out of range, when the result holds no
            team embeddings of two teams' player tracks, or when a
            goalkeeper is found but the teams' sides cannot be told.
    """
    if outlier_spread <= 0:
        raise ValueError(f"outlier_spread must be positive, got {outlier_spread}")
    if not 0 <= min_box_share <= 1:
        raise ValueError(f"min_box_share must be in [0, 1], got {min_box_share}")
    if off_pitch_margin < 0:
        raise ValueError(f"off_pitch_margin must be >= 0, got {off_pitch_margin}")
    scale = metres_per_unit(result.pitch)
    size = scale * [result.pitch.length, result.pitch.width]
    tracks = _tracks(result, scale)
    off_pitch = {t.id for t in tracks if _off_pitch(t, size, off_pitch_margin)}
    on_pitch = [t for t in tracks if t.id not in off_pitch]
    others = _kit_outliers(on_pitch, outlier_spread)
    roles = {t.id: (REFEREE, NO_TEAM) for t in others}
    keepers = _goalkeepers(others, result, min_box_share)
    if keepers:
        # tv.analysis imports the pipeline, so it is imported only when needed here
        from tactifoot_vision.analysis.direction import attack_direction

        try:
            directions = attack_direction(_relabel(result, off_pitch, roles))
        except ValueError as error:
            raise ValueError(
                f"assign_roles cannot tell which goal each team defends: {error}"
            ) from error
        for attacks, track in keepers.items():
            # the team attacking towards the other goal defends this one
            roles[track.id] = (GOALKEEPER, _team_attacking(directions, -attacks))
    logger.info(
        "Roles: %d goalkeepers, %d referees or staff, %d people off the pitch",
        len(keepers),
        len(others) - len(keepers),
        len(off_pitch),
    )
    return _relabel(result, off_pitch, roles)


def _tracks(result: PipelineResult, scale: np.ndarray) -> list[_Track]:
    names: defaultdict[int, Counter[str]] = defaultdict(Counter)
    teams: defaultdict[int, Counter[int]] = defaultdict(Counter)
    positions: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    originals: defaultdict[int, set[int]] = defaultdict(set)
    for frame in result:
        people = frame.detections
        if people.tracker_id is None:
            continue
        own_ids = people.data.get("original_tracker_id", people.tracker_id)
        xy = frame.pitch_xy * scale
        for track, original, name, team, point in zip(
            people.tracker_id,
            own_ids,
            people.data["class_name"],
            frame.team_ids,
            xy,
            strict=True,
        ):
            if track == UNTRACKED:
                continue
            track = int(track)
            names[track][str(name)] += 1
            teams[track][int(team)] += 1
            originals[track].add(int(original))
            if np.isfinite(point).all():
                positions[track].append(point)
    tracks = []
    for track in sorted(names):
        embeddings = [
            result.track_embeddings[o]
            for o in sorted(originals[track])
            if o in result.track_embeddings
        ]
        tracks.append(
            _Track(
                id=track,
                name=names[track].most_common(1)[0][0],
                team=teams[track].most_common(1)[0][0],
                xy=np.array(positions[track], dtype=float).reshape(-1, 2),
                embedding=np.mean(embeddings, axis=0) if embeddings else None,
            )
        )
    return tracks


def _off_pitch(track: _Track, size: np.ndarray, margin: float) -> bool:
    if len(track.xy) == 0:
        return False
    median = np.median(track.xy, axis=0)
    return bool(np.any(median < -margin) or np.any(median > size + margin))


def _kit_outliers(tracks: list[_Track], outlier_spread: float) -> list[_Track]:
    """Player tracks whose embedding is far from every team's centre, by its spread."""
    players = [t for t in tracks if t.name == PLAYER and t.embedding is not None]
    teams = sorted({t.team for t in players if t.team != NO_TEAM})
    if len(teams) < 2:
        raise ValueError(
            "assign_roles needs the team embeddings of two teams' player tracks; "
            f"the result has them for {len(teams)} team(s). Run the pipeline with "
            "a team classifier and a tracker"
        )
    embeddings = np.stack([t.embedding for t in players])
    ratios = np.full(len(players), np.inf)
    for team in teams:
        members = embeddings[[t.team == team for t in players]]
        centre = np.median(members, axis=0)
        spread = np.median(np.linalg.norm(members - centre, axis=1))
        distance = np.linalg.norm(embeddings - centre, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.nan_to_num(distance / spread, nan=0.0, posinf=np.inf)
        ratios = np.minimum(ratios, ratio)
    return [
        t for t, ratio in zip(players, ratios, strict=True) if ratio > outlier_spread
    ]


def _goalkeepers(
    others: list[_Track], result: PipelineResult, min_box_share: float
) -> dict[int, _Track]:
    """The goalkeeper among ``others`` of each goal, by the direction attacking it.

    ``-1`` is the goal at ``x = 0`` (attacked towards ``-x``), ``1`` the goal
    at ``x = length``.
    """
    pitch, scale = result.pitch, metres_per_unit(result.pitch)
    length, centre = pitch.length * scale[0], pitch.width * scale[1] / 2
    box_length = pitch.penalty_box_length * scale[0]
    box_half_width = pitch.penalty_box_width * scale[1] / 2
    keepers = {}
    for attacks, goal_x in ((-1, 0.0), (1, length)):
        nearest, best = None, np.inf
        for track in others:
            if len(track.xy) == 0:
                continue
            x, y = track.xy.T
            in_box = (np.abs(x - goal_x) <= box_length) & (
                np.abs(y - centre) <= box_half_width
            )
            distance = np.linalg.norm(np.median(track.xy, axis=0) - [goal_x, centre])
            if in_box.mean() >= min_box_share and distance < best:
                nearest, best = track, distance
        if nearest is not None:
            keepers[attacks] = nearest
    return keepers


def _team_attacking(directions: dict[int, int], direction: int) -> int:
    return next(team for team, d in directions.items() if d == direction)


def _relabel(
    result: PipelineResult,
    off_pitch: set[int],
    roles: dict[int, tuple[str, int]],
) -> PipelineResult:
    """Copy of ``result`` without the ``off_pitch`` tracks and with ``roles`` written."""
    frames = []
    for frame in result:
        people = frame.detections
        if people.tracker_id is None:
            frames.append(frame)
            continue
        keep = ~np.isin(people.tracker_id, list(off_pitch))
        people = people[keep]  # boolean indexing copies every array
        names = [str(name) for name in people.data["class_name"]]
        teams = frame.team_ids[keep]
        for i, track in enumerate(people.tracker_id):
            if (role := roles.get(int(track))) is not None:
                names[i], teams[i] = role
                if people.class_id is not None and role[0] in result.class_names:
                    people.class_id[i] = result.class_names.index(role[0])
        people.data["class_name"] = np.array(names, dtype=str)
        people.data["team_id"] = teams
        masks = frame.masks
        if masks is not None:
            masks = ObjectMasks(
                origins=masks.origins[keep],
                crops=[crop for crop, k in zip(masks.crops, keep, strict=True) if k],
            )
        frames.append(dataclasses.replace(frame, detections=people, masks=masks))
    return dataclasses.replace(result, frames=frames)
