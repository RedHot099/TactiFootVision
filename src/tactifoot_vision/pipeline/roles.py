"""The ``assign_roles`` post-process: goalkeepers, referees and people off the pitch.

On wide footage the detector seldom predicts ``goalkeeper`` or ``referee``, so
the team vote puts goalkeepers, referees and linesmen into a team, and people
beside the pitch count as players. :func:`assign_roles` finds them from the
team embeddings the pass kept (``PipelineResult.track_embeddings``), grouped
into kits, and from where they stand, with no colour threshold of any match.

Distances on the pitch are in metres whatever the pitch units, as in
:func:`~tactifoot_vision.pipeline.join_tracks`.
"""

import dataclasses
import logging
from collections import Counter, defaultdict

import numpy as np

from tactifoot_vision.enums import DetectionKey, PostProcessName, Role
from tactifoot_vision.pipeline.postprocess import POSTPROCESSES, metres_per_unit
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    UNTRACKED,
    ObjectMasks,
    PipelineResult,
)
from tactifoot_vision.pitch.pitch import SoccerPitch

logger = logging.getLogger(__name__)

# fewer player tracks than this give no robust team kit
_MIN_TEAM_TRACKS = 3


@dataclasses.dataclass(frozen=True)
class _Track:
    """One track of people (a player track after joining): class, team, positions, kit."""

    id: int
    name: str
    team: int
    xy: np.ndarray  # (K, 2) metres of its rows with a pitch position
    embedding: np.ndarray | None  # mean of its tracks' team embeddings, by samples
    rows: int  # its rows: how long it is seen


@POSTPROCESSES.register(PostProcessName.ASSIGN_ROLES)
def assign_roles(
    result: PipelineResult,
    n_kits: int = 6,
    outlier_spread: float = 3.0,
    min_box_share: float = 0.5,
    off_pitch_margin: float = 1.5,
) -> PipelineResult:
    """Copy of ``result`` with goalkeepers and referees found and people off the pitch left out.

    A track is judged as a whole (a player track after
    :func:`~tactifoot_vision.pipeline.join_tracks`), by its most frequent
    ``class_name`` and ``team_id``; its embedding is the mean of the team
    embeddings of the tracks it joins (``result.track_embeddings``), each
    weighted by its number of samples. Kits are compared in the embedder's
    own space, not in the team classifier's UMAP space.

    1. A track whose median pitch position lies more than
       ``off_pitch_margin`` m outside the pitch (stewards, cameramen,
       substitutes behind the boards) is left out: its rows are dropped.
    2. The ``player`` and detector ``goalkeeper`` tracks are grouped into
       ``n_kits`` kits by k-means on their embeddings, each weighted by its
       rows. A match has five kits
       on the pitch (two teams, two goalkeepers, the officials); one more
       lets a team split in two without taking an official's kit. There
       are at least two kits and two fewer than tracks, so some kits hold
       two tracks and have a spread. Each team's kit is the kit holding
       most rows of the tracks voted into it, and its spread its tracks'
       mean distance to its centre, at least the teams' mean spread by rows
       and a kit's spread (a part of a team k-means split off measures
       less than the team has, and one track none). A kit
       whose centre lies farther than ``outlier_spread`` times the spread
       from both teams' kits is of neither team: its tracks are outliers.
       Other kits are parts of a team. The team vote cannot do this: it
       puts every official into a team, and a team centre and spread taken
       over them would count them in. Measuring against the teams' spreads
       keeps the rule free of colour thresholds that would hold for one
       match only.
    3. For each goal, the outlier kit with most positioned rows inside its
       penalty area is that goal's goalkeeper kit, or the next one when no
       track of it is mostly in the area (detector goalkeepers are grouped
       too, and a goal with one with an embedding in its penalty area takes
       its kit, or none when that kit is a team's): its tracks in at least
       ``min_box_share`` of their positioned rows inside the area (one
       goalkeeper, often tracked in pieces) become the ``goalkeeper`` of the
       team defending it, which
       :func:`~tactifoot_vision.analysis.attack_direction` tells from where
       the teams stand. One track keeps one goal at most. When the teams' sides
       cannot be told, a warning is logged and the goalkeeper candidates stay
       players of their voted team. The remaining outliers become
       ``referee`` with ``NO_TEAM`` (linesmen and staff on the pitch included).

    A run covers one period: the teams change ends at half-time, so call
    this on each period's result, as ``attack_direction``.

    Tracks the detector calls ``goalkeeper`` or ``referee`` keep their
    class and team, and tracks without an embedding or a pitch position are
    not judged by what they lack. ``class_id`` follows a new class name when
    ``result.class_names`` holds it, and stays the player's (logged) when it
    does not. People without a tracker id are left alone.

    Args:
        result: a result of a pipeline run with a team classifier and a tracker.
        n_kits: kits the player tracks are grouped into, at least 2.
        outlier_spread: how many team spreads from both teams' kits a kit
            is of neither team.
        min_box_share: least share, above 0, of a goalkeeper's positioned
            rows inside the penalty area.
        off_pitch_margin: metres outside the pitch a median position may lie
            and still count as on it (linesmen walk the touchline).

    Raises:
        ValueError: on an option out of range, or when the result holds no
            team embeddings of two teams' player tracks.
    """
    if n_kits < 2:
        raise ValueError(f"n_kits must be >= 2, got {n_kits}")
    if outlier_spread <= 0:
        raise ValueError(f"outlier_spread must be positive, got {outlier_spread}")
    if not 0 < min_box_share <= 1:
        raise ValueError(f"min_box_share must be in (0, 1], got {min_box_share}")
    if off_pitch_margin < 0:
        raise ValueError(f"off_pitch_margin must be >= 0, got {off_pitch_margin}")
    if not result.track_embeddings:
        raise ValueError(
            "assign_roles needs the team embeddings the pass keeps, and the result "
            "holds none: run the pipeline (again) with a team classifier and a tracker"
        )
    scale = metres_per_unit(result.pitch)
    size = scale * [result.pitch.length, result.pitch.width]
    tracks = _tracks(result, scale)
    off_pitch = {t.id for t in tracks if _off_pitch(t, size, off_pitch_margin)}
    on_pitch = [t for t in tracks if t.id not in off_pitch]
    kits = _outlier_kits(on_pitch, n_kits, outlier_spread)
    roles = {
        t.id: (Role.REFEREE, NO_TEAM)
        for kit in kits
        for t in kit
        if t.name == Role.PLAYER
    }
    keepers = _goalkeepers(kits, on_pitch, result.pitch, scale, min_box_share)
    if keepers:
        # tv.analysis imports the pipeline, so it is imported only when needed here
        from tactifoot_vision.analysis.direction import attack_direction

        try:
            directions = attack_direction(_relabel(result, off_pitch, roles))
        except ValueError as error:
            logger.warning(
                "assign_roles cannot tell which goal each team defends (%s); "
                "the %d goalkeeper candidates stay players",
                error,
                len(keepers),
            )
            for tracks in keepers.values():
                for track in tracks:
                    del roles[track.id]
            keepers = {}
        for attacks, tracks in keepers.items():
            # the team attacking towards the other goal defends this one
            for track in tracks:
                roles[track.id] = (
                    Role.GOALKEEPER,
                    _team_attacking(directions, -attacks),
                )
    for name in sorted({name for name, _ in roles.values()}):
        if name not in result.class_names:
            logger.info(
                "The result's class names have no %r: its rows keep the player's "
                "class_id",
                name,
            )
    logger.info(
        "Roles: %d goalkeeper tracks, %d referee or staff tracks, %d people off "
        "the pitch",
        sum(name == Role.GOALKEEPER for name, _ in roles.values()),
        sum(name == Role.REFEREE for name, _ in roles.values()),
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
        own_ids = people.data.get(DetectionKey.ORIGINAL_TRACKER_ID, people.tracker_id)
        xy = frame.pitch_xy * scale
        for track, original, name, team, point in zip(
            people.tracker_id,
            own_ids,
            people.data[DetectionKey.CLASS_NAME],
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
        embedded = [o for o in sorted(originals[track]) if o in result.track_embeddings]
        embedding = None
        if embedded:
            embedding = np.average(
                [result.track_embeddings[o].astype(np.float32) for o in embedded],
                axis=0,
                weights=[result.track_embedding_samples[o] for o in embedded],
            )
        tracks.append(
            _Track(
                id=track,
                name=names[track].most_common(1)[0][0],
                team=teams[track].most_common(1)[0][0],
                xy=np.array(positions[track], dtype=float).reshape(-1, 2),
                embedding=embedding,
                rows=names[track].total(),
            )
        )
    return tracks


def _off_pitch(track: _Track, size: np.ndarray, margin: float) -> bool:
    if len(track.xy) == 0:
        return False
    median = np.median(track.xy, axis=0)
    return bool(np.any(median < -margin) or np.any(median > size + margin))


def _outlier_kits(
    tracks: list[_Track], n_kits: int, outlier_spread: float
) -> list[list[_Track]]:
    """Kits far from both teams' kits, by the teams' spreads.

    Player and detector goalkeeper tracks are grouped; a kit's tracks may
    hold both.
    """
    from sklearn.cluster import KMeans

    embedded = [
        t
        for t in tracks
        if t.name in (Role.PLAYER, Role.GOALKEEPER) and t.embedding is not None
    ]
    voted = Counter(
        t.team for t in embedded if t.team != NO_TEAM and t.name == Role.PLAYER
    )
    if len(voted) < 2:
        raise ValueError(
            "assign_roles needs the team embeddings of two teams' player tracks; "
            f"the result has them for {len(voted)} team(s). Run the pipeline with "
            "a team classifier and a tracker"
        )
    for team, count in sorted(voted.items()):
        if count < _MIN_TEAM_TRACKS:
            logger.warning(
                "Team %d has %d player tracks with an embedding; too few to tell "
                "its kit from the others, so kits may be missed",
                team,
                count,
            )
    embeddings = np.stack([t.embedding for t in embedded]).astype(np.float64)
    weights = np.array([t.rows for t in embedded], dtype=float)
    # two kits fewer than tracks: some kits hold two tracks and have a spread
    k = min(n_kits, max(2, len(embedded) - 2))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(embeddings, sample_weight=weights)
    teams = {
        team: int(
            np.argmax(
                np.bincount(
                    labels,
                    weights=weights
                    * [t.team == team and t.name == Role.PLAYER for t in embedded],
                    minlength=k,
                )
            )
        )
        for team in sorted(voted)
    }
    team_kits = set(teams.values())
    if len(team_kits) < 2:
        logger.warning(
            "assign_roles finds one kit for every team; no kit is of neither team"
        )
        return []
    # Grow each team from its kit: the kit nearest a team, by that team's
    # spread, joins it while within outlier_spread; a team's centre and spread
    # follow its kits, so a team k-means split in parts is put back together.
    # Such a part measures a smaller spread than its team has (none for one
    # track), so neither team's spread is taken below the teams' mean spread,
    # by rows, nor below a kit's.
    kit_spread = _kit_spread(embeddings, weights, labels)
    members = {team: labels == kit for team, kit in teams.items()}
    rest = set(labels.tolist()) - team_kits
    kit_centres = {
        kit: _centre_and_spread(embeddings[labels == kit], weights[labels == kit])[0]
        for kit in rest
    }
    while rest:
        ratio, team, kit = _nearest_kit(
            members,
            {kit: kit_centres[kit] for kit in rest},
            embeddings,
            weights,
            kit_spread,
        )
        logger.debug(
            "Kit of tracks %s: %.1f team spreads from team %d",
            [t.id for t, label in zip(embedded, labels, strict=True) if label == kit],
            ratio,
            team,
        )
        if ratio > outlier_spread:
            break
        members[team] = members[team] | (labels == kit)
        rest.remove(kit)
    return [
        [t for t, label in zip(embedded, labels, strict=True) if label == kit]
        for kit in sorted(rest)
    ]


def _nearest_kit(
    members: dict[int, np.ndarray],
    kit_centres: dict[int, np.ndarray],
    embeddings: np.ndarray,
    weights: np.ndarray,
    kit_spread: float,
) -> tuple[float, int, int]:
    """``(ratio, team, kit)`` of the kit nearest a team, in that team's spreads.

    A team's spread is at least the teams' mean spread, by rows, and
    ``kit_spread``.
    """
    shapes = {
        team: _centre_and_spread(embeddings[member], weights[member])
        for team, member in members.items()
    }
    rows = {team: weights[member].sum() for team, member in members.items()}
    mean_spread = sum(rows[team] * shapes[team][1] for team in shapes) / sum(
        rows.values()
    )
    nearest = None
    for team, (centre, spread) in shapes.items():
        spread = max(spread, mean_spread, kit_spread)
        for kit, kit_centre in kit_centres.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.nan_to_num(
                    np.linalg.norm(kit_centre - centre) / spread,
                    nan=0.0,
                    posinf=np.inf,
                )
            if nearest is None or ratio < nearest[0]:
                nearest = (float(ratio), team, kit)
    return nearest


def _centre_and_spread(
    embeddings: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Weighted centre and mean distance to it of a team's tracks."""
    centre = np.average(embeddings, axis=0, weights=weights)
    distances = np.linalg.norm(embeddings - centre, axis=1)
    return centre, float(np.average(distances, weights=weights))


def _kit_spread(
    embeddings: np.ndarray, weights: np.ndarray, labels: np.ndarray
) -> float:
    """Spread of a kit: rows-weighted mean distance of tracks to their kit's centre.

    Kits of one track are left out; a kit of ``n`` tracks has its distances
    scaled by ``sqrt(n / (n - 1))``, as its own centre lies nearer them than
    the kit's true centre. 0 when no kit holds two tracks.
    """
    total = rows = 0.0
    for kit in np.unique(labels):
        inside = labels == kit
        n = int(inside.sum())
        if n < 2:
            continue
        _, spread = _centre_and_spread(embeddings[inside], weights[inside])
        total += weights[inside].sum() * spread * np.sqrt(n / (n - 1))
        rows += weights[inside].sum()
    return total / rows if rows else 0.0


def _goalkeepers(
    kits: list[list[_Track]],
    tracks: list[_Track],
    pitch: SoccerPitch,
    scale: np.ndarray,
    min_box_share: float,
) -> dict[int, list[_Track]]:
    """The goalkeeper tracks among ``kits`` of each goal, by the direction attacking it.

    ``-1`` is the goal at ``x = 0`` (attacked towards ``-x``), ``1`` the goal
    at ``x = length``. A goal's goalkeeper kit is the kit with most positioned
    rows inside its penalty area that has a player track in the area, among
    the kits of the detector goalkeepers there with an embedding when it
    holds any (none of them in ``kits``: no goalkeeper kit). The goal at
    ``x = 0``'s goalkeepers are no candidates for the other goal.
    """
    length, centre = pitch.length * scale[0], pitch.width * scale[1] / 2
    box = np.array([pitch.penalty_box_length, pitch.penalty_box_width / 2]) * scale

    def inside(track: _Track, goal_x: float) -> np.ndarray:
        return np.all(np.abs(track.xy - [goal_x, centre]) <= box, axis=1)

    def in_box(track: _Track, goal_x: float) -> bool:
        return len(track.xy) > 0 and bool(inside(track, goal_x).mean() >= min_box_share)

    def rows_in(kit: list[_Track], goal_x: float) -> int:
        return sum(int(inside(t, goal_x).sum()) for t in kit)

    keepers: dict[int, list[_Track]] = {}
    for attacks, goal_x in ((-1, 0.0), (1, length)):
        # a detector goalkeeper without an embedding is in no kit
        detected = {
            t.id
            for t in tracks
            if t.name == Role.GOALKEEPER
            and t.embedding is not None
            and in_box(t, goal_x)
        }
        candidates = [
            kit for kit in kits if not detected or any(t.id in detected for t in kit)
        ]
        taken = {t.id for chosen in keepers.values() for t in chosen}
        # a kit with most rows in the area but no track mostly in it (a
        # referee passing through) gives way to the next one
        for kit in sorted(candidates, key=lambda kit: -rows_in(kit, goal_x)):
            if rows_in(kit, goal_x) == 0:
                break
            chosen = [
                t
                for t in kit
                if t.name == Role.PLAYER and t.id not in taken and in_box(t, goal_x)
            ]
            if chosen:
                keepers[attacks] = chosen
                break
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
        names = [str(name) for name in people.data[DetectionKey.CLASS_NAME]]
        teams = frame.team_ids[keep]
        for i, track in enumerate(people.tracker_id):
            if (role := roles.get(int(track))) is not None:
                names[i], teams[i] = role
                if people.class_id is not None and role[0] in result.class_names:
                    people.class_id[i] = result.class_names.index(role[0])
        people.data[DetectionKey.CLASS_NAME] = np.array(names, dtype=str)
        people.data[DetectionKey.TEAM_ID] = teams
        masks = frame.masks
        if masks is not None:
            masks = ObjectMasks(
                origins=masks.origins[keep],
                crops=[crop for crop, k in zip(masks.crops, keep, strict=True) if k],
            )
        frames.append(dataclasses.replace(frame, detections=people, masks=masks))
    return dataclasses.replace(result, frames=frames)
