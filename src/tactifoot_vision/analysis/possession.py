"""Ball carrier, possession and passes: who has the ball, frame by frame."""

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline.result import (
    BALL_ANCHOR,
    NO_TEAM,
    PERSON_ANCHOR,
    UNTRACKED,
    FrameResult,
    PipelineResult,
)

_PASS_COLUMNS = ["team_id", "from_track_id", "to_track_id", "from_frame", "to_frame"]


def ball_carriers(
    result: PipelineResult, max_distance: float = 0.7, hysteresis: int = 3
) -> list[int | None]:
    """The ball carrier of each frame of ``result.frames``: a ``tracker_id`` or ``None``.

    In each frame with a ball (one filled in by :func:`track_ball
    <tactifoot_vision.pipeline.track_ball>` too), the candidate is the person
    with a team (players and goalkeepers: referees and staff have
    ``NO_TEAM``, see :func:`~tactifoot_vision.pipeline.assign_roles`) and a
    ``tracker_id`` whose box foot (bottom centre) is nearest the ball's
    centre, in image pixels over the person's box height, when that is under
    ``max_distance`` (0.7); otherwise the ball is loose. The image is used
    because the ball's height is unknown, so the rule also holds with the
    ball in the air. Run :func:`join_tracks
    <tactifoot_vision.pipeline.join_tracks>` first so a carrier is a player
    track.

    The carrier changes, to another person or to nobody, once a candidate
    other than the carrier has been seen in ``hysteresis`` (3) frames with a
    ball in a row, so a shorter flicker changes nothing: another candidate,
    the carrier or a loose ball, in between starts the count again. The
    count is in processed frames, whatever the stride; a frame without a
    ball neither confirms nor breaks it, so the carrier is held while the
    ball is unseen.
    """
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError(
            f"max_distance must be a finite number > 0, got {max_distance}"
        )
    if isinstance(hysteresis, bool) or not isinstance(hysteresis, int | np.integer):
        raise ValueError(
            f"hysteresis must be a whole number of frames, got {hysteresis}"
        )
    if hysteresis < 1:
        raise ValueError(f"hysteresis must be >= 1, got {hysteresis}")
    carriers: list[int | None] = []
    current: int | None = None
    # The candidate challenging the carrier and the frames in a row it has
    # been seen; no challenger while count is 0 (None is a loose ball).
    challenger: int | None = None
    count = 0
    for frame in result:
        if len(frame.ball):
            candidate = _nearest(frame, max_distance)
            if candidate == current:
                count = 0
            else:
                count = count + 1 if count and candidate == challenger else 1
                challenger = candidate
                if count >= hysteresis:
                    current, count = candidate, 0
        carriers.append(current)
    return carriers


def possession(
    result: PipelineResult,
    carriers: Sequence[int | None],
    frames: int | slice | None = None,
) -> dict[int, float]:
    """Each team's share of the frames with a ball carrier, by team id.

    ``carriers`` are :func:`ball_carriers` of ``result``, computed over the
    whole result so the hysteresis holds at the range's edges. ``frames`` is
    a position in ``result.frames`` (as ``result[i]``, not the video frame
    index), a slice of them, or ``None`` for every frame. Every team of the
    result is a key, and the shares add up to 1. A range without a carrier
    is a ``ValueError``.
    """
    teams = _carrier_teams(result, carriers)
    every = range(len(result.frames))
    selected = every[frames] if frames is not None else every
    selected = [selected] if isinstance(selected, int) else selected
    counts = Counter(teams[p] for p in selected if teams[p] is not None)
    total = sum(counts.values())
    if not total:
        raise ValueError(
            f"Possession needs a ball carrier; there is no ball carrier in the "
            f"{len(selected)} frame{'' if len(selected) == 1 else 's'} selected"
        )
    return {team: counts[team] / total for team in _teams(result)}


def passes(result: PipelineResult, carriers: Sequence[int | None]) -> pd.DataFrame:
    """The passes of ``result``: one row per change of carrier within a team.

    ``carriers`` are :func:`ball_carriers` of ``result``. Columns: ``team_id,
    from_track_id, to_track_id, from_frame, to_frame``, where ``from_frame``
    is the video frame index of the passer's last frame as carrier and
    ``to_frame`` that of the receiver's first. Frames without a carrier
    between the two are skipped, so a pass is a change of carrier, not a
    measured ball flight: a teammate taking a loose ball counts, a carrier
    regaining a loose ball does not, and a pass whose receiver never becomes
    the carrier (out of the picture, or for fewer frames than the
    hysteresis) is missed. A carrier of the other team in between breaks it.
    """
    teams = _carrier_teams(result, carriers)
    rows = []
    last = None  # position of the previous carrier's last frame
    for position, carrier in enumerate(carriers):
        if carrier is None:
            continue
        if last is not None and carrier != carriers[last]:
            if teams[position] == teams[last]:
                rows.append({
                    "team_id": teams[position],
                    "from_track_id": carriers[last], "to_track_id": carrier,
                    "from_frame": result[last].index,
                    "to_frame": result[position].index,
                })  # fmt: skip
        last = position
    return pd.DataFrame(rows, columns=_PASS_COLUMNS)


def pass_counts(
    result: PipelineResult, carriers: Sequence[int | None]
) -> dict[int, int]:
    """The number of :func:`passes` of each team, by team id (every team of the result)."""
    counts = Counter(passes(result, carriers)["team_id"])
    return {team: counts[team] for team in _teams(result)}


def carrier_table(
    result: PipelineResult, carriers: Sequence[int | None]
) -> pd.DataFrame:
    """One row per frame: ``frame, time, track_id, team_id`` of its ball carrier.

    ``carriers`` are :func:`ball_carriers` of ``result``; ``track_id`` and
    ``team_id`` are empty (``pd.NA``) in a frame without a carrier.
    """
    teams = _carrier_teams(result, carriers)
    return pd.DataFrame({
        "frame": [frame.index for frame in result],
        "time": [frame.timestamp for frame in result],
        "track_id": pd.array(list(carriers), dtype="Int64"),
        "team_id": pd.array(teams, dtype="Int64"),
    })  # fmt: skip


def _nearest(frame: FrameResult, max_distance: float) -> int | None:
    """The tracker id of the frame's carrier candidate, ``None`` for a loose ball."""
    detections = frame.detections[_eligible(frame)]
    feet = detections.get_anchors_coordinates(PERSON_ANCHOR)
    ball = frame.ball.get_anchors_coordinates(BALL_ANCHOR)[0]
    heights = np.maximum(detections.xyxy[:, 3] - detections.xyxy[:, 1], 1.0)
    distances = np.hypot(*(feet - ball).T) / heights
    close = np.flatnonzero(distances < max_distance)  # NaN boxes are never close
    if not len(close):
        return None
    return int(detections.tracker_id[close[np.argmin(distances[close])]])


def _eligible(frame: FrameResult) -> np.ndarray:
    """Which of the frame's people can carry the ball: those with a team and a tracker id."""
    ids = frame.detections.tracker_id
    if ids is None:
        return np.zeros(len(frame.detections), dtype=bool)
    return (frame.team_ids != NO_TEAM) & (ids != UNTRACKED)


def _carrier_teams(
    result: PipelineResult, carriers: Sequence[int | None]
) -> list[int | None]:
    """The team of each frame's carrier: their track's most frequent team."""
    if len(carriers) != len(result.frames):
        raise ValueError(
            f"carriers must hold one carrier per frame of the result "
            f"({len(result.frames)}), got {len(carriers)}; pass ball_carriers(result)"
        )
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame in result:
        people = _eligible(frame)
        if people.any():
            ids, team_ids = frame.detections.tracker_id[people], frame.team_ids[people]
            for track, team in zip(ids, team_ids, strict=True):
                votes[int(track)][int(team)] += 1
    teams = {track: vote.most_common(1)[0][0] for track, vote in votes.items()}
    if unknown := sorted({c for c in carriers if c is not None and c not in teams}):
        raise ValueError(
            f"Carrier {unknown[0]} is nobody of the result with a team and a "
            "tracker id; pass ball_carriers(result)"
        )
    return [None if c is None else teams[c] for c in carriers]


def _teams(result: PipelineResult) -> list[int]:
    """Every team id of the result's people, in order."""
    return sorted({int(t) for f in result for t in f.team_ids if t != NO_TEAM})
