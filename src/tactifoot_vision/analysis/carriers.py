"""Ball carrier, possession and passes: who has the ball, frame by frame."""

import math
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

_EPS = 1e-6  # timestamps are index / fps, so equal spans may differ in the last bit
_PASS_COLUMNS = ["team_id", "from_track_id", "to_track_id", "from_frame", "to_frame"]


def ball_carriers(
    result: PipelineResult,
    max_distance: float = 0.7,
    hysteresis: float = 0.12,
    max_gap: float = 1.0,
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
    other than the carrier has been seen for ``hysteresis`` seconds (0.12:
    three frames at 25 fps) in frames with a ball in a row, each frame
    counting one of the result's median time steps, so any stride waits the
    same time and a coarse one at least one frame. A shorter flicker changes
    nothing: another candidate, the carrier or a loose ball, in between
    starts the count again. A frame without a ball neither confirms nor
    breaks it, so the carrier is held while the ball is unseen, for up to
    ``max_gap`` seconds (1.0) after the ball was last seen; then nobody
    carries it and the count starts again (``np.inf`` holds it for good).
    """
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError(
            f"max_distance must be a finite number > 0, got {max_distance}"
        )
    if not np.isfinite(hysteresis) or hysteresis < 0:
        raise ValueError(
            f"hysteresis must be a finite number of seconds >= 0, got {hysteresis}"
        )
    if not max_gap > 0:
        raise ValueError(f"max_gap must be > 0 seconds, got {max_gap}")
    times = [frame.timestamp for frame in result]
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    step = step if step > 0 else 1 / result.fps
    frames_needed = max(1, math.ceil(hysteresis / step - _EPS))
    carriers: list[int | None] = []
    current: int | None = None
    # The candidate challenging the carrier and the frames in a row it has
    # been seen; no challenger while count is 0 (None is a loose ball).
    challenger: int | None = None
    count = 0
    last_seen = -math.inf  # timestamp of the last frame with a ball
    for frame in result:
        if len(frame.ball):
            last_seen = frame.timestamp
            candidate = _nearest(frame, max_distance)
            if candidate == current:
                count = 0
            else:
                count = count + 1 if count and candidate == challenger else 1
                challenger = candidate
                if count >= frames_needed:
                    current, count = candidate, 0
        elif frame.timestamp - last_seen >= max_gap - _EPS:
            current, count = None, 0
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
    return _passes(result, carriers, _carrier_teams(result, carriers))


def _passes(
    result: PipelineResult, carriers: Sequence[int | None], teams: list[int | None]
) -> pd.DataFrame:
    """:func:`passes` with the carriers' ``teams`` already found."""
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
    return _carrier_table(result, carriers, _carrier_teams(result, carriers))


def _carrier_table(
    result: PipelineResult, carriers: Sequence[int | None], teams: list[int | None]
) -> pd.DataFrame:
    """:func:`carrier_table` with the carriers' ``teams`` already found."""
    return pd.DataFrame({
        "frame": [frame.index for frame in result],
        "time": [frame.timestamp for frame in result],
        "track_id": pd.array(list(carriers), dtype="Int64"),
        "team_id": pd.array(teams, dtype="Int64"),
    })  # fmt: skip


def _run_folder_tables(
    result: PipelineResult, **options: float
) -> dict[str, pd.DataFrame]:
    """``carriers.csv`` and ``passes.csv`` of the run folder, from ``ball_carriers(result, **options)``."""
    carriers = ball_carriers(result, **options)
    teams = _carrier_teams(result, carriers)
    return {
        "carriers.csv": _carrier_table(result, carriers, teams),
        "passes.csv": _passes(result, carriers, teams),
    }


def _nearest(frame: FrameResult, max_distance: float) -> int | None:
    """The tracker id of the frame's carrier candidate, ``None`` for a loose ball."""
    detections = frame.detections[_eligible(frame)]
    feet = detections.get_anchors_coordinates(PERSON_ANCHOR)
    # ``ball`` holds at most one detection: the pipeline's best candidate or
    # track_ball's choice.
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
