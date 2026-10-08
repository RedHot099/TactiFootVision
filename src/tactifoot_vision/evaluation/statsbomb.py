"""Compare pipeline pitch positions with StatsBomb 360 freeze frames."""

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tactifoot_vision.pipeline.result import PipelineResult

logger = logging.getLogger(__name__)

_TIME = ["period", "minute", "second"]
# StatsBomb 360 freeze frames hold only these; other detections never match.
_CANDIDATE_TYPES = ("player", "goalkeeper")
# freeze-frame column -> column added to the StatsBomb table
_DETECTED = {
    "location": "detected_location",
    "type": "detected_type",
    "player_id": "detected_player_id",
    "frame_id": "detected_frame_id",
    "frame_bbox": "detected_frame_bbox",
    "confidence": "detected_confidence",
    "visible_area": "detected_visible_area",
}


def compare_with_statsbomb(
    freeze_frames: "pd.DataFrame | PipelineResult",
    statsbomb: pd.DataFrame,
    period: int = 1,
) -> pd.DataFrame:
    """Match every StatsBomb 360 object to the nearest detected player or goalkeeper.

    Rows of both tables are grouped by ``(period, minute, second)``. Only
    detections of type ``player`` or ``goalkeeper`` with a pitch location are
    candidates (StatsBomb freeze frames hold nothing else), so the ball,
    referees and other classes never match. Each StatsBomb object gets the
    closest candidate (Euclidean distance between pitch locations) in **one
    frame**: the processed frame closest in time to its event, when both
    tables carry sub-second times (``timestamp_seconds`` in both and
    ``frame_id`` in the freeze frames, as
    :func:`tactifoot_vision.data.load_statsbomb` and ``to_freeze_frames``
    give). That frame is picked among all freeze-frame rows of the second,
    candidates or not, so when it holds no candidate (only a referee, or
    players without a pitch location) the object stays unmatched rather than
    borrowing a later frame. Without sub-second times every processed frame
    of that second competes, and ``euclidean_distance`` is the minimum over
    the whole second, an optimistic figure at high frame rates.

    Both tables must use the same pitch coordinates: build the pipeline with
    ``SoccerPitch(120, 80)`` to compare with StatsBomb units. A
    :class:`PipelineResult` is converted with ``to_freeze_frames(period)``,
    which assumes the video starts at the period's kick-off; otherwise pass
    ``result.to_freeze_frames(period, period_start=...)``.

    Args:
        freeze_frames: pipeline freeze frames (columns ``period, minute, second,
            location`` plus optionally ``type, player_id, frame_id,
            timestamp_seconds, frame_bbox, confidence, visible_area``) or the
            :class:`PipelineResult` producing them.
        statsbomb: StatsBomb objects with ``period, minute, second,
            pitch_location, type`` and optionally ``timestamp_seconds`` (see
            :func:`tactifoot_vision.data.load_statsbomb`); locations may be
            ``[x, y]`` lists or JSON strings.
        period: match period to compare.

    Returns:
        The StatsBomb rows of ``period`` that have a location, with
        ``detected_location``, ``detected_type``, ``detected_player_id``,
        ``detected_frame_id`` (the frame the match comes from),
        ``detected_frame_bbox``, ``detected_confidence``,
        ``detected_visible_area`` and ``euclidean_distance`` (missing when the
        matched frame, or the second, holds no player or goalkeeper).
    """
    if not isinstance(freeze_frames, pd.DataFrame):
        freeze_frames = freeze_frames.to_freeze_frames(period=period)
    _require(statsbomb, [*_TIME, "pitch_location", "type"], "statsbomb")
    _require(freeze_frames, [*_TIME, "location"], "freeze_frames")
    reference = _prepare(statsbomb, "pitch_location", period)
    reference = reference[reference["_xy"].notna()]
    # Every object of every frame: the closest frame is picked among all of them.
    detected = _prepare(freeze_frames, "location", period)
    detected["_candidate"] = detected["_xy"].notna()
    if "type" in detected:
        detected["_candidate"] &= detected["type"].isin(_CANDIDATE_TYPES)
    by_frame = (
        "timestamp_seconds" in reference
        and "timestamp_seconds" in detected
        and "frame_id" in detected
    )

    matches = []
    groups = dict(list(detected.groupby(_TIME)))
    for key, rows in reference.groupby(_TIME):
        frames = groups.get(key)
        if frames is None:
            continue
        if by_frame:
            for _, event_rows in rows.groupby(
                rows["timestamp_seconds"] % 1, dropna=False
            ):
                event_time = event_rows["timestamp_seconds"].iloc[0] % 1
                matches.append(_nearest(event_rows, _closest_frame(frames, event_time)))
        else:
            matches.append(_nearest(rows, frames))

    columns = [*_DETECTED.values(), "euclidean_distance"]
    found = pd.concat(matches) if matches else pd.DataFrame(columns=columns)
    result = reference.drop(columns="_xy").join(found.reindex(columns=columns))
    result["detected_confidence"] = pd.to_numeric(result["detected_confidence"])
    result["euclidean_distance"] = pd.to_numeric(result["euclidean_distance"])
    for column in ("detected_player_id", "detected_frame_id"):
        ids = result[column]
        if ids.isna().all() or pd.api.types.is_numeric_dtype(ids):
            result[column] = ids.astype("Int64")

    matched_count = int(result["euclidean_distance"].notna().sum())
    logger.info(
        "Matched %d of %d StatsBomb objects in period %d (mean distance %.2f)",
        matched_count,
        len(result),
        period,
        result["euclidean_distance"].mean() if matched_count else np.nan,
    )
    return result.reset_index(drop=True)


def _closest_frame(frames: pd.DataFrame, event_time: float) -> pd.DataFrame:
    """The rows of the frame whose time within the second is closest to ``event_time``.

    A missing event time (NaN) keeps every frame of the second.
    """
    if not np.isfinite(event_time):
        return frames
    offsets = (frames["timestamp_seconds"] % 1 - event_time).abs()
    frame = frames["frame_id"].loc[offsets.idxmin()]
    return frames[frames["frame_id"] == frame]


def _nearest(rows: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The ``_DETECTED`` columns of the nearest candidate in ``frames`` per row, plus the distance.

    An empty table (the rows stay unmatched) when ``frames`` holds no candidate.
    """
    candidates = frames[frames["_candidate"]]
    if candidates.empty:
        return pd.DataFrame()
    distances = np.linalg.norm(
        np.stack(rows["_xy"].to_list())[:, None]
        - np.stack(candidates["_xy"].to_list())[None],
        axis=2,
    )
    nearest = distances.argmin(axis=1)
    found = candidates.iloc[nearest].reindex(columns=list(_DETECTED))
    found = found.rename(columns=_DETECTED).set_index(rows.index)
    found["euclidean_distance"] = distances[np.arange(len(rows)), nearest]
    return found


def _require(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in table]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _prepare(table: pd.DataFrame, location: str, period: int) -> pd.DataFrame:
    """Rows of ``period`` with integer time columns and a parsed ``_xy`` location.

    ``_xy`` is ``None`` where the location is missing or invalid.
    """
    table = table.reset_index(drop=True)
    for column in _TIME:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[table["period"] == period].dropna(subset=_TIME).copy()
    table[_TIME] = table[_TIME].astype(int)
    table["_xy"] = table[location].map(_parse_location)
    missing = int(table["_xy"].isna().sum())
    if missing:
        logger.debug("%d rows have no valid %r", missing, location)
    return table


def _parse_location(value: Any) -> np.ndarray | None:
    """``[x, y]`` from a list/array or its JSON string; ``None`` if absent or invalid."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if value is None or isinstance(value, float):  # missing values are None or NaN
        return None
    try:
        xy = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return None
    return xy
