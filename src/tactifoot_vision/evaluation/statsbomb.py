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
_PERIOD_START_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105}  # StatsBomb's match clock
# freeze-frame column -> column added to the StatsBomb table
_DETECTED = {
    "location": "detected_location",
    "type": "detected_type",
    "player_id": "detected_player_id",
    "frame_bbox": "detected_frame_bbox",
    "confidence": "detected_confidence",
    "visible_area": "detected_visible_area",
}


def compare_with_statsbomb(
    freeze_frames: "pd.DataFrame | PipelineResult",
    statsbomb: pd.DataFrame,
    period: int = 1,
) -> pd.DataFrame:
    """Match every StatsBomb 360 object to the nearest detected object in the same second.

    Rows of both tables are grouped by ``(period, minute, second)``; each
    StatsBomb object of ``period`` gets the closest detected object of its
    group (Euclidean distance between pitch locations).

    Both tables must use the same pitch coordinates: build the pipeline with
    ``SoccerPitch(120, 80)`` to compare with StatsBomb units. A
    :class:`PipelineResult` is converted with ``to_freeze_frames``, assuming the
    video starts at the period's kick-off on StatsBomb's clock (0', 45', 90',
    105'); otherwise pass ``result.to_freeze_frames(period, period_start=...)``.

    Args:
        freeze_frames: pipeline freeze frames (columns ``period, minute, second,
            location`` plus optionally ``type, player_id, frame_bbox, confidence,
            visible_area``) or the :class:`PipelineResult` producing them.
        statsbomb: StatsBomb objects with ``period, minute, second,
            pitch_location, type`` (see :func:`tactifoot_vision.data.load_statsbomb`);
            locations may be ``[x, y]`` lists or JSON strings.
        period: match period to compare.

    Returns:
        The StatsBomb rows of ``period`` that have a location, with
        ``detected_location``, ``detected_type``, ``detected_player_id``,
        ``detected_frame_bbox``, ``detected_confidence``,
        ``detected_visible_area`` and ``euclidean_distance`` (missing when
        nothing was detected in that second).
    """
    if not isinstance(freeze_frames, pd.DataFrame):
        kick_off = _PERIOD_START_MINUTE.get(period, 0) * 60.0
        freeze_frames = freeze_frames.to_freeze_frames(
            period=period, period_start=kick_off
        )
    _require(statsbomb, [*_TIME, "pitch_location", "type"], "statsbomb")
    _require(freeze_frames, [*_TIME, "location"], "freeze_frames")
    reference = _prepare(statsbomb, "pitch_location", period)
    detected = _prepare(freeze_frames, "location", period)

    matches = []
    groups = dict(list(detected.groupby(_TIME)))
    for key, rows in reference.groupby(_TIME):
        candidates = groups.get(key)
        if candidates is None:
            continue
        distances = np.linalg.norm(
            np.stack(rows["_xy"].to_list())[:, None]
            - np.stack(candidates["_xy"].to_list())[None],
            axis=2,
        )
        nearest = distances.argmin(axis=1)
        found = candidates.iloc[nearest].reindex(columns=list(_DETECTED))
        found = found.rename(columns=_DETECTED).set_index(rows.index)
        found["euclidean_distance"] = distances[np.arange(len(rows)), nearest]
        matches.append(found)

    columns = [*_DETECTED.values(), "euclidean_distance"]
    found = pd.concat(matches) if matches else pd.DataFrame(columns=columns)
    result = reference.drop(columns="_xy").join(found.reindex(columns=columns))
    result["detected_confidence"] = pd.to_numeric(result["detected_confidence"])
    result["euclidean_distance"] = pd.to_numeric(result["euclidean_distance"])
    ids = result["detected_player_id"]
    if ids.isna().all() or pd.api.types.is_numeric_dtype(ids):
        result["detected_player_id"] = ids.astype("Int64")

    matched_count = int(result["euclidean_distance"].notna().sum())
    logger.info(
        "Matched %d of %d StatsBomb objects in period %d (mean distance %.2f)",
        matched_count,
        len(result),
        period,
        result["euclidean_distance"].mean() if matched_count else np.nan,
    )
    return result.reset_index(drop=True)


def _require(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in table]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _prepare(table: pd.DataFrame, location: str, period: int) -> pd.DataFrame:
    """Rows of ``period`` with integer time columns and a parsed ``_xy`` location."""
    table = table.reset_index(drop=True)
    for column in _TIME:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[table["period"] == period].dropna(subset=_TIME)
    table[_TIME] = table[_TIME].astype(int)
    table["_xy"] = table[location].map(_parse_location)
    missing = int(table["_xy"].isna().sum())
    if missing:
        logger.debug("Ignoring %d rows without a valid %r", missing, location)
    return table[table["_xy"].notna()]


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
