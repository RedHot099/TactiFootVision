"""StatsBomb 360 freeze frames as a flat table (one row per player seen in a frame)."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EVENT_COLUMNS = [
    "event_uuid",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "type_name",
    "possession",
    "possession_team_name",
    "play_pattern_name",
    "team_name",
    "player_name",
    "under_pressure",
]
_OUTPUT_COLUMNS = [
    *_EVENT_COLUMNS[:4],
    "timestamp_seconds",
    *_EVENT_COLUMNS[4:],
    "pitch_location",
    "teammate",
    "actor",
    "keeper",
    "type",
    "visible_area",
]


def load_statsbomb(path: str | Path) -> pd.DataFrame:
    """Load StatsBomb 360 data for one match.

    ``path`` is either a folder with ``*_events.json`` and ``*_360.json`` files
    (StatsBomb open-data layout) or a CSV previously written from this table.

    Returns one row per freeze-frame object with the event context, the object's
    ``pitch_location`` ``[x, y]`` in StatsBomb units (120 x 80), flags
    ``teammate`` / ``actor`` / ``keeper``, ``type`` (``"player"`` or
    ``"goalkeeper"``) and the event's ``visible_area`` polygon (StatsBomb's flat
    ``[x1, y1, x2, y2, ...]`` list, like ``PipelineResult.to_freeze_frames``).
    """
    path = Path(path)
    if path.suffix == ".csv":
        table = pd.read_csv(path)
        for column in ("pitch_location", "visible_area"):
            if column in table:
                table[column] = table[column].map(
                    lambda v: json.loads(v) if isinstance(v, str) else None
                )
        return table
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a StatsBomb folder or CSV, got {path}")

    events = _read_json_lists(sorted(path.glob("*_events.json")))
    frames = _read_json_lists(sorted(path.glob("*_360.json")))
    if not events or not frames:
        raise FileNotFoundError(f"{path} needs *_events.json and *_360.json files")

    events_table = pd.json_normalize(events, sep="_").rename(
        columns={"id": "event_uuid"}
    )
    events_table = events_table[[c for c in _EVENT_COLUMNS if c in events_table]]
    events_table["timestamp_seconds"] = pd.to_timedelta(
        events_table["timestamp"]
    ).dt.total_seconds()

    objects = (
        pd.DataFrame(frames)[["event_uuid", "visible_area", "freeze_frame"]]
        .explode("freeze_frame", ignore_index=True)
        .dropna(subset=["freeze_frame"])
    )
    details = pd.json_normalize(objects["freeze_frame"].tolist())
    objects = pd.concat(
        [objects.drop(columns="freeze_frame").reset_index(drop=True), details], axis=1
    )
    objects = objects.rename(columns={"location": "pitch_location"})
    objects["type"] = objects["keeper"].map({True: "goalkeeper", False: "player"})

    table = objects.merge(events_table, on="event_uuid", how="left")
    logger.info(
        "Loaded %d freeze-frame objects from %d events", len(table), len(frames)
    )
    return table[[c for c in _OUTPUT_COLUMNS if c in table]]


def _read_json_lists(files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{file} should contain a JSON list")
        records += data
    return records
