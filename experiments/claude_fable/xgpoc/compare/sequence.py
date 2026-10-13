"""Rebounds from the raw StatsBomb event sequence (xg-compare ticket 01; spec item 2).

`previous_shots` is a pure function of a shots table with the columns `SHOT_COLUMNS` (one row per StatsBomb Shot event,
penalties and every other shot of the match included). For each shot it finds the previous shot: the latest Shot event
with a smaller event index in the same match, period and StatsBomb possession. It reports that shot's id, the time
since it (ms, both timestamps within the period), whether it was by the same team and its StatsBomb outcome.

A shot is a rebound within w seconds when its previous shot exists, is by the same team and lies 0 to w seconds
earlier (`rebound`). The C4 design uses w = 5 (3 as a sensitivity check). The previous shot's outcome is information
from before the current shot. It names the subtype (after a save, after a block) and is never the current shot's
label.

`load_shots` reads the raw events of the given matches from the pinned store (`xgpoc.paths.STATSBOMB`) through
`xgpoc.vendor.statsbomb_fetch.load_raw`, which checks every file against its recorded hashes. It keeps the shot rows
of `xgpoc.vendor.statsbomb_tables.event_rows`.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.vendor import statsbomb_fetch as fetching
from xgpoc.vendor import statsbomb_tables as tables

SHOT_COLUMNS = ["match_id", "event_id", "index", "period", "timestamp_ms", "minute", "team_id", "possession", "outcome"]
REBOUND_SECONDS = 5.0
SENSITIVITY_SECONDS = 3.0
SEQUENCE_COLUMNS = ["event_id", "minute", "prev_event_id", "prev_dt_ms", "prev_same_team", "prev_outcome", "rebound_5s", "rebound_3s"]


def load_shots(root: Path, match_ids: Iterable[int]) -> pd.DataFrame:
    """`SHOT_COLUMNS` of every Shot event of the given matches, from the verified raw event files."""
    sha = fetching.read_pin(root)
    if sha is None:
        raise ValueError(f"{root} has no pin.json")
    rows = []
    for match_id in sorted(set(int(m) for m in match_ids)):
        events = json.loads(fetching.load_raw(fetching.events_path(root, match_id), sha, f"data/events/{match_id}.json"))
        rows += tables.event_rows(match_id, events)[0]
    return pd.DataFrame(rows, columns=tables.SHOT_COLUMNS)[SHOT_COLUMNS]


def previous_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Per shot event, in the input order: `event_id`, `minute` and the previous shot's id, time gap, same-team flag
    and outcome (module docstring); NaN / None when the shot opens its possession's shots."""
    missing = [c for c in SHOT_COLUMNS if c not in shots.columns]
    if missing:
        raise ValueError(f"shots table lacks {missing}")
    if shots.event_id.duplicated().any():
        raise ValueError("duplicate event_id in the shots table")
    if shots.duplicated(["match_id", "index"]).any():
        raise ValueError("two shots share a match and an event index")
    ordered = shots.sort_values(["match_id", "period", "possession", "index"])
    groups = ordered.groupby(["match_id", "period", "possession"], sort=False)
    out = pd.DataFrame({"event_id": ordered.event_id, "minute": ordered.minute, "prev_event_id": groups.event_id.shift(),
                        "prev_dt_ms": ordered.timestamp_ms - groups.timestamp_ms.shift(),
                        "prev_same_team": ordered.team_id.eq(groups.team_id.shift()), "prev_outcome": groups.outcome.shift()})
    out["prev_dt_ms"] = out.prev_dt_ms.astype(float)
    return out.loc[shots.index].reset_index(drop=True)


def rebound(previous: pd.DataFrame, seconds: float) -> np.ndarray:
    """Whether each shot of `previous_shots` is a rebound within `seconds`."""
    dt = previous.prev_dt_ms.to_numpy(float)
    return previous.prev_event_id.notna().to_numpy() & previous.prev_same_team.to_numpy(bool) & (dt >= 0) & (dt <= 1000 * seconds)


def sequence_table(shots: pd.DataFrame) -> pd.DataFrame:
    """`SEQUENCE_COLUMNS` per shot event: `previous_shots` plus the 5 s and 3 s rebound flags."""
    previous = previous_shots(shots)
    return previous.assign(rebound_5s=rebound(previous, REBOUND_SECONDS), rebound_3s=rebound(previous, SENSITIVITY_SECONDS))[SEQUENCE_COLUMNS]
