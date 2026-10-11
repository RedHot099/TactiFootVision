"""Derived tables from the raw StatsBomb events: matches, shots, context events and timing events.

Pure functions of the raw JSON. Rows are plain dicts; `build_tables` assembles the frames in a fixed
order (by match id, then event index), so a rebuild from the same raw files gives the same tables.

shots     one row per Shot event (penalties and shoot-out kicks included, as in the prototype's build):
          flat columns, the raw event JSON and the freeze frame as a separate JSON column.
context   the Pass events that a shot names as its key pass, and Own Goal For/Against events: raw JSON.
timing    non-shot events a timing test needs, with timestamp, period, team and type: substitutions,
          fouls, offsides, set-piece passes (throw-in, corner, kick-off, free kick, goal kick),
          half starts and ends, stoppages, passes ruled offside, and every event that sent the ball out of play.
          `soccernet_label` names the SoccerNet label an event is comparable with
          (Substitution, Throw-in, Ball out of play, Corner, Kick-off, Foul, Offside), else None.
          A throw-in whose pass also went out keeps the label Throw-in; `ball_out` is True for it too.
"""

import json
from collections.abc import Iterable, Mapping

import pandas as pd

MATCH_COLUMNS = ["match_id", "match_date", "kick_off", "competition", "season", "home_team", "away_team", "home_score",
                 "away_score", "competition_id", "season_id", "home_team_id", "away_team_id", "match_status"]
SHOT_COLUMNS = ["match_id", "event_id", "index", "period", "timestamp", "minute", "second", "team", "player", "play_pattern",
                "x", "y", "under_pressure", "statsbomb_xg", "outcome", "body_part", "shot_type", "technique", "first_time",
                "key_pass_id", "n_freeze_frame", "event_json", "timestamp_ms", "team_id", "player_id", "possession",
                "freeze_frame_json"]
CONTEXT_COLUMNS = ["match_id", "event_id", "type", "event_json", "index", "period", "timestamp", "timestamp_ms", "team"]
TIMING_COLUMNS = ["match_id", "event_id", "index", "period", "timestamp", "timestamp_ms", "minute", "second", "type", "detail",
                  "team", "team_id", "player", "ball_out", "soccernet_label"]

TIMING_TYPES = {"Substitution": "Substitution", "Offside": "Offside", "Foul Committed": "Foul", "Foul Won": None,
                "Half Start": None, "Half End": None, "Injury Stoppage": None, "Referee Ball-Drop": None}
PASS_LABELS = {"Throw-in": "Throw-in", "Corner": "Corner", "Kick Off": "Kick-off", "Free Kick": None, "Goal Kick": None}
BALL_OUT_LABEL = "Ball out of play"
OFFSIDE_PASS = "Pass Offside"   # a pass to a player in an offside position: the whistle comes with it


def timestamp_ms(timestamp: str) -> int:
    """"00:09:59.015" -> 599015: milliseconds since the start of the period."""
    hours, minutes, seconds = timestamp.split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def dump(event: Mapping) -> str:
    return json.dumps(event, separators=(",", ":"))


def matches_table(match_records: Iterable[Mapping]) -> pd.DataFrame:
    rows = [{"match_id": m["match_id"], "match_date": m["match_date"], "kick_off": m["kick_off"],
             "competition": m["competition"]["competition_name"], "season": m["season"]["season_name"],
             "home_team": m["home_team"]["home_team_name"], "away_team": m["away_team"]["away_team_name"],
             "home_score": m["home_score"], "away_score": m["away_score"],
             "competition_id": m["competition"]["competition_id"], "season_id": m["season"]["season_id"],
             "home_team_id": m["home_team"]["home_team_id"], "away_team_id": m["away_team"]["away_team_id"],
             "match_status": m.get("match_status")} for m in match_records]
    return pd.DataFrame(rows, columns=MATCH_COLUMNS).sort_values("match_id").reset_index(drop=True)


def shot_row(match_id: int, event: Mapping) -> dict:
    shot = event["shot"]
    freeze = shot.get("freeze_frame", [])
    return {"match_id": match_id, "event_id": event["id"], "index": event["index"], "period": event["period"],
            "timestamp": event["timestamp"], "minute": event["minute"], "second": event["second"],
            "team": event["team"]["name"], "player": event["player"]["name"], "play_pattern": event["play_pattern"]["name"],
            "x": event["location"][0], "y": event["location"][1], "under_pressure": bool(event.get("under_pressure", False)),
            "statsbomb_xg": shot.get("statsbomb_xg"), "outcome": shot["outcome"]["name"], "body_part": shot["body_part"]["name"],
            "shot_type": shot["type"]["name"], "technique": shot["technique"]["name"],
            "first_time": bool(shot.get("first_time", False)), "key_pass_id": shot.get("key_pass_id"),
            "n_freeze_frame": len(freeze), "event_json": dump(event), "timestamp_ms": timestamp_ms(event["timestamp"]),
            "team_id": event["team"]["id"], "player_id": event["player"]["id"], "possession": event["possession"],
            "freeze_frame_json": dump(freeze) if "freeze_frame" in shot else None}


def context_row(match_id: int, event: Mapping) -> dict:
    return {"match_id": match_id, "event_id": event["id"], "type": event["type"]["name"], "event_json": dump(event),
            "index": event["index"], "period": event["period"], "timestamp": event["timestamp"],
            "timestamp_ms": timestamp_ms(event["timestamp"]), "team": event["team"]["name"]}


def timing_row(match_id: int, event: Mapping) -> dict | None:
    """The timing-event row of `event`, or None when it is not one the timing test uses."""
    kind = event["type"]["name"]
    detail, label, ball_out = "", None, bool(event.get("out", False))
    if kind in TIMING_TYPES:
        label = TIMING_TYPES[kind]
        if kind == "Substitution":
            detail = event["substitution"]["outcome"]["name"]
    elif kind == "Pass":
        pass_type = event["pass"].get("type", {}).get("name", "")
        outcome = event["pass"].get("outcome", {}).get("name", "")
        ball_out = ball_out or outcome == "Out"
        if pass_type in PASS_LABELS:
            detail, label = pass_type, PASS_LABELS[pass_type]
        elif outcome == OFFSIDE_PASS:
            detail, label = outcome, "Offside"
    if kind not in TIMING_TYPES and not detail and not ball_out:
        return None
    return {"match_id": match_id, "event_id": event["id"], "index": event["index"], "period": event["period"],
            "timestamp": event["timestamp"], "timestamp_ms": timestamp_ms(event["timestamp"]), "minute": event["minute"],
            "second": event["second"], "type": kind, "detail": detail, "team": event["team"]["name"],
            "team_id": event["team"]["id"], "player": event.get("player", {}).get("name"), "ball_out": ball_out,
            "soccernet_label": label or (BALL_OUT_LABEL if ball_out else None)}


def event_rows(match_id: int, events: list[Mapping]) -> tuple[list[dict], list[dict], list[dict]]:
    """(shots, context, timing) rows of one match, in event order."""
    key_ids = {e["shot"].get("key_pass_id") for e in events if e["type"]["name"] == "Shot"} - {None}
    shots, context, timing = [], [], []
    for event in sorted(events, key=lambda e: e["index"]):
        kind = event["type"]["name"]
        if kind == "Shot":
            shots.append(shot_row(match_id, event))
            continue
        if (kind == "Pass" and event["id"] in key_ids) or kind.startswith("Own Goal"):
            context.append(context_row(match_id, event))
        if (row := timing_row(match_id, event)) is not None:
            timing.append(row)
    return shots, context, timing


def build_tables(match_records: Iterable[Mapping], events_by_match: Iterable[tuple[int, list[Mapping]]]) -> dict[str, pd.DataFrame]:
    """Frames "matches", "shots", "context", "timing" from the match records and (match_id, events) pairs.
    The pairs are consumed one match at a time, in the order given (the caller sorts by match id)."""
    shots, context, timing = [], [], []
    for match_id, events in events_by_match:
        s, c, t = event_rows(match_id, events)
        shots += s
        context += c
        timing += t
    return {"matches": matches_table(match_records),
            "shots": pd.DataFrame(shots, columns=SHOT_COLUMNS),
            "context": pd.DataFrame(context, columns=CONTEXT_COLUMNS),
            "timing": pd.DataFrame(timing, columns=TIMING_COLUMNS)}
