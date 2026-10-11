"""Reasons for the unpaired SoccerNet shots (xg-vendor ticket 04, descriptive; the verdict does not read them).

Every unpaired SoccerNet index shot of ticket 02 gets the nearest StatsBomb event of any type from the raw events of
its match and half (signed dt = event - shot on the StatsBomb clock), the last play-stopping event before it, and the
first reason of `REASONS` that applies (the rule note `research/xg_vendor_timing_rule.md`, "Opis bez wpływu na
werdykt"):

    no_half_offset            guard: the half has no half offset (none in the frozen pairing; not in the note's list)
    sb_shot_within_10s        a StatsBomb shot lies within 10 s
    dead_ball                 the last play-stopping event lies at most 10 s before the shot and play has not restarted
    outside_half              the shot lies before Half Start or after Half End of the StatsBomb period
    other_event_within_2_5s   some StatsBomb event lies within 2.5 s
    no_event_within_2_5s      otherwise
"""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from xgpoc.vendor.statsbomb_tables import timestamp_ms

SHOT_NEAR_MS = 10_000
DEAD_BALL_MS = 10_000
EVENT_NEAR_MS = 2_500
REASONS = ("no_half_offset", "sb_shot_within_10s", "dead_ball", "outside_half", "other_event_within_2_5s", "no_event_within_2_5s")
STOP_TYPES = frozenset({"Offside", "Foul Committed", "Injury Stoppage", "Half End"})
RESTART_PASSES = frozenset({"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"})
RESTART_TYPES = frozenset({"Referee Ball-Drop", "Half Start"})
OFFSIDE_PASS = "Pass Offside"
EVENT_COLUMNS = ["match_id", "event_id", "index", "period", "timestamp_ms", "type", "detail", "team", "ball_out", "stops_play",
                 "restarts_play"]


def event_frame(match_id: int, events: Iterable[Mapping]) -> pd.DataFrame:
    """Every raw StatsBomb event of one match: time, type, detail, team and whether it stops or restarts play.

    detail: the pass type (else its outcome), the shot outcome, else empty. ball_out: `out` true, a pass with outcome
    Out or a shot with outcome Off T. stops_play: Offside, Foul Committed, Injury Stoppage, Half End, a ball-out event
    or a pass with outcome Pass Offside. restarts_play: a Throw-in, Corner, Free Kick, Goal Kick or Kick Off pass, a
    Referee Ball-Drop or a Half Start."""
    rows = []
    for e in events:
        kind = e["type"]["name"]
        pass_type = e.get("pass", {}).get("type", {}).get("name", "") if kind == "Pass" else ""
        outcome = (e.get("pass") or e.get("shot") or {}).get("outcome", {}).get("name", "") if kind in ("Pass", "Shot") else ""
        ball_out = bool(e.get("out", False)) or (kind == "Pass" and outcome == "Out") or (kind == "Shot" and outcome == "Off T")
        rows.append({"match_id": match_id, "event_id": e["id"], "index": e["index"], "period": e["period"],
                     "timestamp_ms": timestamp_ms(e["timestamp"]), "type": kind, "detail": pass_type or outcome,
                     "team": e.get("team", {}).get("name"), "ball_out": ball_out,
                     "stops_play": kind in STOP_TYPES or ball_out or (kind == "Pass" and outcome == OFFSIDE_PASS),
                     "restarts_play": pass_type in RESTART_PASSES or kind in RESTART_TYPES})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _reason(clock: float, nearest_shot_ms: float, ev: pd.DataFrame) -> tuple[str, dict]:
    """(reason, details) of one unpaired shot at `clock` on the StatsBomb clock; `ev` its half's events in order."""
    details = {"nearest_event_id": pd.NA, "nearest_event_type": pd.NA, "nearest_event_detail": pd.NA, "nearest_event_team": pd.NA,
               "nearest_event_dt_ms": np.nan, "stoppage_type": pd.NA, "stoppage_dt_ms": np.nan}
    dead = False
    if np.isnan(clock):
        return "no_half_offset", details
    if len(ev):
        dt = ev.timestamp_ms.to_numpy(dtype=float) - clock
        k = int(np.abs(dt).argmin())
        details |= {"nearest_event_id": ev.event_id.iat[k], "nearest_event_type": ev.type.iat[k], "nearest_event_detail": ev.detail.iat[k],
                    "nearest_event_team": ev.team.iat[k], "nearest_event_dt_ms": float(dt[k])}
        before = ev[ev.timestamp_ms.le(clock)]
        stops = before[before.stops_play]
        if len(stops):
            last = stops.iloc[-1]
            restarted = before.restarts_play & (before["index"] > last["index"])
            details |= {"stoppage_type": last.type if last.type != "Pass" else f"Pass ({last.detail})",
                        "stoppage_dt_ms": float(last.timestamp_ms - clock)}
            dead = clock - last.timestamp_ms <= DEAD_BALL_MS and not restarted.any()
    if nearest_shot_ms <= SHOT_NEAR_MS:
        return "sb_shot_within_10s", details
    if dead:
        return "dead_ball", details
    start, end = ev.timestamp_ms[ev.type.eq("Half Start")], ev.timestamp_ms[ev.type.eq("Half End")]
    if (len(start) and clock < start.min()) or (len(end) and clock > end.max()):
        return "outside_half", details
    if len(ev) and abs(details["nearest_event_dt_ms"]) <= EVENT_NEAR_MS:
        return "other_event_within_2_5s", details
    return "no_event_within_2_5s", details


def unpaired_reasons(unpaired_sn: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per unpaired SoccerNet index shot: the nearest StatsBomb event of any type (signed dt = event - shot on
    the StatsBomb clock), the last play-stopping event before it and the reason (`REASONS`, first that applies)."""
    events = events.sort_values(["match_id", "period", "index"])
    halves = {key: part.reset_index(drop=True) for key, part in events.groupby(["match_id", "period"])}
    empty = events.iloc[:0]
    rows = []
    for shot in unpaired_sn.itertuples():
        ev = halves.get((int(shot.sb_match_id), int(shot.half)), empty) if not pd.isna(shot.sb_match_id) else empty
        nearest_shot = np.inf if pd.isna(shot.nearest_sb_ms) else float(shot.nearest_sb_ms)
        reason, details = _reason(float(shot.sb_clock_ms), nearest_shot, ev)
        rows.append({"shot_id": shot.shot_id, "game": shot.game, "split": shot.split, "half": shot.half, "position_ms": shot.position_ms,
                     "label": shot.label, "team": shot.team, "goal": shot.goal, "sb_match_id": shot.sb_match_id,
                     "sb_clock_ms": shot.sb_clock_ms, "nearest_sb_shot_ms": shot.nearest_sb_ms,
                     "no_sb_shot_within_10s": not nearest_shot <= SHOT_NEAR_MS} | details | {"reason": reason})
    table = pd.DataFrame(rows)
    return table.astype({"sb_match_id": "Int64"}) if len(table) else table


def reason_summary(reasons: pd.DataFrame) -> dict:
    """Counts of the reasons, for all unpaired shots and for those without a StatsBomb shot within 10 s."""
    far_from_shots = reasons[reasons.no_sb_shot_within_10s] if len(reasons) else reasons
    return {"unpaired_shots": int(len(reasons)), "unpaired_shots_without_sb_shot_within_10s": int(len(far_from_shots)),
            "reasons": reasons.reason.value_counts().to_dict() if len(reasons) else {},
            "reasons_without_sb_shot_within_10s": far_from_shots.reason.value_counts().to_dict() if len(far_from_shots) else {}}
