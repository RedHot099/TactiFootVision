"""Probe targets of xg-dino in metres (ticket 04; spec §Probe targets, council C1 and C2, which win where they differ).

One definition (`position_targets`) on aligned positions (attacked goal at x = 105, centre y = 34, `conventions`),
applied to two sources:

    automatic   the feet_t0 E3 frames (`automatic_targets`): the t0 frame replayed with the frozen feet_t0 rule
                (`location.FEET_T0`: its ball order and `feet_situation`), every role as `location.situation_features`
                picks it. Probes are fitted on these targets only. Label-free; runs inside `no_statsbomb_reads()`.
    statsbomb   the freeze frames of the paired shots (`statsbomb_targets`), converted with the existing goal-centred
                `conventions.statsbomb_to_metres`, roles as the detector audit reads them (opponents `teammate` false,
                the goalkeeper the first opponent at position "Goalkeeper"). Evaluation only: nothing is fitted to
                it (`probes` refuses a table whose `attrs["source"]` is "statsbomb"), and it is the only StatsBomb
                read of xg-dino, so the guard refuses it inside the block.

Targets (`TARGETS`)
    distance             shot location to the goal centre (105, 34), m
    angle                angle subtended by the posts at the shot location, rad
    def_cone             outfield opponents (goalkeeper excluded) in the triangle shot - post - post, edges included;
                         count, and `def_cone_ge1` (>= 1, binary)
    near_opp_m           shot location to the nearest outfield opponent (goalkeeper excluded), m
    opp_3m               outfield opponents (goalkeeper excluded) within 3.0 m of the shot location, boundary inclusive;
                         count, and `opp_3m_ge1` (>= 1, binary)
    gk_depth_m           |105 - x| of the goalkeeper, m
    gk_lateral_m         |y - 34| of the goalkeeper (offset from the goal centre along the goal line), m; primary
    gk_lateral_signed_m  y - 34 in aligned coordinates (positive = the attacker's right), m; secondary

Missing is missing, never zero (C1): no shot location -> every target missing; unknown shooter team (`teams_ok` = 0)
or no freeze frame -> the opponent targets missing; no goalkeeper -> the goalkeeper targets missing. "No opponent
observed" (teams known, no outfield opponent on the pitch) is distinct: the counts are 0 and `near_opp_m` is missing
with `near_opp_status` "no_opponent" (else "observed" / "missing").

Goalkeeper: automatic = the detected goalkeeper nearest the attacked goal in its half, else the E3 proxy (the player
other than the shooter nearest the attacked goal; stated, C1); `gk_detected` / `gk_proxy` record which, so the
goalkeeper targets can be split. StatsBomb: the freeze-frame goalkeeper (`gk_detected` = 1).
"""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import NO_TEAM
from xgpoc.detector import location as loc
from xgpoc.detector.conventions import GOAL, POSTS, statsbomb_to_metres
from xgpoc.detector.features import in_cone

NEAR_M = 3.0
TARGETS = ("distance", "angle", "def_cone", "def_cone_ge1", "near_opp_m", "opp_3m", "opp_3m_ge1", "gk_depth_m",
           "gk_lateral_m", "gk_lateral_signed_m")
BINARY = ("def_cone_ge1", "opp_3m_ge1")
COUNTS = ("def_cone", "opp_3m")
OPPONENT = ("def_cone", "def_cone_ge1", "near_opp_m", "opp_3m", "opp_3m_ge1")
GOALKEEPER = ("gk_depth_m", "gk_lateral_m", "gk_lateral_signed_m")
STRATA = ("shot_ok", "teams_ok", "gk_ok", "gk_detected", "gk_proxy", "n_opponents", "near_opp_status")
COLUMNS = (*TARGETS, *STRATA)
SOURCES = ("automatic", "statsbomb")
PROBE_ONLY_SOURCES = ("caption", "gpt")           # header labels of xg-dino-info: probe targets, never a model design
BINARY_EXTRA: set[str] = set()                    # binary targets of xg-dino-info, registered by `register_binary`
GOALKEEPER_POSITION = "Goalkeeper"


def kind(target: str) -> str:
    """"binary" (AUC) or "continuous" (R², MAE; counts included)."""
    return "binary" if target in BINARY or target in BINARY_EXTRA else "continuous"


def register_binary(table: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Register the `columns` of `table` whose finite values are all 0 or 1 as binary targets (AUC); returns them."""
    found = [c for c in columns if table[c].notna().any() and table[c].dropna().isin([0.0, 1.0]).all()]
    BINARY_EXTRA.update(found)
    return found


def position_targets(shot, defenders=None, goalkeeper=None, *, gk_detected: bool = False) -> dict:
    """`COLUMNS` of one shot from aligned positions in metres (module docstring). `defenders`: `(N, 2)` outfield
    opponents, None when unknown (teams unknown or no freeze frame), empty when none is observed; `goalkeeper`: `(2,)`
    or None."""
    out = dict.fromkeys(TARGETS, np.nan) | {"shot_ok": 0.0, "teams_ok": 0.0, "gk_ok": 0.0, "gk_detected": 0.0,
                                            "gk_proxy": 0.0, "n_opponents": np.nan, "near_opp_status": "missing"}
    if shot is None or not np.isfinite(np.asarray(shot, float)).all():
        return out
    x, y = (float(v) for v in shot)
    a1, a2 = np.arctan2(POSTS[0][1] - y, POSTS[0][0] - x), np.arctan2(POSTS[1][1] - y, POSTS[1][0] - x)
    out |= {"shot_ok": 1.0, "distance": float(np.hypot(GOAL[0] - x, GOAL[1] - y)), "angle": float(abs(a2 - a1))}
    if defenders is not None:
        d = np.asarray(defenders, float).reshape(-1, 2)
        near = np.hypot(d[:, 0] - x, d[:, 1] - y)
        cone, close = float(in_cone(d, (x, y)).sum()), float((near <= NEAR_M).sum())
        out |= {"teams_ok": 1.0, "n_opponents": float(len(d)), "def_cone": cone, "def_cone_ge1": float(cone >= 1),
                "opp_3m": close, "opp_3m_ge1": float(close >= 1),
                "near_opp_m": float(near.min()) if len(d) else np.nan,
                "near_opp_status": "observed" if len(d) else "no_opponent"}
    if goalkeeper is not None:
        gx, gy = (float(v) for v in goalkeeper)
        out |= {"gk_ok": 1.0, "gk_detected": float(gk_detected), "gk_proxy": float(not gk_detected),
                "gk_depth_m": abs(GOAL[0] - gx), "gk_lateral_m": abs(gy - GOAL[1]), "gk_lateral_signed_m": gy - GOAL[1]}
    return out


def frame_targets(frame, rule: loc.LocationRule = loc.FEET_T0) -> dict:
    """`COLUMNS` of one filled frame (`location.filled_frame`) under `rule`; every target missing for None. The roles
    are those of `location.situation_features`: defenders = outfield players (shooter and goalkeeper proxy excluded)
    of a known team other than the shooter's."""
    if frame is None:
        return position_targets(None)
    s = rule.situation(frame)
    if s.shot is None:
        return position_targets(None)
    aligned = s.aligned(s.xy)
    defenders = None
    if s.shooter_team != NO_TEAM:
        outfield = s.outfield
        known = outfield[s.teams[outfield] != NO_TEAM]
        defenders = aligned[known[s.teams[known] != s.shooter_team]]
    goalkeeper = aligned[s.goalkeeper] if s.goalkeeper is not None else None
    return position_targets(s.aligned(s.shot), defenders, goalkeeper, gk_detected=s.goalkeeper is not None and not s.proxy)


def tagged(table: pd.DataFrame, source: str) -> pd.DataFrame:
    if source not in (*SOURCES, *PROBE_ONLY_SOURCES):
        raise ValueError(f"unknown target source {source}")
    table.attrs["source"] = source
    return table


def automatic_targets(frames: pd.DataFrame, shot_ids: Iterable[str], offset: int = 0,
                      rule: loc.LocationRule = loc.FEET_T0) -> pd.DataFrame:
    """Automatic targets (index shot_id, `COLUMNS`, sorted) of `shot_ids` at `offset` from the cached E3 frame rows
    (`frames.parquet` of an E3 cache), replayed under `rule`. Every shot must have frame rows."""
    ids = sorted(set(shot_ids))
    frames = frames[frames.shot_id.isin(ids)]
    if missing := set(ids) - set(frames.shot_id):
        raise ValueError(f"{len(missing)} shots have no E3 frame rows, e.g. {sorted(missing)[0]}")
    rows = [{"shot_id": shot_id} | frame_targets(filled, rule)
            for shot_id, o, _, filled, _ in loc.replay(frames, rule.source) if o == offset]
    return tagged(pd.DataFrame(rows).set_index("shot_id")[list(COLUMNS)], "automatic")


def freeze_frame_positions(frame: list[dict]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(outfield opponents, goalkeeper) of a StatsBomb freeze frame in aligned metres; (None, None) without one."""
    if not frame:
        return None, None
    opponents = [p for p in frame if not p["teammate"]]
    is_gk = np.array([(p.get("position") or {}).get("name") == GOALKEEPER_POSITION for p in opponents], bool)
    xy = statsbomb_to_metres(np.array([p["location"][:2] for p in opponents], float).reshape(-1, 2))
    return xy[~is_gk], (xy[is_gk][0] if is_gk.any() else None)


def statsbomb_table_targets(pop: pd.DataFrame, freeze_frames: pd.Series) -> pd.DataFrame:
    """StatsBomb targets (index shot_id) of a paired population (`shot_id`, `sb_event_id`, `sb_x`, `sb_y`) given the
    freeze-frame JSON by event id. Pure: no file is read."""
    rows = []
    for shot in pop.itertuples(index=False):
        raw = freeze_frames.get(shot.sb_event_id)
        defenders, goalkeeper = freeze_frame_positions(json.loads(raw) if isinstance(raw, str) else [])
        rows.append({"shot_id": shot.shot_id} | position_targets(statsbomb_to_metres([shot.sb_x, shot.sb_y]), defenders,
                                                                 goalkeeper, gk_detected=goalkeeper is not None))
    return tagged(pd.DataFrame(rows).set_index("shot_id")[list(COLUMNS)], "statsbomb")


def statsbomb_targets(shot_ids: Iterable[str], pairs: Path | None = None, shots: Path | None = None) -> pd.DataFrame:
    """StatsBomb targets of the paired shots `shot_ids` (evaluation only; the one StatsBomb reader of xg-dino). Reads
    the pairing and the freeze frames through the detector audit's readers (named columns, no outcome); refused by
    the guard inside `no_statsbomb_reads()`."""
    from xgpoc.detector import (
        audit,  # the audit's readers; imported here so no fitting module imports it
    )

    pop = audit.read_population(pairs or audit.PAIRS).set_index("shot_id")
    ids = sorted(set(shot_ids))
    if missing := set(ids) - set(pop.index):
        raise ValueError(f"{len(missing)} shots are not paired non-penalty shots, e.g. {sorted(missing)[0]}")
    pop = pop.loc[ids].reset_index()
    return statsbomb_table_targets(pop, audit.read_freeze_frames(shots or audit.SHOTS, pop.sb_event_id))


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Per target: shots, present, share; for near_opp_m also the "no_opponent" count; GK targets by detected/proxy."""
    rows = []
    for t in TARGETS:
        present = table[t].notna()
        row = {"target": t, "shots": len(table), "present": int(present.sum()), "share": float(present.mean())}
        if t == "near_opp_m":
            row["no_opponent"] = int((table.near_opp_status == "no_opponent").sum())
        if t in GOALKEEPER:
            row["gk_detected"] = int((present & (table.gk_detected == 1)).sum())
            row["gk_proxy"] = int((present & (table.gk_proxy == 1)).sum())
        rows.append(row)
    return pd.DataFrame(rows)
