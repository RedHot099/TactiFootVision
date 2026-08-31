"""Parse SoccerNet Labels-v2.json files into shot events with binary goal outcome.

SoccerNet action spotting labels of interest: "Shots on target", "Shots off target",
"Goal", "Penalty". A "Goal" annotation marks the moment the ball crosses the line; the
preceding shot annotation (if any) marks the strike. We build one ShotEvent per shot
and label it goal=True when a "Goal" of the same team follows within `goal_window_ms`.
Goals without a preceding shot annotation become their own ShotEvent at goal_time -
`goal_backoff_ms` (the strike happened shortly before the ball crossed the line).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

SHOT_LABELS = {"Shots on target", "Shots off target"}
GOAL_LABEL = "Goal"
PENALTY_LABEL = "Penalty"


@dataclass(frozen=True)
class ShotEvent:
    game: str
    half: int
    position_ms: int
    label: str
    team: str
    visibility: str
    goal: bool
    is_penalty: bool
    source: str  # "shot" or "goal_only"


def _parse_annotation(ann: dict) -> tuple[int, int]:
    half = int(ann["gameTime"].split(" - ")[0])
    return half, int(ann["position"])


def parse_labels(labels_path: Path, goal_window_ms: int = 6000, goal_backoff_ms: int = 1000,
                 penalty_window_ms: int = 60000) -> list[ShotEvent]:
    game = labels_path.parent.as_posix()
    data = json.loads(labels_path.read_text())
    anns = data["annotations"]
    parsed = []
    for a in anns:
        half, pos = _parse_annotation(a)
        parsed.append((half, pos, a["label"], a.get("team", "not applicable"), a.get("visibility", "unknown")))
    parsed.sort(key=lambda x: (x[0], x[1]))

    goals = [(h, p, t) for h, p, lbl, t, _ in parsed if lbl == GOAL_LABEL]
    penalties = [(h, p, t) for h, p, lbl, t, _ in parsed if lbl == PENALTY_LABEL]
    used_goals: set[tuple[int, int, str]] = set()
    events: list[ShotEvent] = []

    for half, pos, label, team, vis in parsed:
        if label not in SHOT_LABELS:
            continue
        goal = False
        for g in goals:
            if g[0] == half and g[2] == team and 0 <= g[1] - pos <= goal_window_ms and g not in used_goals:
                goal = True
                used_goals.add(g)
                break
        is_pen = any(p[0] == half and p[2] == team and 0 <= pos - p[1] <= penalty_window_ms for p in penalties)
        events.append(ShotEvent(game, half, pos, label, team, vis, goal, is_pen, "shot"))

    for g in goals:
        if g in used_goals:
            continue
        half, pos, team = g
        is_pen = any(p[0] == half and p[2] == team and 0 <= pos - p[1] <= penalty_window_ms for p in penalties)
        vis = next((v for h, p, lbl, t, v in parsed if (h, p, t) == g and lbl == GOAL_LABEL), "unknown")
        events.append(ShotEvent(game, half, max(0, pos - goal_backoff_ms), GOAL_LABEL, team, vis, True, is_pen, "goal_only"))

    events.sort(key=lambda e: (e.half, e.position_ms))
    return events


def build_shot_table(root: Path, **kwargs) -> pd.DataFrame:
    rows = []
    for labels_path in sorted(root.rglob("Labels-v2.json")):
        for ev in parse_labels(labels_path, **kwargs):
            d = asdict(ev)
            d["game"] = labels_path.parent.relative_to(root).as_posix()
            d["split"] = d["game"].split("/")[0] if "/" in d["game"] else ""
            rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df["shot_id"] = (df["game"].str.replace("/", "__") + "__" + df["half"].astype(str) + "__" + df["position_ms"].astype(str))
    return df
