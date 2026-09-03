"""Parse SoccerNet Labels-v2.json files into shot events with binary goal outcome.

SoccerNet action spotting labels of interest: "Shots on target", "Shots off target",
"Goal", "Penalty". A "Goal" annotation marks the moment the ball crosses the line; the
strike that scored is the *nearest preceding* shot annotation of the same team (on the
500 games the gap is 0.6 s median, 2 s at p99). Earlier shots inside the window are
saves, blocks or posts followed by a rebound and must stay goal=False. Crediting the
earliest shot instead - the rule used up to index v1.1 - flipped ~200 labels in rebound
chains (see research/review_2026-09-02.md, D1). Goals without any preceding shot
annotation become their own ShotEvent at goal_time - `goal_backoff_ms`; the collector
excludes them from the trusted index.

A "Penalty" annotation shares its timestamp with the penalty kick's shot annotation
(the gap is 0 ms up to p90), so the flag goes to the single nearest same-team shot in
[-penalty_lead_ms, +penalty_window_ms]. The 60 s window used before v1.2 also flagged
rebounds and the next attack.
"""

import json
from dataclasses import asdict, dataclass
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
                 penalty_window_ms: int = 2000, penalty_lead_ms: int = 1000) -> list[ShotEvent]:
    game = labels_path.parent.as_posix()
    data = json.loads(labels_path.read_text())
    parsed = []
    for a in data["annotations"]:
        half, pos = _parse_annotation(a)
        parsed.append((half, pos, a["label"], a.get("team", "not applicable"), a.get("visibility", "unknown")))
    parsed.sort(key=lambda x: (x[0], x[1]))

    shots = [x for x in parsed if x[2] in SHOT_LABELS]
    goals = [(h, p, t) for h, p, lbl, t, _ in parsed if lbl == GOAL_LABEL]
    penalties = [(h, p, t) for h, p, lbl, t, _ in parsed if lbl == PENALTY_LABEL]

    # goal -> nearest preceding same-team shot that is not already credited
    credited: dict[int, tuple[int, int, str]] = {}
    for g in goals:
        cands = [i for i, s in enumerate(shots)
                 if s[0] == g[0] and s[3] == g[2] and 0 <= g[1] - s[1] <= goal_window_ms and i not in credited]
        if cands:
            credited[max(cands, key=lambda i: shots[i][1])] = g

    # penalty label -> the single nearest same-team shot around its timestamp
    penalty_shots: set[int] = set()
    for p in penalties:
        cands = [i for i, s in enumerate(shots)
                 if s[0] == p[0] and s[3] == p[2] and -penalty_lead_ms <= s[1] - p[1] <= penalty_window_ms]
        if cands:
            penalty_shots.add(min(cands, key=lambda i: abs(shots[i][1] - p[1])))

    events = [ShotEvent(game, half, pos, label, team, vis, i in credited, i in penalty_shots, "shot")
              for i, (half, pos, label, team, vis) in enumerate(shots)]

    used = set(credited.values())
    for g in goals:
        if g in used:
            continue
        half, pos, team = g
        is_pen = any(p[0] == half and p[2] == team and -penalty_lead_ms <= pos - p[1] <= goal_window_ms for p in penalties)
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
