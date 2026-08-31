"""Parse SoccerTrack v2 BAS (12-class) event files into shot attempts.

Follows the codex-branch audit conclusions (docs/COMPLETE_XG_DATASET_GUIDE_PL.md on
branch codex/xg-video-poc):
  * `GOAL` is evidence of the outcome, never the contact timestamp;
  * `SHOT` (and flagged `HEADER` / `FREE KICK`) are attempt candidates and frame anchors;
  * a goal is credited to the nearest preceding same-team candidate within
    `goal_window_ms` with no other same-team candidate in between;
  * goals with no such candidate become `weak` records without a frame anchor
    (excluded from the trusted set, never anchored via a fixed backoff);
  * `position` is a global match clock in ms; second-half video time needs a
    nominal 45-minute subtraction.
"""

import json
from dataclasses import dataclass
from pathlib import Path

HALF_OFFSET_MS = 45 * 60 * 1000
CANDIDATE_LABELS = {"SHOT", "HEADER", "FREE KICK"}


@dataclass(frozen=True)
class Attempt:
    match_id: str
    half: int
    position_ms: int      # global match clock
    video_ms: int         # clock within the half video file
    label: str
    team: str
    player_id: str | None
    goal: bool
    trusted: bool
    reason: str           # "" | "goal_without_candidate"


def _half(action: dict) -> int:
    gt = action["gameTime"]
    if " - " in gt:
        return int(gt.split(" - ")[0])
    # Some entries (stoppage time past 90:00) lack the half prefix; infer from the
    # global clock. Events past the released video length fail frame extraction and
    # end up in the weak bucket, which is the intended accounting.
    return 1 if int(action["position"]) < HALF_OFFSET_MS else 2


def parse_bas(path: Path, goal_window_ms: int = 10000) -> list[Attempt]:
    data = json.loads(path.read_text())
    match_id = str(data["match_id"])
    actions = sorted(data["actions"], key=lambda a: (_half(a), int(a["position"])))
    candidates = [a for a in actions if a["label"] in CANDIDATE_LABELS]
    goals = [a for a in actions if a["label"] == "GOAL"]

    def video_ms(half: int, pos: int) -> int:
        return pos - HALF_OFFSET_MS if half == 2 else pos

    matched_goals: set[int] = set()
    attempts: list[Attempt] = []
    for c in candidates:
        ch, cp = _half(c), int(c["position"])
        goal = False
        for gi, g in enumerate(goals):
            gh, gp = _half(g), int(g["position"])
            if gi in matched_goals or gh != ch or g.get("team") != c.get("team"):
                continue
            if not 0 <= gp - cp <= goal_window_ms:
                continue
            between = [x for x in candidates if x is not c and _half(x) == ch
                       and x.get("team") == c.get("team") and cp < int(x["position"]) < gp]
            if between:
                continue  # a later candidate of the same team will claim this goal
            goal = True
            matched_goals.add(gi)
            break
        attempts.append(Attempt(match_id, ch, cp, video_ms(ch, cp), c["label"], c.get("team", "?"),
                                c.get("player_id"), goal, True, ""))
    for gi, g in enumerate(goals):
        if gi in matched_goals:
            continue
        gh, gp = _half(g), int(g["position"])
        attempts.append(Attempt(match_id, gh, gp, video_ms(gh, gp), "GOAL", g.get("team", "?"),
                                g.get("player_id"), True, False, "goal_without_candidate"))
    attempts.sort(key=lambda a: (a.half, a.position_ms))
    return attempts
