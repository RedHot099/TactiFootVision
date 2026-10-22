"""Shot-location rules: who shoots and where the shot is placed (2026-10-22 error decomposition; frozen rule
`research/feet_location_rule.md`).

A rule pairs a ball-source order (`extract3.SourceConfig`: which frame's ball a feature row uses) with a situation
function (`features.situation` or `feet_situation`: the shooter and the shot location given that ball), and names the
offsets at which a stack table takes E3's ball-source columns instead of its own (`e3_flags_at`, see
`xgpoc.stack_location.inputs`).

    E3        xg-detector's rule as frozen: E3's order (-400, -800, 0); the shooter is the player whose PITCH position is
              nearest the ball, the shot location is the ball.
    feet_t0   order (0, -400, -800), so the t0 row's shooter is picked by the t0 frame's own ball first; the shooter is
              the player whose IMAGE box is nearest the ball's image centre (`image_shooter`), and the shot location is
              that player's projected feet. A stack table keeps E3's t0 ball-source columns (`e3_flags_at` = (0,)):
              the development ablation kept the dd2c t0 ball flags, so the gain is the location's, not the flags'.

Without a ball, or without a player on the pitch, `feet_situation` is `features.situation` (shooter nearest a goal).

The frozen modules (`features`, `extract3`, `stack.inputs`) are pinned by hash in several completion records and hash
lists, so this module adds code beside them and never edits them: `situation_features` is a copy of
`features.shot_features` that takes the situation as an argument, and `filled_frame` / `replay` are copies of
`stack.inputs.filled_frame` / `replay` that take the order as an argument. With the E3 rule they reproduce the frozen
functions exactly (tests on planted frames; `scripts/location_check.py` on the full E3 caches).
"""

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import NO_TEAM, FrameResult
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.detector.conventions import GOAL, LENGTH, align, attacks_right
from xgpoc.detector.features import (
    COLUMNS,
    Situation,
    _nearest,
    geometry,
    missing_features,
    situation,
)

OFFSETS = x3.STUDY_OFFSETS                      # t0, -400 ms


@dataclass(frozen=True)
class LocationRule:
    """A shot-location rule (module docstring). `e3_flags_at`: the offsets whose stack-table ball-source columns (the
    sanitised ball source one-hot and `shot_from_ball`) are E3's, i.e. computed from the E3-filled frame with
    `features.situation`, instead of the rule's own."""

    name: str
    source: x3.SourceConfig
    situation: Callable[[FrameResult], Situation]
    e3_flags_at: tuple[int, ...] = ()


# ------------------------------------------------------------------ the feet rule
def box_distance(boxes: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Distance from `point` (x, y) to each `(N, 4)` xyxy box, 0 inside or on the edge (image pixels)."""
    dx = np.maximum(np.maximum(boxes[:, 0] - point[0], 0), point[0] - boxes[:, 2])
    dy = np.maximum(np.maximum(boxes[:, 1] - point[1], 0), point[1] - boxes[:, 3])
    return np.hypot(dx, dy)


def image_shooter(frame: FrameResult, s: Situation) -> int:
    """The player (index into the frame's people) whose image box is nearest the image centre of the ball `s` uses (the
    frame's first ball with a pitch position, as `features.situation` takes it); the first such player on a tie. The
    ball box may come from another frame of the order (`extract3.with_ball` keeps its own image coordinates)."""
    finite = np.isfinite(frame.ball_xy.astype(float)).all(axis=1)
    box = frame.ball.xyxy[finite][0]
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return int(s.players[np.argmin(box_distance(frame.detections.xyxy[s.players].astype(float), centre))])


def feet_situation(frame: FrameResult) -> Situation:
    """`features.situation` with the feet rule (module docstring): shooter = `image_shooter`, shot location = his
    projected feet, the attacked goal and the goalkeeper (detected, else the proxy) chosen from that location as
    `features.situation` chooses them. The ball and its source stay those of `features.situation`."""
    s = situation(frame)
    if s.ball is None or not len(s.players):
        return s
    shooter = image_shooter(frame, s)
    shot = s.xy[shooter]
    right = attacks_right(shot)
    aligned = align(s.xy, right)
    attacked_half = aligned[:, 0] >= LENGTH / 2
    goalkeeper, proxy = None, False
    candidates = s.keepers[attacked_half[s.keepers]]
    if len(candidates):
        goalkeeper = int(candidates[_nearest(aligned[candidates], GOAL)])
    else:
        others = s.players[(s.players != shooter) & attacked_half[s.players]]
        if len(others):
            goalkeeper, proxy = int(others[_nearest(aligned[others], GOAL)]), True
    return dataclasses.replace(s, shooter=shooter, shot=shot, attack_right=right, goalkeeper=goalkeeper, proxy=proxy)


E3 = LocationRule("E3", x3.CONFIGS["E3"], situation)
FEET_T0 = LocationRule("feet_t0", x3.SourceConfig("feet_t0", (0, -400, -800)), feet_situation, e3_flags_at=(0,))
RULES = {r.name: r for r in (E3, FEET_T0)}


# ------------------------------------------------------------------ features
def situation_features(frame: FrameResult, s: Situation, keypoint_confidence: float = 0.5) -> dict[str, float]:
    """`features.COLUMNS` of an analysed frame given its situation `s`: `features.shot_features` step for step, with
    `s` an argument instead of `features.situation(frame)` (equal to it for that situation)."""
    out = missing_features()
    out["homography_ok"] = float(frame.homography is not None)
    confidence = frame.keypoints.confidence if frame.keypoints is not None else None
    out["n_keypoints"] = float((confidence[0] >= keypoint_confidence).sum()) if confidence is not None and len(confidence) else 0.0
    out["n_players"], out["n_goalkeepers"] = float(len(s.players)), float(len(s.keepers))
    out["ball_ok"] = float(s.ball is not None)
    out["ball_fallback"] = float(s.ball is not None and s.ball_source > 0)
    out["ball_source"] = float(s.ball_source) if s.ball is not None else np.nan
    if s.shooter is not None:
        if s.ball is not None:
            out["shooter_ball_dist"] = float(np.hypot(*(s.xy[s.shooter] - s.ball)))
        out["shooter_by_goal"] = float(s.shooter_by_goal)
    out["shooter_ok"] = float(s.shooter is not None)
    if s.shot is None:
        return out
    shot, aligned = s.aligned(s.shot), s.aligned(s.xy)
    out |= {"shot_ok": 1.0, "shot_from_ball": float(s.ball is not None), "shot_x": float(shot[0]), "shot_y": float(shot[1]),
            "attack_right": float(s.attack_right)}

    goalkeeper = s.goalkeeper
    out["gk_ok"], out["gk_detected"], out["gk_proxy"] = (float(goalkeeper is not None), float(goalkeeper is not None and not s.proxy),
                                                         float(s.proxy))
    if goalkeeper is not None:
        out["gk_x"], out["gk_y"] = (float(v) for v in aligned[goalkeeper])

    shooter_team, teams = s.shooter_team, s.teams
    if s.proxy and shooter_team != NO_TEAM and teams[goalkeeper] != NO_TEAM:
        out["gk_proxy_same_team"] = float(teams[goalkeeper] == shooter_team)
    outfield = s.outfield
    known = outfield[teams[outfield] != NO_TEAM]
    out["n_unknown_team"] = float(len(outfield) - len(known))
    defenders_xy = opponents_xy = None
    if shooter_team != NO_TEAM:
        defenders = known[teams[known] != shooter_team]
        out |= {"teams_ok": 1.0, "n_defenders": float(len(defenders)), "n_attackers": float(len(known) - len(defenders))}
        defenders_xy = aligned[defenders]
        opponents_xy = np.vstack([defenders_xy, aligned[[goalkeeper]]]) if goalkeeper is not None else defenders_xy
    out |= geometry(shot, defenders_xy=defenders_xy, opponents_xy=opponents_xy,
                    goalkeeper_xy=aligned[goalkeeper] if goalkeeper is not None else None)
    return out


def rule_features(frame: FrameResult | None, rule: LocationRule) -> dict[str, float]:
    """`features.COLUMNS` of a filled frame (`filled_frame`) under `rule`, at the extractor's keypoint threshold;
    `missing_features()` for an unreadable frame."""
    if frame is None:
        return missing_features()
    return situation_features(frame, rule.situation(frame), x3.COMMON["homography_min_confidence"])


# ------------------------------------------------------------------ replay of cached frames
def filled_frame(offset: int, cached: dict[int, FrameResult | None], source: x3.SourceConfig) -> FrameResult | None:
    """The row's frame at `offset` with the ball of `source`'s order (`stack.inputs.filled_frame` with the order as an
    argument), None when unreadable."""
    result = cached.get(offset)
    if result is None:
        return None
    order = source.order(offset)
    picked = x3.chosen_ball(cached, order)
    if picked is None:
        return x3.without_ball(result)
    others = [o for o in order if o != offset]
    return x3.with_ball(result, cached[picked[1]], 0 if picked[1] == offset else others.index(picked[1]) + 1)


def replay(frames: pd.DataFrame, source: x3.SourceConfig
           ) -> Iterator[tuple[str, int, dict[int, FrameResult | None], FrameResult | None, float]]:
    """(shot_id, offset, the shot's cached frames, the filled frame or None, keypoint instance confidence) for every
    shot (sorted) and study offset (`stack.inputs.replay` with the order as an argument, plus the cached frames)."""
    for shot_id, group in frames.groupby("shot_id", sort=True):
        rows = {int(r["offset_ms"]): r for r in group.to_dict("records")}
        cached = {o: ex.frame_from_row(r) for o, r in rows.items()}
        for offset in OFFSETS:
            confidence = float(rows[offset]["kp_instance_conf"]) if rows[offset]["readable"] else np.nan
            yield shot_id, offset, cached, filled_frame(offset, cached, source), confidence


def feature_rows(frames: pd.DataFrame, rule: LocationRule) -> pd.DataFrame:
    """The raw feature rows of a cache under `rule` (shot_id, offset_ms, `features.COLUMNS`; every shot of `frames`,
    sorted, at t0 and -400 ms): the `features.parquet` of an E3-style cache built from `frames`."""
    rows = [{"shot_id": shot_id, "offset_ms": offset} | rule_features(filled, rule)
            for shot_id, offset, _, filled, _ in replay(frames, rule.source)]
    return pd.DataFrame(rows, columns=["shot_id", "offset_ms", *COLUMNS])
