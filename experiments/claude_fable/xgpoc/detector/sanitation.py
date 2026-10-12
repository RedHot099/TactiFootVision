"""Label-free sanitation of the detector features and the raster points of approach 1 (xg-detector ticket 03, council C1).

The rule uses detector outputs and pitch geometry only (never a label, StatsBomb or an audit result), applies the same
way at t0 and -400 ms, and was committed before any training run:

1. Roles are those of the extraction (`features.situation`: ball, shooter, shot location, attacked goal, goalkeeper).
   Sanitation never chooses again; it only masks positions and what depends on them.
2. A projected position more than `MARGIN` (3 m) outside the 105 x 68 m pitch is missing (`on_pitch`). This applies to
   every player and goalkeeper, the ball and the shot location.
3. The goalkeeper (detected or proxy) is missing for the goalkeeper features (`gk_depth`, `gk_off_line_abs`,
   `gk_dist_goal`, the gk flags) when its position is missing or more than 3 m outside the attacked penalty area
   (`in_goalkeeper_zone`). It keeps its role otherwise: out of the outfield counts and, when its position is on the
   pitch, an opponent in `ff_opp_3m`, as in the extraction.
4. A feature that depends on a missing position is missing. A masked shot location makes every geometry feature
   missing (it also fixes the attacked goal). A masked shooter makes the shooter's team unknown.
5. Counts (`ff_def_cone`, `def_cone_ge1/2`, `ff_opp_3m`; also `ff_near_opp`) use players with a valid position only,
   and are missing, never 0, when the frame has no homography or the shooter's team is unknown (council C1, condition
   2). `n_valid` counts the players and goalkeepers with a valid position so a model can tell few detections from few
   players; `n_masked` counts those the pitch rule masked.
6. Every shot is kept. Indicators say what was masked: `shot_masked`, `gk_masked`, `n_masked`.

`SANITISED` lists the columns of a sanitised row. The flags mean the extraction's flag and a valid position: `ball_ok`
is a ball on the pitch, `shot_ok` a valid shot location, `teams_ok` a valid shooter with a known team, `gk_ok` a
goalkeeper inside the zone.

Raster points (`raster_points`, approach A2): every player and goalkeeper with a valid position and the valid ball, in
aligned coordinates, each in one channel (`CHANNELS`):
    attacker    the shooter, and players of the shooter's team
    defender    players of the other team
    goalkeeper  the goalkeeper, when it passes rule 3 (otherwise it is drawn by its team like any player)
    ball        the ball (own or fallback, as extracted)
    unknown     players whose team is unknown, and every player but the shooter when the shooter's team is unknown
Referees are never drawn. Without a valid shot location the frame cannot be aligned and has no raster points.
"""

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import NO_TEAM
from xgpoc.detector.conventions import GOAL, LENGTH, PITCH, WIDTH
from xgpoc.detector.extract import STUDY_OFFSETS, ExtractorConfig, replay_frames
from xgpoc.detector.features import FLAGS, GEOMETRY, Situation, geometry, situation

MARGIN = 3.0
BOX_DEPTH, BOX_HALF_WIDTH = PITCH.penalty_box_length, PITCH.penalty_box_width / 2   # 16.5 m, 20.16 m
INDICATORS = ["shot_masked", "gk_masked"]
COUNTS = ["n_valid", "n_masked"]
SANITISED = [*GEOMETRY, *FLAGS, "ball_source", *INDICATORS, *COUNTS]
TEAM_COUNTS = ["ff_def_cone", "def_cone_ge1", "def_cone_ge2", "ff_near_opp", "ff_opp_3m"]
CHANNELS = ("attacker", "defender", "goalkeeper", "ball", "unknown")
ATTACKER, DEFENDER, KEEPER, BALL, UNKNOWN = range(len(CHANNELS))


def on_pitch(xy) -> np.ndarray:
    """Whether pitch points ((2,) or (N, 2), metres) are finite and at most `MARGIN` outside the 105 x 68 m pitch. The
    band is symmetric under the 180-degree alignment, so raw and aligned points give the same answer."""
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(p).all(axis=1) & (p[:, 0] >= -MARGIN) & (p[:, 0] <= LENGTH + MARGIN)
                & (p[:, 1] >= -MARGIN) & (p[:, 1] <= WIDTH + MARGIN))


def in_goalkeeper_zone(aligned_xy) -> np.ndarray:
    """Whether aligned points lie at most `MARGIN` outside the attacked penalty area (16.5 x 40.32 m at x = 105); the
    zone ends `MARGIN` behind the goal line, like the pitch band."""
    p = np.asarray(aligned_xy, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(p).all(axis=1) & (p[:, 0] >= LENGTH - BOX_DEPTH - MARGIN) & (p[:, 0] <= LENGTH + MARGIN)
                & (np.abs(p[:, 1] - GOAL[1]) <= BOX_HALF_WIDTH + MARGIN))


def _people(s: Situation) -> np.ndarray:
    return np.union1d(s.players, s.keepers)


def _valid_goalkeeper(s: Situation, valid: np.ndarray) -> bool:
    g = s.goalkeeper
    return g is not None and bool(valid[g]) and bool(in_goalkeeper_zone(s.aligned(s.xy[g]))[0])


def sanitised_features(s: Situation, homography_ok: bool) -> dict[str, float]:
    """The `SANITISED` columns of a frame's situation (module docstring, rules 1-6)."""
    out = dict.fromkeys(SANITISED, np.nan) | dict.fromkeys([*FLAGS, *INDICATORS], 0.0)
    valid = on_pitch(s.xy)
    people = _people(s)
    out |= {"homography_ok": float(homography_ok), "n_valid": float(valid[people].sum()), "n_masked": float((~valid[people]).sum())}
    ball_ok = s.ball is not None and bool(on_pitch(s.ball)[0])
    out |= {"ball_ok": float(ball_ok), "ball_fallback": float(ball_ok and s.ball_source > 0),
            "ball_source": float(s.ball_source) if ball_ok else np.nan}
    if s.shot is None:
        return out
    if not on_pitch(s.shot)[0]:
        out["shot_masked"] = 1.0
        return out
    shooter_ok = s.shooter is not None and bool(valid[s.shooter])
    out |= {"shot_ok": 1.0, "shot_from_ball": float(s.ball is not None), "shooter_ok": float(shooter_ok),
            "shooter_by_goal": float(shooter_ok and s.shooter_by_goal)}
    aligned = s.aligned(s.xy)
    goalkeeper_xy = None
    if s.goalkeeper is not None:
        if _valid_goalkeeper(s, valid):
            goalkeeper_xy = aligned[s.goalkeeper]
            out |= {"gk_ok": 1.0, "gk_detected": float(not s.proxy), "gk_proxy": float(s.proxy)}
        else:
            out["gk_masked"] = 1.0
    team = s.shooter_team if shooter_ok else NO_TEAM
    defenders_xy = opponents_xy = None
    if homography_ok and team != NO_TEAM:
        outfield = s.outfield[valid[s.outfield]]
        defenders = outfield[(s.teams[outfield] != NO_TEAM) & (s.teams[outfield] != team)]
        defenders_xy = aligned[defenders]
        keeper_on_pitch = s.goalkeeper is not None and bool(valid[s.goalkeeper])
        opponents_xy = np.vstack([defenders_xy, aligned[[s.goalkeeper]]]) if keeper_on_pitch else defenders_xy
        out["teams_ok"] = 1.0
    out |= geometry(s.aligned(s.shot), defenders_xy=defenders_xy, opponents_xy=opponents_xy, goalkeeper_xy=goalkeeper_xy)
    if not out["teams_ok"]:
        out |= dict.fromkeys(TEAM_COUNTS, np.nan)     # rule 5, stated once more: never a silent 0
    return out


def raster_points(s: Situation) -> tuple[np.ndarray, np.ndarray]:
    """(aligned points (K, 2), channel per point (K,)) of a frame's situation (module docstring); empty without a valid
    shot location."""
    valid = on_pitch(s.xy)
    if s.shot is None or not on_pitch(s.shot)[0]:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    aligned = s.aligned(s.xy)
    shooter_ok = s.shooter is not None and bool(valid[s.shooter])
    team = s.shooter_team if shooter_ok else NO_TEAM
    keeper = s.goalkeeper if _valid_goalkeeper(s, valid) else None
    points, channels = [], []
    for i in _people(s):
        if not valid[i]:
            continue
        if i == keeper:
            channel = KEEPER
        elif i == s.shooter:
            channel = ATTACKER
        elif team == NO_TEAM or s.teams[i] == NO_TEAM:
            channel = UNKNOWN
        else:
            channel = ATTACKER if s.teams[i] == team else DEFENDER
        points.append(aligned[i])
        channels.append(channel)
    if s.ball is not None and on_pitch(s.ball)[0]:
        points.append(s.aligned(s.ball))
        channels.append(BALL)
    return np.asarray(points, dtype=float).reshape(-1, 2), np.asarray(channels, dtype=int)


def missing_row() -> dict[str, float]:
    """Sanitised columns of an unreadable frame: no position, every flag, indicator and count 0."""
    return dict.fromkeys(SANITISED, np.nan) | dict.fromkeys([*FLAGS, *INDICATORS, *COUNTS], 0.0)


def sanitised_table(frames: pd.DataFrame, config: ExtractorConfig, offsets=STUDY_OFFSETS) -> pd.DataFrame:
    """One row per shot and offset of a configuration's cached frames: the `SANITISED` columns, `aligned` (the frame
    has raster points' alignment, i.e. a valid shot location) and the raster points (`pts_x`, `pts_y`, `pts_channel`).
    The frames and their ball fallback are the extraction's (`extract.replay_frames`)."""
    rows = []
    for shot_id, offset, result in replay_frames(frames, config, offsets):
        if result is None:
            row, xy, channel = missing_row(), np.zeros((0, 2)), np.zeros(0, dtype=int)
        else:
            s = situation(result)
            row = sanitised_features(s, result.homography is not None)
            xy, channel = raster_points(s)
        rows.append({"shot_id": shot_id, "offset_ms": offset} | row
                    | {"aligned": bool(row["shot_ok"]), "pts_x": xy[:, 0].tolist(), "pts_y": xy[:, 1].tolist(),
                       "pts_channel": channel.tolist()})
    return pd.DataFrame(rows)
