"""Shot situation and H3 geometry in metres from one analysed shot frame (xg-detector spec, Extraction and story 12).

`geometry` is a pure function of aligned positions (attacked goal at x = 105, `conventions`), so the audit can apply
the same definitions to StatsBomb freeze frames converted to metres. The definitions follow the xg-vendor probe targets
(`xgpoc.vendor.probe_targets`, `comparator_features`) in metres instead of StatsBomb units:

    dist             distance from the shot location to the goal centre (105, 34)
    angle            angle subtended by the posts (34 -/+ 3.66) at the shot location, radians
    ff_def_cone      outfield opponents inside the triangle shot location - post - post, edges included
    def_cone_ge1/2   ff_def_cone >= 1 / >= 2
    ff_near_opp      distance from the shot location to the nearest outfield opponent (NaN when none is visible)
    ff_opp_3m        opponents, the goalkeeper included, within 3 m of the shot location (the probe target used 3
                     StatsBomb units, 2.74 m)
    gk_depth         105 - x of the goalkeeper
    gk_off_line_abs  distance of the goalkeeper from the line shot location -> goal centre (lateral offset)
    gk_dist_goal     goalkeeper distance to the goal centre

What the detector adds (`shot_features`, decisions of the spec):
    shooter          the detected player (class "player") nearest the ball; without a ball, the player nearest either
                     goal centre (flag shooter_by_goal)
    shot location    the ball (its own, or the fallback ball of an earlier frame), else the shooter's feet
    attacked goal    the goal nearer the shot location; everything is then aligned (`conventions.align`)
    teams            single-frame teams from the package; opponents are the players whose team is known and differs
                     from the shooter's. An unknown shooter team makes every team-dependent feature NaN with
                     teams_ok = 0: an unknown team never turns every player into a defender (the v1.x bug).
    goalkeeper       the detected goalkeeper nearest the attacked goal, in the attacked half; else the proxy, the
                     player other than the shooter nearest the attacked goal, in the attacked half (gk_proxy = 1). The
                     detector rarely labels goalkeepers, so the proxy is usually the goalkeeper labelled "player": it
                     plays the goalkeeper's role, out of the outfield counts and in ff_opp_3m. No team filter (the
                     spec's "player nearest the goal"): the goalkeeper's own kit falls into either k = 2 kit cluster,
                     so a filter would drop him whenever it lands in the shooter's; gk_proxy_same_team records it
                     (in the C4 run, 51% of proxies) so the audit can compare the two kinds.
    referees         never anyone's team, never counted.

Every feature is NaN when its inputs are missing; the flags (`FLAGS`) say which inputs existed. `shot_x/y` and
`gk_x/y` are the aligned positions themselves (descriptive; the audit compares them with StatsBomb freeze frames).
"""

from dataclasses import dataclass

import numpy as np

from tactifoot_vision.pipeline import NO_TEAM, FrameResult
from xgpoc.detector.conventions import (
    GOAL,
    LENGTH,
    OTHER_GOAL,
    POSTS,
    align,
    attacks_right,
)

NEAR = 3.0
PLAYER, GOALKEEPER = "player", "goalkeeper"
GEOMETRY = ["dist", "angle", "ff_def_cone", "def_cone_ge1", "def_cone_ge2", "ff_near_opp", "ff_opp_3m", "gk_depth",
            "gk_off_line_abs", "gk_dist_goal"]
FLAGS = ["homography_ok", "ball_ok", "ball_fallback", "shot_ok", "shot_from_ball", "shooter_ok", "shooter_by_goal",
         "teams_ok", "gk_ok", "gk_detected", "gk_proxy"]
DESCRIPTIVE = ["shot_x", "shot_y", "attack_right", "ball_source", "n_keypoints", "n_players", "n_goalkeepers",
               "n_defenders", "n_attackers", "n_unknown_team", "shooter_ball_dist", "gk_proxy_same_team", "gk_x", "gk_y"]
COLUMNS = [*GEOMETRY, *FLAGS, *DESCRIPTIVE]


def in_cone(points, shot) -> np.ndarray:
    """Which `(N, 2)` points lie in the triangle shot - post - post, edges included (as `probe_targets.in_triangle`)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    a, b, c = np.asarray(shot, dtype=float), POSTS[0], POSTS[1]

    def side(q, r):
        return (p[:, 0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[:, 1] - r[1])
    d = np.column_stack([side(a, b), side(b, c), side(c, a)])
    return ~((d < 0).any(axis=1) & (d > 0).any(axis=1))


def geometry(shot_xy, *, defenders_xy=None, opponents_xy=None, goalkeeper_xy=None) -> dict[str, float]:
    """The `GEOMETRY` features of a shot at `shot_xy` (aligned metres).

    `defenders_xy`: outfield opponents, `opponents_xy`: every opponent including the goalkeeper; `None` means the
    teams are unknown (NaN), an empty array means none is visible. `goalkeeper_xy`: `None` when there is none.
    """
    out = dict.fromkeys(GEOMETRY, np.nan)
    if shot_xy is None or not np.isfinite(shot_xy).all():
        return out
    x, y = (float(v) for v in shot_xy)
    a1, a2 = np.arctan2(POSTS[0][1] - y, POSTS[0][0] - x), np.arctan2(POSTS[1][1] - y, POSTS[1][0] - x)
    out |= {"dist": float(np.hypot(GOAL[0] - x, GOAL[1] - y)), "angle": float(abs(a2 - a1))}
    if defenders_xy is not None:
        defenders = np.asarray(defenders_xy, dtype=float).reshape(-1, 2)
        cone = float(in_cone(defenders, (x, y)).sum())
        near = np.hypot(defenders[:, 0] - x, defenders[:, 1] - y)
        out |= {"ff_def_cone": cone, "def_cone_ge1": float(cone >= 1), "def_cone_ge2": float(cone >= 2),
                "ff_near_opp": float(near.min()) if len(near) else np.nan}
    if opponents_xy is not None:
        opponents = np.asarray(opponents_xy, dtype=float).reshape(-1, 2)
        out["ff_opp_3m"] = float((np.hypot(opponents[:, 0] - x, opponents[:, 1] - y) <= NEAR).sum())
    if goalkeeper_xy is not None:
        gx, gy = (float(v) for v in goalkeeper_xy)
        vx, vy = GOAL[0] - x, GOAL[1] - y
        out |= {"gk_depth": GOAL[0] - gx, "gk_dist_goal": float(np.hypot(GOAL[0] - gx, GOAL[1] - gy)),
                "gk_off_line_abs": float(abs((gx - x) * vy - (gy - y) * vx) / (np.hypot(vx, vy) + 1e-9))}
    return out


def _nearest(points: np.ndarray, target) -> int:
    return int(np.argmin(np.hypot(points[:, 0] - target[0], points[:, 1] - target[1])))


def missing_features() -> dict[str, float]:
    """Feature row without a frame: every flag 0, every value NaN."""
    return dict.fromkeys(COLUMNS, np.nan) | dict.fromkeys(FLAGS, 0.0)


@dataclass(frozen=True)
class Situation:
    """The roles of one analysed shot frame, chosen by the rules of the module docstring. Positions are pitch metres as
    projected (not aligned; NaN without a projection); indices point into the frame's people detections."""

    xy: np.ndarray                  # (N, 2) every person
    names: np.ndarray               # (N,) class names
    teams: np.ndarray               # (N,) team ids, NO_TEAM when unknown
    players: np.ndarray             # class "player" with a pitch position
    keepers: np.ndarray             # class "goalkeeper" with a pitch position
    ball: np.ndarray | None         # (2,) the ball used (own or fallback), None without one on the pitch
    ball_source: int                # 0 the frame's own ball, k the k-th earlier frame's
    shooter: int | None
    shooter_by_goal: bool           # no ball: the shooter is the player nearest either goal centre
    shot: np.ndarray | None         # (2,) shot location: the ball, else the shooter's feet
    attack_right: bool | None       # the attacked goal is at x = 105 (None without a shot location)
    goalkeeper: int | None          # the attacked goal's goalkeeper (detected, else the proxy)
    proxy: bool                     # the goalkeeper is the proxy (a "player")

    @property
    def outfield(self) -> np.ndarray:
        """Players other than the shooter and the goalkeeper proxy."""
        drop = [i for i in (self.shooter, self.goalkeeper if self.proxy else None) if i is not None]
        return self.players[~np.isin(self.players, drop)]

    @property
    def shooter_team(self) -> int:
        return int(self.teams[self.shooter]) if self.shooter is not None else NO_TEAM

    def aligned(self, xy) -> np.ndarray:
        """Pitch points in aligned coordinates (attacked goal at x = 105)."""
        return align(xy, bool(self.attack_right))


def situation(frame: FrameResult) -> Situation:
    """The `Situation` of an analysed shot frame (after `fill_ball` when the ball may fall back)."""
    names = frame.detections.data["class_name"] if len(frame.detections) else np.zeros(0, dtype=str)
    xy, teams = frame.pitch_xy.astype(float), frame.team_ids
    on_pitch = np.isfinite(xy).all(axis=1)
    players = np.flatnonzero((names == PLAYER) & on_pitch)
    keepers = np.flatnonzero((names == GOALKEEPER) & on_pitch)

    balls = frame.ball_xy.astype(float)
    finite_ball = np.isfinite(balls).all(axis=1)
    ball = balls[finite_ball][0] if finite_ball.any() else None
    source = int(frame.ball.data["fallback"][finite_ball][0]) if ball is not None and "fallback" in frame.ball.data else 0

    shooter, by_goal = None, False
    if len(players):
        if ball is not None:
            shooter = int(players[_nearest(xy[players], ball)])
        else:
            to_goal = np.minimum(np.hypot(*(xy[players] - GOAL).T), np.hypot(*(xy[players] - OTHER_GOAL).T))
            shooter, by_goal = int(players[np.argmin(to_goal)]), True
    shot = ball if ball is not None else (xy[shooter] if shooter is not None else None)
    right = attacks_right(shot) if shot is not None else None

    goalkeeper, proxy = None, False
    if shot is not None:
        aligned = align(xy, right)
        attacked_half = aligned[:, 0] >= LENGTH / 2
        candidates = keepers[attacked_half[keepers]]
        if len(candidates):
            goalkeeper = int(candidates[_nearest(aligned[candidates], GOAL)])
        else:
            others = players[(players != shooter) & attacked_half[players]]
            if len(others):
                goalkeeper, proxy = int(others[_nearest(aligned[others], GOAL)]), True
    return Situation(xy, names, teams, players, keepers, ball, source, shooter, by_goal, shot, right, goalkeeper, proxy)


def shot_features(frame: FrameResult, keypoint_confidence: float = 0.5) -> dict[str, float]:
    """`COLUMNS` of one analysed shot frame (a package `FrameResult`, after `fill_ball` when the ball may fall back).

    `keypoint_confidence` only counts the confident keypoints (`n_keypoints`), the homography's own threshold.
    """
    out = missing_features()
    out["homography_ok"] = float(frame.homography is not None)
    confidence = frame.keypoints.confidence if frame.keypoints is not None else None
    out["n_keypoints"] = float((confidence[0] >= keypoint_confidence).sum()) if confidence is not None and len(confidence) else 0.0
    s = situation(frame)
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
