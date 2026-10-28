"""Automatic attributes Xa of xg-dino-info (ticket 09; spec-info §Attributes, council C3.1, C3.3, C3.8, which win).

Every column comes from the feet_t0 E3 cache (`cache/xgd_full4_feet_t0/E3/{frames,features}.parquet`, 720p frames,
1280 x 720 px) replayed without a GPU under `location.FEET_T0`, from the SoccerNet-v2 labels, and (motion only) from
the camera-cut flag of the stored offset frames. No StatsBomb, caption or GPT data is read (`no_label_reads`).

Roles (C3.8). The t0 and -400 ms rows are replayed with `location.replay` and `FEET_T0.situation` (`feet_situation`):
shooter, goalkeeper (detected, else the proxy) and the ball the rule used are indices into the frame's detections. The
replay must reproduce the cached `shot_x` / `shot_y` exactly (both missing counts as reproduced); a row that does not
keeps every role-dependent column missing (`roles_ok` = 0), nothing is guessed.

Columns per offset (`<name>@<offset>`, offsets 0 and -400; pixels are 720p pixels; missing stays NaN)

    quality (Xq; C3.1: "may carry broadcast/annotation artefacts", descriptive)
        n_keypoints        confident pitch keypoints (cached)
        kp_instance_conf   the keypoint instance confidence
        n_players          class "player" detections on the pitch (cached)
        player_h_med_px    median height of the class "player" boxes
        px_per_m           homography scale at the shot location: sqrt of the area scale pitch -> image, px per metre
        box_share          share of the frame inside the attacked penalty area (16.5 x 40.32 m), 128 x 72 grid
        ball_ok, ball_fallback, shot_from_ball
                           the fallback-filled ball flags of the rule (cached columns, not the own-frame ball)
    crowding (content, metres)
        box_players        players and goalkeepers in the attacked penalty area (shooter included, referees never)
        near5_players      players and goalkeepers within 5 m of the shot location, any team, shooter excluded
        box_attackers      the shooter's team-mates in the penalty area, shooter excluded; missing for an unknown team
    shooter (content, image; a crude posture proxy, not pose)
        shooter_h_px, shooter_aspect (height / width), shooter_h_rel (height / player_h_med_px),
        shooter_overlap    largest share of the shooter box covered by another person box
    ball (content; own-frame detection only, C3.3)
        ball_own           the rule's ball is the frame's own detection (0 for a fallback or no ball)
        ball_above         ball box centre above the shooter box top
        ball_height_rel    (shooter box bottom - ball centre y) / shooter box height
        ball_dist_rel      distance from the ball centre to the shooter box (0 inside) / shooter box height
        At -400 ms ball_above / ball_height_rel / ball_dist_rel exist only where the ball is own at both offsets.

Motion between -400 ms and t0 (content; one column each, no offset suffix)
    motion_shot_disp_m   displacement of the shot location (the shooter's feet), aligned metres
    motion_gk_disp_m     displacement of the goalkeeper
    motion_box_change    box_players@0 - box_players@-400
    Valid only with: both rows reproduced, homographies at both offsets, the same attacked goal, no camera cut
    (`cut_between_anchor` False at -400 ms; unknown counts as a cut) and the same shooter: the shooter of each offset is
    the other's nearest player (players and goalkeepers on the pitch, mutual nearest neighbours) within
    `MAX_STEP_M`. The goalkeeper displacement also needs the same goalkeeper by the same test. `motion_status` says why
    a row is missing.

SoccerNet flags (content; `FLAG_WINDOWS`, ms relative to the shot annotation, inclusive, C3.3)
    sn_direct_fk (-3000..+300; the +300 ms end is stated as retrospective context), sn_direct_fk_le0 (-3000..0, the
    sensitivity window), sn_indirect_fk_3s, sn_corner_20s, sn_team_unknown. `sn_prev_shot_5s` is excluded (C5). The
    rules are `stack.inputs.shot_flags` (Goal records dropped, both shot label types one kind) with the windows as an
    argument.
"""

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import NO_TEAM, FrameResult
from xgpoc.detector import guard
from xgpoc.detector import location as loc
from xgpoc.detector.conventions import LENGTH, WIDTH, align
from xgpoc.detector.features import Situation
from xgpoc.stack import inputs as si

RULE = loc.FEET_T0
OFFSETS = si.OFFSETS                            # 0, -400
IMAGE_WH = (1280, 720)
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32
NEAR_M = 5.0
MAX_STEP_M = 4.0                                # 10 m/s for 400 ms
GRID = (128, 72)                                # sampling grid of box_share
PERSON_CLASSES = ("player", "goalkeeper")

QUALITY = ["n_keypoints", "kp_instance_conf", "n_players", "player_h_med_px", "px_per_m", "box_share", "ball_ok",
           "ball_fallback", "shot_from_ball"]
CROWDING = ["box_players", "near5_players", "box_attackers"]
SHOOTER = ["shooter_h_px", "shooter_aspect", "shooter_h_rel", "shooter_overlap"]
BALL = ["ball_own", "ball_above", "ball_height_rel", "ball_dist_rel"]
PER_OFFSET = [*QUALITY, *CROWDING, *SHOOTER, *BALL]
MOTION = ["motion_shot_disp_m", "motion_gk_disp_m", "motion_box_change"]
FLAG_WINDOWS = {k: v for k, v in si.WINDOWS.items() if k != "sn_prev_shot_5s"} | {
    "sn_direct_fk_le0": ("Direct free-kick", -3000, 0)}
FLAGS = [*FLAG_WINDOWS, "sn_team_unknown"]


def col(name: str, offset: int) -> str:
    return si.col(name, offset)


BLOCKS = {                                      # block -> design columns (C3.1: quality descriptive, the rest content)
    "quality": [col(c, o) for o in OFFSETS for c in QUALITY],
    "crowding": [col(c, o) for o in OFFSETS for c in CROWDING],
    "shooter": [col(c, o) for o in OFFSETS for c in SHOOTER],
    "ball_t0": [col(c, 0) for c in BALL],
    "ball_m400": [col(c, -400) for c in BALL],
    "motion": MOTION,
    "flags": FLAGS,
}
CONTENT_BLOCKS = ("crowding", "shooter", "ball_t0", "ball_m400", "motion", "flags")
COLUMNS = [c for cols in BLOCKS.values() for c in cols]


# ------------------------------------------------------------------ guard
LABEL_PATHS = r"caption|gpt"


@contextlib.contextmanager
def no_label_reads() -> Iterator[None]:
    """`guard.no_statsbomb_reads` with caption and GPT label paths refused as well (Xa is never built from them)."""
    original = guard.FORBIDDEN
    guard.FORBIDDEN = re.compile(f"{original.pattern}|{LABEL_PATHS}", re.IGNORECASE)
    try:
        with guard.no_statsbomb_reads():
            yield
    finally:
        guard.FORBIDDEN = original


# ------------------------------------------------------------------ projective helpers
def to_pitch(H: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pitch xy, on the ground side) of image points `(N, 2)`: the ground side of the horizon is the side of the
    image's bottom centre (a broadcast camera looks down at the pitch)."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    p = np.column_stack([uv, np.ones(len(uv))]) @ H.T
    ref = H[2] @ np.array([IMAGE_WH[0] / 2, IMAGE_WH[1], 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:], p[:, 2] * ref > 0


def to_image(H: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(image uv, visible) of pitch points `(N, 2)`: visible = finite and on the ground side (a pitch point behind the
    camera lands on the other side of the horizon)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    p = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(H).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[:, :2] / p[:, 2:]
    ok = np.isfinite(uv).all(axis=1)
    side = np.zeros(len(uv), bool)
    side[ok] = to_pitch(H, uv[ok])[1]
    return uv, ok & side


def px_per_metre(H: np.ndarray, xy: np.ndarray, step: float = 0.5) -> float:
    """sqrt of the area scale of pitch -> image at pitch point `xy` (central differences), NaN when not visible."""
    xy = np.asarray(xy, float)
    pts = xy + np.array([[step, 0], [-step, 0], [0, step], [0, -step]])
    uv, ok = to_image(H, pts)
    if not ok.all():
        return np.nan
    jx, jy = (uv[0] - uv[1]) / (2 * step), (uv[2] - uv[3]) / (2 * step)
    return float(np.sqrt(abs(jx[0] * jy[1] - jx[1] * jy[0])))


def in_box(aligned: np.ndarray) -> np.ndarray:
    """Aligned points inside the attacked penalty area (edges included)."""
    a = np.asarray(aligned, float).reshape(-1, 2)
    return (a[:, 0] >= LENGTH - BOX_DEPTH) & (a[:, 0] <= LENGTH) & (np.abs(a[:, 1] - WIDTH / 2) <= BOX_WIDTH / 2)


def box_share(H: np.ndarray, attack_right: bool) -> float:
    """Share of the frame (`GRID` sample points) whose ground point lies in the attacked penalty area."""
    us = (np.arange(GRID[0]) + 0.5) * IMAGE_WH[0] / GRID[0]
    vs = (np.arange(GRID[1]) + 0.5) * IMAGE_WH[1] / GRID[1]
    uv = np.stack(np.meshgrid(us, vs), -1).reshape(-1, 2)
    xy, ground = to_pitch(H, uv)
    inside = np.zeros(len(uv), bool)
    inside[ground] = in_box(align(xy[ground], attack_right))
    return float(inside.mean())


def box_distance(box: np.ndarray, point: np.ndarray) -> float:
    return float(loc.box_distance(np.asarray(box, float).reshape(1, 4), np.asarray(point, float))[0])


def covered_share(box: np.ndarray, others: np.ndarray) -> float:
    """Largest share of `box` covered by one of `others` `(N, 4)` (0 without others)."""
    if not len(others):
        return 0.0
    w = np.clip(np.minimum(others[:, 2], box[2]) - np.maximum(others[:, 0], box[0]), 0, None)
    h = np.clip(np.minimum(others[:, 3], box[3]) - np.maximum(others[:, 1], box[1]), 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return float((w * h).max() / area)


# ------------------------------------------------------------------ roles
@dataclass(frozen=True)
class Roles:
    """One replayed row: the filled frame, its feet_t0 situation, and whether the replay reproduced the cache."""

    frame: FrameResult | None
    s: Situation | None
    reproduced: bool                # cached shot_x / shot_y reproduced (both missing counts)
    kp_instance_conf: float

    @property
    def ok(self) -> bool:
        """Roles usable: a readable frame whose replay reproduced the cache."""
        return self.frame is not None and self.s is not None and self.reproduced

    @property
    def shooter(self) -> int | None:
        return self.s.shooter if self.ok else None

    @property
    def goalkeeper(self) -> int | None:
        return self.s.goalkeeper if self.ok else None

    @property
    def gk_detected(self) -> bool:
        return self.ok and self.s.goalkeeper is not None and not self.s.proxy

    @property
    def own_ball(self) -> np.ndarray | None:
        """The xyxy box of the rule's ball when it is the frame's own detection, else None."""
        if not self.ok or self.s.ball is None or self.s.ball_source != 0:
            return None
        finite = np.isfinite(self.frame.ball_xy.astype(float)).all(axis=1)
        return self.frame.ball.xyxy[finite][0].astype(float)

    def boxes(self) -> np.ndarray:
        return self.frame.detections.xyxy.astype(float).reshape(-1, 4)

    def persons(self) -> np.ndarray:
        """Indices of players and goalkeepers with a pitch position."""
        return np.sort(np.concatenate([self.s.players, self.s.keepers]))


def aligned_shot(s: Situation) -> tuple[float, float]:
    if s is None or s.shot is None:
        return np.nan, np.nan
    x, y = s.aligned(s.shot)
    return float(x), float(y)


def reproduces(s: Situation | None, cached_x: float, cached_y: float) -> bool:
    x, y = aligned_shot(s)
    same = [(np.isnan(a) and np.isnan(b)) or a == b for a, b in ((x, cached_x), (y, cached_y))]
    return bool(all(same))


def replay_roles(frames: pd.DataFrame, cached: pd.DataFrame) -> Iterator[tuple[str, dict[int, Roles]]]:
    """(shot_id, {offset: Roles}) for every shot of `frames` (sorted). `cached`: the cache's features.parquet rows,
    index (shot_id, offset_ms), with shot_x and shot_y."""
    current, out = None, {}
    for shot_id, offset, _, filled, confidence in loc.replay(frames, RULE.source):
        if current is not None and shot_id != current:
            yield current, out
            out = {}
        current = shot_id
        s = RULE.situation(filled) if filled is not None else None
        cx, cy = cached.loc[(shot_id, offset), ["shot_x", "shot_y"]]
        out[offset] = Roles(filled, s, reproduces(s, float(cx), float(cy)), confidence)
    if current is not None:
        yield current, out


# ------------------------------------------------------------------ per-offset attributes
def offset_attributes(r: Roles, cached: pd.Series) -> dict[str, float]:
    """`PER_OFFSET` of one row (module docstring). `cached`: the row's cached features (n_keypoints, n_players and
    the fallback-filled ball flags)."""
    out = dict.fromkeys(PER_OFFSET, np.nan)
    if r.frame is None:
        return out
    out |= {"n_keypoints": cached["n_keypoints"], "n_players": cached["n_players"], "kp_instance_conf": r.kp_instance_conf,
            "ball_ok": cached["ball_ok"], "ball_fallback": cached["ball_fallback"], "shot_from_ball": cached["shot_from_ball"]}
    names = r.frame.detections.data["class_name"] if len(r.frame.detections) else np.zeros(0, str)
    boxes = r.boxes()
    heights = boxes[:, 3] - boxes[:, 1]
    player_h = heights[names == "player"]
    if len(player_h):
        out["player_h_med_px"] = float(np.median(player_h))
    if not r.ok:
        return out
    s, H = r.s, r.frame.homography
    out["ball_own"] = float(r.own_ball is not None)
    if s.shot is not None:
        aligned = s.aligned(s.xy)
        persons = r.persons()
        shot = s.aligned(s.shot)
        if H is not None:
            out["px_per_m"] = px_per_metre(H, s.shot)
            out["box_share"] = box_share(H, s.attack_right)
        inside = in_box(aligned[persons])
        out["box_players"] = float(inside.sum())
        others = persons[persons != s.shooter]
        out["near5_players"] = float((np.hypot(*(aligned[others] - shot).T) <= NEAR_M).sum())
        if s.shooter is not None and s.shooter_team != NO_TEAM:
            mates = persons[(s.teams[persons] == s.shooter_team) & (persons != s.shooter)]
            out["box_attackers"] = float(in_box(aligned[mates]).sum())
    if s.shooter is None:
        return out
    box = boxes[s.shooter]
    h, w = box[3] - box[1], box[2] - box[0]
    out |= {"shooter_h_px": float(h), "shooter_aspect": float(h / w) if w > 0 else np.nan,
            "shooter_overlap": covered_share(box, np.delete(boxes, s.shooter, axis=0))}
    if len(player_h):
        out["shooter_h_rel"] = float(h / np.median(player_h))
    ball = r.own_ball
    if ball is not None and h > 0:
        centre = np.array([(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2])
        out |= {"ball_above": float(centre[1] < box[1]), "ball_height_rel": float((box[3] - centre[1]) / h),
                "ball_dist_rel": box_distance(box, centre) / h}
    return out


# ------------------------------------------------------------------ motion
def mutual_nearest(a: Roles, i: int, b: Roles, j: int) -> bool:
    """Person `i` of row `a` and person `j` of row `b` are each other's nearest person (players and goalkeepers on the
    pitch, pitch metres) and at most `MAX_STEP_M` apart."""
    pa, pb = a.persons(), b.persons()
    if i not in pa or j not in pb:
        return False
    xa, xb = a.s.xy[i], b.s.xy[j]
    d = float(np.hypot(*(xa - xb)))
    near_b = pb[np.argmin(np.hypot(*(b.s.xy[pb] - xa).T))]
    near_a = pa[np.argmin(np.hypot(*(a.s.xy[pa] - xb).T))]
    return bool(near_b == j and near_a == i and d <= MAX_STEP_M)


def motion(r0: Roles, r4: Roles, box0: float, box4: float, cut: bool | None) -> dict:
    """`MOTION` and `motion_status` (module docstring)."""
    out = dict.fromkeys(MOTION, np.nan)
    if not (r0.ok and r4.ok):
        return out | {"motion_status": "roles_missing"}
    if r0.frame.homography is None or r4.frame.homography is None:
        return out | {"motion_status": "no_homography"}
    if r0.s.shot is None or r4.s.shot is None or r0.s.shooter is None or r4.s.shooter is None:
        return out | {"motion_status": "no_shooter"}
    if r0.s.attack_right != r4.s.attack_right:
        return out | {"motion_status": "direction_differs"}
    if cut is None or cut:
        return out | {"motion_status": "camera_cut" if cut else "cut_unknown"}
    if not mutual_nearest(r0, r0.s.shooter, r4, r4.s.shooter):
        return out | {"motion_status": "shooter_differs"}
    right = r0.s.attack_right
    out["motion_shot_disp_m"] = float(np.hypot(*(align(r0.s.shot, right) - align(r4.s.shot, right))))
    out["motion_box_change"] = box0 - box4
    g0, g4 = r0.s.goalkeeper, r4.s.goalkeeper
    if g0 is not None and g4 is not None and mutual_nearest(r0, g0, r4, g4):
        out["motion_gk_disp_m"] = float(np.hypot(*(r0.s.xy[g0] - r4.s.xy[g4])))
    return out | {"motion_status": "ok"}


# ------------------------------------------------------------------ SoccerNet flags
def shot_flags(annotations: list[si.Annotation], half: int, position_ms: int,
               windows: dict = FLAG_WINDOWS) -> dict[str, float]:
    """`stack.inputs.shot_flags` with `windows` as an argument (`FLAGS`)."""
    teams = {a.team for a in annotations if a.kind == si.SHOT and a.half == half and a.position_ms == position_ms}
    if len(teams) != 1 or None in teams:
        return dict.fromkeys(windows, 0.0) | {"sn_team_unknown": 1.0}
    team = teams.pop()
    same = [a for a in annotations if a.half == half and a.team == team]
    return {flag: float(any(a.kind == kind and lo <= a.position_ms - position_ms <= hi for a in same))
            for flag, (kind, lo, hi) in windows.items()} | {"sn_team_unknown": 0.0}


def flags_table(shots: pd.DataFrame, annotations: dict[str, list[dict]]) -> pd.DataFrame:
    """`FLAGS` per shot (index as `shots`: game, half, position_ms); `annotations`: raw lists by game."""
    parsed = {game: si.parse_annotations(raw) for game, raw in annotations.items()}
    rows = [shot_flags(parsed[r.game], int(r.half), int(r.position_ms)) for r in shots.itertuples()]
    return pd.DataFrame(rows, index=shots.index, columns=FLAGS)


# ------------------------------------------------------------------ table
def index_row(shot_id: str, offset: int, r: Roles) -> dict:
    """Persisted roles of one row: indices into the frame's detections and their 720p boxes (C3.8)."""
    row = {"shot_id": shot_id, "offset_ms": offset, "readable": r.frame is not None, "reproduced": r.reproduced,
           "roles_ok": r.ok, "shooter_idx": -1, "gk_idx": -1, "gk_detected": r.gk_detected, "ball_own": r.own_ball is not None,
           "attack_right": np.nan, "shot_x": np.nan, "shot_y": np.nan}
    for role in ("shooter", "gk", "ball"):
        row |= {f"{role}_{k}": np.nan for k in ("x0", "y0", "x1", "y1")}
    if not r.ok:
        return row
    boxes = r.boxes()
    row["shot_x"], row["shot_y"] = aligned_shot(r.s)
    if r.s.attack_right is not None:
        row["attack_right"] = float(r.s.attack_right)
    for role, index in (("shooter", r.shooter), ("gk", r.goalkeeper)):
        if index is not None:
            row[f"{role}_idx"] = int(index)
            row |= dict(zip([f"{role}_{k}" for k in ("x0", "y0", "x1", "y1")], map(float, boxes[index]), strict=True))
    if r.own_ball is not None:
        row |= dict(zip([f"ball_{k}" for k in ("x0", "y0", "x1", "y1")], map(float, r.own_ball), strict=True))
    return row


def build(frames: pd.DataFrame, features: pd.DataFrame, shots: pd.DataFrame, annotations: dict[str, list[dict]],
          cuts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Roles]]:
    """(attributes: index shot_id sorted, `COLUMNS` + motion_status + roles_ok@<offset>; roles: one row per shot and
    offset, `index_row`; the t0 `Roles` by shot id, for the mask regions). `shots`: index shot_id with game, half, position_ms; `frames` / `features`: the feet_t0 E3
    cache rows; `cuts`: `cut_between_anchor` of the -400 ms offset frame by shot id (missing -> unknown)."""
    ids = sorted(shots.index)
    frames = frames[frames.shot_id.isin(ids)]
    if missing := set(ids) - set(frames.shot_id):
        raise ValueError(f"{len(missing)} shots have no E3 frame rows, e.g. {sorted(missing)[0]}")
    cached = features[features.shot_id.isin(ids)].set_index(["shot_id", "offset_ms"]).sort_index()
    rows, roles_rows, t0 = {}, [], {}
    for shot_id, roles in replay_roles(frames, cached):
        row, per = {}, {}
        t0[shot_id] = roles[0]
        for offset in OFFSETS:
            per[offset] = offset_attributes(roles[offset], cached.loc[(shot_id, offset)])
            roles_rows.append(index_row(shot_id, offset, roles[offset]))
        if not (per[0]["ball_own"] == 1.0 and per[-400]["ball_own"] == 1.0):
            per[-400] |= dict.fromkeys(["ball_above", "ball_height_rel", "ball_dist_rel"], np.nan)
        for offset in OFFSETS:
            row |= {col(k, offset): float(v) for k, v in per[offset].items()}
            row[col("roles_ok", offset)] = float(roles[offset].ok)
        cut = cuts.get(shot_id)
        cut = None if cut is None or pd.isna(cut) else bool(cut)
        row |= motion(roles[0], roles[-400], per[0]["box_players"], per[-400]["box_players"], cut)
        rows[shot_id] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "shot_id"
    table = pd.concat([table, flags_table(shots.loc[ids], annotations)], axis=1)
    extra = [col("roles_ok", o) for o in OFFSETS] + ["motion_status"]
    return table[[*COLUMNS, *extra]].sort_index(), pd.DataFrame(roles_rows), t0


# ------------------------------------------------------------------ coverage
def replay_match(roles: pd.DataFrame) -> dict:
    """Per offset: rows, readable, reproduced (count, share of readable)."""
    out = {}
    for offset, g in roles.groupby("offset_ms"):
        readable = g[g.readable]
        out[str(offset)] = {"rows": len(g), "readable": len(readable), "reproduced": int(readable.reproduced.sum()),
                            "share": float(readable.reproduced.mean()) if len(readable) else np.nan}
    return out


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Per column: block, shots, present, share (and the mean of a 0/1 column)."""
    block_of = {c: b for b, cols in BLOCKS.items() for c in cols}
    rows = []
    for c in COLUMNS:
        present = table[c].notna()
        rows.append({"column": c, "block": block_of[c], "kind": "quality" if block_of[c] == "quality" else "content",
                     "shots": len(table), "present": int(present.sum()), "share": float(present.mean()),
                     "mean": float(table[c].mean()) if present.any() else np.nan})
    return pd.DataFrame(rows)
