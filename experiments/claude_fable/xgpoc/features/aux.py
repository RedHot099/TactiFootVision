"""Auxiliary high-precision features built from the cached positional tokens/boxes.

Implements the P0/P1.1 plan (research/feature_implementation_plan.md): is_penalty,
Gaussian goal-mouth occlusion, GK-proxy geometry, corrected defender features and
the ball-in-air proxy. Every feature ships with a validity flag; invalid values are
zeroed so models can learn the missingness explicitly.
"""

import numpy as np
import pandas as pd

from xgpoc.features.cache import BOX_DIM  # noqa: F401  (documents the box layout dependency)

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
GOAL_Y, GOAL_HALF = PITCH_WIDTH / 2, 7.32 / 2
BODY_W, GK_W = 0.8, 1.6  # StatsBomb-style blocker widths in metres

AUX_NAMES = [
    "is_penalty",
    "occlusion", "occlusion_gk", "n_shadow_players", "occl_valid",
    "gkp_depth", "gkp_lateral", "gkp_dist_ball", "gkp_valid",
    "def_in_cone_fixed", "nearest_def_fixed", "teams_known",
    "ball_in_air", "ball_air_valid",
]

# token columns (see cache.freeze_frame_tokens): x, y, is_ball, is_gk, is_att, is_def, team_known, valid
TX, TY, TBALL, TGK, TATT, TDEF, TKNOWN, TVALID = range(8)


def _grid_occlusion(ball: np.ndarray, blockers: list[tuple[float, float, float]]) -> float:
    """blockers: (x, y, body_width). Gaussian shadow coverage of the goal mouth, 1 cm grid."""
    ys = np.linspace(GOAL_Y - GOAL_HALF, GOAL_Y + GOAL_HALF, 733)
    free = np.ones_like(ys)
    for x, y, w in blockers:
        dxb = PITCH_LENGTH - ball[0]
        dxp = x - ball[0]
        if dxp <= 0.1 or dxp >= dxb:
            continue  # behind the ball or behind the goal line
        centre = ball[1] + (y - ball[1]) * dxb / dxp
        width = w * dxb / dxp
        sigma = max(width / 2, 1e-3)
        free *= 1 - np.exp(-0.5 * ((ys - centre) / sigma) ** 2)
    return float(1 - free.mean())


def shot_aux_features(tokens: np.ndarray, boxes: np.ndarray, is_penalty: bool) -> dict[str, float]:
    t = tokens[tokens[:, TVALID] > 0]
    ball_rows = t[t[:, TBALL] > 0]
    has_ball = len(ball_rows) > 0
    ball = ball_rows[0, :2] * [PITCH_LENGTH, PITCH_WIDTH] if has_ball else None
    players = t[t[:, TBALL] == 0]
    xy = players[:, :2] * [PITCH_LENGTH, PITCH_WIDTH]
    defending = players[:, TDEF] > 0
    gk_mask = players[:, TGK] > 0
    teams_known = bool(defending.any() or (players[:, TATT] > 0).any())

    f = dict.fromkeys(AUX_NAMES, 0.0)
    f["is_penalty"] = float(is_penalty)
    f["teams_known"] = float(teams_known)

    # --- goal-mouth occlusion (needs ball + known teams) ---
    if has_ball and teams_known:
        blockers = [(x, y, GK_W if g else BODY_W) for (x, y), d, g in zip(xy, defending, gk_mask) if d or g]
        shadows = [b for b in blockers if 0.1 < b[0] - ball[0]]
        f["occlusion"] = _grid_occlusion(ball, blockers)
        f["occlusion_gk"] = _grid_occlusion(ball, [b for b in blockers if b[2] == GK_W])
        f["n_shadow_players"] = float(len(shadows))
        f["occl_valid"] = 1.0

    # --- GK proxy: detector GK first, else deepest defending player ---
    gk_xy = None
    if gk_mask.any():
        gk_xy = xy[gk_mask][np.argmax(xy[gk_mask][:, 0])]
    elif defending.any():
        gk_xy = xy[defending][np.argmax(xy[defending][:, 0])]
    if gk_xy is not None and has_ball:
        f["gkp_depth"] = float(PITCH_LENGTH - gk_xy[0])
        goal = np.array([PITCH_LENGTH, GOAL_Y])
        d = goal - ball
        n = np.linalg.norm(d)
        f["gkp_lateral"] = float(abs(np.cross(d / max(n, 1e-6), gk_xy - ball)))
        f["gkp_dist_ball"] = float(np.linalg.norm(gk_xy - ball))
        f["gkp_valid"] = 1.0

    # --- corrected defender features (only with known teams) ---
    if has_ball and teams_known and defending.any():
        dxy = xy[defending]
        dists = np.linalg.norm(dxy - ball, axis=1)
        f["nearest_def_fixed"] = float(dists.min())
        p1 = np.array([PITCH_LENGTH, GOAL_Y - GOAL_HALF])
        p2 = np.array([PITCH_LENGTH, GOAL_Y + GOAL_HALF])
        def side(a, b, c):
            return (b[0] - a[0]) * (c[:, 1] - a[1]) - (b[1] - a[1]) * (c[:, 0] - a[0])
        s1, s2, s3 = side(ball, p1, dxy), side(p1, p2, dxy), side(p2, ball, dxy)
        cone = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
        f["def_in_cone_fixed"] = float(cone.sum())

    # --- ball-in-air proxy from image-space boxes ---
    valid_boxes = boxes[boxes[:, 8] > 0]
    ball_boxes = valid_boxes[valid_boxes[:, 4] > 0]      # one-hot cls: col4 = ball
    person_boxes = valid_boxes[valid_boxes[:, 4] == 0]
    if len(ball_boxes) and len(person_boxes):
        bx = (ball_boxes[0, 0] + ball_boxes[0, 2]) / 2
        by = (ball_boxes[0, 1] + ball_boxes[0, 3]) / 2
        pc = (person_boxes[:, 0] + person_boxes[:, 2]) / 2
        heights = person_boxes[:, 3] - person_boxes[:, 1]
        near = np.argsort(np.abs(pc - bx))[:3]
        # airborne if the ball centre sits above the upper quarter of nearby players' boxes
        top_q = person_boxes[near, 1] + 0.25 * heights[near]
        f["ball_in_air"] = float((by < top_q).all())
        f["ball_air_valid"] = 1.0
    return f


def build_aux_table(index: pd.DataFrame, tokens: dict[str, np.ndarray], boxes: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for r in index.itertuples():
        if r.shot_id not in tokens:
            continue
        f = shot_aux_features(tokens[r.shot_id], boxes[r.shot_id], bool(r.is_penalty))
        f["shot_id"] = r.shot_id
        rows.append(f)
    return pd.DataFrame(rows)[["shot_id", *AUX_NAMES]]
