"""Engineered xG features from a canonical FreezeFrame (attacked goal at x = 105)."""

import numpy as np

from xgpoc.features.freeze_frame import CLS_GK, CLS_PLAYER, PITCH_LENGTH, PITCH_WIDTH, FreezeFrame

GOAL_Y = PITCH_WIDTH / 2
GOAL_HALF_WIDTH = 7.32 / 2
FEATURE_NAMES = [
    "has_ball", "homography_ok", "n_keypoints", "n_players",
    "dist_to_goal", "angle_to_goal", "ball_x", "ball_y_abs",
    "n_defenders_in_cone", "n_attackers_in_cone", "nearest_defender_dist", "nearest_attacker_dist",
    "gk_dist_to_ball", "gk_dist_to_goal_line", "gk_present",
    "n_defenders_in_box", "n_attackers_in_box", "n_players_in_box",
    "n_opponents_within_3m", "attack_team_known",
]


def _shot_angle(ball: np.ndarray) -> float:
    dx = PITCH_LENGTH - ball[0]
    a = np.arctan2(GOAL_Y + GOAL_HALF_WIDTH - ball[1], dx) - np.arctan2(GOAL_Y - GOAL_HALF_WIDTH - ball[1], dx)
    return float(abs(a))


def _in_cone(pts: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Points inside the triangle ball -> two goal posts."""
    if len(pts) == 0:
        return np.zeros(0, bool)
    p1, p2 = np.array([PITCH_LENGTH, GOAL_Y - GOAL_HALF_WIDTH]), np.array([PITCH_LENGTH, GOAL_Y + GOAL_HALF_WIDTH])
    def side(a, b, c):
        return (b[0] - a[0]) * (c[:, 1] - a[1]) - (b[1] - a[1]) * (c[:, 0] - a[0])
    s1, s2, s3 = side(ball, p1, pts), side(p1, p2, pts), side(p2, ball, pts)
    return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))


def freeze_frame_features(ff: FreezeFrame) -> np.ndarray:
    ff = ff.canonical()
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    f["homography_ok"], f["n_keypoints"], f["n_players"] = float(ff.homography_ok), ff.n_keypoints, len(ff.players_xy)
    f["attack_team_known"] = float(ff.attacking_team >= 0)
    if ff.ball_xy is None:
        # Fall back to the attacking centroid so the row is still usable.
        ball = ff.players_xy.mean(0) if len(ff.players_xy) else np.array([PITCH_LENGTH - 20, GOAL_Y])
    else:
        ball = ff.ball_xy
        f["has_ball"] = 1.0
    ball = np.clip(ball, [0, 0], [PITCH_LENGTH, PITCH_WIDTH])
    f["dist_to_goal"] = float(np.hypot(PITCH_LENGTH - ball[0], GOAL_Y - ball[1]))
    f["angle_to_goal"] = _shot_angle(ball)
    f["ball_x"], f["ball_y_abs"] = float(ball[0]), float(abs(ball[1] - GOAL_Y))

    outfield = ff.players_cls == CLS_PLAYER
    gk = ff.players_cls == CLS_GK
    if ff.attacking_team >= 0:
        att = outfield & (ff.players_team == ff.attacking_team)
        de = outfield & (ff.players_team >= 0) & (ff.players_team != ff.attacking_team)
    else:
        att = np.zeros(len(outfield), bool)
        de = outfield
    xy = ff.players_xy
    cone = _in_cone(xy, ball)
    box = (xy[:, 0] > PITCH_LENGTH - 16.5) & (np.abs(xy[:, 1] - GOAL_Y) < 40.32 / 2) if len(xy) else np.zeros(0, bool)
    d = np.linalg.norm(xy - ball, axis=1) if len(xy) else np.zeros(0)
    f["n_defenders_in_cone"], f["n_attackers_in_cone"] = float((cone & de).sum()), float((cone & att).sum())
    f["nearest_defender_dist"] = float(d[de].min()) if de.any() else 50.0
    f["nearest_attacker_dist"] = float(np.sort(d[att])[1]) if att.sum() > 1 else 50.0  # skip the shooter
    f["n_defenders_in_box"], f["n_attackers_in_box"], f["n_players_in_box"] = float((box & de).sum()), float((box & att).sum()), float(box.sum())
    f["n_opponents_within_3m"] = float(((d < 3) & de).sum())
    if gk.any():
        g = xy[gk][np.argmax(xy[gk][:, 0])]  # the GK nearest the attacked goal
        f["gk_present"], f["gk_dist_to_ball"], f["gk_dist_to_goal_line"] = 1.0, float(np.linalg.norm(g - ball)), float(PITCH_LENGTH - g[0])
    else:
        f["gk_dist_to_ball"], f["gk_dist_to_goal_line"] = f["dist_to_goal"], 0.0
    return np.array([f[k] for k in FEATURE_NAMES], np.float32)
