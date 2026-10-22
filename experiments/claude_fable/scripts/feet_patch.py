"""Shot location from the shooter's feet (2026-10-22 error decomposition, descriptive variant; not a frozen definition).

When a ball exists, the shooter is the player whose IMAGE box is nearest the ball's image centre (a ball in the air
projects metres away on the pitch, so the pitch-nearest player and the ball itself are wrong for headers and volleys),
and the shot location is that player's feet (his projected position). Without a ball nothing changes. `apply()` swaps
`features.situation` (and the names imported from it) for this rule."""

import dataclasses

import numpy as np
from xgpoc.detector import features as ft

_original = ft.situation


def _rect_dist(boxes: np.ndarray, pt: np.ndarray) -> np.ndarray:
    dx = np.maximum(np.maximum(boxes[:, 0] - pt[0], 0), pt[0] - boxes[:, 2])
    dy = np.maximum(np.maximum(boxes[:, 1] - pt[1], 0), pt[1] - boxes[:, 3])
    return np.hypot(dx, dy)


def situation(frame):
    s = _original(frame)
    if s.ball is None or not len(s.players):
        return s
    finite = np.isfinite(frame.ball_xy.astype(float)).all(axis=1)
    box = frame.ball.xyxy[finite][0]
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    shooter = int(
        s.players[
            np.argmin(
                _rect_dist(frame.detections.xyxy[s.players].astype(float), centre)
            )
        ]
    )
    shot = s.xy[shooter]
    right = ft.attacks_right(shot)
    aligned = ft.align(s.xy, right)
    attacked_half = aligned[:, 0] >= ft.LENGTH / 2
    goalkeeper, proxy = None, False
    candidates = s.keepers[attacked_half[s.keepers]]
    if len(candidates):
        goalkeeper = int(candidates[ft._nearest(aligned[candidates], ft.GOAL)])
    else:
        others = s.players[(s.players != shooter) & attacked_half[s.players]]
        if len(others):
            goalkeeper, proxy = int(others[ft._nearest(aligned[others], ft.GOAL)]), True
    return dataclasses.replace(
        s,
        shooter=shooter,
        shot=shot,
        attack_right=right,
        goalkeeper=goalkeeper,
        proxy=proxy,
    )


def apply(t0_ball: bool = False) -> None:
    """`t0_ball`: the t0 row picks its shooter by the t0 frame's own ball first (E3 order -400, -800, 0 becomes 0, -400,
    -800); the shot location stays the shooter's feet at t0, so the ball's flight after contact is never a position."""
    from xgpoc.detector import extract3 as x3
    from xgpoc.stack import inputs

    ft.situation = situation
    inputs.situation = situation
    if t0_ball:
        inputs.E3_CONFIG = x3.SourceConfig("E3_feet_t0", (0, -400, -800))
