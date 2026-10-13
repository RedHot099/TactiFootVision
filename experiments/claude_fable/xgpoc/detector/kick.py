"""The kick frame of a shot from the ball's track (pitch-keypoints 04, council C3 Q3).

A shot is struck where the ball, after moving slowly with the shooter or arriving from a pass, leaves at shot speed.
The rule finds that frame in the ball's pitch track alone, with no player, label or StatsBomb input:

1. `prepare`: the track (one pitch position per frame, NaN without a ball) with gaps of at most `MAX_GAP` frames filled
   linearly, then a 3-frame running median per coordinate (one wrong detection between two right ones is removed, a
   monotone run is kept).
2. Per frame t, the forward velocity `v(t) = (p(t + SPAN) - p(t)) / (SPAN * frame time)` and the backward one
   `u(t) = (p(t) - p(t - SPAN)) / (SPAN * frame time)`.
3. The kick is the first frame of the window where the ball leaves fast and changes its motion: `|v(t)| >= V_SHOT` and
   `|v(t) - u(t)| >= DV_KICK`, and it keeps going: `|p(t + 2 * SPAN) - p(t)| >= V_SHOT * 2 * SPAN * frame time` (a single
   wrong position or one jittery homography does not make a kick). Without such a frame there is no kick (`None`).

The shot location is the prepared track at the kick frame. Every frame needs both neighbours `SPAN` away (and the one
`2 * SPAN` ahead), so the earliest kick is `SPAN` frames into the window. The thresholds are physical, fixed before
any audit: a shot leaves the foot at well over 10 m/s, a dribble or a rolling ball is slower, and the velocity change of
a first-time shot (ball arriving, ball leaving towards the goal) is large too.
"""

import numpy as np

FRAME_MS = 40.0          # 25 fps
SPAN = 3                 # frames of each velocity (120 ms)
V_SHOT = 10.0            # m/s, the ball's speed after a kick
DV_KICK = 8.0            # m/s, the velocity change at a kick
MAX_GAP = 2              # missing frames filled by interpolation


def fill_gaps(xy: np.ndarray, max_gap: int = MAX_GAP) -> np.ndarray:
    """`(n, 2)` positions with runs of at most `max_gap` NaN rows between two known rows filled linearly; longer runs
    and the ends stay NaN."""
    out = np.array(xy, dtype=float)
    known = np.flatnonzero(np.isfinite(out).all(axis=1))
    for a, b in zip(known[:-1], known[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            weights = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = out[a] + weights[:, None] * (out[b] - out[a])
    return out


def running_median(xy: np.ndarray) -> np.ndarray:
    """3-frame running median per coordinate over the known neighbours (an end frame keeps the median of two, i.e.
    their mean); a NaN row stays NaN."""
    xy = np.asarray(xy, dtype=float)
    out = np.full_like(xy, np.nan)
    for t in range(len(xy)):
        if not np.isfinite(xy[t]).all():
            continue
        window = xy[max(t - 1, 0):t + 2]
        window = window[np.isfinite(window).all(axis=1)]
        out[t] = np.median(window, axis=0)
    return out


def prepare(xy: np.ndarray) -> np.ndarray:
    """A raw track filled (`fill_gaps`) and smoothed (`running_median`), as `kick_index` reads it."""
    return running_median(fill_gaps(xy))


def kick_index(xy: np.ndarray, frame_ms: float = FRAME_MS) -> int | None:
    """The kick frame of a prepared track (`prepare`), or None."""
    p = np.asarray(xy, dtype=float)
    span_s = SPAN * frame_ms / 1000.0          # seconds between a frame and its neighbour SPAN frames away
    for t in range(SPAN, len(p) - 2 * SPAN):
        ahead, behind, far = p[t + SPAN], p[t - SPAN], p[t + 2 * SPAN]
        if not (np.isfinite(p[t]).all() and np.isfinite(ahead).all() and np.isfinite(behind).all() and np.isfinite(far).all()):
            continue
        v, u = (ahead - p[t]) / span_s, (p[t] - behind) / span_s
        if (np.hypot(*v) >= V_SHOT and np.hypot(*(v - u)) >= DV_KICK
                and np.hypot(*(far - p[t])) >= V_SHOT * 2 * span_s):
            return t
    return None


def kick(xy: np.ndarray, frame_ms: float = FRAME_MS) -> tuple[int | None, np.ndarray | None]:
    """(kick frame, shot location) of a raw track (`(n, 2)` pitch positions, NaN without a ball), or (None, None)."""
    prepared = prepare(xy)
    index = kick_index(prepared, frame_ms)
    return (index, prepared[index]) if index is not None else (None, None)
