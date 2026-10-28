"""Mask regions of xg-dino-info in G-combo input space (ticket 09; spec-info §(d), council C3.5, which wins).

G-combo reads the t0 frame at 448 x 252 px (the 1280 x 720 detector frame scaled by 0.35 on both axes), as a 32 x 18
grid of 14-px patches, with the overlay mask (`datasets.overlay_mask(..., "overlay")`: graphics and the top band)
filled grey. A region is a boolean patch mask `[18, 32]`: the patches a shape touches at input resolution, dilated by
`MARGIN` patch (8-neighbourhood), so every region has at least one whole patch of margin.

    shooter      the shooter box of the feet_t0 replay (`attributes.Roles`)
    goalkeeper   the goalkeeper box, only when detected as a goalkeeper (`gk_detected`; never the proxy)
    ball         the own-frame ball box (no fallback ball: its box belongs to another frame)
    goal         approximate (C3.5): the attacked goal's post bases projected from the pitch plane with the frame's
                 homography, the posts drawn straight up with height = post-to-post pixel distance x 2.44 / 7.32. Valid
                 when both bases are visible (ground side of the horizon), the bases are at least `MIN_GOAL_PX` apart
                 and the quadrilateral touches the frame; else absent with `reason`.
    players      every person box (players, goalkeepers, referees)
    background   every patch outside persons, ball and goal (the dilated regions); descriptive only, no placebo

A region is `present` when its shape exists. Size is the number of patches. `overlay_share`: share of its patches the
overlay mask blanks anyway (masking them changes nothing).

Placebos (C3.5): the same patch mask translated, shape preserved: vertical shift within `Y_BAND` patch rows, any
horizontal shift, every translated patch on the projected pitch (`PITCH_SHARE` of its 3 x 3 sample points), outside
the overlay and off every patch a person box, the own ball or the goal shape touches (undilated: the placebo carries
its own margin, so it may border a region's margin but never covers a person, the ball or the goal). `N_PLACEBO` draws from the feasible
shifts, seeded by (shot, region): without replacement when there are at least `N_PLACEBO`, else with replacement, and
`n_feasible` is recorded. No feasible shift -> `placebo_status` "no_valid_placebo"; a mask is never shrunk.

Keep-only variants (C3.5) are complements: keep shooter + ball = ~(shooter | ball), keep persons = ~players, keep goal
region = ~goal; they need no table of their own.
"""

import hashlib

import cv2
import numpy as np
import pandas as pd

from xgpoc.data.datasets import overlay_mask
from xgpoc.detector.conventions import POSTS, align
from xgpoc.dino import attributes as at

INPUT_WH = (448, 252)
PATCH = 14
GRID_HW = (INPUT_WH[1] // PATCH, INPUT_WH[0] // PATCH)    # 18 rows, 32 columns
SCALE = INPUT_WH[0] / at.IMAGE_WH[0]
MARGIN = 1
GOAL_HEIGHT_M, GOAL_WIDTH_M = 2.44, 7.32
MIN_GOAL_PX = 4.0                                         # 720p
Y_BAND = 1
PITCH_SHARE = 0.5
N_PLACEBO = 3
SEED = 9
REGIONS = ("shooter", "goalkeeper", "ball", "goal", "players", "background")
PLACEBO_REGIONS = ("shooter", "goalkeeper", "ball", "goal", "players")
if INPUT_WH[1] / at.IMAGE_WH[1] != SCALE or INPUT_WH[0] % PATCH or INPUT_WH[1] % PATCH:
    raise ValueError("the G-combo input is not the detector frame scaled evenly onto whole patches")


# ------------------------------------------------------------------ patch masks
def empty() -> np.ndarray:
    return np.zeros(GRID_HW, bool)


def dilate(mask: np.ndarray, margin: int = MARGIN) -> np.ndarray:
    out = mask.copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            out |= shift(mask, dx, dy)
    return out


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """`mask` moved by (dx, dy) patches; what leaves the grid is dropped."""
    out = empty()
    h, w = GRID_HW
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
    out[ys, xs] = mask[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def pixels_to_patches(pixels: np.ndarray) -> np.ndarray:
    """Patches holding any pixel of an input-resolution boolean image `[252, 448]`."""
    h, w = GRID_HW
    return pixels.reshape(h, PATCH, w, PATCH).any(axis=(1, 3))


def box_patches(box: np.ndarray) -> np.ndarray:
    """Patches touched by a 720p xyxy box, before the margin."""
    x0, y0, x1, y1 = np.asarray(box, float) * SCALE
    h, w = GRID_HW
    c0, c1 = int(np.clip(np.floor(x0 / PATCH), 0, w - 1)), int(np.clip(np.ceil(x1 / PATCH) - 1, 0, w - 1))
    r0, r1 = int(np.clip(np.floor(y0 / PATCH), 0, h - 1)), int(np.clip(np.ceil(y1 / PATCH) - 1, 0, h - 1))
    out = empty()
    if x1 <= 0 or y1 <= 0 or x0 >= INPUT_WH[0] or y0 >= INPUT_WH[1]:
        return out
    out[r0:r1 + 1, c0:c1 + 1] = True
    return out


def polygon_patches(polygon: np.ndarray) -> np.ndarray:
    """Patches touched by a 720p polygon `(K, 2)`, before the margin."""
    img = np.zeros((INPUT_WH[1], INPUT_WH[0]), np.uint8)
    cv2.fillPoly(img, [np.round(np.asarray(polygon, float) * SCALE * 16).astype(np.int32)], 1, shift=4)
    return pixels_to_patches(img.astype(bool))


def overlay_patches() -> np.ndarray:
    """Patches the G-combo overlay mask blanks for more than half of their pixels."""
    blank = overlay_mask(INPUT_WH, "overlay")
    h, w = GRID_HW
    return blank.reshape(h, PATCH, w, PATCH).mean(axis=(1, 3)) > 0.5


def pitch_patches(H: np.ndarray) -> np.ndarray:
    """Patches whose 3 x 3 sample points lie on the pitch (ground side, inside 105 x 68 m) for `PITCH_SHARE` or more."""
    h, w = GRID_HW
    offsets = (np.arange(3) + 0.5) / 3 * PATCH
    us = (np.arange(w)[:, None] * PATCH + offsets).ravel() / SCALE
    vs = (np.arange(h)[:, None] * PATCH + offsets).ravel() / SCALE
    uv = np.stack(np.meshgrid(us, vs), -1).reshape(-1, 2)
    xy, ground = at.to_pitch(H, uv)
    with np.errstate(invalid="ignore"):
        on = ground & (xy[:, 0] >= 0) & (xy[:, 0] <= at.LENGTH) & (xy[:, 1] >= 0) & (xy[:, 1] <= at.WIDTH)
    return on.reshape(h, 3, w, 3).mean(axis=(1, 3)) >= PITCH_SHARE


# ------------------------------------------------------------------ goal
def goal_polygon(H: np.ndarray, attack_right: bool) -> tuple[np.ndarray | None, str]:
    """(720p quadrilateral base0, base1, top1, top0 of the attacked goal, "ok") or (None, reason) (module docstring)."""
    bases = align(np.vstack(POSTS), attack_right)          # align is its own inverse
    uv, visible = at.to_image(H, bases)
    if not visible.all():
        return None, "post_not_visible"
    width = float(np.hypot(*(uv[1] - uv[0])))
    if width < MIN_GOAL_PX:
        return None, "posts_too_close"
    up = np.array([0.0, width * GOAL_HEIGHT_M / GOAL_WIDTH_M])
    polygon = np.vstack([uv[0], uv[1], uv[1] - up, uv[0] - up])
    if not polygon_patches(polygon).any():
        return None, "outside_frame"
    return polygon, "ok"


# ------------------------------------------------------------------ placebos
def rng_for(shot_id: str, region: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{shot_id}:{region}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def feasible_shifts(mask: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Shifts (dx, dy), |dy| <= `Y_BAND`, that keep every patch of `mask` on the grid and inside `allowed`."""
    n = int(mask.sum())
    h, w = GRID_HW
    out = []
    for dy in range(-Y_BAND, Y_BAND + 1):
        for dx in range(-(w - 1), w):
            if dx == 0 and dy == 0:
                continue
            moved = shift(mask, dx, dy)
            if int(moved.sum()) == n and not (moved & ~allowed).any():
                out.append((dx, dy))
    return out


def placebos(mask: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> tuple[list[tuple[int, int]], int]:
    """(`N_PLACEBO` shifts, number of feasible shifts); no shift when none is feasible."""
    shifts = feasible_shifts(mask, allowed)
    if not shifts:
        return [], 0
    picks = rng.choice(len(shifts), N_PLACEBO, replace=len(shifts) < N_PLACEBO)
    return [shifts[i] for i in picks], len(shifts)


# ------------------------------------------------------------------ one shot
def pack(mask: np.ndarray) -> bytes:
    return np.packbits(mask.ravel()).tobytes()


def unpack(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, np.uint8))[: GRID_HW[0] * GRID_HW[1]].astype(bool).reshape(GRID_HW)


def shot_regions(shot_id: str, r: at.Roles, overlay: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    """(summary rows: one per region; mask rows: region masks and placebo masks; goal geometry) of one shot's t0 row."""
    masks, reasons = dict.fromkeys(REGIONS), dict.fromkeys(REGIONS, "ok")
    goal = {"goal_status": "no_frame", **{f"goal_{k}{i}": np.nan for i in range(4) for k in ("u", "v")}}
    pitch, exclusion = empty(), empty()
    if r.frame is None:
        reasons = dict.fromkeys(REGIONS, "no_frame")
    else:
        boxes = r.boxes()
        if len(boxes):
            persons = empty()
            for b in boxes:
                persons |= box_patches(b)
            masks["players"] = dilate(persons)
            exclusion |= persons
        else:
            reasons["players"] = "no_person"
        if r.shooter is not None:
            masks["shooter"] = dilate(box_patches(boxes[r.shooter]))
        else:
            reasons["shooter"] = "roles_missing" if not r.ok else "no_shooter"
        if r.gk_detected:
            masks["goalkeeper"] = dilate(box_patches(boxes[r.goalkeeper]))
        else:
            reasons["goalkeeper"] = "roles_missing" if not r.ok else ("proxy_only" if r.goalkeeper is not None else "no_goalkeeper")
        if r.own_ball is not None:
            exclusion |= box_patches(r.own_ball)
            masks["ball"] = dilate(box_patches(r.own_ball))
        else:
            reasons["ball"] = "roles_missing" if not r.ok else "no_own_ball"
        H = r.frame.homography
        if H is None:
            reasons["goal"] = goal["goal_status"] = "no_homography"
        elif not r.ok or r.s.attack_right is None:
            reasons["goal"] = goal["goal_status"] = "roles_missing" if not r.ok else "no_shot_location"
        else:
            polygon, status = goal_polygon(H, r.s.attack_right)
            goal["goal_status"] = status
            if polygon is None:
                reasons["goal"] = status
            else:
                exclusion |= polygon_patches(polygon)
                masks["goal"] = dilate(polygon_patches(polygon))
                goal |= {f"goal_{k}{i}": float(polygon[i, j]) for i in range(4) for j, k in enumerate(("u", "v"))}
        if H is not None:
            pitch = pitch_patches(H)
        dilated = empty()
        for name in ("players", "ball", "goal"):
            if masks[name] is not None:
                dilated |= masks[name]
        masks["background"] = ~dilated
    allowed = pitch & ~overlay & ~exclusion
    summary, rows = [], []
    for region in REGIONS:
        m = masks[region]
        row = {"shot_id": shot_id, "region": region, "present": m is not None, "reason": reasons[region],
               "n_patches": int(m.sum()) if m is not None else 0,
               "overlay_share": float((m & overlay).sum() / max(m.sum(), 1)) if m is not None else np.nan,
               "placebo_status": "not_applicable" if region not in PLACEBO_REGIONS else "region_absent",
               "n_feasible": 0}
        if m is not None:
            rows.append({"shot_id": shot_id, "region": region, "kind": "region", "draw": 0, "dx": 0, "dy": 0, "mask": pack(m)})
            if region in PLACEBO_REGIONS:
                shifts, n = placebos(m, allowed, rng_for(shot_id, region))
                row |= {"placebo_status": "ok" if shifts else "no_valid_placebo", "n_feasible": n}
                rows += [{"shot_id": shot_id, "region": region, "kind": "placebo", "draw": k, "dx": dx, "dy": dy,
                          "mask": pack(shift(m, dx, dy))} for k, (dx, dy) in enumerate(shifts)]
        summary.append(row)
    return summary, rows, {"shot_id": shot_id} | goal


def build(t0: dict[str, at.Roles]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(summary: shot x region; masks: region and placebo masks, packed; goal geometry per shot) for every shot."""
    overlay = overlay_patches()
    summary, masks, goals = [], [], []
    for shot_id in sorted(t0):
        s, m, g = shot_regions(shot_id, t0[shot_id], overlay)
        summary += s
        masks += m
        goals.append(g)
    return pd.DataFrame(summary), pd.DataFrame(masks), pd.DataFrame(goals)


def coverage(summary: pd.DataFrame) -> pd.DataFrame:
    """Per region: shots, present (count, share), size in patches among present (median, p10, p90, mean), overlay
    share (mean), placebo status counts, median feasible shifts."""
    rows = []
    for region, g in summary.groupby("region", sort=False):
        p = g[g.present]
        sizes = p.n_patches.to_numpy()
        row = {"region": region, "shots": len(g), "present": len(p), "share": float(len(p) / len(g)),
               "patches_median": float(np.median(sizes)) if len(p) else np.nan,
               "patches_p10": float(np.percentile(sizes, 10)) if len(p) else np.nan,
               "patches_p90": float(np.percentile(sizes, 90)) if len(p) else np.nan,
               "patches_mean": float(sizes.mean()) if len(p) else np.nan,
               "overlay_share_mean": float(p.overlay_share.mean()) if len(p) else np.nan,
               "n_feasible_median": float(p.n_feasible.median()) if len(p) else np.nan}
        row |= {f"placebo_{k}": int((g.placebo_status == k).sum())
                for k in ("ok", "no_valid_placebo", "region_absent", "not_applicable")}
        row |= {f"absent_{k}": int(v) for k, v in g[~g.present].reason.value_counts().items()}
        rows.append(row)
    return pd.DataFrame(rows)
