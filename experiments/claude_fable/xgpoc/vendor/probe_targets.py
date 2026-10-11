"""Probe targets of H3: positional quantities of each StatsBomb shot (xg-vendor ticket 09).

A pure function of the derived StatsBomb shots table (`xgpoc.vendor.statsbomb_tables`, columns `event_id`, `x`, `y`,
`body_part`, `freeze_frame_json`). The geometry is the prototype's (`s2_ladder.shot_features`), restricted to the
quantities the probes decode. Coordinates are StatsBomb pitch units (120 x 80, yards); the goal is at x = 120 with posts
at y = 36 and y = 44, so "within 3" is the prototype's "opponents within 3 m" in StatsBomb units.

Targets
    dist             distance from the shot location to the goal centre (120, 40)
    angle            angle subtended by the posts at the shot location (radians)
    ff_def_cone      outfield opponents inside the shooter-post-post triangle (edges included)
    def_cone_ge1/2   ff_def_cone >= 1 / >= 2 (binary)
    gk_depth         120 - x of the goalkeeper
    gk_off_line_abs  absolute distance of the goalkeeper from the line shooter -> goal centre (lateral offset)
    gk_dist_goal     goalkeeper distance to the goal centre
    ff_opp_3m        opponents (goalkeeper included) within 3 units of the shooter
    head             body part "Head" (binary)

Freeze-frame targets are NaN for a shot without a freeze frame, goalkeeper targets also for a freeze frame without an
opposing goalkeeper (the first opponent whose position is "Goalkeeper" is the goalkeeper).
"""

import json

import numpy as np
import pandas as pd

GOAL = (120.0, 40.0)
POSTS = ((120.0, 36.0), (120.0, 44.0))
NEAR = 3.0
TARGET_COLUMNS = ["dist", "angle", "ff_def_cone", "def_cone_ge1", "def_cone_ge2", "gk_depth", "gk_off_line_abs", "gk_dist_goal",
                  "ff_opp_3m", "head"]
BINARY_TARGETS = ("def_cone_ge1", "def_cone_ge2", "head")
SHOT_INPUT_COLUMNS = ["event_id", "x", "y", "body_part", "freeze_frame_json"]


def in_triangle(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    """Whether p lies in the triangle abc, edges included (signs of the three cross products agree)."""
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d = (side(p, a, b), side(p, b, c), side(p, c, a))
    return not (any(v < 0 for v in d) and any(v > 0 for v in d))


def freeze_frame_targets(x: float, y: float, freeze_frame: list[dict]) -> dict:
    """Freeze-frame targets of one shot at (x, y); an empty freeze frame gives NaN everywhere."""
    nan = dict.fromkeys(["ff_def_cone", "ff_opp_3m", "gk_depth", "gk_off_line_abs", "gk_dist_goal"], np.nan)
    if not freeze_frame:
        return nan
    shooter = (x, y)
    opponents = [p for p in freeze_frame if not p["teammate"]]
    is_gk = [(p.get("position") or {}).get("name") == "Goalkeeper" for p in opponents]
    locations = [(float(p["location"][0]), float(p["location"][1])) for p in opponents]
    outfield = [q for q, gk in zip(locations, is_gk, strict=True) if not gk]
    out = nan | {"ff_def_cone": float(sum(in_triangle(q, shooter, *POSTS) for q in outfield)),
                 "ff_opp_3m": float(sum(np.hypot(q[0] - x, q[1] - y) <= NEAR for q in locations))}
    keepers = [q for q, gk in zip(locations, is_gk, strict=True) if gk]
    if keepers:
        gx, gy = keepers[0]
        vx, vy = GOAL[0] - x, GOAL[1] - y
        out |= {"gk_depth": GOAL[0] - gx, "gk_dist_goal": float(np.hypot(GOAL[0] - gx, GOAL[1] - gy)),
                "gk_off_line_abs": float(abs((gx - x) * vy - (gy - y) * vx) / (np.hypot(vx, vy) + 1e-9))}
    return out


def probe_targets(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per shot, indexed by `event_id`, with the `TARGET_COLUMNS` (module docstring)."""
    missing = [c for c in SHOT_INPUT_COLUMNS if c not in shots.columns]
    if missing:
        raise ValueError(f"shots table lacks {missing}")
    if shots.event_id.duplicated().any():
        raise ValueError("duplicate event_id in the shots table")
    rows = []
    for event_id, x, y, body_part, frame in shots[SHOT_INPUT_COLUMNS].itertuples(index=False):
        x, y = float(x), float(y)
        a1, a2 = np.arctan2(POSTS[0][1] - y, POSTS[0][0] - x), np.arctan2(POSTS[1][1] - y, POSTS[1][0] - x)
        row = {"event_id": event_id, "dist": float(np.hypot(GOAL[0] - x, GOAL[1] - y)), "angle": float(abs(a2 - a1)),
               "head": float(body_part == "Head")}
        row |= freeze_frame_targets(x, y, json.loads(frame) if isinstance(frame, str) else [])
        rows.append(row)
    out = pd.DataFrame(rows).set_index("event_id")
    cone = out.ff_def_cone
    out["def_cone_ge1"] = (cone >= 1).astype(float).where(cone.notna())
    out["def_cone_ge2"] = (cone >= 2).astype(float).where(cone.notna())
    return out[TARGET_COLUMNS]


def targets_for_table(table: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """The targets of the table's rows, in table order (index = table positions); raises when a row's event is missing."""
    missing = ~table.statsbomb_event_id.isin(targets.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} table rows have no StatsBomb shot, e.g. {table.statsbomb_event_id[missing].iloc[0]}")
    return targets.loc[table.statsbomb_event_id.to_numpy(), TARGET_COLUMNS].reset_index(drop=True)


def compare_with_features(targets: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Per target: shots compared, NaN-pattern disagreements and max |difference| against the prototype's
    `sb_features.parquet` (columns `event_id`, `body`, `ff_def_cone`, ...; its binary targets are derived here as the
    prototype's `s3_probes.py` derived them)."""
    f = features.set_index("event_id")
    proto = pd.DataFrame({"dist": f.dist, "angle": f.angle, "ff_def_cone": f.ff_def_cone, "def_cone_ge1": (f.ff_def_cone >= 1).astype(float),
                          "def_cone_ge2": (f.ff_def_cone >= 2).astype(float), "gk_depth": f.gk_depth, "gk_off_line_abs": f.gk_off_line.abs(),
                          "gk_dist_goal": f.gk_dist_goal, "ff_opp_3m": f.ff_opp_3m, "head": f.body.eq("Head").astype(float)})
    # the prototype turned a missing cone count into 0 for the binary targets; compare only where it had a count
    proto.loc[f.ff_def_cone.isna(), ["def_cone_ge1", "def_cone_ge2"]] = np.nan
    common = targets.index.intersection(proto.index)
    rows = []
    for column in TARGET_COLUMNS:
        a, b = targets.loc[common, column], proto.loc[common, column]
        both = a.notna() & b.notna()
        rows.append({"target": column, "shots": len(common), "compared": int(both.sum()), "nan_mismatch": int((a.isna() != b.isna()).sum()),
                     "max_abs_diff": float((a[both] - b[both]).abs().max()) if both.any() else np.nan})
    return pd.DataFrame(rows)
