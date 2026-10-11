"""Features of the positional comparator (xg-vendor ticket 06): what StatsBomb knew at the strike, nothing after it.

`feature_table` is a pure function of the derived StatsBomb tables (`xgpoc.vendor.statsbomb_tables`): the shots table
(`event_json`, `freeze_frame_json`) and the context table (the key passes). It returns one row per shot with the
features below and no label. The geometry is the prototype's (`s2_ladder.shot_features`). Coordinates are StatsBomb
pitch units (120 x 80, yards); the goal is at x = 120, posts at y = 36 and y = 44.

The frozen feature lists (`RUNG_FEATURES`)
    dist_angle     dist, angle
    event          + x, abs_dy; first_time, under_pressure, one_on_one, open_goal, aerial_won, follows_dribble;
                   key pass: kp_has, kp_cross, kp_through, kp_cutback, kp_length; categorical body, technique, stype
                   (shot type), pattern (play pattern), kp_height, kp_type
    freeze_frame   + ff_n (players in the freeze frame), ff_def_cone / ff_mate_cone (outfield opponents / teammates in
                   the shooter-post-post triangle), ff_opp_1m / 3m / 5m (opponents within 1.5 / 3 / 5 units),
                   ff_near_opp (nearest outfield opponent), ff_opp_goalside (outfield opponents closer to the goal
                   centre than the shooter), ff_n_opp, ff_n_mate, ff_has_gk; goalkeeper: gk_dist_goal,
                   gk_dist_shooter, gk_in_cone, gk_off_line (signed distance from the line shooter -> goal centre),
                   gk_depth (120 - x)

Post-contact exclusions (`POST_CONTACT`): the outcome, the end location, StatsBomb's xG and everything recorded after
the strike (deflected, redirect, saved off target, saved to post, the event's duration, "out", the related events,
off camera), and the key pass's goal_assist / shot_assist flags, which name the shot's outcome. They are removed from
every event before a feature is computed (`pre_contact_shot`, `pre_contact_pass`), so a feature that reads one fails
with a KeyError, and `check_features` refuses a feature list that names one (`FORBIDDEN_TOKENS`).
"""

import json
import re

import numpy as np
import pandas as pd

GOAL = (120.0, 40.0)
POSTS = ((120.0, 36.0), (120.0, 44.0))

# Raw event fields recorded after the strike: (where, field) -> why it is excluded.
POST_CONTACT = {
    ("shot", "outcome"): "the result of the shot",
    ("shot", "end_location"): "where the ball ended (incl. its height at the goal line)",
    ("shot", "statsbomb_xg"): "the vendor's xG: the reference, never a feature",
    ("shot", "deflected"): "the shot was deflected after the strike",
    ("shot", "redirect"): "the shot was redirected after the strike",
    ("shot", "saved_off_target"): "goalkeeper save of a shot that was going wide",
    ("shot", "saved_to_post"): "goalkeeper save onto the post",
    ("event", "duration"): "time from the strike to the end of the ball's flight",
    ("event", "out"): "the ball went out after the shot",
    ("event", "related_events"): "links to later events (goalkeeper, block)",
    ("event", "off_camera"): "a broadcast-coverage flag of the event, not of the situation",
    ("key_pass", "goal_assist"): "the assisted shot was a goal",
    ("key_pass", "shot_assist"): "the assisted shot was not a goal (the complement of goal_assist)",
    ("key_pass", "assisted_shot_id"): "a link to the shot itself",
}
# Feature-name tokens (split on "_") that point at post-contact information.
FORBIDDEN_TOKENS = frozenset({"outcome", "end", "xg", "deflected", "redirect", "saved", "duration", "out", "related", "assist", "label",
                              "result", "scored", "statsbomb", "camera"})
FORBIDDEN_NAMES = frozenset({"goal", "is_goal", "y_true", "target"})

CATEGORICAL = ("body", "technique", "stype", "pattern", "kp_height", "kp_type")
DIST_ANGLE = ["dist", "angle"]
EVENT = [*DIST_ANGLE, "x", "abs_dy", "first_time", "under_pressure", "one_on_one", "open_goal", "aerial_won", "follows_dribble",
         "kp_has", "kp_cross", "kp_through", "kp_cutback", "kp_length", *CATEGORICAL]
FREEZE_FRAME = [*EVENT, "ff_n", "ff_def_cone", "ff_mate_cone", "ff_opp_1m", "ff_opp_3m", "ff_opp_5m", "ff_near_opp", "ff_opp_goalside",
                "ff_n_opp", "ff_n_mate", "ff_has_gk", "gk_dist_goal", "gk_dist_shooter", "gk_in_cone", "gk_off_line", "gk_depth"]
RUNG_FEATURES = {"dist_angle": DIST_ANGLE, "event": EVENT, "freeze_frame": FREEZE_FRAME}
KEY_COLUMNS = ["event_id", "match_id", "period", "shot_type"]
SHOT_INPUT_COLUMNS = ["event_id", "match_id", "period", "shot_type", "event_json", "freeze_frame_json"]


class PostContactFeature(ValueError):
    """A feature list names a post-contact quantity."""


def check_features(columns) -> None:
    """Raise `PostContactFeature` when a feature name is, or contains a token of, a post-contact quantity, and
    ValueError when it is not one of the frozen features (so nothing unreviewed reaches a model)."""
    known = set(FREEZE_FRAME)
    for column in columns:
        tokens = set(re.split(r"[_\W]+", column.lower()))
        if column.lower() in FORBIDDEN_NAMES or tokens & FORBIDDEN_TOKENS:
            raise PostContactFeature(f"feature {column!r} names post-contact information")
        if column not in known:
            raise ValueError(f"feature {column!r} is not in the frozen feature list")


def pre_contact_shot(event: dict) -> dict:
    """The shot event without its post-contact fields (`POST_CONTACT`); the input is not changed."""
    out = {k: v for k, v in event.items() if ("event", k) not in POST_CONTACT}
    out["shot"] = {k: v for k, v in event["shot"].items() if ("shot", k) not in POST_CONTACT}
    return out


def pre_contact_pass(event: dict) -> dict:
    """The key-pass event without the fields that name the assisted shot's outcome."""
    out = {k: v for k, v in event.items() if ("event", k) not in POST_CONTACT}
    out["pass"] = {k: v for k, v in event["pass"].items() if ("key_pass", k) not in POST_CONTACT}
    return out


def in_triangle(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    """Whether p lies in the triangle abc, edges included (signs of the three cross products agree)."""
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d = (side(p, a, b), side(p, b, c), side(p, c, a))
    return not (any(v < 0 for v in d) and any(v > 0 for v in d))


def event_features(shot: dict, key_pass: dict | None) -> dict:
    """Location, shot qualifiers and key-pass features of one pre-contact shot event (and its pre-contact key pass)."""
    x, y = (float(v) for v in shot["location"][:2])
    q = shot["shot"]
    a1, a2 = np.arctan2(POSTS[0][1] - y, POSTS[0][0] - x), np.arctan2(POSTS[1][1] - y, POSTS[1][0] - x)
    f = {"x": x, "y": y, "dist": float(np.hypot(GOAL[0] - x, GOAL[1] - y)), "angle": float(abs(a2 - a1)), "abs_dy": abs(y - GOAL[1]),
         "body": q["body_part"]["name"], "technique": q["technique"]["name"], "stype": q["type"]["name"],
         "pattern": shot["play_pattern"]["name"], "first_time": float(bool(q.get("first_time"))),
         "under_pressure": float(bool(shot.get("under_pressure"))), "one_on_one": float(bool(q.get("one_on_one"))),
         "open_goal": float(bool(q.get("open_goal"))), "aerial_won": float(bool(q.get("aerial_won"))),
         "follows_dribble": float(bool(q.get("follows_dribble")))}
    if key_pass is None:
        return f | {"kp_has": 0.0, "kp_cross": 0.0, "kp_through": 0.0, "kp_cutback": 0.0, "kp_height": "None", "kp_length": np.nan,
                    "kp_type": "None"}
    p = key_pass["pass"]
    through = bool(p.get("through_ball")) or p.get("technique", {}).get("name") == "Through Ball"
    return f | {"kp_has": 1.0, "kp_cross": float(bool(p.get("cross"))), "kp_through": float(through),
                "kp_cutback": float(bool(p.get("cut_back"))), "kp_height": p.get("height", {}).get("name", "NA"),
                "kp_length": float(p.get("length", np.nan)), "kp_type": p.get("type", {}).get("name", "Regular")}


def freeze_frame_features(x: float, y: float, frame: list[dict]) -> dict:
    """Freeze-frame features of a shot at (x, y). An empty frame gives ff_n = 0 and NaN elsewhere; goalkeeper features
    are NaN when no opponent is a goalkeeper (the first one that is, is the goalkeeper)."""
    out = dict.fromkeys(FREEZE_FRAME[len(EVENT):], np.nan) | {"ff_n": float(len(frame))}
    if not frame:
        return out
    opponents = [((float(p["location"][0]), float(p["location"][1])), p.get("position", {}).get("name") == "Goalkeeper")
                 for p in frame if not p["teammate"]]
    mates = [(float(p["location"][0]), float(p["location"][1])) for p in frame if p["teammate"]]
    outfield = [q for q, gk in opponents if not gk]
    keepers = [q for q, gk in opponents if gk]
    dist = float(np.hypot(GOAL[0] - x, GOAL[1] - y))

    def cone(q):
        return in_triangle(q, (x, y), *POSTS)

    def near(q):
        return float(np.hypot(q[0] - x, q[1] - y))

    out |= {"ff_def_cone": float(sum(map(cone, outfield))), "ff_mate_cone": float(sum(map(cone, mates))),
            "ff_opp_1m": float(sum(near(q) <= 1.5 for q, _ in opponents)), "ff_opp_3m": float(sum(near(q) <= 3 for q, _ in opponents)),
            "ff_opp_5m": float(sum(near(q) <= 5 for q, _ in opponents)), "ff_near_opp": min(map(near, outfield), default=np.nan),
            "ff_opp_goalside": float(sum(np.hypot(GOAL[0] - q[0], GOAL[1] - q[1]) < dist for q in outfield)),
            "ff_n_opp": float(len(opponents)), "ff_n_mate": float(len(mates)), "ff_has_gk": float(bool(keepers))}
    if keepers:
        gx, gy = keepers[0]
        vx, vy = GOAL[0] - x, GOAL[1] - y
        out |= {"gk_dist_goal": float(np.hypot(GOAL[0] - gx, GOAL[1] - gy)), "gk_dist_shooter": near((gx, gy)),
                "gk_in_cone": float(cone((gx, gy))), "gk_off_line": float(((gx - x) * vy - (gy - y) * vx) / (np.hypot(vx, vy) + 1e-9)),
                "gk_depth": GOAL[0] - gx}
    return out


def feature_table(shots: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """One row per shot, in the shots table's order: `KEY_COLUMNS`, then y and every feature of `FREEZE_FRAME`.
    Each event (and its key pass from `context`) passes through `pre_contact_*` first; no label is carried over."""
    missing = [c for c in SHOT_INPUT_COLUMNS if c not in shots.columns]
    if missing:
        raise ValueError(f"shots table lacks {missing}")
    if shots.event_id.duplicated().any():
        raise ValueError("duplicate event_id in the shots table")
    passes = context[context.type.eq("Pass")]
    key_passes = dict(zip(passes.event_id, passes.event_json, strict=True))
    rows = []
    for event_id, match_id, period, shot_type, event_json, frame_json in shots[SHOT_INPUT_COLUMNS].itertuples(index=False):
        shot = pre_contact_shot(json.loads(event_json))
        kp_id = shot["shot"].get("key_pass_id")
        if kp_id is not None and kp_id not in key_passes:
            raise ValueError(f"shot {event_id} names key pass {kp_id}, which the context table lacks")
        key_pass = pre_contact_pass(json.loads(key_passes[kp_id])) if kp_id is not None else None
        f = event_features(shot, key_pass)
        f |= freeze_frame_features(f["x"], f["y"], json.loads(frame_json) if isinstance(frame_json, str) else [])
        rows.append({"event_id": event_id, "match_id": int(match_id), "period": int(period), "shot_type": shot_type} | f)
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, "y", *FREEZE_FRAME])
