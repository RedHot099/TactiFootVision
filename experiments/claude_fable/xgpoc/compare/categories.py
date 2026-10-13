"""The category dictionary of xg-compare (ticket 01): every category, its levels, its rule and its source column.

The dictionary is part of the frozen protocol (`research/xgc_compare_protocol.json`). It was fixed before any
category-level outcome or gap was computed. Every rule reads only descriptors that exist before the strike: StatsBomb
pre-contact features (`xgpoc.vendor.comparator_features`), the StatsBomb event sequence (`xgpoc.compare.sequence`), the
offset-0 detector descriptors of `cache/xgd_full3/E3` and the pairing data. The one exception is the post-shot group,
whose levels come from the label file and are flagged as information after the contact.

A category column holds one level name per shot, or None where the category is undefined (no goalkeeper in the freeze
frame, no detector descriptor for a StatsBomb-only shot, no key pass). An analysis of a level compares the shots in
the level with the other shots where the category is defined; undefined shots are left out of both sides.

Coordinates are StatsBomb pitch units (120 x 80, yards; the goal at x = 120 between y = 36 and y = 44). The C4
design's "4 m" for the goalkeeper (item 6a) is the comparator's `gk_depth`, in these units.

`label_free_cuts` computes the cut points from the primary population's descriptors, never from a label. The combined
population reuses them unchanged. There are two: the `n_players` terciles and the top quartile of the absolute timing
residual. The rule is `nearest_share_cut`. For each cut it takes the observed value whose share of shots is nearest to
the target share, because integer counts and millisecond residuals tie.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

GROUPS = ("events", "geometry", "freeze_frame", "sequence", "camera_view", "context", "post_shot")
ROLES = ("category", "sensitivity", "negative_control", "post_shot")
YES_NO = ("yes", "no")
PREFIX = "cat_"                       # the table column of category c is PREFIX + c
SAVE_OUTCOMES = frozenset({"Saved", "Saved to Post", "Saved Off Target"})
BLOCK_OUTCOMES = frozenset({"Blocked"})
FOOT = ("Right Foot", "Left Foot")
BOX_X = 102.0                         # the penalty area: x >= 102 and |y - 40| <= 22
BOX_HALF_WIDTH = 22.0
GOAL_Y = 40.0
TERCILE, QUARTILE = 1 / 3, 1 / 4


@dataclass(frozen=True)
class Category:
    """One category of the dictionary. `scan_levels` are the levels the open scan contrasts with the rest of the
    defined shots (empty: not in the scan); a two-level category lists only one, since the other is its mirror.
    `populations` names where the category is defined at all."""

    name: str
    group: str
    source: str
    rule: str
    levels: tuple[str, ...]
    scan_levels: tuple[str, ...]
    populations: tuple[str, ...] = ("primary", "combined")
    role: str = "category"

    def __post_init__(self) -> None:
        if self.group not in GROUPS or self.role not in ROLES:
            raise ValueError(f"category {self.name}: unknown group or role")
        if not set(self.scan_levels) <= set(self.levels):
            raise ValueError(f"category {self.name}: scan levels outside its levels")

    @property
    def column(self) -> str:
        return PREFIX + self.name


@dataclass(frozen=True)
class Cuts:
    """Label-free cut points from the primary population (module docstring)."""

    players_low_max: int        # tight view: n_players <= this
    players_high_min: int       # wide view: n_players >= this
    timing_top_min_ms: float    # |timing residual| >= this is the top quartile


def nearest_share_cut(values: np.ndarray, share: float, side: str) -> float:
    """The observed value c whose share of `values` at or below it (side "low") or at or above it (side "high") is
    nearest to `share`; ties go to the more extreme cut (smaller for "low", larger for "high")."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values) or side not in ("low", "high"):
        raise ValueError("need finite values and side 'low' or 'high'")
    candidates = np.unique(values)
    shares = np.array([(values <= c).mean() if side == "low" else (values >= c).mean() for c in candidates])
    distance = np.round(np.abs(shares - share), 12)
    best = np.flatnonzero(distance == distance.min())
    return float(candidates[best[0] if side == "low" else best[-1]])


def label_free_cuts(primary: pd.DataFrame) -> Cuts:
    """The cut points from the primary population's descriptor columns `e3_n_players` and `timing_residual_ms`."""
    players = primary.e3_n_players.to_numpy(float)
    return Cuts(players_low_max=int(nearest_share_cut(players, TERCILE, "low")),
                players_high_min=int(nearest_share_cut(players, TERCILE, "high")),
                timing_top_min_ms=nearest_share_cut(np.abs(primary.timing_residual_ms.to_numpy(float)), QUARTILE, "high"))


# ------------------------------------------------------------------ level rules
def _levels(values: pd.Series, mapping: Callable[[object], str | None]) -> pd.Series:
    return pd.Series([None if pd.isna(v) else mapping(v) for v in values], index=values.index, dtype=object)


def flag(values: pd.Series) -> pd.Series:
    """1/0 (or True/False) as "yes"/"no"; missing stays undefined."""
    return _levels(values, lambda v: "yes" if bool(v) else "no")


def bins(values: pd.Series, edges: tuple[float, ...], labels: tuple[str, ...]) -> pd.Series:
    """Left-closed bins [edges[i], edges[i + 1]); the last bin is open to the right."""
    if len(labels) != len(edges):
        raise ValueError("one label per left edge")
    return _levels(values, lambda v: labels[int(np.searchsorted(edges, v, side="right")) - 1] if v >= edges[0] else None)


def text(values: pd.Series) -> pd.Series:
    return _levels(values, str)


def outside_box(x: pd.Series, y: pd.Series) -> pd.Series:
    """"yes" when the shot is outside the penalty area: x < 102 or |y - 40| > 22."""
    outside = (x.to_numpy(float) < BOX_X) | (np.abs(y.to_numpy(float) - GOAL_Y) > BOX_HALF_WIDTH)
    return pd.Series(np.where(outside, "yes", "no"), index=x.index, dtype=object)


def rebound_subtype(rebound: pd.Series, previous_outcome: pd.Series) -> pd.Series:
    """For a 5 s rebound: after_save, after_block or after_other by the previous shot's StatsBomb outcome; none for a
    shot that is not a rebound."""
    def one(is_rebound, outcome):
        if is_rebound != "yes":
            return "none"
        return "after_save" if outcome in SAVE_OUTCOMES else "after_block" if outcome in BLOCK_OUTCOMES else "after_other"
    return pd.Series([one(r, o) for r, o in zip(rebound, previous_outcome, strict=True)], index=rebound.index, dtype=object)


def minute_bin(minute: pd.Series, period: pd.Series) -> pd.Series:
    """First half 0-14, 15-29, 30+ (stoppage included); second half 45-59, 60-74, 75+; extra time apart."""
    def one(m, p):
        if p == 1:
            return "1h_00_14" if m < 15 else "1h_15_29" if m < 30 else "1h_30_end"
        if p == 2:
            return "2h_45_59" if m < 60 else "2h_60_74" if m < 75 else "2h_75_end"
        return "extra_time"
    return pd.Series([one(m, p) for m, p in zip(minute, period, strict=True)], index=minute.index, dtype=object)


def players_tercile(n_players: pd.Series, cuts: Cuts) -> pd.Series:
    return _levels(n_players, lambda v: "low" if v <= cuts.players_low_max else "high" if v >= cuts.players_high_min else "mid")


def timing_top(residual_ms: pd.Series, cuts: Cuts) -> pd.Series:
    return _levels(residual_ms, lambda v: "yes" if abs(v) >= cuts.timing_top_min_ms else "no")


def half_name(period: pd.Series) -> pd.Series:
    return _levels(period, lambda p: {1: "first", 2: "second"}.get(int(p), "extra_time"))


# ------------------------------------------------------------------ the dictionary
SB = "StatsBomb pre-contact feature (xgpoc.vendor.comparator_features)"
E3 = "detector descriptor at offset 0 (cache/xgd_full3/E3, offset_ms == 0)"
DIST_EDGES, DIST_LABELS = (0.0, 6.0, 12.0, 18.0, 25.0), ("00_06", "06_12", "12_18", "18_25", "25_plus")
ANGLE_EDGES_DEG, ANGLE_LABELS = (0.0, 15.0, 20.0, 30.0, 45.0), ("00_15", "15_20", "20_30", "30_45", "45_plus")
GK_DEPTH_EDGES, GK_DEPTH_LABELS = (0.0, 2.0, 4.0), ("00_02", "02_04", "04_plus")
GK_LATERAL_EDGES, GK_LATERAL_LABELS = (0.0, 0.5, 1.5), ("00_0.5", "0.5_1.5", "1.5_plus")
NEAREST_EDGES, NEAREST_LABELS = (0.0, 1.5, 3.0), ("00_1.5", "1.5_3", "3_plus")
KEYPOINT_EDGES, KEYPOINT_LABELS = (0.0, 9.0, 10.0, 11.0), ("le_8", "9", "10", "ge_11")
LEAGUE_SEASONS = ("england_epl/2015-2016", "france_ligue-1/2015-2016", "germany_bundesliga/2015-2016", "italy_serie-a/2015-2016",
                  "spain_laliga/2014-2015", "spain_laliga/2015-2016", "spain_laliga/2016-2017")
MINUTE_LEVELS = ("1h_00_14", "1h_15_29", "1h_30_end", "2h_45_59", "2h_60_74", "2h_75_end", "extra_time")
SB_OUTCOMES = ("Goal", "Saved", "Saved to Post", "Saved Off Target", "Blocked", "Off T", "Post", "Wayward")
PRIMARY_ONLY = ("primary",)


def _binary(name: str, group: str, source: str, rule: str, scan: bool = True, **kw) -> Category:
    return Category(name, group, source, rule, YES_NO, ("yes",) if scan else (), **kw)


CATEGORIES: tuple[Category, ...] = (
    # events
    Category("body", "events", f"sb_body: {SB} `body`", "StatsBomb body part", ("Right Foot", "Left Foot", "Head", "Other"),
             ("Right Foot", "Left Foot", "Head", "Other")),
    Category("technique", "events", f"sb_technique: {SB} `technique`", "StatsBomb shot technique",
             ("Normal", "Half Volley", "Volley", "Lob", "Overhead Kick", "Diving Header", "Backheel"),
             ("Normal", "Half Volley", "Volley", "Lob", "Overhead Kick", "Diving Header", "Backheel")),
    Category("shot_type", "events", f"sb_stype: {SB} `stype`", "StatsBomb shot type (penalties are out of every population)",
             ("Open Play", "Free Kick", "Corner", "Kick Off"), ("Free Kick",)),
    Category("play_pattern", "events", f"sb_pattern: {SB} `pattern`", "StatsBomb play pattern of the possession",
             ("Regular Play", "From Free Kick", "From Throw In", "From Corner", "From Counter", "From Goal Kick", "From Keeper",
              "From Kick Off", "Other"),
             ("Regular Play", "From Free Kick", "From Throw In", "From Corner", "From Counter", "From Goal Kick", "From Keeper",
              "From Kick Off", "Other")),
    _binary("key_pass_cross", "events", f"sb_kp_cross: {SB} `kp_cross`", "the key pass is a cross (pass.cross)"),
    _binary("key_pass_through", "events", f"sb_kp_through: {SB} `kp_through`", "the key pass is a through ball"),
    _binary("key_pass_cutback", "events", f"sb_kp_cutback: {SB} `kp_cutback`", "the key pass is a cut-back (pass.cut_back)"),
    _binary("key_pass_none", "events", f"sb_kp_has: {SB} `kp_has`", "the shot has no key pass (kp_has == 0)"),
    Category("key_pass_height", "events", f"sb_kp_height: {SB} `kp_height`",
             "StatsBomb height of the key pass; undefined without a key pass", ("Ground Pass", "Low Pass", "High Pass"),
             ("Ground Pass", "Low Pass", "High Pass")),
    _binary("first_time", "events", f"sb_first_time: {SB} `first_time`", "StatsBomb first-time flag"),
    _binary("under_pressure", "events", f"sb_under_pressure: {SB} `under_pressure`", "StatsBomb under-pressure flag"),
    _binary("one_on_one", "events", f"sb_one_on_one: {SB} `one_on_one`", "StatsBomb one-on-one flag"),
    _binary("open_goal", "events", f"sb_open_goal: {SB} `open_goal`", "StatsBomb open-goal flag (item 6b, descriptive)"),
    _binary("follows_dribble", "events", f"sb_follows_dribble: {SB} `follows_dribble`", "StatsBomb follows-dribble flag"),
    _binary("aerial_won", "events", f"sb_aerial_won: {SB} `aerial_won`", "StatsBomb aerial-won flag"),
    # geometry
    Category("distance", "geometry", f"sb_dist: {SB} `dist` (distance to the goal centre, yards)",
             "bins [0, 6), [6, 12), [12, 18), [18, 25), [25, inf)", DIST_LABELS, DIST_LABELS),
    Category("angle", "geometry", f"sb_angle: {SB} `angle` (goal-mouth angle, radians, binned in degrees)",
             "bins [0, 15), [15, 20), [20, 30), [30, 45), [45, 180] degrees", ANGLE_LABELS, ANGLE_LABELS),
    _binary("outside_box", "geometry", f"sb_x, sb_y: {SB} `x`, `y`", "outside the penalty area: x < 102 or |y - 40| > 22"),
    # freeze frame
    Category("defenders_in_cone", "freeze_frame", f"sb_ff_def_cone: {SB} `ff_def_cone`",
             "outfield opponents in the shooter-post-post triangle: 0, 1, 2 or more", ("0", "1", "2_plus"), ("0", "1", "2_plus")),
    Category("gk_depth", "freeze_frame", f"sb_gk_depth: {SB} `gk_depth` (120 - goalkeeper x, yards)",
             "bins [0, 2), [2, 4), [4, inf), a goalkeeper behind the line counts as 0; undefined without a goalkeeper in the freeze frame", GK_DEPTH_LABELS, GK_DEPTH_LABELS),
    Category("gk_off_line", "freeze_frame", f"sb_gk_off_line: {SB} `gk_off_line` (signed distance from the shooter-goal line)",
             "bins of |gk_off_line| [0, 0.5), [0.5, 1.5), [1.5, inf); undefined without a goalkeeper", GK_LATERAL_LABELS,
             GK_LATERAL_LABELS),
    Category("nearest_opponent", "freeze_frame", f"sb_ff_near_opp: {SB} `ff_near_opp` (nearest outfield opponent, yards)",
             "bins [0, 1.5), [1.5, 3), [3, inf)", NEAREST_LABELS, NEAREST_LABELS),
    # sequence
    _binary("rebound_5s", "sequence", "seq_prev_*: raw StatsBomb events (xgpoc.compare.sequence)",
            "the previous shot of the same match, period and possession is by the same team and 0-5 s earlier"),
    _binary("rebound_3s", "sequence", "seq_prev_*: raw StatsBomb events (xgpoc.compare.sequence)",
            "as rebound_5s with 0-3 s (sensitivity check of item 2)", scan=False, role="sensitivity"),
    Category("rebound_subtype", "sequence", "seq_prev_outcome: StatsBomb outcome of the previous shot",
             "for a 5 s rebound: after_save (Saved, Saved to Post, Saved Off Target), after_block (Blocked), after_other; "
             "none otherwise", ("after_save", "after_block", "after_other", "none"), ("after_save", "after_block", "after_other")),
    # camera view (primary only: StatsBomb-only shots have no detector descriptor)
    Category("homography", "camera_view", f"e3_homography_ok: {E3}", "the pitch homography was available",
             ("ok", "failed"), ("failed",), PRIMARY_ONLY),
    Category("players_visible", "camera_view", f"e3_n_players: {E3}",
             "terciles of the number of detected players: low <= players_low_max, high >= players_high_min (label-free cuts)",
             ("low", "mid", "high"), ("low", "mid", "high"), PRIMARY_ONLY),
    Category("keypoints_visible", "camera_view", f"e3_n_keypoints: {E3} (a zoom proxy)",
             "number of detected pitch keypoints: <= 8, 9, 10, >= 11", KEYPOINT_LABELS, KEYPOINT_LABELS, PRIMARY_ONLY),
    _binary("gk_visible", "camera_view", f"e3_gk_detected: {E3}", "the goalkeeper was detected", populations=PRIMARY_ONLY),
    _binary("ball_detected", "camera_view", f"e3_ball_ok: {E3}", "the ball was detected", populations=PRIMARY_ONLY),
    # context
    Category("league_season", "context", "league_season: the SoccerNet match path", "league and season of the match",
             LEAGUE_SEASONS, LEAGUE_SEASONS),
    Category("minute", "context", "sb_minute, half: StatsBomb minute and period of the shot",
             "first half 0-14, 15-29, 30 to the end; second half 45-59, 60-74, 75 to the end; extra time", MINUTE_LEVELS,
             MINUTE_LEVELS),
    _binary("timing_residual_top", "context", "timing_residual_ms: SoccerNet anchor minus aligned StatsBomb time",
            "|timing residual| >= timing_top_min_ms (label-free top-quartile cut)", populations=PRIMARY_ONLY),
    Category("half", "context", "half: the period of the shot", "first or second half (negative control)",
             ("first", "second", "extra_time"), (), role="negative_control"),
    # post-shot: from the label file only, information after the contact; descriptive, never in the scan
    _binary("on_target_sn", "post_shot", "labels: on_target_sn (SoccerNet 'Shots on target', counts blocked shots)",
            "SoccerNet on target", scan=False, populations=PRIMARY_ONLY, role="post_shot"),
    Category("sb_outcome", "post_shot", "labels: sb_outcome (StatsBomb outcome of the shot)", "StatsBomb outcome", SB_OUTCOMES,
             (), role="post_shot"),
)
BY_NAME = {c.name: c for c in CATEGORIES}
PRE_SHOT = tuple(c for c in CATEGORIES if c.role != "post_shot")


def derive(descriptors: pd.DataFrame, cuts: Cuts) -> pd.DataFrame:
    """The pre-shot category columns (`PRE_SHOT`, named `cat_<name>`) of a table of descriptor columns, same index.
    A level outside the category's levels raises, so an unforeseen value never slips in silently."""
    d = descriptors
    angle_deg = pd.Series(np.degrees(d.sb_angle.to_numpy(float)), index=d.index)
    cone = _levels(d.sb_ff_def_cone, lambda v: "2_plus" if v >= 2 else str(int(v)))
    rebound = flag(d.seq_rebound_5s)
    has_kp = d.sb_kp_has.astype(float).eq(1)
    columns = {
        "body": text(d.sb_body), "technique": text(d.sb_technique), "shot_type": text(d.sb_stype), "play_pattern": text(d.sb_pattern),
        "key_pass_cross": flag(d.sb_kp_cross), "key_pass_through": flag(d.sb_kp_through), "key_pass_cutback": flag(d.sb_kp_cutback),
        "key_pass_none": flag(1 - d.sb_kp_has.astype(float)), "key_pass_height": text(d.sb_kp_height.where(has_kp)),
        "first_time": flag(d.sb_first_time), "under_pressure": flag(d.sb_under_pressure), "one_on_one": flag(d.sb_one_on_one),
        "open_goal": flag(d.sb_open_goal), "follows_dribble": flag(d.sb_follows_dribble), "aerial_won": flag(d.sb_aerial_won),
        "distance": bins(d.sb_dist, DIST_EDGES, DIST_LABELS), "angle": bins(angle_deg, ANGLE_EDGES_DEG, ANGLE_LABELS),
        "outside_box": outside_box(d.sb_x, d.sb_y), "defenders_in_cone": cone,
        "gk_depth": bins(d.sb_gk_depth.clip(lower=0), GK_DEPTH_EDGES, GK_DEPTH_LABELS),
        "gk_off_line": bins(d.sb_gk_off_line.abs(), GK_LATERAL_EDGES, GK_LATERAL_LABELS),
        "nearest_opponent": bins(d.sb_ff_near_opp, NEAREST_EDGES, NEAREST_LABELS),
        "rebound_5s": rebound, "rebound_3s": flag(d.seq_rebound_3s),
        "rebound_subtype": rebound_subtype(rebound, d.seq_prev_outcome),
        "homography": _levels(d.e3_homography_ok, lambda v: "ok" if v else "failed"),
        "players_visible": players_tercile(d.e3_n_players, cuts),
        "keypoints_visible": bins(d.e3_n_keypoints, KEYPOINT_EDGES, KEYPOINT_LABELS),
        "gk_visible": flag(d.e3_gk_detected), "ball_detected": flag(d.e3_ball_ok),
        "league_season": text(d.league_season), "minute": minute_bin(d.sb_minute, d.half),
        "timing_residual_top": timing_top(d.timing_residual_ms, cuts), "half": half_name(d.half),
    }
    if set(columns) != {c.name for c in PRE_SHOT}:
        raise ValueError("derive() and the dictionary disagree on the pre-shot categories")
    out = pd.DataFrame({BY_NAME[name].column: values for name, values in columns.items()}, index=d.index)
    for category in PRE_SHOT:
        unknown = set(out[category.column].dropna()) - set(category.levels)
        if unknown:
            raise ValueError(f"category {category.name}: values outside its levels: {sorted(unknown)}")
    return out[[c.column for c in PRE_SHOT]]


def dictionary_record(cuts: Cuts) -> dict:
    """The dictionary as JSON-ready data, with the label-free cuts filled in."""
    return {"cuts": asdict(cuts), "categories": [{**asdict(c), "column": c.column} for c in CATEGORIES]}
