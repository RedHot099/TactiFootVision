"""Timing-source test (xg-vendor ticket 04): do the SoccerNet anchor and the StatsBomb timestamp share a clock?

The rule is fixed in `research/xg_vendor_timing_rule.md` (Polish), written and hashed before any non-shot event was
compared, and pinned here by `RULE_SHA256`; `scripts/vendor_timing_source.py` refuses another note. Everything here
is a pure function of its inputs:

    labels        Labels-v2 records (label, gameTime half, position, team, visibility of every annotation)
    half_offsets  the frozen half offsets of ticket 02 (game, half, sb_match_id, offset_ms, offset_source)
    match_map     ticket 02's match map (game, sb_match_id, sb_home, sb_away, reason)
    pairs         ticket 02's index shots (the paired ones give the shot residuals)
    timing        the StatsBomb timing table (xgpoc.vendor.statsbomb_tables)
    shots         the StatsBomb shots (match_id, event_id, period, timestamp_ms, team, outcome, event_json)

Residual r = (SoccerNet position - half offset) - StatsBomb time, in ms; positive: SoccerNet later.

Matching: per (match, half, label), the SoccerNet labels and the StatsBomb events of the mapped kind (`sb_label_events`)
are assigned one to one by `shot_pairing.assign` (minimal total |r|, an "unpaired" option costing the window,
pairs with |r| <= `WINDOW_MS[label]` kept). Team is not used. A substitution pair leaves the rule's set when its
StatsBomb substitution is a half-time one (period 2, at or before the period's first Kick Off pass) or when the
SoccerNet label lies before the StatsBomb period starts.

Verdict (`decide`), in order: too few substitutions -> "unresolved"; >= 90% of the rule's substitutions within
+-200 ms -> "shared"; median |r| of substitutions >= 1000 ms while the kicks agree (shots with leave-one-out offset
and corners, >= 80% within +-200 ms each) -> "independent"; else "unresolved".
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.vendor import match_pairing as mp
from xgpoc.vendor import shot_pairing as sp
from xgpoc.vendor import timing_verdict

RULE_NOTE = Path("research/xg_vendor_timing_rule.md")   # relative to experiments/claude_fable
RULE_SHA256 = "42dcf71018f28c078e5fa6ff83e71f137ac77e13c45fdb2534608314882029c5"

SUBSTITUTION, CORNER, BALL_OUT = "Substitution", "Corner", "Ball out of play"
LABELS = (SUBSTITUTION, "Throw-in", BALL_OUT, CORNER, "Kick-off", "Foul", "Offside")
WINDOW_MS = {SUBSTITUTION: 120_000} | {label: 10_000 for label in LABELS if label != SUBSTITUTION}
SENSITIVITY_WINDOWS_MS = (30_000, 60_000, 120_000, 300_000)
AGREE_MS = 200
LAG_MS = 1000
SHARED_SHARE = 0.90
KICK_SHARE = 0.80
MIN_SUBSTITUTIONS = 50
MIN_SUBSTITUTION_MATCH_RATE = 0.50
MIN_CORNERS = 50
HALF_TIME = "half-time substitution"
BEFORE_PERIOD = "SoccerNet label before the StatsBomb period starts"

SN_LABEL_COLUMNS = ["game", "half", "position_ms", "label", "team", "visibility"]
SB_EVENT_COLUMNS = ["label", "match_id", "period", "timestamp_ms", "event_id", "type", "detail", "team", "side", "half_time"]


def rule_note_sha256(root: Path) -> str:
    """sha256 of the rule note; raises when it is not the pinned note."""
    digest = hashlib.sha256((Path(root) / RULE_NOTE).read_bytes()).hexdigest()
    if digest != RULE_SHA256:
        raise ValueError(f"{RULE_NOTE} has sha256 {digest}, the rule pinned {RULE_SHA256}")
    return digest


# ---------------------------------------------------------------------------------------------------- inputs
def sn_labels(records: Iterable[Mapping], games: Iterable[str]) -> pd.DataFrame:
    """The Labels-v2 annotations of the compared labels (`LABELS`) in the given matches, every visibility."""
    wanted = set(games)
    rows = [{"game": mp.game_id(r), "half": int(a["gameTime"].split(" - ")[0]), "position_ms": int(a["position"]),
             "label": a["label"], "team": a["team"], "visibility": a.get("visibility", "")}
            for r in records if mp.game_id(r) in wanted for a in r["annotations"] if a["label"] in LABELS]
    return pd.DataFrame(rows, columns=SN_LABEL_COLUMNS)


def shot_out(shots: pd.DataFrame) -> pd.Series:
    """StatsBomb shots that left the pitch: `out` true in the event JSON, or outcome Off T."""
    flagged = shots.event_json.map(lambda text: bool(json.loads(text).get("out", False)))
    return flagged | shots.outcome.eq("Off T")


def sb_label_events(timing: pd.DataFrame, shots: pd.DataFrame, match_map: pd.DataFrame) -> pd.DataFrame:
    """The StatsBomb events each SoccerNet label is compared with (the rule note's table), in the paired matches.

    One row per (label, event); a throw-in or corner whose pass went out is both its own label and Ball out of play.
    `side` is home/away from the match map; `half_time` marks the half-time substitutions."""
    paired = match_map[match_map.reason.eq("paired")]
    home = dict(zip(paired.sb_match_id.astype(int), paired.sb_home, strict=True))
    away = dict(zip(paired.sb_match_id.astype(int), paired.sb_away, strict=True))
    timing = timing[timing.match_id.isin(home)]
    shots = shots[shots.match_id.isin(home)]
    is_pass = timing.type.eq("Pass")
    selections = {
        SUBSTITUTION: timing.type.eq("Substitution"),
        "Throw-in": is_pass & timing.detail.eq("Throw-in"),
        BALL_OUT: timing.ball_out,
        CORNER: is_pass & timing.detail.eq("Corner"),
        "Kick-off": is_pass & timing.detail.eq("Kick Off"),
        "Foul": timing.type.eq("Foul Committed"),
        "Offside": timing.soccernet_label.eq("Offside"),
    }
    keep = ["match_id", "period", "timestamp_ms", "event_id", "type", "detail", "team"]
    parts = [timing.loc[mask, keep].assign(label=label) for label, mask in selections.items()]
    out_shots = shots[shot_out(shots)].assign(type="Shot", detail=lambda s: s.outcome)
    parts.append(out_shots[keep].assign(label=BALL_OUT))
    events = pd.concat(parts, ignore_index=True)
    events["side"] = np.select([events.team.eq(events.match_id.map(home)), events.team.eq(events.match_id.map(away))],
                               ["home", "away"], default="?")
    kick_off = timing[is_pass & timing.detail.eq("Kick Off") & timing.period.eq(2)].groupby("match_id").timestamp_ms.min()
    second_start = events.match_id.map(kick_off).fillna(0)
    events["half_time"] = events.label.eq(SUBSTITUTION) & events.period.eq(2) & events.timestamp_ms.le(second_start)
    events = events.sort_values(["match_id", "period", "timestamp_ms", "label", "event_id"])
    return events.reset_index(drop=True)[SB_EVENT_COLUMNS]


# ---------------------------------------------------------------------------------------------------- matching
def aligned_halves(half_offsets: pd.DataFrame) -> pd.DataFrame:
    """The halves with a half offset: game, half, sb_match_id, offset_ms."""
    known = half_offsets[half_offsets.offset_source.ne("none") & half_offsets.offset_ms.notna()]
    return known[["game", "half", "sb_match_id", "offset_ms"]].astype({"sb_match_id": int})


def match_labels(sn: pd.DataFrame, sb: pd.DataFrame, half_offsets: pd.DataFrame, windows: Mapping[str, float] = WINDOW_MS) -> pd.DataFrame:
    """One row per SoccerNet label of an aligned half: its StatsBomb partner (or none), residual and exclusion.

    Labels of halves without a half offset are left out (the caller counts them). `excluded` is empty, `HALF_TIME`
    or `BEFORE_PERIOD` for substitution pairs, empty for every other label."""
    sn = sn[sn.label.isin(list(windows))].merge(aligned_halves(half_offsets), on=["game", "half"], how="inner")
    sn = sn.sort_values(["game", "half", "label", "position_ms", "team"]).reset_index(drop=True)
    sn["sb_clock_ms"] = sn.position_ms - sn.offset_ms
    candidates = sb.groupby(["match_id", "period", "label"]).indices
    partner = np.full(len(sn), -1)
    residual = np.full(len(sn), np.nan)
    sb_ms = sb.timestamp_ms.to_numpy(dtype=float)
    for (_, half, label), rows in sn.groupby(["game", "half", "label"]).indices.items():
        rows = np.asarray(rows)
        found = candidates.get((int(sn.sb_match_id.iat[rows[0]]), half, label))
        if found is None:
            continue
        found = np.asarray(found)
        for i, j, r in sp.assign(sn.position_ms.to_numpy(dtype=float)[rows], sb_ms[found], float(sn.offset_ms.iat[rows[0]]), windows[label]):
            partner[rows[i]], residual[rows[i]] = found[j], r
    table = sn.copy()
    table["matched"] = partner >= 0
    table["residual_ms"] = residual
    taken = sb.reset_index(drop=True).reindex(partner)   # partner -1: a row of NA
    for column in ("event_id", "type", "detail", "timestamp_ms", "team", "side", "half_time"):
        table[f"sb_{column}"] = taken[column].to_numpy()
    table["sb_timestamp_ms"] = table.sb_timestamp_ms.astype("Int64")
    table["sb_half_time"] = table.sb_half_time.astype("boolean")
    sides = table.team.isin(["home", "away"]) & table.sb_side.isin(["home", "away"])
    table["team_agree"] = table.team.eq(table.sb_side).astype("boolean").where(table.matched & sides)
    substitution = table.matched & table.label.eq(SUBSTITUTION)
    table["excluded"] = np.select([substitution & table.sb_half_time.fillna(False).astype(bool), substitution & table.sb_clock_ms.lt(0)],
                                  [HALF_TIME, BEFORE_PERIOD], default="")
    return table


def rule_substitutions(matches: pd.DataFrame) -> pd.DataFrame:
    """The rule's substitution set Z: matched substitution labels, half-time and pre-period pairs excluded."""
    return matches[matches.label.eq(SUBSTITUTION) & matches.matched & matches.excluded.eq("")]


def shot_residuals(pairs: pd.DataFrame) -> pd.DataFrame:
    """The paired shots of ticket 02 with the frozen residual and the leave-one-out one.

    The leave-one-out offset of a shot is the median raw difference (SoccerNet - StatsBomb) of the other paired shots
    of its half (NaN for a half's only pair)."""
    shots = pairs[pairs.paired.astype(bool)].sort_values(["game", "half", "position_ms", "shot_id"]).reset_index(drop=True)
    shots = shots[["shot_id", "game", "half", "split", "position_ms", "sb_event_id", "sb_timestamp_ms", "offset_ms", "residual_ms"]].copy()
    shots["raw_diff_ms"] = shots.position_ms - shots.sb_timestamp_ms.astype(float)
    loo = np.full(len(shots), np.nan)
    for rows in shots.groupby(["game", "half"]).indices.values():
        diff = shots.raw_diff_ms.to_numpy()[rows]
        if len(rows) > 1:
            loo[rows] = [np.median(np.delete(diff, k)) for k in range(len(rows))]
    shots["loo_offset_ms"] = loo
    shots["loo_residual_ms"] = shots.raw_diff_ms - loo
    return shots


# ---------------------------------------------------------------------------------------------------- statistics
def residual_stats(residual) -> dict:
    """Distribution of residuals (NaN dropped): quantiles, median |r|, shares within 200/500/1000 ms, exact zeros."""
    r = np.asarray(residual, dtype=float)
    r = r[~np.isnan(r)]
    names = ["median_ms", "q25_ms", "q75_ms", "p05_ms", "p10_ms", "p90_ms", "p95_ms", "median_abs_ms",
             "share_within_200ms", "share_within_500ms", "share_within_1000ms", "share_zero"]
    if not len(r):
        return {"n": 0} | dict.fromkeys(names, float("nan"))
    a = np.abs(r)
    values = [*np.percentile(r, [50, 25, 75, 5, 10, 90, 95]), np.median(a), (a <= AGREE_MS).mean(), (a <= 500).mean(),
              (a <= 1000).mean(), (r == 0).mean()]
    return {"n": int(len(r))} | {k: float(v) for k, v in zip(names, values, strict=True)}


def _agreement_row(event: str, subset: str, labels: pd.DataFrame | None, residual, candidates=None) -> dict:
    """One row of the agreement table; without `labels` (pairs only) the label count is the pair count."""
    stats = residual_stats(residual)
    n_matched = stats.pop("n")
    row = {"event": event, "subset": subset, "n_labels": len(labels) if labels is not None else n_matched,
           "n_candidates": candidates, "n_matched": n_matched}
    row["match_rate"] = row["n_matched"] / row["n_labels"] if labels is not None and len(labels) else float("nan")
    agree = labels.team_agree.dropna() if labels is not None else pd.Series(dtype=bool)
    row["n_team_known"], row["team_agree_share"] = len(agree), float(agree.mean()) if len(agree) else float("nan")
    return row | stats


def agreement_table(matches: pd.DataFrame, sb: pd.DataFrame, shots: pd.DataFrame, half_offsets: pd.DataFrame) -> pd.DataFrame:
    """Agreement distribution per event type: all labels, by visibility, and for substitutions the rule's set and the
    excluded pairs. Shots: the paired shots with the frozen and the leave-one-out offset."""
    aligned = aligned_halves(half_offsets).rename(columns={"sb_match_id": "match_id", "half": "period"})
    in_aligned = sb.merge(aligned[["match_id", "period"]], on=["match_id", "period"], how="inner")
    rows = []
    for label in LABELS:
        of_label = matches[matches.label.eq(label)]
        n_candidates = int(in_aligned.label.eq(label).sum())
        subsets = {"all": of_label, "visible": of_label[of_label.visibility.eq("visible")],
                   "not shown": of_label[of_label.visibility.eq("not shown")]}
        for subset, part in subsets.items():
            rows.append(_agreement_row(label, subset, part, part.residual_ms[part.matched], n_candidates if subset == "all" else None))
        if label == SUBSTITUTION:
            rule = rule_substitutions(matches)
            rows.append(_agreement_row(label, "rule (half-time and pre-period pairs excluded)", None, rule.residual_ms))
            excluded = of_label[of_label.excluded.ne("")]
            rows.append(_agreement_row(label, "excluded pairs", None, excluded.residual_ms))
    rows.append(_agreement_row("Shot (paired)", "frozen half offset", None, shots.residual_ms))
    rows.append(_agreement_row("Shot (paired)", "leave-one-out offset", None, shots.loo_residual_ms))
    return pd.DataFrame(rows).astype({"n_candidates": "Int64"})


def substitution_sensitivity(sn: pd.DataFrame, sb: pd.DataFrame, half_offsets: pd.DataFrame) -> pd.DataFrame:
    """The rule's substitution statistics under other windows and for "visible" labels only (descriptive)."""
    rows = []
    subs = sn[sn.label.eq(SUBSTITUTION)]
    for subset, labels in (("all", subs), ("visible", subs[subs.visibility.eq("visible")])):
        for window in SENSITIVITY_WINDOWS_MS:
            matches = match_labels(labels, sb, half_offsets, {SUBSTITUTION: window})
            rule = rule_substitutions(matches)
            rows.append({"subset": subset, "window_ms": window, "n_labels": len(matches), "n_matched": int(matches.matched.sum()),
                         "match_rate": float(matches.matched.mean()) if len(matches) else float("nan"),
                         "frozen_window": window == WINDOW_MS[SUBSTITUTION]} | residual_stats(rule.residual_ms))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- verdict
def decide(rule_residuals, substitution_match_rate: float, shot_loo_residuals, corner_residuals) -> dict:
    """The verdict of the rule note: verdict, deciding step (1-4), reason and the statistics it used."""
    z, s, c = (residual_stats(x) for x in (rule_residuals, shot_loo_residuals, corner_residuals))   # the note's Z, S and R
    statistics = {"n_substitutions": z["n"], "substitution_match_rate": float(substitution_match_rate),
                  "substitution_share_within_200ms": z["share_within_200ms"], "substitution_median_abs_ms": z["median_abs_ms"],
                  "substitution_median_ms": z["median_ms"], "n_shots": s["n"], "shot_loo_share_within_200ms": s["share_within_200ms"],
                  "n_corners": c["n"], "corner_share_within_200ms": c["share_within_200ms"]}
    kicks_agree = s["share_within_200ms"] >= KICK_SHARE and c["n"] >= MIN_CORNERS and c["share_within_200ms"] >= KICK_SHARE
    if z["n"] < MIN_SUBSTITUTIONS or not substitution_match_rate >= MIN_SUBSTITUTION_MATCH_RATE:
        verdict, step = "unresolved", 1
        reason = (f"too few substitutions: {z['n']} in the rule's set (needs {MIN_SUBSTITUTIONS}), "
                  f"match rate {substitution_match_rate:.3f} (needs {MIN_SUBSTITUTION_MATCH_RATE})")
    elif z["share_within_200ms"] >= SHARED_SHARE:
        verdict, step = "shared", 2
        reason = f"{z['share_within_200ms']:.3f} of substitutions within +-{AGREE_MS} ms (>= {SHARED_SHARE})"
    elif z["median_abs_ms"] >= LAG_MS and kicks_agree:
        verdict, step = "independent", 3
        reason = (f"substitutions lag: median |r| {z['median_abs_ms']:.0f} ms (>= {LAG_MS}); kicks agree: shots "
                  f"{s['share_within_200ms']:.3f}, corners {c['share_within_200ms']:.3f} within +-{AGREE_MS} ms (>= {KICK_SHARE})")
    else:
        verdict, step = "unresolved", 4
        reason = (f"neither: substitutions {z['share_within_200ms']:.3f} within +-{AGREE_MS} ms, median |r| {z['median_abs_ms']:.0f} ms; "
                  f"kicks agree: {kicks_agree} (shots {s['share_within_200ms']:.3f}, corners {c['share_within_200ms']:.3f}, n {c['n']})")
    return {"verdict": verdict, "step": step, "reason": reason, "statistics": statistics}


def thresholds() -> dict:
    return {"agree_ms": AGREE_MS, "lag_ms": LAG_MS, "shared_share": SHARED_SHARE, "kick_share": KICK_SHARE,
            "min_substitutions": MIN_SUBSTITUTIONS, "min_substitution_match_rate": MIN_SUBSTITUTION_MATCH_RATE,
            "min_corners": MIN_CORNERS, "windows_ms": dict(WINDOW_MS)}


def verdict_record(decision: Mapping, rule_sha256: str) -> dict:
    """The record ticket 09 reads (`timing_verdict`, schema `xgv-timing-verdict-1`), validated before it is returned."""
    record = {"schema": timing_verdict.SCHEMA, "verdict": decision["verdict"], "rule_sha256": rule_sha256, "rule_note": str(RULE_NOTE),
              "step": decision["step"], "reason": decision["reason"], "statistics": dict(decision["statistics"]),
              "thresholds": thresholds()}
    timing_verdict.parse_timing_verdict(record, RULE_SHA256)
    return record


# ---------------------------------------------------------------------------------------------------- entry point
@dataclass(frozen=True)
class TimingSource:
    label_matches: pd.DataFrame
    shot_residuals: pd.DataFrame
    agreement: pd.DataFrame
    substitution_sensitivity: pd.DataFrame
    verdict: dict
    summary: dict


def timing_source_test(labels: Iterable[Mapping], half_offsets: pd.DataFrame, match_map: pd.DataFrame, pairs: pd.DataFrame,
                       timing: pd.DataFrame, shots: pd.DataFrame, rule_sha256: str) -> TimingSource:
    """Run the test of the rule note on the given inputs (module docstring); `rule_sha256` must be the pinned note's."""
    if rule_sha256 != RULE_SHA256:
        raise ValueError(f"rule sha256 {rule_sha256} is not the pinned rule {RULE_SHA256}")
    games = match_map.game[match_map.reason.eq("paired")]
    sn = sn_labels(labels, games)
    sb = sb_label_events(timing, shots, match_map)
    matches = match_labels(sn, sb, half_offsets)
    shot_table = shot_residuals(pairs)
    subs = matches[matches.label.eq(SUBSTITUTION)]
    decision = decide(rule_substitutions(matches).residual_ms, float(subs.matched.mean()) if len(subs) else float("nan"),
                      shot_table.loo_residual_ms, matches.residual_ms[matches.label.eq(CORNER) & matches.matched])
    summary = {"verdict": decision["verdict"], "step": decision["step"], "sn_labels": int(len(sn)),
               "sn_labels_in_aligned_halves": int(len(matches)), "matched": matches.groupby("label").matched.sum().astype(int).to_dict()}
    return TimingSource(matches, shot_table, agreement_table(matches, sb, shot_table, half_offsets),
                        substitution_sensitivity(sn, sb, half_offsets), verdict_record(decision, rule_sha256), summary)
