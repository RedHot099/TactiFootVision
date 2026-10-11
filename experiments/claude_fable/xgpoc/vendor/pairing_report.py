"""The pairing report: counts, window sensitivity, disagreement lists with reasons, kill rule K0.

`build_report` turns the tables of one pairing run into a JSON-able dict; `check_k0` enforces the kill rule on it and
`report_text` serialises it. Nothing here reads a file.
"""

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from xgpoc.vendor import match_pairing as mp
from xgpoc.vendor import shot_pairing as sp

WINDOWS_MS = (250, 500, 1000, 2500, 5000)
K0_MIN_PAIRED = 0.85
K0_MIN_GOAL_AGREEMENT = 0.99
ANCHOR_CONVENTION = "statsbomb_time_plus_half_offset"
NEAR_GOAL_MS = 10_000      # reasons for label disagreements look this far for a neighbouring goal
RESIDUAL_TRUST_MS = 200    # a pair this tight is the same event for practical purposes


class K0Failed(RuntimeError):
    """Kill rule K0: fewer than 85% of the index shots paired, or goal agreement below 99%."""


def check_k0(report: Mapping) -> None:
    k0 = report["k0"]
    if not k0["passed"]:
        raise K0Failed(f"K0 failed: {k0['paired_rate']:.4f} of the index shots paired (need >= {K0_MIN_PAIRED}), "
                       f"goal agreement {k0['goal_agreement']:.4f} (need >= {K0_MIN_GOAL_AGREEMENT})")


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else float("nan")


def _quantiles(values, qs=(10, 50, 90), digits=0) -> list:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return np.percentile(values, qs).round(digits).tolist() if len(values) else []


def _sensitivity(halves: sp.Halves, offsets: dict, main: pd.DataFrame) -> dict:
    base = set(zip(main.sn_row, main.sb_row, strict=True))
    total = len(halves.sn)
    out = {}
    for window in WINDOWS_MS:
        for name, kwargs in (("offset", {}), ("no_offset", {"use_offset": False}), ("offset_same_side", {"same_side": True})):
            found = sp.run_pairing(halves, offsets, window, **kwargs)
            agree = halves.sn.goal.to_numpy()[found.sn_row] == halves.sb.goal.to_numpy()[found.sb_row]
            out[f"{name}_w{window}"] = {
                "window_ms": window, "rule": name, "pairs": len(found), "rate": _rate(len(found), total),
                "goal_agreement": round(float(agree.mean()), 4) if len(found) else None,
                "same_pairs_as_main": len(base & set(zip(found.sn_row, found.sb_row, strict=True)))}
    return out


def _team_reason(row, halves: sp.Halves) -> str:
    """Why a pair's SoccerNet team differs from the StatsBomb shot's side."""
    sb_rows = halves.sb_rows.get((row.game, row.half), np.empty(0, dtype=int))
    sb = halves.sb.iloc[sb_rows]
    near = sb[(np.abs(row.position_ms - row.offset_ms - sb.timestamp_ms) <= sp.WINDOW_MS) & sb.side.eq(row.team)]
    other = near[near.event_id != row.sb_event_id]
    if len(other):
        gap = float(np.abs(row.position_ms - row.offset_ms - other.timestamp_ms).min())
        return f"StatsBomb has a {row.team} shot {gap:.0f} ms from this shot: possible pairing error between two shots"
    if abs(row.residual_ms) <= RESIDUAL_TRUST_MS:
        return f"tight pair (|residual| {abs(row.residual_ms):.0f} ms) and no other shot of that team: the two sources disagree on the team"
    return f"loose pair (|residual| {abs(row.residual_ms):.0f} ms) and no other shot of that team: possible pairing error or team error"


def _goal_reason(row, halves: sp.Halves, pairs: pd.DataFrame, goal_labels: pd.DataFrame) -> str:
    """Why a pair's SoccerNet goal flag differs from the StatsBomb outcome."""
    clock = row.position_ms - row.offset_ms
    if row.goal:   # SoccerNet goal, StatsBomb not
        sb = halves.sb.iloc[halves.sb_rows.get((row.game, row.half), np.empty(0, dtype=int))]
        goals = sb[sb.goal & (np.abs(clock - sb.timestamp_ms) <= NEAR_GOAL_MS)]
        if len(goals):
            lag = clock - goals.timestamp_ms
            return (f"StatsBomb outcome {row.sb_outcome}, but StatsBomb credits a goal {lag.iloc[np.abs(lag).argmin()]:+.0f} ms from it: "
                    "probably a rebound chain: the goal is attributed to another shot")
        return f"StatsBomb outcome {row.sb_outcome} and no StatsBomb goal within {NEAR_GOAL_MS // 1000} s: the sources label this shot differently"
    labels = goal_labels[goal_labels.game.eq(row.game) & goal_labels.half.eq(row.half)]
    lag = labels.position_ms - row.position_ms
    labels, lag = labels[lag.between(*sp.RECOVER_MS)], lag[lag.between(*sp.RECOVER_MS)]
    if len(labels):
        nearest = np.abs(lag).argmin()
        team, gap = labels.team.iloc[nearest], lag.iloc[nearest]
        if team != row.team:
            return (f"StatsBomb goal; the SoccerNet Goal label at {gap:+.0f} ms is for the {team} team but the shot is labelled {row.team}: "
                    "SoccerNet did not credit the shot (team label of shot and goal differ)")
        return f"StatsBomb goal; a SoccerNet Goal label at {gap:+.0f} ms exists but was attributed to another shot"
    here = pairs[pairs.game.eq(row.game) & pairs.half.eq(row.half) & pairs.goal.astype(bool) & pairs.shot_id.ne(row.shot_id)]
    near = here[(here.position_ms - row.position_ms).abs() <= NEAR_GOAL_MS]
    if len(near):
        return "StatsBomb goal; SoccerNet credits a goal to another shot nearby (probably a rebound chain)"
    return "StatsBomb goal and no SoccerNet Goal label within the recovery window: SoccerNet has no goal here"


def _disagreements(pairs: pd.DataFrame, halves: sp.Halves, goal_labels: pd.DataFrame) -> dict:
    paired = pairs[pairs.paired]
    team = paired[paired.team_agree.eq(False).fillna(False).astype(bool)]
    label = paired[paired.goal.astype(bool) != paired.sb_goal.astype(bool)]
    keep = ["shot_id", "game", "half", "position_ms", "residual_ms", "team", "sb_side", "sb_event_id", "goal", "sb_goal", "sb_outcome"]

    def listing(frame, reason):
        rows = frame[keep].astype(object).where(frame[keep].notna(), None).to_dict("records")
        for record, row in zip(rows, frame.itertuples(), strict=True):
            record["reason"] = reason(row)
        return rows

    return {"team_side": listing(team, lambda r: _team_reason(r, halves)),
            "goal_label": listing(label, lambda r: _goal_reason(r, halves, pairs, goal_labels))}


def _match_report(match_map: pd.DataFrame, rejected: pd.DataFrame, clubs: Mapping) -> dict:
    paired = match_map[match_map.reason.eq("paired")]
    by_season = match_map.assign(p=match_map.reason.eq("paired")).groupby(["league", "season"]).p.agg(["sum", "count"]).astype(int)
    return {"sn_matches": len(match_map), "paired": len(paired), "reasons": match_map.reason.value_counts().to_dict(),
            "paired_by_league_season": {"/".join(k): f"{r['sum']}/{r['count']}" for k, r in by_season.iterrows()},
            "min_name_sim_paired": round(float(paired.name_sim.min()), 3) if len(paired) else None,
            "lowest_name_sim_paired": paired.nsmallest(5, "name_sim")[["sn_home", "sb_home", "sn_away", "sb_away", "name_sim"]].round(2).values.tolist(),
            "sb_match_used_twice": int(paired.sb_match_id.duplicated().sum()),
            "name_rejected_candidates": rejected.astype(object).where(rejected.notna(), None).to_dict("records"),
            "clubs_without_counterpart": {k: [list(v) for v in values] for k, values in clubs.items()}}


def _shot_report(pairs: pd.DataFrame, halves: sp.Halves, offsets: dict, half_offsets: pd.DataFrame) -> dict:
    paired = pairs[pairs.paired]
    unpaired = pairs[~pairs.paired]
    goal, sb_goal = paired.goal.astype(bool), paired.sb_goal.astype(bool)
    on_sb = paired.sb_outcome.isin(["Goal", "Saved", "Saved To Post"])
    on_sn = paired.label.eq("Shots on target")
    known = half_offsets.offset_ms.dropna()
    residual = paired.residual_ms.to_numpy(float)
    return {
        "window_ms": sp.WINDOW_MS, "index_shots": len(pairs), "index_goals": int(pairs.goal.sum()), "index_penalties": int(pairs.is_penalty.sum()),
        "index_by_split": pairs.split.value_counts().to_dict(), "paired": len(paired), "paired_rate": _rate(len(paired), len(pairs)),
        "paired_by_split": paired.split.value_counts().to_dict(), "paired_goals_sn": int(goal.sum()),
        "paired_matches_by_split": paired.drop_duplicates("game").split.value_counts().to_dict(),
        "paired_non_penalty": int((~paired.is_penalty).sum()),
        "goal_agreement": _rate(int((goal == sb_goal).sum()), len(paired)), "goal_agree": f"{int((goal == sb_goal).sum())}/{len(paired)}",
        "goal_disagreements": {"sn_goal_sb_not": int((goal & ~sb_goal).sum()), "sb_goal_sn_not": int((~goal & sb_goal).sum())},
        "penalty_agree": f"{int((paired.is_penalty.astype(bool) == paired.sb_penalty.astype(bool)).sum())}/{len(paired)}",
        "sn_penalties_paired_to_sb_penalties": f"{int((paired.is_penalty.astype(bool) & paired.sb_penalty.astype(bool)).sum())}/{int(pairs.is_penalty.sum())}",
        "team_agree": f"{int(paired.team_agree.sum())}/{len(paired)}", "team_disagree": int(paired.team_agree.eq(False).sum()),
        "team_ambiguous": int(paired.team.eq(sp.AMBIGUOUS_TEAM).sum()),
        "on_target_agree": f"{int((on_sb == on_sn).sum())}/{len(paired)}",
        "residual_ms": {"median": float(np.median(residual)), "iqr": np.percentile(residual, [25, 75]).round(1).tolist(),
                        "share_abs_le_200": round(float(np.mean(np.abs(residual) <= 200)), 4)} if len(residual) else {},
        "half_offset_ms": {"median": float(known.median()), "min": float(known.min()), "max": float(known.max()), "halves": len(known),
                            "halves_without_offset": int(len(half_offsets) - len(known))},
        "halves_with_at_most_2_pairs": int((half_offsets.n_pairs <= 2).sum()),
        "unpaired_by_label": unpaired.label.value_counts().to_dict(),
        "unpaired_nearest_sb_ms_quantiles_10_50_90": _quantiles(unpaired.nearest_sb_ms),
        "unpaired_no_sb_shot_within_10s": int((unpaired.nearest_sb_ms > 10_000).sum()),
        "unpaired_in_halves_without_offset": int(unpaired.nearest_sb_ms.isna().sum()),
        "halves_without_offset_detail": half_offsets[half_offsets.offset_ms.isna()].astype(object).where(
            half_offsets[half_offsets.offset_ms.isna()].notna(), None).to_dict("records"),
    }


def _sb_report(halves: sp.Halves, main: pd.DataFrame, sb_unpaired: pd.DataFrame, recovered: pd.DataFrame, own_goals: pd.DataFrame,
               cut_targets: pd.DataFrame) -> dict:
    sb = halves.sb
    used = sb.index.isin(main.sb_row)
    secondary = sb_unpaired[sb_unpaired.population.eq("secondary")]
    usable = cut_targets[cut_targets.usable]
    return {
        "sb_shots_in_paired_matches": len(sb), "sb_goals": int(sb.goal.sum()), "sb_penalties": int(sb.penalty.sum()),
        "paired": int(used.sum()), "unpaired": int((~used).sum()),
        "pairing_rate_by_outcome": sb.assign(p=used).groupby("outcome").p.agg(["sum", "count"]).astype(int).apply(
            lambda r: f"{r['sum']}/{r['count']}", axis=1).to_dict(),
        "unpaired_by_population": sb_unpaired.population.value_counts().to_dict(),
        "unpaired_by_reason": sb_unpaired.reason.value_counts().to_dict(),
        "own_goals_in_paired_matches": int(len(own_goals)),
        "recovery": {"dropped_goals": len(recovered), "with_candidate_in_window": int(recovered.any_candidate.sum()),
                     "recovered": int(recovered.recovered.sum()),
                     "recovered_team_agrees": f"{int(recovered.team_agree.sum())}/{int(recovered.recovered.sum())}", "reasons": recovered.reason.value_counts().to_dict(),
                     "lag_ms_quantiles_10_50_90": _quantiles(recovered.lag_ms),
                     "label_lag_ms_quantiles_10_50_90": _quantiles(recovered.label_lag_ms)},
        "secondary": {"shots": len(secondary), "by_outcome": secondary.outcome.value_counts().to_dict(),
                      "goals_without_sn_goal": int(secondary.sb_goal_without_sn_goal.sum()),
                      "possible_mispairs_left_out": int(sb_unpaired.population.eq("possible_mispair").sum()),
                      "anchor_convention": ANCHOR_CONVENTION},
        "cut_targets": {"targets": len(cut_targets), "usable": len(usable), "unusable": int((~cut_targets.usable).sum()),
                        "by_kind": cut_targets.kind.value_counts().to_dict(),
                        "halves": int(usable.drop_duplicates(["game", "half"]).shape[0]),
                        "matches": int(usable.game.nunique())},
    }


def _k0(shot: Mapping) -> dict:
    passed = bool(shot["paired_rate"] >= K0_MIN_PAIRED and shot["goal_agreement"] >= K0_MIN_GOAL_AGREEMENT)
    return {"rule": f"paired rate >= {K0_MIN_PAIRED} and goal agreement >= {K0_MIN_GOAL_AGREEMENT}", "paired_rate": shot["paired_rate"],
            "goal_agreement": shot["goal_agreement"], "passed": passed}


def jsonable(value):
    """The report with numpy scalars as Python numbers and NaN as null, for `json.dumps`."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def report_text(report: Mapping) -> str:
    return json.dumps(jsonable(report), indent=1, sort_keys=False, allow_nan=False) + "\n"


def build_report(match_map, rejected, clubs, halves, offsets, main, half_offsets, pairs, sb_unpaired, recovered, own_goals, cut_targets,
                 goal_labels, n_index_shots) -> dict:
    """The report of one pairing run (see `xgpoc.vendor.pairing.pair_sources` for the tables)."""
    shot = _shot_report(pairs, halves, offsets, half_offsets)
    return {
        "rules": {"window_ms": sp.WINDOW_MS, "windows_ms": list(WINDOWS_MS), "offset_search_ms": sp.OFFSET_SEARCH_MS,
                  "offset_dense_window_ms": sp.DENSE_MS, "offset_refits": sp.OFFSET_REFITS, "recover_ms": list(sp.RECOVER_MS),
                  "mispair_ms": sp.MISPAIR_MS, "name_min": mp.NAME_MIN, "day_tolerance": mp.DAY_TOLERANCE,
                  "team_constraint": False, "k0_min_paired": K0_MIN_PAIRED, "k0_min_goal_agreement": K0_MIN_GOAL_AGREEMENT},
        "match_pairing": _match_report(match_map, rejected, clubs),
        "sn_side": {"index_shots_in_paired_matches": n_index_shots, "dropped_goals": len(recovered)},
        "shot_pairing": shot,
        "window_sensitivity": _sensitivity(halves, offsets, main),
        "disagreements": _disagreements(pairs, halves, goal_labels),
        "sb_side": _sb_report(halves, main, sb_unpaired, recovered, own_goals, cut_targets),
        "k0": _k0(shot),
    }
