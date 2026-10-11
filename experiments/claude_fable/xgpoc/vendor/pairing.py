"""The frozen pairing of SoccerNet with StatsBomb (xg-vendor study, test seam 1).

`pair_sources` is a pure function of its inputs; it reads no file and no prediction:

    index        SoccerNet shot index v1.2 (one row per shot_id: game, half, position_ms, label, goal, is_penalty, split)
    labels       Labels-v2 records (the JSON content): date, teams, score and the annotations that give each shot's team
    attempts     attempts table next to the index; read only for the Goal labels without a shot annotation
    statsbomb    the StatsBomb tables "matches", "shots" and "context" (xgpoc.vendor.statsbomb_tables)

and returns a `Pairing`:

    match_map       every SoccerNet match, its StatsBomb match or the reason it has none
    pairs           every index shot of a paired match: paired or not, residual, StatsBomb partner's fields
    half_offsets    per (game, half): half offset SoccerNet - StatsBomb and how it was fitted
    recovered_goals every dropped SoccerNet Goal label of a paired match: the StatsBomb goal it took (lag) or why not
    sb_unpaired     every StatsBomb shot of a paired match that is not paired: population and reason
    unpaired_sn     the unpaired index shots with the nearest StatsBomb shot
    cut_targets     where the secondary population and the recovered goals sit in the SoccerNet halves
    report          counts, window sensitivity, disagreement lists with reasons, K0

Rules (see the spec, "Pairing"): `match_pairing` for matches, `shot_pairing` for the half-level algorithms.
Populations of `sb_unpaired` (column `population`):

    recovered_goal    taken by a dropped SoccerNet goal
    extra_time        period 3 or later: SoccerNet halves hold periods 1 and 2 only
    penalty           penalties are reported apart
    unaligned_half    the half has no half offset (no SoccerNet shot within the offset search), nothing was paired in it
    possible_mispair  within sp.MISPAIR_MS of an unpaired index shot of the same half: left out so no shot counts twice
    secondary         everything else; `sb_goal_without_sn_goal` marks the goals among them

Own goals are no Shot events in StatsBomb, so the secondary population has none by construction; the report counts them.
Times in the cut targets follow the secondary population's anchor convention: StatsBomb time plus the half offset.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.vendor import match_pairing as mp
from xgpoc.vendor import pairing_report as rep
from xgpoc.vendor import shot_pairing as sp

OWN_GOAL_EVENT = "Own Goal For"
# Population of an unpaired StatsBomb shot -> its reason, in the order the rules apply; None: the reason names the outcome.
POPULATION_REASONS = {
    "recovered_goal": "goal taken by a dropped SoccerNet Goal label",
    "extra_time": "extra time or shoot-out: the SoccerNet halves hold periods 1 and 2",
    "penalty": "penalty, reported separately",
    "unaligned_half": "the half has no half offset, so its shots were not paired: unknown whether SoccerNet annotated it",
    "possible_mispair": f"within {sp.MISPAIR_MS // 1000} s of an unpaired SoccerNet shot of the same half: possible mispair, left out",
    "secondary": None,
}
GOAL_BACKOFF_MS = 1000    # attempts_all anchors a goal without shot annotation this long before its Goal label (soccernet_labels.parse_labels)

SN_COLUMNS = ["shot_id", "game", "league", "split", "half", "position_ms", "label", "team", "goal", "is_penalty"]
SB_FIELDS = {"event_id": "sb_event_id", "match_id": "sb_match_id", "timestamp_ms": "sb_timestamp_ms", "team": "sb_team",
             "side": "sb_side", "statsbomb_xg": "sb_xg", "outcome": "sb_outcome", "goal": "sb_goal", "penalty": "sb_penalty",
             "body_part": "sb_body_part", "shot_type": "sb_shot_type", "technique": "sb_technique",
             "first_time": "sb_first_time", "under_pressure": "sb_under_pressure", "x": "sb_x", "y": "sb_y",
             "play_pattern": "sb_play_pattern", "n_freeze_frame": "sb_n_freeze_frame"}


@dataclass(frozen=True)
class Pairing:
    match_map: pd.DataFrame
    pairs: pd.DataFrame
    half_offsets: pd.DataFrame
    recovered_goals: pd.DataFrame
    sb_unpaired: pd.DataFrame
    unpaired_sn: pd.DataFrame
    cut_targets: pd.DataFrame
    report: dict

    @property
    def secondary(self) -> pd.DataFrame:
        """The secondary population: the StatsBomb shots SoccerNet never annotated, mispair candidates left out."""
        return self.sb_unpaired[self.sb_unpaired.population.eq("secondary")].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------- inputs
def annotation_frame(records: Iterable[Mapping], games: Iterable[str]) -> pd.DataFrame:
    """The Labels-v2 annotations of the given matches: game, half, position_ms, label, team."""
    wanted = set(games)
    rows = [{"game": mp.game_id(r), "half": int(a["gameTime"].split(" - ")[0]), "position_ms": int(a["position"]),
             "label": a["label"], "team": a["team"]}
            for r in records if mp.game_id(r) in wanted for a in r["annotations"]]
    return pd.DataFrame(rows, columns=["game", "half", "position_ms", "label", "team"])


def _team_lookup(annotations: pd.DataFrame) -> pd.DataFrame:
    """One team per (game, half, position_ms, label). Labels-v2 holds a few annotations with the same time and label but
    opposite teams; their team is "ambiguous" (the shot pairing does not use the team)."""
    key = ["game", "half", "position_ms", "label"]
    teams = annotations.groupby(key).team.agg(lambda t: t.iloc[0] if t.nunique() == 1 else sp.AMBIGUOUS_TEAM)
    return teams.reset_index()


def _sn_shots(index: pd.DataFrame, annotations: pd.DataFrame, paired_games: set) -> pd.DataFrame:
    shots = index[index.game.isin(paired_games)].copy()
    shots["league"] = shots.game.str.split("/").str[0]
    shots = shots.merge(_team_lookup(annotations), on=["game", "half", "position_ms", "label"], how="left")
    if not shots.shot_id.is_unique:
        raise ValueError("The index holds a shot_id twice")
    if shots.team.isna().any():
        raise ValueError(f"{int(shots.team.isna().sum())} index shots have no Labels-v2 annotation to take the team from")
    return shots.sort_values(["game", "half", "position_ms", "shot_id"]).reset_index(drop=True)[SN_COLUMNS]


def _sb_shots(shots: pd.DataFrame, match_map: pd.DataFrame, sb_matches: pd.DataFrame) -> pd.DataFrame:
    paired = match_map[match_map.reason.eq("paired")]
    game_of = dict(zip(paired.sb_match_id.astype(int), paired.game, strict=True))
    teams = sb_matches.set_index("match_id")
    out = shots[shots.match_id.isin(game_of)].copy()
    out["game"] = out.match_id.map(game_of)
    home = out.match_id.map(teams.home_team)
    away = out.match_id.map(teams.away_team)
    out["side"] = np.select([out.team == home, out.team == away], ["home", "away"], default="?")
    out["goal"] = out.outcome.eq("Goal")
    out["penalty"] = out.shot_type.eq("Penalty")
    keep = ["match_id", "event_id", "game", "period", "timestamp_ms", "team", "side", "statsbomb_xg", "outcome", "goal", "penalty",
            "body_part", "shot_type", "technique", "first_time", "under_pressure", "x", "y", "play_pattern", "n_freeze_frame",
            "player", "index"]
    return out.sort_values(["match_id", "index"]).reset_index(drop=True)[keep]


# ---------------------------------------------------------------------------------------------------- tables
def _half_offsets_table(halves: sp.Halves, initial: dict, final: dict, refitted: set, pairs: pd.DataFrame, match_map: pd.DataFrame) -> pd.DataFrame:
    sb_match = dict(zip(match_map.game, match_map.sb_match_id, strict=True))
    n_pairs = pairs.groupby(["game", "half"]).size().to_dict()
    rows = []
    for key in halves.keys:
        sn_rows, sb_rows = halves.of(key)
        known = not np.isnan(final[key])
        rows.append({"game": key[0], "half": key[1], "sb_match_id": sb_match[key[0]], "n_sn_shots": len(sn_rows),
                     "n_sb_shots": len(sb_rows), "initial_offset_ms": initial[key], "offset_ms": final[key],
                     "n_pairs": n_pairs.get(key, 0),
                     "offset_source": "refit" if key in refitted else ("densest_window" if known else "none")})
    return pd.DataFrame(rows).astype({"sb_match_id": "Int64"})


def _nearest_sb(halves: sp.Halves, offsets: dict) -> np.ndarray:
    """Per SN shot: ms to the nearest StatsBomb shot of its half after the offset (NaN without one)."""
    distance = np.full(len(halves.sn), np.nan)
    for key, sn_rows in halves.sn_rows.items():
        sb_rows = halves.sb_rows.get(key)
        if sb_rows is not None and len(sb_rows):
            distance[sn_rows] = np.abs(halves.sn_ms[sn_rows][:, None] - offsets[key] - halves.sb_ms[sb_rows][None, :]).min(axis=1)
    return distance


def _nullable(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Columns that may be missing for unpaired rows get a nullable dtype that keeps bool and integers."""
    for column in frame.columns:
        kind = source[column].dtype.kind
        if kind == "b":
            frame[column] = frame[column].astype("boolean")
        elif kind in "iu":
            frame[column] = frame[column].astype("Int64")
    return frame


def _pairs_table(halves: sp.Halves, main: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    sn, sb = halves.sn, halves.sb
    table = sn.copy()
    table["paired"] = table.index.isin(main.sn_row)
    table["residual_ms"] = main.set_index("sn_row").residual_ms.reindex(table.index)
    table["offset_ms"] = [offsets[(g, h)] for g, h in zip(sn.game, sn.half, strict=True)]
    table["nearest_sb_ms"] = _nearest_sb(halves, offsets)
    partner = sb[list(SB_FIELDS)].rename(columns=SB_FIELDS).iloc[main.sb_row.to_numpy()].set_index(main.sn_row.to_numpy())
    table = table.join(_nullable(partner.reindex(table.index), sb.rename(columns=SB_FIELDS)))
    table["team_agree"] = table.team.eq(table.sb_side).astype("boolean").where(table.paired & table.team.ne(sp.AMBIGUOUS_TEAM))
    return table


def _unpaired_sn_table(pairs: pd.DataFrame, halves: sp.Halves, offsets: dict) -> pd.DataFrame:
    """The unpaired index shots: position on the StatsBomb clock and the nearest StatsBomb shot of the half."""
    table = pairs[~pairs.paired].copy()
    table["sb_clock_ms"] = table.position_ms - table.offset_ms
    sb = halves.sb
    match_of = dict(zip(sb.game, sb.match_id, strict=True))
    table["sb_match_id"] = table.game.map(match_of).astype("Int64")
    event, outcome = [], []
    for row in table.itertuples():
        rows = halves.sb_rows.get((row.game, row.half))
        if rows is None or not len(rows) or np.isnan(row.offset_ms):
            event.append(pd.NA)
            outcome.append(pd.NA)
            continue
        k = rows[np.abs(row.position_ms - offsets[(row.game, row.half)] - halves.sb_ms[rows]).argmin()]
        event.append(sb.event_id.iat[k])
        outcome.append(sb.outcome.iat[k])
    table["nearest_sb_event_id"], table["nearest_sb_outcome"] = event, outcome
    cols = ["shot_id", "game", "league", "split", "half", "position_ms", "label", "team", "goal", "is_penalty", "offset_ms",
            "sb_clock_ms", "sb_match_id", "nearest_sb_ms", "nearest_sb_event_id", "nearest_sb_outcome"]
    return table[cols].reset_index(drop=True)


def _dropped_goals(attempts: pd.DataFrame, annotations: pd.DataFrame, paired_games: set) -> pd.DataFrame:
    """The Goal labels without a shot annotation in paired matches: game, half, anchor_ms, goal_label_ms, team.

    The attempts table anchors such a goal `GOAL_BACKOFF_MS` before its Goal label (xgpoc.data.soccernet_labels); the
    team comes from the Goal annotation at anchor + backoff, and a goal without one is an error."""
    dropped = attempts[attempts.status.eq("goal_without_shot_annotation") & attempts.game.isin(paired_games)]
    dropped = dropped.drop_duplicates(["game", "half", "position_ms"])
    dropped = dropped.assign(goal_label_ms=dropped.position_ms + GOAL_BACKOFF_MS).rename(columns={"position_ms": "anchor_ms"})
    teams = _team_lookup(annotations).rename(columns={"position_ms": "goal_label_ms"})
    dropped = dropped.merge(teams, on=["game", "half", "goal_label_ms", "label"], how="left")
    if dropped.team.isna().any():
        raise ValueError(f"{int(dropped.team.isna().sum())} dropped goals have no Goal annotation {GOAL_BACKOFF_MS} ms after their anchor")
    return dropped.sort_values(["game", "half", "anchor_ms"]).reset_index(drop=True)[["game", "half", "anchor_ms", "goal_label_ms", "team"]]


def _recovered_table(dropped: pd.DataFrame, halves: sp.Halves, offsets: dict, free_goal: np.ndarray, own_goals: pd.DataFrame,
                     match_map: pd.DataFrame) -> pd.DataFrame:
    """One row per dropped goal. `free_goal` marks StatsBomb shot rows that are unpaired goals (the candidates)."""
    sb = halves.sb
    sb_match = dict(zip(match_map.game, match_map.sb_match_id, strict=True))
    own = {key: g.timestamp_ms.to_numpy() for key, g in own_goals.groupby(["match_id", "period"])}
    rows = []
    for (game, half), group in dropped.groupby(["game", "half"], sort=True):
        offset = offsets.get((game, half), np.nan)
        candidates = halves.sb_rows.get((game, half), np.empty(0, dtype=int))
        candidates = candidates[free_goal[candidates]]
        chosen, lag = sp.recover(group.anchor_ms.to_numpy(), halves.sb_ms[candidates], offset)
        window_ok = (lag >= sp.RECOVER_MS[0]) & (lag <= sp.RECOVER_MS[1])
        for i, goal in enumerate(group.itertuples()):
            j = chosen[i]
            any_candidate = bool(window_ok[i].any()) if lag.size else False
            og = own.get((sb_match[game], half), np.empty(0))
            og_lag = goal.anchor_ms - offset - og
            is_own_goal = bool(((og_lag >= sp.RECOVER_MS[0]) & (og_lag <= sp.RECOVER_MS[1])).any())
            if j is not None:
                reason = "recovered"
            elif np.isnan(offset):
                reason = "half has no half offset"
            elif any_candidate:
                reason = "candidate taken by another dropped goal"
            elif is_own_goal:
                reason = "own goal in StatsBomb"
            else:
                reason = "no unpaired StatsBomb goal in the window"
            row = candidates[j] if j is not None else None
            rows.append({"game": game, "half": half, "anchor_ms": goal.anchor_ms, "goal_label_ms": goal.goal_label_ms, "team": goal.team,
                         "sb_match_id": sb_match[game], "offset_ms": offset, "any_candidate": any_candidate,
                         "recovered": j is not None, "sb_event_id": sb.event_id.iat[row] if row is not None else pd.NA,
                         "sb_timestamp_ms": sb.timestamp_ms.iat[row] if row is not None else pd.NA,
                         "lag_ms": float(lag[i, j]) if j is not None else np.nan,
                         "label_lag_ms": float(lag[i, j]) + GOAL_BACKOFF_MS if j is not None else np.nan,
                         "sb_side": sb.side.iat[row] if row is not None else pd.NA,
                         "team_agree": goal.team == sb.side.iat[row] if row is not None else pd.NA, "reason": reason})
    columns = ["game", "half", "anchor_ms", "goal_label_ms", "team", "sb_match_id", "offset_ms", "any_candidate", "recovered", "sb_event_id",
               "sb_timestamp_ms", "lag_ms", "label_lag_ms", "sb_side", "team_agree", "reason"]
    table = pd.DataFrame(rows, columns=columns).astype({"sb_match_id": "Int64", "sb_timestamp_ms": "Int64", "recovered": bool,
                                                         "team_agree": "boolean"})
    table["target_position_ms"] = _target_ms(table.sb_timestamp_ms, table.offset_ms)
    return table


def _target_ms(timestamp: pd.Series, offset: pd.Series) -> pd.Series:
    """StatsBomb time plus the half offset: the position in the SoccerNet half, in whole ms (NA without an offset)."""
    return (pd.to_numeric(timestamp).astype("Float64") + pd.to_numeric(offset).astype("Float64")).round().astype("Int64")


def _nearest_signed(values: np.ndarray, centre: float) -> float:
    """The signed distance (value - centre) of the value nearest to `centre`; NaN without values."""
    if not len(values):
        return float("nan")
    d = values - centre
    return float(d[np.abs(d).argmin()])


def _sb_unpaired_table(halves: sp.Halves, main: pd.DataFrame, offsets: dict, recovered: pd.DataFrame, unpaired_sn: pd.DataFrame,
                       goal_labels: pd.DataFrame) -> pd.DataFrame:
    sb = halves.sb
    table = sb[~sb.index.isin(main.sb_row)].copy()
    recovered_events = set(recovered.sb_event_id.dropna())
    table["offset_ms"] = [offsets.get((g, p), np.nan) for g, p in zip(table.game, table.period, strict=True)]
    # distance to the nearest unpaired index shot of the same half, on the StatsBomb clock
    clock = {k: g.sb_clock_ms.to_numpy() for k, g in unpaired_sn.groupby(["game", "half"])}
    table["nearest_unpaired_sn_ms"] = [
        abs(_nearest_signed(clock.get((g, p), np.empty(0)), t)) for g, p, t in zip(table.game, table.period, table.timestamp_ms, strict=True)]
    labels = {k: g.position_ms.to_numpy() for k, g in goal_labels.groupby(["game", "half"])}
    table["nearest_sn_goal_label_lag_ms"] = [
        _nearest_signed(labels.get((g, p), np.empty(0)) - o, t) if goal else np.nan
        for g, p, o, t, goal in zip(table.game, table.period, table.offset_ms, table.timestamp_ms, table.goal, strict=True)]
    near_unpaired = table.nearest_unpaired_sn_ms.le(sp.MISPAIR_MS)
    first_match = [table.event_id.isin(recovered_events), table.period.gt(2), table.penalty, table.offset_ms.isna(), near_unpaired]
    table["population"] = np.select(first_match, list(POPULATION_REASONS)[:-1], default="secondary")
    table["sb_goal_without_sn_goal"] = table.goal & table.population.eq("secondary")
    table["reason"] = [POPULATION_REASONS[p] or f"not annotated by SoccerNet: outcome {o}" for p, o in zip(table.population, table.outcome, strict=True)]
    table["anchor_convention"] = rep.ANCHOR_CONVENTION
    table["target_position_ms"] = _target_ms(table.timestamp_ms, table.offset_ms)
    return table.drop(columns=["index"]).reset_index(drop=True)


def _cut_targets(secondary: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    """Every frame the secondary population and the recovered goals need: where in which SoccerNet half."""
    shots = secondary.assign(kind="secondary", half=secondary.period).rename(
        columns={"match_id": "sb_match_id", "event_id": "sb_event_id", "timestamp_ms": "sb_timestamp_ms"})
    goals = recovered[recovered.recovered].assign(kind="recovered_goal")
    cols = ["kind", "game", "half", "sb_match_id", "sb_event_id", "sb_timestamp_ms", "offset_ms", "target_position_ms"]
    parts = [part[cols] for part in (shots, goals) if len(part)]
    table = pd.concat(parts, ignore_index=True) if parts else shots[cols]
    table["anchor_convention"] = rep.ANCHOR_CONVENTION
    table["usable"] = table.target_position_ms.notna() & table.target_position_ms.ge(0)
    table["reason_unusable"] = np.where(table.offset_ms.isna(), "half has no half offset (no SoccerNet shot to align with)",
                                        np.where(table.usable, "", "target before the start of the half"))
    return table.sort_values(["game", "half", "target_position_ms", "sb_event_id"]).reset_index(drop=True)


def halves_to_cut(cut_targets: pd.DataFrame) -> pd.DataFrame:
    """The (game, half) SoccerNet halves that hold usable targets, with counts per kind and the target time range."""
    grouped = cut_targets[cut_targets.usable].groupby(["game", "half"])
    table = grouped.agg(sb_match_id=("sb_match_id", "first"), secondary_shots=("kind", lambda k: int(k.eq("secondary").sum())),
                        recovered_goals=("kind", lambda k: int(k.eq("recovered_goal").sum())),
                        first_target_ms=("target_position_ms", "min"), last_target_ms=("target_position_ms", "max"))
    return table.reset_index()


# ---------------------------------------------------------------------------------------------------- entry point
def pair_sources(index: pd.DataFrame, labels: Iterable[Mapping], attempts: pd.DataFrame, statsbomb: Mapping[str, pd.DataFrame]) -> Pairing:
    """Pair SoccerNet with StatsBomb. See the module docstring; raises when a StatsBomb match would be paired twice or
    the index and Labels-v2 disagree (a game without metadata, a shot without a team, a repeated shot id)."""
    labels = list(labels)
    games = mp.sn_games(labels)
    if unknown := sorted(set(index.game) - set(games.game)):
        raise ValueError(f"{len(unknown)} index games have no Labels-v2 record, e.g. {unknown[0]}")
    sb_matches = statsbomb["matches"]
    match_map = mp.pair_matches(games, sb_matches)
    paired_games = set(match_map.game[match_map.reason.eq("paired")])
    annotations = annotation_frame(labels, paired_games)

    sn = _sn_shots(index, annotations, paired_games)
    sb = _sb_shots(statsbomb["shots"], match_map, sb_matches)
    halves = sp.Halves(sn, sb)
    initial, offsets, main, refitted = sp.fit(halves, sp.WINDOW_MS, sp.OFFSET_REFITS)
    half_offsets = _half_offsets_table(halves, initial, offsets, refitted, main, match_map)
    pairs = _pairs_table(halves, main, offsets)
    unpaired_sn = _unpaired_sn_table(pairs, halves, offsets)

    own_goals = statsbomb["context"]
    own_goals = own_goals[own_goals.type.eq(OWN_GOAL_EVENT) & own_goals.match_id.isin(sb.match_id.unique())]
    free_goal = (sb.goal & ~sb.index.isin(main.sb_row)).to_numpy()
    recovered = _recovered_table(_dropped_goals(attempts, annotations, paired_games), halves, offsets, free_goal, own_goals, match_map)
    goal_labels = annotations[annotations.label.eq("Goal")]
    sb_unpaired = _sb_unpaired_table(halves, main, offsets, recovered, unpaired_sn, goal_labels)
    cut_targets = _cut_targets(sb_unpaired[sb_unpaired.population.eq("secondary")], recovered)

    report = rep.build_report(match_map, mp.rejected_candidates(games[~games.game.isin(paired_games)], sb_matches),
                              mp.unmatched_clubs(games, sb_matches), halves, offsets, main, half_offsets, pairs, sb_unpaired, recovered,
                              own_goals, cut_targets, goal_labels, len(sn))
    return Pairing(match_map, pairs, half_offsets, recovered, sb_unpaired, unpaired_sn, cut_targets, report)


