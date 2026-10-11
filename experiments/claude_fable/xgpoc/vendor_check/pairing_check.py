"""Pure functions for an independent SoccerNet/StatsBomb shot pairing check."""

from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

COMPETITIONS = {
    "england_epl": "Premier League",
    "spain_laliga": "La Liga",
    "italy_serie-a": "Serie A",
    "france_ligue-1": "Ligue 1",
    "germany_bundesliga": "1. Bundesliga",
    "europe_uefa-champions-league": "Champions League",
}
ALIASES = {
    "utd": "united",
    "sg": "saint germain",
    "psg": "paris saint germain",
    "atl.": "atletico",
    "ath": "athletic",
    "man": "manchester",
    "spurs": "tottenham",
}
STOP_TOKENS = {
    "fc",
    "cf",
    "afc",
    "ac",
    "as",
    "ss",
    "ssc",
    "sc",
    "rc",
    "cd",
    "ud",
    "sd",
    "rcd",
    "club",
    "calcio",
    "de",
    "the",
    "deportivo",
    "1899",
    "1846",
    "04",
    "05",
    "1.",
}
IMPOSSIBLE_COST = 1e12


def normalize_team_name(name: str) -> str:
    """Apply the frozen token-level team-name normalisation."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = folded.lower().replace("-", " ").split()
    expanded = [
        piece for token in tokens for piece in ALIASES.get(token, token).split()
    ]
    return " ".join(token for token in expanded if token not in STOP_TOKENS)


def name_similarity(left: str, right: str) -> float:
    """Return the maximum frozen similarity score for two already-normalised names."""
    ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    subset_score = (
        0.9
        if left_tokens
        and right_tokens
        and (left_tokens <= right_tokens or right_tokens <= left_tokens)
        else 0.0
    )
    prefix_score = (
        0.9
        if min(len(left), len(right)) >= 4
        and (left.startswith(right) or right.startswith(left))
        else 0.0
    )
    return max(ratio, subset_score, prefix_score)


def pair_similarity(
    sn_home: str,
    sn_away: str,
    sb_home: str,
    sb_away: str,
) -> float:
    """Score a home/away team pair after normalising all four names."""
    home = name_similarity(normalize_team_name(sn_home), normalize_team_name(sb_home))
    away = name_similarity(normalize_team_name(sn_away), normalize_team_name(sb_away))
    return min(home, away)


def match_games(game_metadata: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Map each SoccerNet game to its unique StatsBomb candidate, if one exists."""
    rows: list[dict] = []
    used_match_ids: set[int] = set()
    sb_dates = pd.to_datetime(matches["match_date"])

    for game in game_metadata.sort_values("game").itertuples(index=False):
        competition = COMPETITIONS[game.league]
        date = pd.Timestamp(game.game_date)
        mask = (
            matches["competition"].eq(competition)
            & sb_dates.sub(date).abs().le(pd.Timedelta(days=1))
            & matches["home_score"].eq(game.home_score)
            & matches["away_score"].eq(game.away_score)
        )
        qualified: list[int] = []
        for candidate in matches.loc[mask].itertuples(index=False):
            score = pair_similarity(
                game.home_team,
                game.away_team,
                candidate.home_team,
                candidate.away_team,
            )
            if score >= 0.6:
                qualified.append(int(candidate.match_id))

        match_id = qualified[0] if len(qualified) == 1 else None
        if match_id is not None:
            if match_id in used_match_ids:
                raise ValueError(
                    f"StatsBomb match {match_id} paired to more than one SoccerNet game"
                )
            used_match_ids.add(match_id)
        rows.append({"game": game.game, "sb_match_id": match_id})

    result = pd.DataFrame(rows, columns=["game", "sb_match_id"])
    result["sb_match_id"] = result["sb_match_id"].astype("Int64")
    return result


def estimate_half_offset(soccer_net_ms, statsbomb_ms) -> float:
    """Estimate SoccerNet minus StatsBomb time from the densest 400 ms difference window."""
    sn = np.asarray(list(soccer_net_ms), dtype=float)
    sb = np.asarray(list(statsbomb_ms), dtype=float)
    if not len(sn) or not len(sb):
        return float("nan")
    differences = (sn[:, None] - sb[None, :]).ravel()
    differences = np.sort(differences[np.abs(differences) <= 120_000])
    if not len(differences):
        return float("nan")
    ends = np.searchsorted(differences, differences + 400, side="right")
    starts = np.arange(len(differences))
    best_start = int(np.argmax(ends - starts))
    return float(np.median(differences[best_start : ends[best_start]]))


def assign_shots(
    soccer_net: pd.DataFrame,
    statsbomb: pd.DataFrame,
    offset_ms: float,
) -> pd.DataFrame:
    """Globally assign SoccerNet shots to StatsBomb shots for one half."""
    columns = ["shot_id", "sb_event_id", "delta_ms", "residual_ms"]
    if soccer_net.empty:
        return pd.DataFrame(columns=columns)

    effective_offset = 0.0 if pd.isna(offset_ms) else float(offset_ms)
    sn_ms = soccer_net["position_ms"].to_numpy(dtype=float)
    sb_ms = statsbomb["timestamp_ms"].to_numpy(dtype=float)
    residuals = sn_ms[:, None] - sb_ms[None, :] - effective_offset
    real_costs = np.where(
        np.abs(residuals) <= 2_500, np.abs(residuals), IMPOSSIBLE_COST
    )
    dummy_costs = np.full((len(sn_ms), len(sn_ms)), IMPOSSIBLE_COST)
    np.fill_diagonal(dummy_costs, 2_500)
    row_indices, column_indices = linear_sum_assignment(
        np.concatenate([real_costs, dummy_costs], axis=1)
    )

    pairs = []
    for sn_index, assigned_column in zip(row_indices, column_indices, strict=True):
        if (
            assigned_column >= len(sb_ms)
            or abs(residuals[sn_index, assigned_column]) > 2_500
        ):
            continue
        pairs.append(
            {
                "shot_id": soccer_net.iloc[sn_index]["shot_id"],
                "sb_event_id": statsbomb.iloc[assigned_column]["event_id"],
                "delta_ms": float(sn_ms[sn_index] - sb_ms[assigned_column]),
                "residual_ms": float(residuals[sn_index, assigned_column]),
            }
        )
    return pd.DataFrame(pairs, columns=columns)


def pair_half(
    soccer_net: pd.DataFrame,
    statsbomb: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Estimate, assign, and perform the two frozen offset refits for one half."""
    offset = estimate_half_offset(soccer_net["position_ms"], statsbomb["timestamp_ms"])
    pairs = assign_shots(soccer_net, statsbomb, offset)
    for _ in range(2):
        if not pairs.empty:
            offset = float(pairs["delta_ms"].median())
        pairs = assign_shots(soccer_net, statsbomb, offset)
    return offset, pairs


def recover_dropped_goals(
    attempts: pd.DataFrame,
    unpaired_goals: pd.DataFrame,
    offset_ms: float,
) -> pd.DataFrame:
    """Assign dropped SoccerNet goal anchors to unpaired StatsBomb goals."""
    columns = ["anchor_ms", "sb_event_id", "lag_ms"]
    if attempts.empty or unpaired_goals.empty or pd.isna(offset_ms):
        return pd.DataFrame(columns=columns)

    anchors = attempts["anchor_ms"].to_numpy(dtype=float)
    goal_ms = unpaired_goals["timestamp_ms"].to_numpy(dtype=float)
    lags = anchors[:, None] - float(offset_ms) - goal_ms[None, :]
    allowed = (lags >= -3_000) & (lags <= 8_000)
    real_costs = np.where(allowed, np.abs(lags), IMPOSSIBLE_COST)
    dummy_costs = np.full((len(anchors), len(anchors)), IMPOSSIBLE_COST)
    np.fill_diagonal(dummy_costs, 8_001)
    anchor_indices, columns_assigned = linear_sum_assignment(
        np.concatenate([real_costs, dummy_costs], axis=1)
    )

    recovered = []
    for anchor_index, assigned_column in zip(
        anchor_indices, columns_assigned, strict=True
    ):
        if (
            assigned_column >= len(goal_ms)
            or not allowed[anchor_index, assigned_column]
        ):
            continue
        recovered.append(
            {
                "anchor_ms": int(anchors[anchor_index]),
                "sb_event_id": unpaired_goals.iloc[assigned_column]["event_id"],
                "lag_ms": float(lags[anchor_index, assigned_column]),
            }
        )
    return pd.DataFrame(recovered, columns=columns)


def classify_unpaired_shots(
    statsbomb: pd.DataFrame,
    *,
    recovered_event_ids: set[str],
    offsets: dict[int, float],
    unpaired_soccer_net: pd.DataFrame,
) -> pd.DataFrame:
    """Classify unpaired StatsBomb shots by the first applicable frozen rule."""
    rows = []
    for shot in statsbomb.itertuples(index=False):
        if shot.event_id in recovered_event_ids:
            population = "recovered_goal"
        elif shot.period > 2:
            population = "extra_time"
        elif shot.shot_type == "Penalty":
            population = "penalty"
        else:
            offset = offsets.get(int(shot.period), float("nan"))
            if math.isnan(offset):
                population = "unaligned_half"
            else:
                same_half = unpaired_soccer_net.loc[
                    unpaired_soccer_net["half"].eq(shot.period), "position_ms"
                ].to_numpy(dtype=float)
                is_near = bool(
                    len(same_half)
                    and np.any(np.abs(same_half - offset - shot.timestamp_ms) <= 10_000)
                )
                population = "possible_mispair" if is_near else "secondary"
        rows.append({"event_id": shot.event_id, "population": population})
    return pd.DataFrame(rows, columns=["event_id", "population"])


def build_pairing_tables(
    index: pd.DataFrame,
    attempts: pd.DataFrame,
    game_metadata: pd.DataFrame,
    matches: pd.DataFrame,
    shots: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all five independent comparison tables from already-loaded inputs."""
    match_map = match_games(game_metadata, matches)
    all_pairs: list[pd.DataFrame] = []
    offset_rows: list[dict] = []
    recovered_rows: list[pd.DataFrame] = []
    population_rows: list[pd.DataFrame] = []

    for mapping in match_map.dropna(subset=["sb_match_id"]).itertuples(index=False):
        game = mapping.game
        match_id = int(mapping.sb_match_id)
        game_sn = index[index["game"].eq(game)]
        game_sb = shots[shots["match_id"].eq(match_id)]
        offsets: dict[int, float] = {}
        game_pairs: list[pd.DataFrame] = []

        for half in (1, 2):
            sn_half = game_sn[game_sn["half"].eq(half)].reset_index(drop=True)
            sb_half = game_sb[game_sb["period"].eq(half)].reset_index(drop=True)
            offset, pairs = pair_half(sn_half, sb_half)
            offsets[half] = offset
            offset_rows.append({"game": game, "half": half, "offset_ms": offset})
            if not pairs.empty:
                pairs.insert(0, "game", game)
                pairs.insert(1, "half", half)
                game_pairs.append(pairs)
                all_pairs.append(pairs)

        paired_ids = {
            event_id
            for frame in game_pairs
            for event_id in frame["sb_event_id"].tolist()
        }
        recovered_ids: set[str] = set()
        for half in (1, 2):
            dropped = attempts[
                attempts["game"].eq(game) & attempts["half"].eq(half)
            ].rename(columns={"position_ms": "anchor_ms"})
            goal_candidates = game_sb[
                game_sb["period"].eq(half)
                & game_sb["outcome"].eq("Goal")
                & ~game_sb["event_id"].isin(paired_ids)
            ].reset_index(drop=True)
            recovered = recover_dropped_goals(dropped, goal_candidates, offsets[half])
            if not recovered.empty:
                recovered.insert(0, "game", game)
                recovered.insert(1, "half", half)
                recovered_rows.append(recovered)
                recovered_ids.update(recovered["sb_event_id"])

        paired_shot_ids = {
            shot_id for frame in game_pairs for shot_id in frame["shot_id"].tolist()
        }
        unpaired_sn = game_sn[~game_sn["shot_id"].isin(paired_shot_ids)]
        unpaired_sb = game_sb[~game_sb["event_id"].isin(paired_ids)].reset_index(
            drop=True
        )
        population_rows.append(
            classify_unpaired_shots(
                unpaired_sb,
                recovered_event_ids=recovered_ids,
                offsets=offsets,
                unpaired_soccer_net=unpaired_sn,
            )
        )

    pair_columns = ["game", "half", "shot_id", "sb_event_id", "delta_ms", "residual_ms"]
    recovered_columns = ["game", "half", "anchor_ms", "sb_event_id", "lag_ms"]
    return {
        "match_map": match_map,
        "pairs": pd.concat(all_pairs, ignore_index=True)
        if all_pairs
        else pd.DataFrame(columns=pair_columns),
        "half_offsets": pd.DataFrame(
            offset_rows, columns=["game", "half", "offset_ms"]
        ),
        "recovered_goals": (
            pd.concat(recovered_rows, ignore_index=True)
            if recovered_rows
            else pd.DataFrame(columns=recovered_columns)
        ),
        "populations": (
            pd.concat(population_rows, ignore_index=True)
            if population_rows
            else pd.DataFrame(columns=["event_id", "population"])
        ),
    }
