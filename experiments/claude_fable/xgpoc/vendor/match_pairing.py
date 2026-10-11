"""Match pairing of the xg-vendor study: which StatsBomb match is which SoccerNet match.

Pure functions of the Labels-v2 metadata records and the StatsBomb `matches` table. Date, teams and score come
from the Labels-v2 content (`gameDate`, `gameHomeTeam`, `gameAwayTeam`, `gameScore`), never from the folder name:
the folder name of Real Madrid 10-2 Rayo says "1 - 0". The record's `UrlLocal` is only the match identifier
(it equals the index column `game`).

A StatsBomb match is accepted for a SoccerNet match when it is in the same competition (`COMPETITION` maps the
SoccerNet league folder), within `DAY_TOLERANCE` days, has the exact score, and both team names are at least
`NAME_MIN` similar (`name_sim`: ASCII fold, stop tokens dropped, alias tokens expanded, then the best of a sequence
ratio, token containment and prefix match). Exactly one candidate may pass; two passing candidates leave the match
unpaired ("ambiguous"). No StatsBomb match is paired twice (`pair_matches` raises).

`unmatched_clubs` is the alias-list test: a club name without a counterpart in the other source is a missing
alias until someone has reviewed it as genuinely absent.
"""

import unicodedata
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# SoccerNet league folder -> StatsBomb competition name
COMPETITION = {
    "england_epl": "Premier League",
    "spain_laliga": "La Liga",
    "italy_serie-a": "Serie A",
    "france_ligue-1": "Ligue 1",
    "germany_bundesliga": "1. Bundesliga",
    "europe_uefa-champions-league": "Champions League",
}
NAME_MIN = 0.6
DAY_TOLERANCE = 1
STOP = frozenset({"fc", "cf", "afc", "ac", "as", "ss", "ssc", "sc", "rc", "cd", "ud", "sd", "rcd", "club", "calcio", "de",
                  "the", "deportivo", "1899", "1846", "04", "05", "1."})
ALIAS = {"utd": "united", "sg": "saint germain", "psg": "paris saint germain", "atl.": "atletico", "ath": "athletic",
         "man": "manchester", "west brom": "west bromwich", "spurs": "tottenham"}
# A competition season with fewer StatsBomb matches than this cannot hold the clubs of a SoccerNet season (the open data
# have the Champions League final only); `unmatched_clubs` skips it.
MIN_SB_SEASON_MATCHES = 10

MATCH_MAP_COLUMNS = ["game", "league", "season", "sn_home", "sn_away", "sn_score", "sb_match_id", "sb_home", "sb_away", "name_sim",
                     "reason"]


def normalise(name: str) -> str:
    """ASCII fold, lower case, hyphens to spaces, stop tokens dropped, alias tokens expanded."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace("-", " ")
    tokens = (ALIAS.get(token, token) for token in folded.split())
    return " ".join(token for token in tokens if token not in STOP)


def name_sim(a: str, b: str) -> float:
    """Similarity of two club names in [0, 1]."""
    a, b = normalise(a), normalise(b)
    tokens_a, tokens_b = set(a.split()), set(b.split())
    contain = 1.0 if tokens_a and tokens_b and (tokens_a <= tokens_b or tokens_b <= tokens_a) else 0.0
    prefix = 1.0 if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)) else 0.0
    return max(SequenceMatcher(None, a, b).ratio(), 0.9 * contain, 0.9 * prefix)


def game_id(record: Mapping) -> str:
    """The match identifier of a Labels-v2 record: `UrlLocal` without the trailing slash."""
    return record["UrlLocal"].rstrip("/")


def sn_games(records: Iterable[Mapping]) -> pd.DataFrame:
    """One row per SoccerNet match from the Labels-v2 metadata: game, league, season, date, home, away, scores."""
    rows = []
    for record in records:
        game = game_id(record)
        home_score, away_score = (int(x) for x in record["gameScore"].split(" - "))
        rows.append({"game": game, "league": game.split("/")[0], "season": game.split("/")[1],
                     "date": pd.to_datetime(record["gameDate"].split(" - ")[0], dayfirst=True),
                     "home": record["gameHomeTeam"], "away": record["gameAwayTeam"],
                     "home_score": home_score, "away_score": away_score})
    games = pd.DataFrame(rows).sort_values("game").reset_index(drop=True)
    if games.game.duplicated().any():
        raise ValueError(f"Duplicate Labels-v2 matches: {games.game[games.game.duplicated()].tolist()}")
    return games


def _pair_sim(game, match) -> float:
    return min(name_sim(game.home, match.home_team), name_sim(game.away, match.away_team))


def _same_day_candidates(game, sb_matches: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb matches of the game's competition within `DAY_TOLERANCE` days of it."""
    date = pd.to_datetime(sb_matches.match_date)
    return sb_matches[(date - game.date).abs().le(pd.Timedelta(days=DAY_TOLERANCE)) & sb_matches.competition.eq(COMPETITION[game.league])]


def _exact_score(game, matches: pd.DataFrame) -> pd.DataFrame:
    return matches[(matches.home_score == game.home_score) & (matches.away_score == game.away_score)]


def pair_matches(games: pd.DataFrame, sb_matches: pd.DataFrame) -> pd.DataFrame:
    """The match map: every SoccerNet match with its StatsBomb match id, or the reason it has none.

    reason: "paired" | "ambiguous: two date+score candidates pass the name rule" | "teams+date match but score differs" |
    "StatsBomb has matches that day, none with these teams" | "no StatsBomb match within 1 day"."""
    out = []
    for game in games.itertuples():
        near = _same_day_candidates(game, sb_matches)
        scored = sorted(((_pair_sim(game, m), m) for m in _exact_score(game, near).itertuples()), key=lambda t: -t[0])
        passing = [m for sim, m in scored if sim >= NAME_MIN]
        best, reason = None, "no StatsBomb match within 1 day"
        if len(passing) == 1:
            best, reason = passing[0], "paired"
        elif len(passing) > 1:
            reason = "ambiguous: two date+score candidates pass the name rule"
        elif any(_pair_sim(game, m) >= NAME_MIN for m in near.itertuples()):
            reason = "teams+date match but score differs"
        elif len(near):
            reason = "StatsBomb has matches that day, none with these teams"
        shown = best if best is not None else (scored[0][1] if scored else None)
        out.append({"game": game.game, "league": game.league, "season": game.season, "sn_home": game.home, "sn_away": game.away,
                    "sn_score": f"{game.home_score} - {game.away_score}",
                    "sb_match_id": best.match_id if best is not None else pd.NA,
                    "sb_home": shown.home_team if shown is not None else "", "sb_away": shown.away_team if shown is not None else "",
                    "name_sim": scored[0][0] if scored else np.nan, "reason": reason})
    match_map = pd.DataFrame(out, columns=MATCH_MAP_COLUMNS).astype({"sb_match_id": "Int64"})
    assert_no_reuse(match_map)
    return match_map


def assert_no_reuse(match_map: pd.DataFrame) -> None:
    """No StatsBomb match may be paired with two SoccerNet matches."""
    ids = match_map.sb_match_id.dropna()
    if ids.duplicated().any():
        reused = sorted(ids[ids.duplicated()].unique().tolist())
        raise ValueError(f"StatsBomb matches paired with more than one SoccerNet match: {reused}")


def rejected_candidates(games: pd.DataFrame, sb_matches: pd.DataFrame) -> pd.DataFrame:
    """The date+competition+score candidates the name rule rejected, with both similarities: the places where a missing
    alias would hide. One row per (SoccerNet match, StatsBomb match). The caller passes the unpaired matches."""
    rows = []
    for game in games.itertuples():
        for m in _exact_score(game, _same_day_candidates(game, sb_matches)).itertuples():
            home, away = name_sim(game.home, m.home_team), name_sim(game.away, m.away_team)
            if min(home, away) < NAME_MIN:
                rows.append({"game": game.game, "sn_home": game.home, "sn_away": game.away, "sb_match_id": m.match_id,
                             "sb_home": m.home_team, "sb_away": m.away_team, "home_sim": round(home, 3), "away_sim": round(away, 3)})
    return pd.DataFrame(rows, columns=["game", "sn_home", "sn_away", "sb_match_id", "sb_home", "sb_away", "home_sim", "away_sim"])


def club_counterparts(club: str, others: Iterable[str]) -> bool:
    return any(name_sim(club, other) >= NAME_MIN for other in others)


def unmatched_clubs(games: pd.DataFrame, sb_matches: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """The alias-list test: clubs that have no counterpart in the other source.

    "sn_without_sb": (competition season, SoccerNet club) with no StatsBomb club of at least `NAME_MIN` similarity in the
    same competition season; only seasons with at least `MIN_SB_SEASON_MATCHES` StatsBomb matches count, because a
    season with fewer cannot hold the SoccerNet clubs.
    "sb_without_sn": (competition, StatsBomb club) with no SoccerNet club of the same competition in any season. A
    StatsBomb club may lack a SoccerNet game in one season (SoccerNet has a sample of the matches), so only the pooled
    competition counts."""
    sn_season = {}
    for game in games.itertuples():
        key = (COMPETITION[game.league], game.season.replace("-", "/"))
        sn_season.setdefault(key, set()).update((game.home, game.away))
    sb_season = {}
    for match in sb_matches.itertuples():
        sb_season.setdefault((match.competition, match.season), set()).update((match.home_team, match.away_team))
    sn_without_sb = sorted(
        (f"{competition} {season}", club) for (competition, season), clubs in sn_season.items()
        if len(sb_matches[(sb_matches.competition == competition) & (sb_matches.season == season)]) >= MIN_SB_SEASON_MATCHES
        for club in clubs if not club_counterparts(club, sb_season.get((competition, season), ())))
    sn_competition = {}
    for (competition, _), clubs in sn_season.items():
        sn_competition.setdefault(competition, set()).update(clubs)
    sb_without_sn = sorted(
        (competition, club) for competition in {c for c, _ in sb_season} & set(sn_competition)
        for club in set().union(*(clubs for (c, _), clubs in sb_season.items() if c == competition))
        if not club_counterparts(club, sn_competition[competition]))
    return {"sn_without_sb": sn_without_sb, "sb_without_sn": sb_without_sn}
