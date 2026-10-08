"""Split plan of the leave-one-league-out transfer test (test seam 2 of the credibility round).

`build_plan(index)` is a pure function of the shot index. For every domestic league L and
match half-set h of L (A or B: a seeded, season-stratified split of L's matches into two
halves of the league, unrelated to match periods) it lists:

* role "test": h's matches, scored by both arms;
* arm "without": `n_matches` matches drawn from L's pool, i.e. every match that involves
  none of L's clubs. That removes L itself and every Champions League match with one of
  its clubs;
* arm "with": all of L's other half-set plus the first `n_matches - |other|` matches of the
  same pool ordering that "without" uses, so the two arms differ only in the league's
  matches and the pool matches they replace.

Inside each arm the matches get the roles "train" (gradient steps), "select" (epoch
selection) and "calibration" (the slice Platt is fitted on) in the fixed proportions of
ROLE_WEIGHTS. Roles follow the pool ordering by smooth weighted round-robin, so every prefix
of the ordering is balanced and the "with" arm's pool matches keep the roles they have in
"without"; the other half-set fills the remaining quota of each role. Both arms therefore
have exactly the same number of matches in every role.

Club names come from the match identifier (identical to gameHomeTeam/gameAwayTeam in the
SoccerNet Labels-v2.json files on all 500 matches of index v1.2) and pass through
`normalise_club`. A plan is refused while two club names look alike and are neither
aliased nor reviewed as distinct clubs, so a spelling change across seasons cannot split
one club into two and leak it into the pool.
"""

import re
import unicodedata
import zlib
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import pandas as pd

DOMESTIC_LEAGUES = ("england_epl", "france_ligue-1", "germany_bundesliga", "italy_serie-a", "spain_laliga")
CHAMPIONS_LEAGUE = "europe_uefa-champions-league"  # never held out; only filtered by club
HALF_SETS = ("A", "B")
ARMS = ("without", "with")
ROLES = ("train", "select", "calibration", "test")
ROLE_WEIGHTS = {"train": 15, "select": 2, "calibration": 3}  # per 20 matches: 75 / 10 / 15 %
N_MATCHES = 300
PLAN_COLUMNS = ["league", "half_set", "arm", "role", "source", "game"]

# "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley": anchored, so the date cannot pass for a club
MATCH_RE = re.compile(r"^\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$")

# Spelling variant -> canonical name, both after `clean_name`. Empty on index v1.2: the audit of
# 2026-10-08 found one spelling per club across all seasons and competitions (133 names).
CLUB_ALIASES: dict[str, str] = {}

# Look-alike pairs on index v1.2 that were checked and are different clubs.
REVIEWED_DISTINCT = frozenset(frozenset(pair) for pair in [
    ("Atl. Madrid", "Real Madrid"),
    ("Dijon", "Gijon"),
    ("Everton", "Verona"),
    ("Manchester City", "Manchester United"),
])
LOOKALIKE_RATIO = 0.75


def clean_name(name: str) -> str:
    """Strip accents and collapse whitespace: "  Atlético  Madrid " -> "Atletico Madrid"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(ascii_name.split())


def normalise_club(name: str, aliases: Mapping[str, str] = CLUB_ALIASES) -> str:
    cleaned = clean_name(name)
    return aliases.get(cleaned, cleaned)


def parse_match(game: str, aliases: Mapping[str, str] = CLUB_ALIASES) -> tuple[str, str, str, str]:
    """`league/season/<date> - <time> <home> <h> - <a> <away>` -> (league, season, home, away)."""
    parts = game.split("/")
    m = MATCH_RE.match(parts[-1]) if len(parts) == 3 else None
    if m is None:
        raise ValueError(f"Not a league/season/match identifier: {game!r}")
    return parts[0], parts[1], normalise_club(m.group(1), aliases), normalise_club(m.group(2), aliases)


def match_table(index: pd.DataFrame, aliases: Mapping[str, str] = CLUB_ALIASES) -> pd.DataFrame:
    """One row per match: game, league, season, home, away, shots, goals."""
    if index.shot_id.duplicated().any():
        raise ValueError("Duplicate shot ids in the index")
    matches = (index.assign(goal=index.goal.astype(bool)).groupby("game", sort=True)
               .agg(shots=("shot_id", "size"), goals=("goal", "sum")).reset_index())
    parsed = pd.DataFrame([parse_match(g, aliases) for g in matches.game],
                          columns=["league", "season", "home", "away"], index=matches.index)
    return pd.concat([matches[["game"]], parsed, matches[["shots", "goals"]]], axis=1)


def _words(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.casefold())


def _abbreviates(short: str, long: str) -> bool:
    """"utd" abbreviates "united": same first letter, letters in order."""
    rest = iter(long[1:])
    return short[0] == long[0] and all(c in rest for c in short[1:])


def lookalike_pairs(clubs: Iterable[str]) -> list[tuple[str, str]]:
    """Pairs of names that may be one club. Four signs: similar spelling; one name's words
    inside the other's; word-by-word abbreviation ("Man Utd" / "Manchester United"); an
    acronym of the other's words ("PSG" / "Paris Saint-Germain"). Nicknames with no shared
    letters ("Spurs" / "Tottenham Hotspur") pass unseen: the manual audit covers those."""
    def similar(a: str, b: str) -> bool:
        wa, wb = _words(a), _words(b)
        if set(wa) <= set(wb) or set(wb) <= set(wa):
            return True
        if len(wa) == len(wb) and all(_abbreviates(x, y) or _abbreviates(y, x) for x, y in zip(wa, wb, strict=True)):
            return True
        if [a.casefold()] == wa and a.casefold() == "".join(w[0] for w in wb):
            return True
        if [b.casefold()] == wb and b.casefold() == "".join(w[0] for w in wa):
            return True
        return SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= LOOKALIKE_RATIO
    return [(a, b) for a, b in combinations(sorted(set(clubs)), 2) if similar(a, b)]


def check_club_names(matches: pd.DataFrame) -> None:
    clubs = set(matches.home) | set(matches.away)
    unreviewed = [p for p in lookalike_pairs(clubs) if frozenset(p) not in REVIEWED_DISTINCT]
    if unreviewed:
        raise ValueError(f"Club names that may be one club; add an alias or review them as distinct: {unreviewed}")


def league_clubs(matches: pd.DataFrame, league: str) -> set[str]:
    own = matches[matches.league == league]
    return set(own.home) | set(own.away)


def training_pool(matches: pd.DataFrame, league: str) -> list[str]:
    """Matches with none of `league`'s clubs, sorted: the league and its clubs' Champions League games are out."""
    clubs = league_clubs(matches, league)
    return sorted(matches.game[~(matches.home.isin(clubs) | matches.away.isin(clubs))])


def _rng(seed: int, *keys: str) -> np.random.Generator:
    # crc32 rather than hash(): stable across interpreter runs
    return np.random.default_rng([seed, zlib.crc32("|".join(keys).encode())])


def half_sets(matches: pd.DataFrame, league: str, seed: int) -> pd.Series:
    """game -> "A" | "B" for the league's matches. Shuffled within each season and dealt
    alternately across the season-ordered list, so A and B differ by at most one match
    in every season and overall."""
    own = matches[matches.league == league].sort_values("game")
    rng = _rng(seed, league, "half_sets")
    order = [g for _, season in own.groupby("season", sort=True) for g in rng.permutation(season.game.to_numpy())]
    return pd.Series([HALF_SETS[i % 2] for i in range(len(order))], index=order, name="half_set")


def _round_robin(n: int, weights: Mapping[str, int]) -> list[str]:
    """Smooth weighted round-robin: every prefix of the result holds each role in its
    proportion to within one match, and n = sum(weights) gives exactly `weights`."""
    active = {r: w for r, w in weights.items() if w > 0}
    total, current, out = sum(active.values()), dict.fromkeys(active, 0), []
    for _ in range(n):
        for r, w in active.items():
            current[r] += w
        pick = max(active, key=current.__getitem__)  # ties go to the first role in `weights`
        current[pick] -= total
        out.append(pick)
    return out


def build_plan(index: pd.DataFrame, seed: int = 0, n_matches: int = N_MATCHES,
               leagues: Iterable[str] = DOMESTIC_LEAGUES, aliases: Mapping[str, str] = CLUB_ALIASES) -> pd.DataFrame:
    """The full plan as rows of PLAN_COLUMNS; "source" is "league" for the held-out league's
    matches and "pool" for matches from its pool."""
    matches = match_table(index, aliases)
    check_club_names(matches)
    pool_roles = _round_robin(n_matches, ROLE_WEIGHTS)
    rows = []
    for league in leagues:
        pool = training_pool(matches, league)
        halves = half_sets(matches, league, seed)
        if halves.empty:
            raise ValueError(f"No matches of {league} in the index")
        if len(pool) < n_matches:
            raise ValueError(f"Pool of {league} has {len(pool)} matches, fewer than {n_matches}")
        for h in HALF_SETS:
            other = sorted(halves.index[halves != h])
            n_pool = n_matches - len(other)
            if n_pool < 0:
                raise ValueError(f"{league}: other half-set ({len(other)}) exceeds {n_matches} matches")
            order = _rng(seed, league, h, "pool").permutation(pool)[:n_matches]
            quotas = {r: pool_roles.count(r) - pool_roles[:n_pool].count(r) for r in ROLE_WEIGHTS}
            other_order = _rng(seed, league, h, "with").permutation(other)
            arms = {
                "without": [(r, "pool", g) for g, r in zip(order, pool_roles, strict=True)],
                "with": [(r, "league", g) for g, r in zip(other_order, _round_robin(len(other), quotas), strict=True)]
                + [(r, "pool", g) for g, r in zip(order[:n_pool], pool_roles[:n_pool], strict=True)],
            }
            test = [("test", "league", g) for g in sorted(halves.index[halves == h])]
            for arm, members in arms.items():
                rows += [(league, h, arm, *member) for member in members + test]
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    plan["arm"] = pd.Categorical(plan.arm, ARMS, ordered=True)
    plan["role"] = pd.Categorical(plan.role, ROLES, ordered=True)
    plan = plan.sort_values(["league", "half_set", "arm", "role", "game"]).reset_index(drop=True)
    return plan.astype({"arm": str, "role": str, "game": str})


def summarise(plan: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Matches, held-out-league matches, shots and goals per league, half-set, arm and role."""
    stats = (index.assign(goal=index.goal.astype(bool)).groupby("game")
             .agg(shots=("shot_id", "size"), goals=("goal", "sum")).reset_index())
    rows = plan.merge(stats, on="game", how="left", validate="many_to_one")
    rows["league_matches"] = rows.source.eq("league").astype(int)
    out = (rows.groupby(["league", "half_set", "arm", "role"], sort=False)
           .agg(matches=("game", "size"), league_matches=("league_matches", "sum"),
                shots=("shots", "sum"), goals=("goals", "sum")).reset_index())
    return out


def pool_summary(index: pd.DataFrame, leagues: Iterable[str] = DOMESTIC_LEAGUES,
                 aliases: Mapping[str, str] = CLUB_ALIASES) -> pd.DataFrame:
    """Per held-out league: its matches, shots, goals and clubs, the Champions League matches
    removed for its clubs, and the size of the pool left. Pass the aliases given to build_plan."""
    matches = match_table(index, aliases)
    rows = []
    for league in leagues:
        clubs = league_clubs(matches, league)
        own = matches[matches.league == league]
        ucl = matches[matches.league == CHAMPIONS_LEAGUE]
        rows.append({"league": league, "matches": len(own), "shots": int(own.shots.sum()),
                     "goals": int(own.goals.sum()), "clubs": len(clubs),
                     "ucl_removed": int((ucl.home.isin(clubs) | ucl.away.isin(clubs)).sum()),
                     "pool": len(training_pool(matches, league))})
    return pd.DataFrame(rows)
