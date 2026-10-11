"""Retraining split plan of the vendor comparison (xg-vendor ticket 05, test seam 1).

`build_plan(index, match_map, seed)` is a pure function of the shot index, the frozen match map of ticket 02 and a seed.
It gives every SoccerNet match exactly one role:

* evaluation: the matches with a StatsBomb partner (`match_map.reason == "paired"`); the retrained G-combo never sees them;
* gate: `N_GATE` of the non-overlap official-test matches (the official test matches that are not evaluation matches),
  drawn stratified by league and season. The retrained ten-model G-combo is checked against the archived G-combo on them;
* training: every other match (non-overlap train/valid matches and the non-overlap official-test matches not drawn for
  the gate), each with a home-team fold.

Gate allocation. A stratum is a (league, season). Stratum s with n_s of the N pool matches gets floor(N_GATE * n_s / N)
matches, and the remaining matches go one each to the strata with the largest remainders (N_GATE * n_s mod N), ties
broken by the larger stratum, then by name. The arithmetic is exact (integers). Inside a stratum the matches are sorted
by name and permuted with a generator seeded from (seed, stratum), so the draw is the same for the same seed and does
not move when another stratum changes.

Folds. `home_folds` is the construction of `run_cv.fold_table` (home team from the match identifier by the same regular
expression, matches sorted by name, `GroupKFold` over the home team) applied to the training matches, so a club's home
matches share a fold. As in the archived folds, away matches are not grouped.

The plan does not touch labels other than through the index's official split; the gate tolerance is fixed here, before
any retrain (decision D7), and written into the plan record by `scripts/vendor_split_plan.py`.
"""

import zlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 0   # fixed before any retrain; not tuned
N_GATE = 37
N_FOLDS = 5
ROLES = ("evaluation", "gate", "training")
NO_FOLD = -1   # fold of a match that is not a training match
PLAN_COLUMNS = ["game", "league", "season", "home", "official_split", "role", "fold"]
OVERLAP_REASON = "paired"
HOME_RE = r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ "   # as scripts/run_cv.py

# Gate tolerance (point estimates, retrained ten-model G-combo minus archived G-combo on the gate matches): the
# retrain passes if log-loss is no more than 0.010 higher and AUC no more than 0.020 lower. The paired match-bootstrap
# interval of both differences is reported next to it and is not part of the decision.
GATE_TOLERANCE = {"log_loss": 0.010, "auc": 0.020}


def home_team(games: pd.Series) -> pd.Series:
    """Home club from the match identifier; the identifier itself when it does not parse (as run_cv)."""
    return games.str.extract(HOME_RE)[0].fillna(games)


def match_table(index: pd.DataFrame, match_map: pd.DataFrame) -> pd.DataFrame:
    """One row per SoccerNet match: game, league, season, home, official_split, overlap. Raises ValueError when the
    index and the match map do not hold the same matches, or a match has no or two official splits."""
    if match_map.game.duplicated().any():
        raise ValueError("Duplicate matches in the match map")
    splits = index.groupby("game").split.agg(["nunique", "first"])
    if (splits["nunique"] != 1).any():
        raise ValueError(f"Matches with more than one official split: {list(splits.index[splits['nunique'] != 1])[:3]}")
    if set(splits.index) != set(match_map.game):
        raise ValueError(f"Index and match map hold different matches: {len(set(splits.index) ^ set(match_map.game))} differ")
    table = pd.DataFrame({"game": splits.index, "official_split": splits["first"].to_numpy()}).sort_values("game", ignore_index=True)
    parts = table.game.str.split("/", expand=True)
    if parts.shape[1] != 3 or parts.isna().any().any():
        raise ValueError("Match identifiers must read league/season/match")
    table["league"], table["season"] = parts[0], parts[1]
    table["home"] = home_team(table.game)
    overlap = set(match_map.game[match_map.reason == OVERLAP_REASON])
    table["overlap"] = table.game.isin(overlap)
    return table[["game", "league", "season", "home", "official_split", "overlap"]]


def largest_remainder(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Proportional allocation of `total` over strata of the given sizes by the largest-remainder rule (exact integers;
    ties go to the larger stratum, then to the name). Never allocates more than a stratum holds."""
    pool = sum(sizes.values())
    if not 0 <= total <= pool:
        raise ValueError(f"Cannot draw {total} from {pool}")
    alloc = {s: total * n // pool for s, n in sizes.items()}
    by_remainder = sorted(sizes, key=lambda s: (-(total * sizes[s] % pool), -sizes[s], s))
    for s in by_remainder[: total - sum(alloc.values())]:
        alloc[s] += 1
    return alloc


def stratum_of(table: pd.DataFrame) -> pd.Series:
    return table.league + "/" + table.season


def draw_gate(pool: pd.DataFrame, n_gate: int, seed: int) -> list[str]:
    """`n_gate` matches of `pool` (a match table), stratified by league and season; sorted."""
    strata = stratum_of(pool)
    alloc = largest_remainder(strata.value_counts().to_dict(), n_gate)
    drawn = []
    for stratum, k in alloc.items():
        games = sorted(pool.game[strata == stratum])
        rng = np.random.default_rng([seed, zlib.crc32(stratum.encode())])
        drawn += [games[i] for i in rng.permutation(len(games))[:k]]
    return sorted(drawn)


def home_folds(games: pd.Series, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """game, home, fold for `games`, as `run_cv.fold_table` builds it: sorted by game, GroupKFold over the home team."""
    folds = pd.DataFrame({"game": sorted(set(games))})
    folds["home"] = home_team(folds.game)
    if folds.home.nunique() < n_folds:
        raise ValueError(f"{folds.home.nunique()} home teams cannot fill {n_folds} folds")
    folds["fold"] = NO_FOLD
    for k, (_, held_out) in enumerate(GroupKFold(n_folds).split(folds, groups=folds.home)):
        folds.loc[held_out, "fold"] = k
    return folds


def build_plan(index: pd.DataFrame, match_map: pd.DataFrame, seed: int = SEED, n_gate: int = N_GATE,
               n_folds: int = N_FOLDS) -> pd.DataFrame:
    """The plan: one row per match with PLAN_COLUMNS, sorted by game. `fold` is NO_FOLD outside training."""
    table = match_table(index, match_map)
    pool = table[(table.official_split == "test") & ~table.overlap]
    gate = set(draw_gate(pool, n_gate, seed))
    table["role"] = np.select([table.overlap, table.game.isin(gate)], ["evaluation", "gate"], "training")
    training = table.role == "training"
    folds = home_folds(table.game[training], n_folds).set_index("game").fold
    table["fold"] = table.game.map(folds).fillna(NO_FOLD).astype(int)
    return table[PLAN_COLUMNS]


def plan_text(plan: pd.DataFrame) -> str:
    """The canonical text of plan.csv; its sha256 is the plan's identity."""
    return plan.to_csv(index=False)


def training_folds(plan: pd.DataFrame) -> pd.DataFrame:
    """The fold file of the retrain: game, home, fold of the training matches (the layout of folds_v12_home5.csv)."""
    return plan.loc[plan.role == "training", ["game", "home", "fold"]].reset_index(drop=True)
