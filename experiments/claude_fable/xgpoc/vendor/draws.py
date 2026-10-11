"""Match-bootstrap draws shared by the analyses of the evaluation table (xg-vendor tickets 08, 09).

The draws are `xgpoc.probe.readability.match_draw_counts` over the matches of the whole table (`np.unique` of the
`match` column, `default_rng(seed)`), so every analysis and every subset of rows uses the same draws: a match drawn k
times weighs k in that draw. A subset of the table simply reads its matches' columns of the counts.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from xgpoc.probe.readability import match_draw_counts

N_BOOT = 5000   # spec: no fewer than 5,000 match draws
SEED = 0


class TableDraws(NamedTuple):
    """`counts` (B, M) over the table's sorted matches `keys`; `match` is each table row's position in `keys`."""

    keys: np.ndarray
    counts: np.ndarray
    match: np.ndarray

    def weights(self, rows: np.ndarray) -> np.ndarray:
        """(B, len(rows)) bootstrap weights of the given table rows."""
        return self.counts[:, self.match[rows]].astype(float)


def table_draws(table: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> TableDraws:
    """The shared draws of `table` (module docstring)."""
    games = table.match.to_numpy()
    keys, counts = match_draw_counts(games, n_boot, seed)
    return TableDraws(keys, counts, np.searchsorted(keys, games))


def split_subsets(table: pd.DataFrame, by_split: bool = False) -> dict[str, np.ndarray]:
    """Row subsets an analysis reports: all rows and, if asked, the official test and train/valid matches separately."""
    subsets = {"all": np.ones(len(table), bool)}
    if by_split:
        subsets |= {"test": table.split.eq("test").to_numpy(), "trainvalid": table.split.isin(["train", "valid"]).to_numpy()}
    return subsets
