"""A synthetic xg-stack population with every input column of every arm, and outcomes with a known truth."""

import numpy as np
import pandas as pd
from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si


def synthetic_table(n_matches: int = 60, per_match: int = 25, seed: int = 0, folds: int = 5) -> pd.DataFrame:
    """Shots of `n_matches` matches in `folds` match-level folds: z ~ N(-2, 1), dist and angle with ~4% missing (with
    their indicator), every other input column noise of the right kind (0/1 indicators and flags, counts, reals)."""
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    match = np.repeat(np.arange(n_matches), per_match)
    table = pd.DataFrame({"game": [f"g{m:03d}" for m in match], "half": 1, "position_ms": np.arange(n) * 1000,
                          "fold": match % folds, sf.Z: rng.normal(-2, 1, n)}, index=[f"s{i:05d}" for i in range(n)])
    for column in si.E3_COLUMNS:
        name = column.split("@")[0]
        if name.endswith("_missing") or name.startswith("ball_") or name in [*si.EXTRA_FLAGS, "shot_from_ball", "shooter_by_net",
                                                                              "shot_masked", "gk_masked"]:
            table[column] = (rng.random(n) < 0.2).astype(float)
        elif name in ("n_keypoints", "n_players", "n_valid", "n_masked"):
            table[column] = rng.integers(0, 20, n).astype(float)
        else:
            table[column] = rng.normal(0, 1, n)
    for offset in si.OFFSETS:
        missing = rng.random(n) < 0.04
        for indicator, columns in si.missing_groups(offset).items():
            gap = missing if indicator == si.col("shot_missing", offset) else np.zeros(n, bool)
            table[indicator] = gap.astype(float)
            for c in columns:
                table.loc[gap, c] = np.nan
        table[si.col("dist", offset)] = np.where(missing, np.nan, rng.uniform(2, 40, n))
        table[si.col("angle", offset)] = np.where(missing, np.nan, rng.uniform(0.05, 1.2, n))
    for flag in si.S_FLAGS:
        table[flag] = (rng.random(n) < 0.1).astype(float)
    table.index.name = "shot_id"
    return table


def standardised_distance(table: pd.DataFrame) -> np.ndarray:
    x = table[si.col("dist", 0)].to_numpy(float)
    finite = np.isfinite(x)
    return np.where(finite, (x - x[finite].mean()) / x[finite].std(), 0.0)


def outcomes(table: pd.DataFrame, beta: float = 0.0, seed: int = 1) -> np.ndarray:
    """y ~ Bernoulli(sigmoid(z + beta * standardised distance)), missing distance 0."""
    rng = np.random.default_rng(seed)
    p = sigmoid(table[sf.Z].to_numpy() + beta * standardised_distance(table))
    return (rng.random(len(p)) < p).astype(int)
