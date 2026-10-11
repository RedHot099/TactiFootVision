"""Permutations of the dry run (xg-vendor ticket 14): the whole confirmatory analysis runs once on labels permuted
within matches, so the code is shown to work without the real answer.

Labels (H1, H2, H4)
    Strata: (match, is_penalty), visited in sorted order; rows inside a stratum in key order. Per stratum, one
    permutation moves `goal_sn` and `sb_outcome` together over all its rows, and a second one moves `on_target_sn`
    over its primary rows (the only rows that carry it). `is_penalty` and the keys stay. Generator
    `numpy.random.default_rng([seed, LABEL_STREAM])`, so goal counts per match and per penalty status are kept and every
    shot-level association with the predictions is destroyed. The second permutation also decouples on-target from
    goal; the dry run needs neither.
Probe targets (H3)
    The probes read their targets from the StatsBomb shots table. The rows of the table's StatsBomb events are permuted
    together (location, body part, freeze frame) within strata (match, penalty shot type), generator
    `default_rng([seed, TARGET_STREAM])`; the probes then derive every target from the permuted rows.
What survives a permutation within matches: differences between matches (goal rate, mean distance). The dry-run
numbers therefore need not be centred on zero; they are never results.
"""

import numpy as np
import pandas as pd

from xgpoc.vendor import evaltable as et

LABEL_STREAM, TARGET_STREAM = 1, 2
PENALTY_SHOT_TYPE = "Penalty"
SHOT_PAYLOAD = ["x", "y", "body_part", "freeze_frame_json"]


def _strata(keys: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each stratum (all columns of `keys` but `order`), strata sorted, rows sorted by `order`."""
    columns = [c for c in keys.columns if c != "order"]
    ordered = keys.reset_index(drop=True).sort_values([*columns, "order"], kind="stable")
    return [group.index.to_numpy() for _, group in ordered.groupby(columns, sort=True)]


def permute_labels(labels: pd.DataFrame, table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """`labels` (key + `LABEL_COLUMNS`, the table's keys in table order) permuted within matches (module docstring)."""
    if not labels.key.reset_index(drop=True).equals(table.key.reset_index(drop=True)):
        raise ValueError("the labels are not in table order")
    rng = np.random.default_rng([seed, LABEL_STREAM])
    out = labels.reset_index(drop=True).copy()
    primary = table.population.eq("primary").to_numpy()
    strata = pd.DataFrame({"match": table.match.to_numpy(), "is_penalty": labels.is_penalty.astype(bool).to_numpy(),
                           "order": labels.key.to_numpy()})
    for rows in _strata(strata):
        moved = rows[rng.permutation(len(rows))]
        for column in ("goal_sn", "sb_outcome"):
            out.loc[rows, column] = labels[column].to_numpy()[moved]
        with_target = rows[primary[rows]]
        moved = with_target[rng.permutation(len(with_target))]
        out.loc[with_target, "on_target_sn"] = labels.on_target_sn.to_numpy()[moved]
    return out[["key", *et.LABEL_COLUMNS]]


def permute_shots(shots: pd.DataFrame, table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The StatsBomb shot rows of the table's events (`event_id` + `SHOT_PAYLOAD`), with the payload permuted within
    strata (match, penalty shot type) (module docstring). Rows come back in table order."""
    rows = shots.set_index("event_id").loc[table.statsbomb_event_id.to_numpy(), SHOT_PAYLOAD].reset_index()
    rng = np.random.default_rng([seed, TARGET_STREAM])
    out = rows.copy()
    strata = pd.DataFrame({"match": table.match.to_numpy(), "penalty": table.sb_shot_type.eq(PENALTY_SHOT_TYPE).to_numpy(),
                           "order": table.statsbomb_event_id.to_numpy()})
    for stratum in _strata(strata):
        moved = stratum[rng.permutation(len(stratum))]
        out.loc[stratum, SHOT_PAYLOAD] = rows.loc[moved, SHOT_PAYLOAD].to_numpy()
    return out
