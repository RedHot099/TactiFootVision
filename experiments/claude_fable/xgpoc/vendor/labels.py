"""Labels of the evaluation table (decision D4; xg-vendor tickets 14, 13, 15).

The label columns of `xgpoc.vendor.evaltable` (`LABEL_COLUMNS`) come from the frozen pairing (ticket 02).

Primary rows (`primary_labels`), from the pair table (`results/xgv_pairing/pairs.parquet`, one row per index v1.2 shot):
    goal_sn          index v1.2 goal of the SoccerNet shot (decision D4: the label G-combo was trained on)
    on_target_sn     SoccerNet label "Shots on target", which counts blocked shots (decision D4)
    sb_outcome       StatsBomb outcome of the paired shot
    is_penalty       index v1.2 penalty flag (on the 2,754 paired shots it equals the StatsBomb penalty flag); a pre-shot
                     condition, used only for routing
    label_exclusion  null: every pair enters the analyses with its SoccerNet label, label disagreements included

Secondary rows (`secondary_labels`, spec user stories 73-74), from the pairing's population table of the unpaired
StatsBomb shots (`results/xgv_pairing/sb_unpaired.parquet`, by StatsBomb event id):
    goal_sn          True for a recovered goal (the recovery rule attributed a SoccerNet goal to it), False for a
                     StatsBomb-only shot (no SoccerNet goal is attributed to it)
    on_target_sn     null (SoccerNet has no label for a shot it did not annotate)
    sb_outcome       StatsBomb outcome
    is_penalty       StatsBomb penalty flag (recovered goals include penalties)
    label_exclusion  "sb_goal_without_sn_goal" for a StatsBomb-only shot that StatsBomb counts as a goal while no
                     SoccerNet goal exists for it (the pairing's `sb_goal_without_sn_goal` flag; it must agree with the
                     outcome): the row stays in the table, listed, and every analysis leaves it out
`table_labels` applies both rules and returns the labels in table order, ready for `evaltable.join_labels`.
"""

import numpy as np
import pandas as pd

from xgpoc.vendor import evaltable as et

ON_TARGET_LABEL = "Shots on target"
PAIR_COLUMNS = ["shot_id", "paired", "label", "goal", "is_penalty", "sb_event_id", "sb_outcome"]
UNPAIRED_POPULATION = {"recovered_goal": "recovered_goal", "statsbomb_only": "secondary"}   # table -> pairing population
GOAL_OUTCOME = "Goal"
SB_GOAL_WITHOUT_SN_GOAL = "sb_goal_without_sn_goal"


def primary_labels(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """`key` + `LABEL_COLUMNS` of the table's primary rows, in table order. Refuses a table with other populations, a
    primary row that is not a pair of the pair table and a pair whose StatsBomb event differs from the table's."""
    if not table.population.eq("primary").all():
        raise ValueError("primary_labels covers primary rows only; secondary rows take secondary_labels")
    paired = pairs.loc[pairs.paired.astype(bool), PAIR_COLUMNS].set_index("shot_id")
    missing = set(table.key) - set(paired.index)
    if missing:
        raise ValueError(f"{len(missing)} primary rows are not pairs of the pair table, e.g. {sorted(missing)[0]}")
    rows = paired.loc[table.key.to_numpy()]
    if not np.array_equal(rows.sb_event_id.to_numpy(), table.statsbomb_event_id.to_numpy()):
        raise ValueError("the pair table pairs a primary row with another StatsBomb event than the table")
    return pd.DataFrame({"key": table.key.to_numpy(), "goal_sn": rows.goal.astype(bool).to_numpy(),
                         "on_target_sn": pd.array(rows.label.eq(ON_TARGET_LABEL).to_numpy(), dtype="boolean"),
                         "sb_outcome": rows.sb_outcome.to_numpy(), "is_penalty": rows.is_penalty.astype(bool).to_numpy(),
                         "label_exclusion": np.full(len(table), None, dtype=object)},
                        columns=["key", *et.LABEL_COLUMNS])


def secondary_labels(table: pd.DataFrame, unpaired: pd.DataFrame) -> pd.DataFrame:
    """`key` + `LABEL_COLUMNS` of the table's secondary rows, in table order (module docstring). Refuses primary rows, an
    event the pairing's population table does not list, a row whose population there differs from the table's, and a
    `sb_goal_without_sn_goal` flag that disagrees with the StatsBomb outcome of a StatsBomb-only shot."""
    if not table.population.isin(list(UNPAIRED_POPULATION)).all():
        raise ValueError("secondary_labels covers secondary rows only; primary rows take primary_labels")
    known = unpaired.set_index("event_id")
    missing = set(table.statsbomb_event_id) - set(known.index)
    if missing:
        raise ValueError(f"{len(missing)} secondary rows are not in the pairing's population table, e.g. {sorted(missing)[0]}")
    rows = known.loc[table.statsbomb_event_id.to_numpy()]
    if not np.array_equal(rows.population.to_numpy(), table.population.map(UNPAIRED_POPULATION).to_numpy()):
        raise ValueError("a secondary row has another population in the pairing than in the table")
    recovered = table.population.eq("recovered_goal").to_numpy()
    flagged = rows.sb_goal_without_sn_goal.astype(bool).to_numpy()
    if not np.array_equal(flagged[~recovered], rows.outcome.eq(GOAL_OUTCOME).to_numpy()[~recovered]) or flagged[recovered].any():
        raise ValueError("the pairing's sb_goal_without_sn_goal flag disagrees with the StatsBomb outcome")
    return pd.DataFrame({"key": table.key.to_numpy(), "goal_sn": recovered,
                         "on_target_sn": pd.array([pd.NA] * len(table), dtype="boolean"),
                         "sb_outcome": rows.outcome.to_numpy(), "is_penalty": rows.penalty.astype(bool).to_numpy(),
                         "label_exclusion": np.where(flagged, SB_GOAL_WITHOUT_SN_GOAL, None).astype(object)},
                        columns=["key", *et.LABEL_COLUMNS])


def table_labels(table: pd.DataFrame, pairs: pd.DataFrame, unpaired: pd.DataFrame | None) -> pd.DataFrame:
    """`key` + `LABEL_COLUMNS` of every row of the table, in table order: `primary_labels` for the primary rows,
    `secondary_labels` for the others (they need `unpaired`, the pairing's population table)."""
    primary = table.population.eq("primary").to_numpy()
    parts = [primary_labels(table[primary], pairs)] if primary.any() else []
    if (~primary).any():
        if unpaired is None:
            raise ValueError("a table with secondary rows needs the pairing's population table (sb_unpaired.parquet)")
        parts.append(secondary_labels(table[~primary], unpaired))
    return pd.concat(parts, ignore_index=True).set_index("key").loc[table.key.to_numpy()].reset_index()[["key", *et.LABEL_COLUMNS]]
