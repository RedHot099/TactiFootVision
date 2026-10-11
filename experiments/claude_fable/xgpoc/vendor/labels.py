"""Labels of the evaluation table (decision D4; xg-vendor tickets 14, 13, 15).

The label columns of `xgpoc.vendor.evaltable` (`LABEL_COLUMNS`) for the primary rows come from the frozen pairing's
pair table (`results/xgv_pairing/pairs.parquet`, one row per index v1.2 shot):

    goal_sn       index v1.2 goal of the SoccerNet shot (decision D4: the label G-combo was trained on)
    on_target_sn  SoccerNet label "Shots on target", which counts blocked shots (decision D4)
    sb_outcome    StatsBomb outcome of the paired shot
    is_penalty    index v1.2 penalty flag (on the 2,754 paired shots it equals the StatsBomb penalty flag)

The rule for the secondary rows (a recovered goal is a goal; a StatsBomb-only shot is a non-goal; StatsBomb goals without
any SoccerNet goal are listed and excluded) belongs to ticket 13 and is not implemented here.
"""

import numpy as np
import pandas as pd

from xgpoc.vendor import evaltable as et

ON_TARGET_LABEL = "Shots on target"
PAIR_COLUMNS = ["shot_id", "paired", "label", "goal", "is_penalty", "sb_event_id", "sb_outcome"]


def primary_labels(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """`key` + `LABEL_COLUMNS` of the table's primary rows, in table order. Refuses a table with other populations, a
    primary row that is not a pair of the pair table and a pair whose StatsBomb event differs from the table's."""
    if not table.population.eq("primary").all():
        raise ValueError("primary_labels covers primary rows only; the secondary label rule is ticket 13's")
    paired = pairs.loc[pairs.paired.astype(bool), PAIR_COLUMNS].set_index("shot_id")
    missing = set(table.key) - set(paired.index)
    if missing:
        raise ValueError(f"{len(missing)} primary rows are not pairs of the pair table, e.g. {sorted(missing)[0]}")
    rows = paired.loc[table.key.to_numpy()]
    if not np.array_equal(rows.sb_event_id.to_numpy(), table.statsbomb_event_id.to_numpy()):
        raise ValueError("the pair table pairs a primary row with another StatsBomb event than the table")
    return pd.DataFrame({"key": table.key.to_numpy(), "goal_sn": rows.goal.astype(bool).to_numpy(),
                         "on_target_sn": pd.array(rows.label.eq(ON_TARGET_LABEL).to_numpy(), dtype="boolean"),
                         "sb_outcome": rows.sb_outcome.to_numpy(), "is_penalty": rows.is_penalty.astype(bool).to_numpy()},
                        columns=["key", *et.LABEL_COLUMNS])
