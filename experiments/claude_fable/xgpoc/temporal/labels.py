"""The only label read of xg-temporal (ticket 04, after the frozen protocol; spec, "Kontrola przecieku" 4).

`load_labels` reads, inside the held-out label seal (`xgpoc.detector.seal.development_labels_only`), the index columns
`goal` (the outcome) and `label` (the SoccerNet shot class; on_target = "Shots on target", needed by the readability
veto) of the population's shots, all in the plan's `training` role. Nothing else of the index is read here, and no
other module of `xgpoc.temporal` reads a label. The labels are an object of their own; fitting code receives them only
as explicit arguments.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.detector import seal
from xgpoc.temporal import inputs as ti

ON_TARGET = "Shots on target"
COLUMNS = ["shot_id", "game", "goal", "label"]


@dataclass(frozen=True)
class Labels:
    """goal and on_target (0/1) by shot id, in a population's shot order."""

    shot_id: np.ndarray
    goal: np.ndarray
    on_target: np.ndarray

    def aligned(self, population: ti.Population) -> "Labels":
        """The labels of `population`'s shots in its order (a subset of the loaded shots)."""
        position = pd.Index(self.shot_id).get_indexer(population.table.index)
        if (position < 0).any():
            raise ValueError("a population shot has no loaded label")
        return Labels(self.shot_id[position], self.goal[position], self.on_target[position])


def load_labels(population: ti.Population, index: Path, plan_csv: Path) -> Labels:
    """goal and on_target of the population's shots, read inside `seal.development_labels_only` (module docstring)."""
    with seal.development_labels_only(plan_csv, index):
        table = pd.read_parquet(index, columns=COLUMNS).set_index("shot_id")
    ids = population.table.index
    if not ids.isin(table.index).all():
        raise ValueError("a population shot has no label")
    rows = table.loc[ids]
    if rows.goal.isna().any() or rows.label.isna().any():
        raise ValueError("a population shot has a missing label")
    if (rows.game.to_numpy() != population.games).any():
        raise ValueError("a label row's game differs from its shot's")
    return Labels(ids.to_numpy(), rows.goal.astype(int).to_numpy(), rows.label.eq(ON_TARGET).astype(int).to_numpy())
