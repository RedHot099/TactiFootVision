"""No label of a held-out shot before the single test (xg-detector ticket 06).

The 172 held-out matches of the split plan (37 gate and 135 evaluation matches) are scored once, by the frozen test
(`scripts/xgd_test.py`, ticket 07). Until then no label of their shots may be read: not by the test rule's power
computation, not by training, not by the preparation of the baseline's test predictions. Inside
`held_out_labels_sealed(training_games, training_shots)`, pandas' parquet and CSV readers refuse to hand over a label
(`LABEL_COLUMNS`) of any shot outside the training pool:

* `pd.read_parquet` of a file with a `game` column and a requested label column reads only the rows of the training
  matches (a pyarrow row filter on `game`, so held-out rows never leave the file). Without a `game` column the result
  must carry `shot_id` and every labelled row must be a training shot.
* `pd.read_csv` with a label column in its result: every row must belong to a training match (`game`) or be a training
  shot (`shot_id`); a table that names neither is refused.

A read that asks for no label column (`columns=[...]`, `usecols=[...]`) passes unchanged. The seal covers pandas'
readers, the only table readers of the sealed paths (tested); byte reads that only hash a file (`sha256_file`) carry
no label into the program and pass. `development_labels_only(plan_csv, index)` builds the seal from the plan's training
role and the index's shot ids, both read without a label column.
"""

import contextlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = frozenset({"goal", "label", "goal_v1_1", "on_target"})   # index outcomes and the derived on-target target


class HeldOutLabelRead(PermissionError):
    """A sealed path tried to read the label of a shot outside the training pool."""


def _path(source) -> Path:
    if not isinstance(source, str | os.PathLike):
        raise HeldOutLabelRead(f"{type(source).__name__}: only file paths can be read inside the label seal")
    return Path(source)


def _check_rows(frame: pd.DataFrame, source, games: frozenset[str], shots: frozenset[str]) -> pd.DataFrame:
    labels = LABEL_COLUMNS & set(frame.columns)
    if not labels:
        return frame
    if "game" in frame.columns:
        outside = ~frame.game.isin(games)
    elif "shot_id" in frame.columns:
        outside = ~frame.shot_id.isin(shots)
    else:
        raise HeldOutLabelRead(f"{source}: labels {sorted(labels)} without a game or shot_id column to check them")
    if outside.any():
        raise HeldOutLabelRead(f"{source}: labels {sorted(labels)} of {int(outside.sum())} shots outside the training pool")
    return frame


@contextlib.contextmanager
def held_out_labels_sealed(training_games: Iterable[str], training_shots: Iterable[str]) -> Iterator[None]:
    """Refuse every label of a shot outside `training_games` / `training_shots` for the duration of the block."""
    import pyarrow.parquet as pq

    games, shots = frozenset(training_games), frozenset(training_shots)
    read_parquet, read_csv = pd.read_parquet, pd.read_csv

    def sealed_parquet(source, *args, columns=None, filters=None, **kwargs):
        names = pq.read_schema(_path(source)).names
        requested = set(names if columns is None else columns)
        if not LABEL_COLUMNS & requested:
            return read_parquet(source, *args, columns=columns, filters=filters, **kwargs)
        if "game" in names:
            allowed = games
            if filters is not None:                  # only the filter of an enclosing seal, intersected with this one
                if not (len(filters) == 1 and len(filters[0]) == 3 and tuple(filters[0][:2]) == ("game", "in")):
                    raise HeldOutLabelRead(f"{source}: row filters of a labelled read are not supported inside the label seal")
                allowed = games & set(filters[0][2])
            filters = [("game", "in", sorted(allowed))]
        frame = read_parquet(source, *args, columns=columns, filters=filters, **kwargs)
        return _check_rows(frame, source, games, shots)

    def sealed_csv(source, *args, **kwargs):
        frame = read_csv(_path(source), *args, **kwargs)
        if not isinstance(frame, pd.DataFrame):
            raise HeldOutLabelRead(f"{source}: chunked CSV reads cannot be checked inside the label seal")
        return _check_rows(frame, source, games, shots)

    pd.read_parquet, pd.read_csv = sealed_parquet, sealed_csv
    try:
        yield
    finally:
        pd.read_parquet, pd.read_csv = read_parquet, read_csv


def training_pool_ids(plan_csv: Path, index: Path) -> tuple[frozenset[str], frozenset[str]]:
    """(games of the plan's training role, shot ids of the index in those games), read without any label column."""
    plan = pd.read_csv(plan_csv, usecols=["game", "role"])
    games = frozenset(plan.game[plan.role == "training"])
    shots = pd.read_parquet(index, columns=["shot_id", "game"])
    return games, frozenset(shots.shot_id[shots.game.isin(games)])


def development_labels_only(plan_csv: Path, index: Path) -> contextlib.AbstractContextManager:
    """The seal of the plan's training pool."""
    return held_out_labels_sealed(*training_pool_ids(plan_csv, index))
