"""No StatsBomb data on the model and feature input paths (xg-detector spec, story 2 and Testing Decisions).

Inside `no_statsbomb_reads()` any read of a path that names StatsBomb data or a table derived from it raises
`StatsBombRead`: Python-level opens through an audit hook (`open`, `io.open`, `os.open`, `numpy.load`, `torch.load`,
...) and the readers that open files in C++ (pandas / pyarrow parquet and csv, OpenCV images). The run script reads
the paired shot ids (which frames the audit needs) before it enters the block, and everything that loads models,
frames or computes features runs inside it. Matching is on the path text, case-insensitive (`FORBIDDEN`); Python
source and bytecode files are code, not data, and pass.

The audit (`xgpoc.detector.audit`, `scripts/xgd_audit.py`, ticket 02) is the one allowed reader of StatsBomb data and
the pairing: it never runs inside the block, no feature or model module imports it, and its outputs are refused here.
"""

import contextlib
import os
import re
import sys
from collections.abc import Iterator

FORBIDDEN = re.compile(
    r"statsbomb"             # open data, derived tables, comparator inputs (results/xgv_statsbomb_data, ...)
    r"|xgv_pairing"          # the SoccerNet <-> StatsBomb pairing and its check (pairs.parquet holds StatsBomb values)
    r"|xgv_release"          # the release package (pair table)
    r"|xgv_comparator"       # the positional comparator, a function of StatsBomb data only
    r"|xgv_eval"             # evaluation tables with the vendor's xG
    r"|xgv_confirmatory|xgv_dryrun|xgv_ladder|xgv_leakage"
    r"|xgv_timing_source"    # per-shot StatsBomb timestamps and residuals
    r"|xgd_audit"            # the detector audit (ticket 02): StatsBomb positions per shot
    r"|open[-_]data",
    re.IGNORECASE)


# Source code is not data: the package's StatsBomb reader (tactifoot_vision/data/statsbomb.py) may be imported and hashed.
CODE_SUFFIXES = (".py", ".pyc")


class StatsBombRead(PermissionError):
    """A model or feature input path tried to read StatsBomb data or the pairing."""


_active = 0
_hook_installed = False


def check_path(path) -> None:
    """Raise `StatsBombRead` when the guard is active and `path` names StatsBomb data."""
    if _active and isinstance(path, str | bytes | os.PathLike):
        text = os.fsdecode(path)
        if FORBIDDEN.search(text) and not text.endswith(CODE_SUFFIXES):
            raise StatsBombRead(f"{text}: StatsBomb data or the pairing is not an input of the detector models")


def _audit(event: str, args: tuple) -> None:
    if _active and event in ("open", "os.listdir", "os.scandir", "glob.glob") and args:
        check_path(args[0])


def _wrap(module, name: str):
    original = getattr(module, name)

    def guarded(path, *args, **kwargs):
        check_path(path)
        return original(path, *args, **kwargs)
    return original, guarded


@contextlib.contextmanager
def no_statsbomb_reads() -> Iterator[None]:
    """Refuse reads of StatsBomb or pairing files for the duration of the block (re-entrant)."""
    global _active, _hook_installed
    import cv2
    import pandas as pd
    import pyarrow.csv
    import pyarrow.parquet

    if not _hook_installed:
        sys.addaudithook(_audit)   # audit hooks cannot be removed; it does nothing while no block is active
        _hook_installed = True
    targets = [(pd, "read_parquet"), (pd, "read_csv"), (pd, "read_json"), (pd, "read_feather"),
               (pyarrow.parquet, "read_table"), (pyarrow.parquet, "ParquetFile"), (pyarrow.parquet, "ParquetDataset"),
               (pyarrow.csv, "read_csv"), (cv2, "imread")]
    originals = []
    for module, name in targets:
        original, guarded = _wrap(module, name)
        setattr(module, name, guarded)
        originals.append((module, name, original))
    _active += 1
    try:
        yield
    finally:
        _active -= 1
        for module, name, original in reversed(originals):
            setattr(module, name, original)
