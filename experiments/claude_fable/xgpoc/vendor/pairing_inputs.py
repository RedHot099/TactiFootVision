"""Reading the inputs of the pairing: Labels-v2 files and the StatsBomb data version, with their hashes.

The pairing itself (`xgpoc.vendor.pairing`) reads no file; this module is its I/O shell. The StatsBomb tables are
used only after their files match the hashes in the data version's completion record (`complete.json`), so a
changed table is refused, not paired.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from xgpoc.vendor.statsbomb_fetch import file_sha256

LABELS_FILE = "Labels-v2.json"
STATSBOMB_TABLES = ("matches", "shots", "context")
# The columns the pairing uses; the raw event JSON and freeze frames stay on disk.
STATSBOMB_COLUMNS = {
    "matches": None,
    "shots": ["match_id", "event_id", "index", "period", "timestamp_ms", "team", "player", "play_pattern", "x", "y", "under_pressure",
              "statsbomb_xg", "outcome", "body_part", "shot_type", "technique", "first_time", "n_freeze_frame"],
    "context": ["match_id", "event_id", "type", "period", "timestamp_ms", "team"],
}


def read_labels(root: Path) -> tuple[list[dict], str]:
    """(Labels-v2 records sorted by path, digest of all files). The digest hashes (relative path, sha256) pairs."""
    paths = sorted(Path(root).rglob(LABELS_FILE))
    if not paths:
        raise FileNotFoundError(f"No {LABELS_FILE} under {root}")
    records = [json.loads(p.read_text()) for p in paths]
    listing = "\n".join(f"{p.relative_to(root)} {file_sha256(p)}" for p in paths)
    return records, hashlib.sha256(listing.encode()).hexdigest()


def read_statsbomb(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """(pairing tables, provenance) of the StatsBomb data version at `root`.

    Raises ValueError when there is no completion record, a table file is missing from it or differs from its hash.
    The provenance holds the hash of `complete.json`, the open-data commit and the sha256 of each table file."""
    root = Path(root)
    record_path = root / "complete.json"
    if not record_path.exists():
        raise ValueError(f"{root} has no completion record: the StatsBomb data version is not finished")
    record = json.loads(record_path.read_text())
    hashes = {}
    for name in STATSBOMB_TABLES:
        relative = f"tables/{name}.parquet"
        if relative not in record["files"]:
            raise ValueError(f"The completion record of {root} does not list {relative}")
        found = file_sha256(root / relative)
        if found != record["files"][relative]:
            raise ValueError(f"{root / relative} differs from the hash in the completion record")
        hashes[relative] = found
    tables = {name: pd.read_parquet(root / f"tables/{name}.parquet", columns=STATSBOMB_COLUMNS[name]) for name in STATSBOMB_TABLES}
    return tables, {"root": str(root), "complete_json_sha256": file_sha256(record_path),
                    "open_data": record["identity"]["open_data"], "tables": hashes}
