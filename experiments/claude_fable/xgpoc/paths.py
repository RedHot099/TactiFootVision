"""Single place for dataset locations used by the scripts (override with env vars).

The index version matters: v1.1 carries the goal-attribution defect documented in
research/review_2026-09-02.md (goal credited to the earliest shot of a rebound chain);
v1.2 re-labels with the nearest-shot rule. Results computed on different versions are
not comparable.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("XGPOC_DATA_ROOT", "/home/kuba/projects/football_data/xg_shots_full"))
SOCCERNET = DATA_ROOT / "soccernet_720p"
LABELS_ROOT = DATA_ROOT / "soccernet_labels"
DATASET_CARD = DATA_ROOT / "dataset_card.md"
INDEX_V1_1 = SOCCERNET / "index_v1_1.parquet"
INDEX = Path(os.environ.get("XGPOC_INDEX", str(SOCCERNET / "index_v1_2.parquet")))

# StatsBomb open-data version (scripts/vendor_fetch_statsbomb.py): raw event JSON, derived tables, completion record.
STATSBOMB = Path(os.environ.get("XGPOC_STATSBOMB", "/home/kuba/projects/football_data/statsbomb_open_v1"))
