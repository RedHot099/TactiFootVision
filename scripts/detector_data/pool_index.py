"""Pool index: one row per image across every set under ``sets/``, with every tag column (``common.SCHEMA``).

    uv run python -m scripts.detector_data.pool_index

Reads each ``sets/<set>/manifest.json`` ``per_image`` list (any converter that follows the schema is picked up) and
the GSR/Tracking converter's ``pool_index_gsr.parquet`` (mapped by ``GSR_COLUMNS``), and writes ``pool_index.parquet`` next to ``sets/``. A training split is a filter on
this table, e.g. the default view::

    pool[pool.xg_role.isin(["evaluation", "gate"]).eq(False) & pool.gsr_split.isin(["valid", "test"]).eq(False)
         & pool.near_dup_of.isna()]
"""

import json
from pathlib import Path

import pandas as pd

from scripts.detector_data import common

GSR_INDEX = common.ROOT / "pool_index_gsr.parquet"
# pool_index_gsr.parquet (GSR/Tracking converter, research/detector-data 2ad8f07) column -> SCHEMA column.
GSR_COLUMNS = {
    "image": "image",
    "image_target": "source_image",
    "width": "width",
    "height": "height",
    "game_id": "game",
    "clip": "clip",
    "frame": "frame",
    "image_source": "image_type",
    "xg_role": "xg_role",
    "gsr_split": "gsr_split",
    "near_dup_of": "near_dup_of",
    "annotated_classes": "annotated_classes",
    **{f"n_{c}": f"n_{c}" for c in common.CLASS_NAMES},
}
GSR_LICENCE = "SoccerNet NDA (research)"


def gsr_rows(path: Path) -> pd.DataFrame:
    """The GSR/Tracking index mapped to :data:`common.SCHEMA` (``set``/``source`` = soccernet_gsr, ``split`` =
    ``gsr_split``, ``image_type`` = gsr / tracking image origin). Its extra columns (label path, stride_group,
    n_other_dropped) stay in the GSR file."""
    gsr = pd.read_parquet(path)
    missing = sorted(set(GSR_COLUMNS) - set(gsr.columns))
    if missing:
        raise ValueError(f"{path}: missing GSR columns {missing}")
    if gsr["near_dup_of"].notna().any():
        raise ValueError(
            f"{path}: near_dup_of set on GSR rows; map near_dup_set/corr before using it"
        )
    rows = gsr[list(GSR_COLUMNS)].rename(columns=GSR_COLUMNS)
    rows = rows.assign(
        set="soccernet_gsr",
        source="soccernet_gsr",
        split=rows["gsr_split"],
        near_dup_set=None,
        near_dup_corr=float("nan"),
        near_dup_hits=0,
        licence=GSR_LICENCE,
        synthetic=False,
    )
    return rows[list(common.SCHEMA)]


def build(sets_dir: Path, gsr_index: Path | None = None) -> pd.DataFrame:
    """Concatenate the ``per_image`` rows of every manifest, plus the GSR index when given.

    Refuses manifest rows that miss or add schema columns; the GSR set is taken from ``gsr_index`` (its manifest has
    no ``per_image`` rows and is skipped).
    """
    frames = []
    for manifest_path in sorted(sets_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if "per_image" not in manifest:  # a set not (yet) written in this schema
            print(f"skipping {manifest_path.parent.name}: no per_image rows")
            continue
        rows = manifest["per_image"]
        for row in rows:
            if set(row) != set(common.SCHEMA):
                missing = sorted(set(common.SCHEMA) - set(row))
                extra = sorted(set(row) - set(common.SCHEMA))
                raise ValueError(
                    f"{manifest_path}: row off schema (missing {missing}, extra {extra})"
                )
        frames.append(pd.DataFrame(rows, columns=list(common.SCHEMA)))
    if gsr_index is not None:
        frames.append(gsr_rows(gsr_index))
    pool = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(common.SCHEMA))
    )
    for column in (
        "game",
        "clip",
        "image_type",
        "xg_role",
        "gsr_split",
        "near_dup_of",
        "near_dup_set",
    ):
        pool[column] = pool[column].astype("string")
    pool["frame"] = pd.to_numeric(pool["frame"], errors="coerce").astype("Int64")
    pool["near_dup_corr"] = pool["near_dup_corr"].astype("float64")
    if pool["image"].duplicated().any():
        raise ValueError("an image path appears twice in the pool")
    return pool


def summary(pool: pd.DataFrame) -> pd.DataFrame:
    """Images and boxes per class, by set and tag group."""
    tags = pool.assign(
        xg_role=pool.xg_role.fillna("null"),
        gsr_split=pool.gsr_split.fillna("null"),
        near_dup=pool.near_dup_of.notna(),
    )
    boxes = [f"n_{c}" for c in common.CLASS_NAMES]
    return tags.groupby(["set", "xg_role", "gsr_split", "near_dup"])[boxes].agg(
        ["count", "sum"]
    )


def main() -> None:
    pool = build(common.ROOT / "sets", GSR_INDEX if GSR_INDEX.is_file() else None)
    out = common.ROOT / "pool_index.parquet"
    pool.to_parquet(out, index=False)
    boxes = [f"n_{c}" for c in common.CLASS_NAMES]
    print(f"{len(pool)} images -> {out}")
    print(
        pool.groupby("set")[boxes]
        .sum()
        .assign(images=pool.groupby("set").size())
        .to_string()
    )


if __name__ == "__main__":
    main()
