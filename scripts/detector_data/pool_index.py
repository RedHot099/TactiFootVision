"""Pool index: one row per image across every set under ``sets/``, with every tag column (``common.SCHEMA``).

    uv run python -m scripts.detector_data.pool_index

Reads each ``sets/<set>/manifest.json`` ``per_image`` list (any converter that follows the schema is picked up, the
GSR/Tracking sets included) and writes ``pool_index.parquet`` next to ``sets/``. A training split is a filter on
this table, e.g. the default view::

    pool[pool.xg_role.isin(["evaluation", "gate"]).eq(False) & pool.gsr_split.isin(["valid", "test"]).eq(False)
         & pool.near_dup_of.isna()]
"""

import json
from pathlib import Path

import pandas as pd

from scripts.detector_data import common


def build(sets_dir: Path) -> pd.DataFrame:
    """Concatenate the ``per_image`` rows of every manifest; refuses rows that miss or add schema columns."""
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
    pool = build(common.ROOT / "sets")
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
