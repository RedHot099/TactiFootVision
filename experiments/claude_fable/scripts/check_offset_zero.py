"""Sanity check: offset-0 frames of an offset-frames version equal the frames the canonical version produced as `t0`.

Compares, for every shot present in both, the decoded frame (index and PTS from the tables) and the stored pixels read
from the two PNG files (the canonical PNG is resized with INTER_AREA when the offset version stores the 448x252 model
input), then the model input (`load_image` at 448x252 with the overlay mask). Recorded hashes are not used.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgpoc.data.datasets import load_image

CHECKS = ("same_frame_index", "same_pts", "same_stored_pixels", "same_model_input")


def stored_pixels_equal(canonical_png: Path, offset_png: Path) -> bool:
    old, new = cv2.imread(str(canonical_png)), cv2.imread(str(offset_png))
    if old is None or new is None:
        return False
    if old.shape != new.shape:
        old = cv2.resize(old, (new.shape[1], new.shape[0]), interpolation=cv2.INTER_AREA)
    return bool(np.array_equal(old, new))


def compare(offsets: Path, canonical: Path) -> dict:
    new = pd.read_csv(offsets / "frames.csv", low_memory=False)
    new = new[new.offset_ms.eq(0)].set_index("shot_id")
    old = pd.read_csv(canonical / "frames.csv")
    old = old[old.variant.eq("t0") & old.availability.eq("available")]
    if old.shot_id.duplicated().any():
        raise ValueError("Canonical frames table has several t0 rows for one shot")
    old = old.set_index("shot_id")
    both = old.index.intersection(new.index)
    rows = []
    for shot_id in both:
        a, b = old.loc[shot_id], new.loc[shot_id]
        row = {"shot_id": shot_id, "offset0_frame_present": isinstance(b.frame_path, str)}
        if row["offset0_frame_present"]:
            old_png, new_png = canonical / a.frame_path, offsets / b.frame_path
            row |= {"same_frame_index": int(a.source_frame_index) == int(b.source_frame_index),
                    "same_pts": int(a.source_pts) == int(b.source_pts),
                    "same_stored_pixels": stored_pixels_equal(old_png, new_png),
                    "same_model_input": bool(np.array_equal(load_image(str(old_png), (448, 252), "overlay"),
                                                            load_image(str(new_png), (448, 252), "overlay")))}
        rows.append(row)
    table = pd.DataFrame(rows, columns=["shot_id", "offset0_frame_present", *CHECKS])
    return {"canonical_shots": len(old), "compared_shots": len(both),
            "offset0_without_frame": int((~table.offset0_frame_present.astype(bool)).sum()),
            **{c: int(table[c].eq(True).sum()) for c in CHECKS},
            "all_identical": bool(len(both) and table.offset0_frame_present.all() and all(table[c].all() for c in CHECKS))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, default=Path(__file__).resolve().parents[1] / "cache/v15/canonical_data_v1")
    args = parser.parse_args()
    result = compare(args.data, args.canonical)
    print(json.dumps(result, indent=2))
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
