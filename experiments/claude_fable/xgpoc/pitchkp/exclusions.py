"""Split hygiene of the pitch keypoint dataset (pitch-keypoints 01, council C3 Q2).

Applied after the label rule, in this order, to the training (`train`) and development (`val`) images only:
1. **xG held-out games**: images of the 172 held-out matches of the xg-vendor split plan (roles `evaluation` and
   `gate`) leave train and val. Game identity is resolvable for Calibration-2023 (`match_info.json`: league, season,
   date, teams); GSR only has a `game_id` index (`unresolved`).
2. **GSR evaluation games**: train/val images of a game that also appears in GSR valid or test. GSR games have no
   names, so this is checked only within GSR (`game_id`); Calibration-2023 vs GSR is unresolved.
3. **Duplicate frames**: candidates are pairs in different splits whose 64-bit difference hashes (`dhash`) differ in
   at most `DUPLICATE_BITS` bits; a candidate is a duplicate when the two pictures also agree pixel by pixel (Pearson
   correlation >= `SAME_PICTURE_CORR` of 64 x 36 grey thumbnails), because the 64-bit hash alone pairs different
   frames dominated by grass stripes. Of a duplicate pair, the copy in the lower-priority split (test_gsr / test_calib >
   val > train) leaves. Frames of one GSR clip or one Calibration game share a split, so their near-identical
   neighbours are not duplicates across splits.
"""

import re
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

HELD_OUT_ROLES = ("evaluation", "gate")
DUPLICATE_BITS = 4
SAME_PICTURE_CORR = 0.95
PRIORITY = {"test_gsr": 0, "test_calib": 0, "val": 1, "train": 2}
_CHUNKS = DUPLICATE_BITS + 1          # pigeonhole: within 4 bits, at least one of 5 chunks is identical

PLAN_GAME = re.compile(r"^(?P<league>[^/]+)/(?P<season>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} "
                       r"(?P<home>.+?) \d+ - \d+ (?P<away>.+)$")
CALIB_GAME = re.compile(r"^(?P<league>[^/]+)/(?P<season>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2}) (?P<home>.+?) - (?P<away>.+)$")


def game_key(game: str, pattern: re.Pattern) -> tuple | None:
    """(league, season, date, home, away) of a game name, or None."""
    m = pattern.match(game)
    return None if m is None else (m["league"], m["season"], m["date"], m["home"].strip(), m["away"].strip())


def plan_roles(plan: pd.DataFrame) -> dict[tuple, str]:
    """Split-plan game key -> role."""
    return {game_key(g, PLAN_GAME): r for g, r in zip(plan["game"], plan["role"], strict=True)}


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash: 9 x 8 grey thumbnail, one bit per horizontal neighbour comparison."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA).astype(int)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def thumbnail(path: str) -> np.ndarray:
    """64 x 36 grey thumbnail (float) of an image file."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(float)


def picture_correlation(path_a: str, path_b: str) -> float:
    """Pearson correlation of two images' grey thumbnails (1 for the same picture at any size or JPEG quality)."""
    a, b = thumbnail(path_a).ravel(), thumbnail(path_b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def near_duplicate_pairs(hashes: np.ndarray, max_bits: int = DUPLICATE_BITS) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose hashes differ in at most `max_bits` bits (exhaustive through chunk buckets)."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    chunks = max_bits + 1
    width = -(-64 // chunks)
    pairs: set[tuple[int, int]] = set()
    for c in range(chunks):
        values = (hashes >> np.uint64(c * width)) & np.uint64((1 << width) - 1)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, ends, strict=True):
            if e - s < 2:
                continue
            for i, j in combinations(sorted(order[s:e].tolist()), 2):
                if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= max_bits:
                    pairs.add((i, j))
    return sorted(pairs)


def apply(kept: pd.DataFrame, plan: pd.DataFrame, correlation=picture_correlation) -> tuple[pd.DataFrame, dict]:
    """`kept` (images that pass the label rule, with `key`, `image`, `export_split`, `source`, `game`, `dhash`) with an
    `exclusion` column ("" = exported) and the evidence of each step. `correlation(image_a, image_b)` confirms a
    duplicate candidate."""
    kept = kept.copy()
    kept["exclusion"] = ""
    trainval = kept["export_split"].isin(["train", "val"])
    roles = plan_roles(plan)
    calib = kept["source"] == "calib"
    kept["plan_role"] = [roles.get(game_key(g, CALIB_GAME), "not in plan") if c else "unresolved"
                         for g, c in zip(kept["game"], calib, strict=True)]
    held = trainval & kept["plan_role"].isin(HELD_OUT_ROLES)
    kept.loc[held, "exclusion"] = "xg held-out game"
    gsr = kept["source"] == "gsr"
    eval_games = set(kept.loc[gsr & kept["export_split"].eq("test_gsr"), "game"]) | \
        set(kept.loc[gsr & kept["export_split"].eq("val"), "game"])
    gsr_test_games = set(kept.loc[gsr & kept["export_split"].eq("test_gsr"), "game"])
    gsr_val_games = set(kept.loc[gsr & kept["export_split"].eq("val"), "game"])
    # GSR train frames of a GSR valid/test game, and GSR valid frames of a GSR test game
    shared = (gsr & kept["export_split"].eq("train") & kept["game"].isin(eval_games)) | \
        (gsr & kept["export_split"].eq("val") & kept["game"].isin(gsr_test_games))
    kept.loc[shared & kept["exclusion"].eq(""), "exclusion"] = "game in GSR evaluation split"
    pairs = near_duplicate_pairs(kept["dhash"].to_numpy())
    split = kept["export_split"].to_numpy()
    images = kept["image"].to_numpy()
    priority = np.array([PRIORITY[s] for s in split])
    candidates, duplicates = [], []
    for i, j in pairs:
        if split[i] == split[j]:
            continue
        r = correlation(images[i], images[j])
        candidates.append((i, j, r))
        if r < SAME_PICTURE_CORR:
            continue
        duplicates.append((i, j, r))
        lower = i if priority[i] > priority[j] else j if priority[j] > priority[i] else None
        if lower is not None and kept.iloc[lower]["exclusion"] == "":
            kept.iloc[lower, kept.columns.get_loc("exclusion")] = "duplicate of a higher-priority split"
    keys = kept["key"].to_numpy()
    evidence = {
        "duplicate_bits": DUPLICATE_BITS, "same_picture_corr": SAME_PICTURE_CORR,
        "hash_pairs": len(pairs),
        "cross_split_candidates": [[keys[i], keys[j], round(r, 4)] for i, j, r in candidates],
        "cross_split_duplicates": [[keys[i], keys[j], round(r, 4)] for i, j, r in duplicates],
        "cross_source_hash_pairs": int(sum(kept.iloc[i]["source"] != kept.iloc[j]["source"] for i, j in pairs)),
        "gsr_games": {"train": sorted(set(kept.loc[gsr & kept["export_split"].eq("train"), "game"])),
                      "val": sorted(gsr_val_games), "test_gsr": sorted(gsr_test_games)},
        "gsr_vs_calibration": "unresolved: GSR game_id has no match name; GSR is the 2019 Swiss Super League "
                              "(SoccerNet-Tracking), Calibration-2023 is SoccerNet-v2 (2014-2017, six other "
                              "competitions), so no game can be shared",
    }
    return kept, evidence
