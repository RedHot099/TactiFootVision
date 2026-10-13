"""The shot-level comparison table of xg-compare (ticket 01) and its separate label file.

`build` is a pure function of frozen inputs. It returns one row per shot of the combined population, which contains
the primary population (flag `in_primary`). Both populations are those of the xg-vendor paper
(`xgpoc.vendor.ladder.analysis_population`): penalties, rows without a scored frame (the 147 StatsBomb-only rows
beyond the half's end), label exclusions and the pre-registered frame exclusion are out. Every join is checked to be
one-to-one and complete.

Columns
    keys        key, soccernet_shot_id, statsbomb_event_id, statsbomb_match_id, match, league_season, half, split
                (official), population, in_primary, scan_half (`scan_halves`)
    methods     p_<m> (raw probability) and z_<m> (raw logit) for m in `METHODS`: vendor, positions, events,
                dist_angle, frame, frame_m400 for every shot, a1 for the primary shots it covers (NaN elsewhere).
                Probabilities are clipped to [EPS, 1 - EPS] and z = logit(p), as the paper's rungs
                (`ladder.raw_rungs`); the frame's z is `ladder.frame_feature` of the stored logit. The main
                (recalibrated) scale needs the labels, so the analysis (ticket 02) fits it; the table does not store it.
    descriptors sb_*: StatsBomb pre-contact features (`xgpoc.vendor.comparator_features`, by statsbomb_event_id) and the
                shot's minute; seq_*: the previous shot of the possession and the rebound flags (`xgpoc.compare.sequence`);
                e3_*: offset-0 detector descriptors of `cache/xgd_full3/E3` (primary shots; NaN for StatsBomb-only
                shots); timing_residual_ms (primary shots)
    categories  cat_<name> for every pre-shot category of `xgpoc.compare.categories` (level name or None)

Labels (`LABEL_COLUMNS`: goal_sn, on_target_sn, sb_outcome) never enter the table. They are written to a separate
label file keyed by `key`, so the label-free analyses run on the table alone.
"""

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import ladder as ld

SCHEMA_VERSION = "xgc-shot-table-1"
TABLE_FILE, LABEL_FILE, COVERAGE_FILE, SIDECAR = "shots.parquet", "labels.parquet", "coverage.csv", "shots.json"
LABEL_COLUMNS = ["goal_sn", "on_target_sn", "sb_outcome"]
PROBABILITY_METHODS = {"vendor": "pred_vendor_xg", "positions": "pred_freeze_frame", "events": "pred_event", "dist_angle": "pred_dist_angle"}
LOGIT_METHODS = {"frame": "frame_logit_0", "frame_m400": "frame_logit_m400"}
METHODS = (*PROBABILITY_METHODS, *LOGIT_METHODS, "a1")
METHOD_COLUMNS = [f"{kind}_{m}" for m in METHODS for kind in ("p", "z")]
KEY_COLUMNS = ["key", "soccernet_shot_id", "statsbomb_event_id", "statsbomb_match_id", "match", "league_season", "half", "split",
               "population", "in_primary", "scan_half"]
SB_FEATURES = ["x", "y", "dist", "angle", "body", "technique", "stype", "pattern", "kp_has", "kp_cross", "kp_through", "kp_cutback",
               "kp_height", "first_time", "under_pressure", "one_on_one", "open_goal", "follows_dribble", "aerial_won", "ff_def_cone",
               "gk_depth", "gk_off_line", "ff_near_opp"]
SEQUENCE_DESCRIPTORS = ["prev_event_id", "prev_dt_ms", "prev_same_team", "prev_outcome", "rebound_5s", "rebound_3s"]
E3_DESCRIPTORS = ["homography_ok", "n_players", "n_keypoints", "gk_detected", "ball_ok"]
DESCRIPTOR_COLUMNS = ([f"sb_{c}" for c in SB_FEATURES] + ["sb_minute"] + [f"seq_{c}" for c in SEQUENCE_DESCRIPTORS]
                      + [f"e3_{c}" for c in E3_DESCRIPTORS] + ["timing_residual_ms"])
CATEGORY_COLUMNS = [c.column for c in cat.PRE_SHOT]
COLUMNS = [*KEY_COLUMNS, *METHOD_COLUMNS, *DESCRIPTOR_COLUMNS, *CATEGORY_COLUMNS]


@dataclass
class ShotTable:
    table: pd.DataFrame
    labels: pd.DataFrame
    cuts: cat.Cuts


def league_season(matches: pd.Series) -> pd.Series:
    """"england_epl/2015-2016/2015-08-08 - ..." -> "england_epl/2015-2016"."""
    return matches.str.split("/").str[:2].str.join("/")


def scan_halves(matches: pd.Series, strata: pd.Series, seed: int = pr.SPLIT_SEED) -> dict[str, str]:
    """Match -> half of the open scan: per stratum in sorted order, the sorted matches permuted by one generator; the
    concatenated list assigned alternately to `HALVES`, so each stratum splits within one match."""
    pairs = pd.DataFrame({"match": matches.to_numpy(), "stratum": strata.to_numpy()}).drop_duplicates()
    if pairs.match.duplicated().any():
        raise ValueError("a match lies in two strata")
    rng = np.random.default_rng(seed)
    order = []
    for _, group in pairs.sort_values(["stratum", "match"]).groupby("stratum", sort=True):
        names = group.match.to_numpy()
        order += list(names[rng.permutation(len(names))])
    return {m: pr.HALVES[i % 2] for i, m in enumerate(order)}


def method_columns(rows: pd.DataFrame, a1: pd.Series) -> pd.DataFrame:
    """p_ and z_ of every method (module docstring); `a1` is the A1 probability per row (NaN where missing)."""
    out = {}
    for name, column in PROBABILITY_METHODS.items():
        p = np.clip(rows[column].to_numpy(float), EPS, 1 - EPS)
        out[f"p_{name}"], out[f"z_{name}"] = p, logit(p)
    for name, column in LOGIT_METHODS.items():
        z = ld.frame_feature(rows[column].to_numpy(float))
        out[f"p_{name}"], out[f"z_{name}"] = sigmoid(z), z
    p = np.clip(a1.to_numpy(float), EPS, 1 - EPS)
    out["p_a1"], out["z_a1"] = p, logit(p)
    return pd.DataFrame(out, index=rows.index)[METHOD_COLUMNS]


def _one_to_one(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, what: str, required: np.ndarray) -> pd.DataFrame:
    """Left join that refuses duplicate keys on either side and a missing match where `required` (n,) is True."""
    if right[right_on].duplicated().any():
        raise ValueError(f"{what}: duplicate {right_on}")
    merged = left.merge(right, left_on=left_on, right_on=right_on, how="left", validate="one_to_one", indicator=True)
    found = merged.pop("_merge").eq("both").to_numpy()
    if (required & ~found).any():
        raise ValueError(f"{what}: no match for {int((required & ~found).sum())} of the rows")
    return merged


def build(joined: pd.DataFrame, frame_exclusions: dict, features: pd.DataFrame, e3: pd.DataFrame, a1: pd.DataFrame,
          sequence: pd.DataFrame, seed: int = pr.SPLIT_SEED) -> ShotTable:
    """The table and the label file (module docstring).

    joined      the confirmatory evaluation table with labels joined (`xgpoc.vendor.evaltable`)
    features    the comparator's StatsBomb feature table (event_id + `SB_FEATURES`)
    e3          the detector descriptors (shot_id, offset_ms, `E3_DESCRIPTORS`); only offset 0 enters the table
    a1          the A1 test predictions (shot_id, p_a1)
    sequence    `xgpoc.compare.sequence.sequence_table` of the StatsBomb shots of the table's matches"""
    combined, _ = ld.analysis_population(joined, et.POPULATIONS, frame_exclusions=frame_exclusions)
    primary, _ = ld.analysis_population(joined, ("primary",))
    if not set(primary.key) <= set(combined.key):
        raise ValueError("a primary shot is missing from the combined population")
    rows = combined.drop(columns=et.LABEL_COLUMNS).reset_index(drop=True)
    labels = combined[["key", *LABEL_COLUMNS]].reset_index(drop=True)
    in_primary = rows.key.isin(set(primary.key)).to_numpy()
    every, none = np.ones(len(rows), bool), np.zeros(len(rows), bool)

    sb = features[["event_id", *SB_FEATURES]].rename(columns={c: f"sb_{c}" for c in SB_FEATURES})
    rows = _one_to_one(rows, sb, "statsbomb_event_id", "event_id", "StatsBomb features", every).drop(columns="event_id")
    seq = sequence.rename(columns={"minute": "sb_minute", **{c: f"seq_{c}" for c in SEQUENCE_DESCRIPTORS}})
    rows = _one_to_one(rows, seq, "statsbomb_event_id", "event_id", "StatsBomb sequence", every).drop(columns="event_id")
    t0 = e3[e3.offset_ms.eq(0)][["shot_id", *E3_DESCRIPTORS]].rename(columns={c: f"e3_{c}" for c in E3_DESCRIPTORS})
    rows = _one_to_one(rows, t0, "key", "shot_id", "detector descriptors", required=in_primary).drop(columns="shot_id")
    a1_rows = a1[["shot_id", "p_a1"]].rename(columns={"p_a1": "a1_stored"})
    rows = _one_to_one(rows, a1_rows, "key", "shot_id", "A1 predictions", none).drop(columns="shot_id")   # where available
    a1_stored = rows.pop("a1_stored")
    if a1_stored.notna().to_numpy()[~in_primary].any():
        raise ValueError("A1 predictions belong to primary shots only")

    rows["league_season"] = league_season(rows.match)
    rows["in_primary"] = in_primary
    rows["scan_half"] = rows.match.map(scan_halves(rows.match, rows.league_season, seed))
    rows["half"] = rows.half.astype(int)
    rows = pd.concat([rows, method_columns(rows, a1_stored)], axis=1)
    cuts = cat.label_free_cuts(rows[rows.in_primary])
    rows = pd.concat([rows, cat.derive(rows, cuts)], axis=1)
    table = rows[COLUMNS].reset_index(drop=True)
    check(table, labels)
    return ShotTable(table, labels, cuts)


def check(table: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Raise unless the table meets its contract: the schema's columns, no label column, unique shots, finite
    predictions, and a label file with exactly the table's keys in the same order."""
    if list(table.columns) != COLUMNS:
        raise ValueError("the table's columns are not the schema's")
    leaked = set(table.columns) & {*LABEL_COLUMNS, *et.LABEL_COLUMNS}
    if leaked:
        raise ValueError(f"label columns in the shot table: {sorted(leaked)}")
    for column in ("key", "statsbomb_event_id"):
        if table[column].duplicated().any():
            raise ValueError(f"duplicate {column} in the shot table")
    if table.soccernet_shot_id.dropna().duplicated().any():
        raise ValueError("a SoccerNet shot appears twice")
    finite = table[[c for c in METHOD_COLUMNS if not c.endswith("_a1")]].to_numpy(float)
    if not np.isfinite(finite).all():
        raise ValueError("a method prediction is missing")
    if not set(table.scan_half) <= set(pr.HALVES):
        raise ValueError("a shot without a scan half")
    if list(labels.columns) != ["key", *LABEL_COLUMNS] or not labels.key.equals(table.key):
        raise ValueError("the label file must hold key + labels for exactly the table's rows, in order")


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Label-free coverage of every level of every pre-shot category: per population, the shots and matches inside,
    the defined shots outside, the undefined shots, and per scan half (primary) the shots inside and outside; plus
    whether the level meets the category floor and the scan floor of the protocol."""
    rows = []
    for population, mask in (("primary", table.in_primary.to_numpy()), ("combined", np.ones(len(table), bool))):
        part = table[mask]
        for category in cat.PRE_SHOT:
            values = part[category.column]
            defined = values.notna()
            for level in category.levels:
                inside = values.eq(level)
                row = {"population": population, "category": category.name, "level": level, "role": category.role,
                       "scan": level in category.scan_levels, "n_in": int(inside.sum()), "matches_in": int(part.match[inside].nunique()),
                       "n_out": int((defined & ~inside).sum()), "n_undefined": int((~defined).sum())}
                for half in pr.HALVES:
                    in_half = part.scan_half.eq(half)
                    row[f"n_in_{half}"] = int((inside & in_half).sum())
                    row[f"n_out_{half}"] = int((defined & ~inside & in_half).sum())
                row["meets_category_floor"] = (row["n_in"] >= pr.FLOORS["category_in"] and row["n_out"] >= pr.FLOORS["category_out"]
                                               and row["matches_in"] >= pr.FLOORS["category_matches_in"])
                row["meets_scan_floor"] = row["meets_category_floor"] and all(
                    row[f"n_in_{h}"] >= pr.FLOORS["scan_half_in"] and row[f"n_out_{h}"] >= pr.FLOORS["scan_half_out"] for h in pr.HALVES)
                rows.append(row)
    return pd.DataFrame(rows)


def write(shot_table: ShotTable, directory: Path, sources: dict) -> dict:
    """Write `TABLE_FILE`, `LABEL_FILE`, `COVERAGE_FILE` and the sidecar `SIDECAR` (schema, hashes of the table and the
    label file, counts, cuts, sources) into `directory`; return the sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_payload, label_payload = et.table_bytes(shot_table.table), et.table_bytes(shot_table.labels)
    (directory / TABLE_FILE).write_bytes(table_payload)
    (directory / LABEL_FILE).write_bytes(label_payload)
    coverage(shot_table.table).to_csv(directory / COVERAGE_FILE, index=False)
    table = shot_table.table
    meta = {"schema_version": SCHEMA_VERSION, "table_sha256": hashlib.sha256(table_payload).hexdigest(),
            "labels_sha256": hashlib.sha256(label_payload).hexdigest(), "columns": list(table.columns),
            "label_columns": ["key", *LABEL_COLUMNS], "rows": len(table),
            "primary": {"shots": int(table.in_primary.sum()), "matches": int(table.match[table.in_primary].nunique())},
            "combined": {"shots": len(table), "matches": int(table.match.nunique())},
            "by_population": {k: int(v) for k, v in table.population.value_counts().sort_index().items()},
            "matches_by_half": {h: int(table.match[table.scan_half.eq(h)].nunique()) for h in pr.HALVES},
            "a1_coverage": {"primary_shots": int(table.p_a1.notna().sum()), "of": int(table.in_primary.sum())},
            "cuts": asdict(shot_table.cuts), "sources": sources}
    (directory / SIDECAR).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def _verified(path: Path, sha256: str) -> pd.DataFrame:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{path} does not match the hash in its sidecar")
    return pd.read_parquet(io.BytesIO(payload))


def read(directory: Path) -> tuple[pd.DataFrame, dict]:
    """The shot table (checked against its sidecar's hash and the contract's columns) and the sidecar. No label."""
    meta = json.loads((Path(directory) / SIDECAR).read_text())
    table = _verified(Path(directory) / TABLE_FILE, meta["table_sha256"])
    if list(table.columns) != COLUMNS:
        raise ValueError("the stored table's columns are not the schema's")
    return table, meta


def read_labels(directory: Path, meta: dict) -> pd.DataFrame:
    """The label file, checked against the sidecar's hash."""
    return _verified(Path(directory) / LABEL_FILE, meta["labels_sha256"])
