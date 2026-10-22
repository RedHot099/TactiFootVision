"""xg-stack's E3 table, population and held-out table under a shot-location rule (`xgpoc.detector.location`).

`xgpoc.stack.inputs` and `xgpoc.stack_heldout.testset` are frozen (their hashes are pinned by the xg-stack protocol,
the xg-stack test's hash list and the re-run records), and they replay E3 with `features.situation` only. These are
their copies with the rule as an argument; with `location.E3` they return exactly what the frozen functions return
(tests on planted frames, `scripts/location_check.py` on the full caches):

    e3_table          `inputs.e3_table`: the replayed raw features must equal the cache's `features.parquet` exactly;
                      the sanitation runs on the rule's situation; at the rule's `e3_flags_at` offsets the ball-source
                      columns (ball one-hot, shot_from_ball) are E3's: the E3-filled frame, `features.situation`
    build_population  `inputs.build_population` with `e3_table` under the rule (inputs gain the rule's name)
    heldout_table     `testset.heldout_table` with `e3_table` under the rule

Label-free: no outcome is read here (`xgpoc.stack.inputs.load_outcomes` reads them, inside the seal).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.detector import location as loc
from xgpoc.detector import records
from xgpoc.detector import sanitation as sn
from xgpoc.detector.features import COLUMNS, situation
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si

E3_FLAG_COLUMNS = ("ball_source", "shot_from_ball")     # sanitised columns behind ball_<level>@o and shot_from_ball@o


def levels(source: x3.SourceConfig, offset: int) -> tuple[str, ...]:
    """The ball-source one-hot levels of `source`'s order at `offset` (`inputs.ball_levels` for any order)."""
    others = [o for o in source.order(offset) if o != offset]
    return ("none", "own", *[f"m{-o}" for o in others])


def check_levels(rule: loc.LocationRule) -> None:
    """Raise unless the rule's ball-source levels are E3's at every offset (`inputs.E3_COLUMNS` names them)."""
    for offset in si.OFFSETS:
        source = si.E3_CONFIG if offset in rule.e3_flags_at else rule.source
        if levels(source, offset) != si.ball_levels(offset):
            raise ValueError(f"{rule.name}: the ball-source levels at {offset} ms are not E3's columns")


def e3_table(frames: pd.DataFrame, cached_features: pd.DataFrame, rule: loc.LocationRule = loc.E3
             ) -> tuple[pd.DataFrame, dict[str, int]]:
    """(`inputs.E3_COLUMNS`, one row per shot of `frames`, index shot_id, sorted; the angle-wrap counts per offset)
    under `rule` (module docstring). Raises unless the replayed raw features equal `cached_features` exactly."""
    check_levels(rule)
    raw_rows, rows = [], {}
    for shot_id, offset, cached, filled, kp_confidence in loc.replay(frames, rule.source):
        raw = loc.rule_features(filled, rule)
        if filled is None:
            clean = sn.missing_row()
        else:
            clean = sn.sanitised_features(rule.situation(filled), filled.homography is not None)
            if offset in rule.e3_flags_at:
                reference = loc.filled_frame(offset, cached, si.E3_CONFIG)
                e3_clean = sn.sanitised_features(situation(reference), reference.homography is not None)
                clean = clean | {k: e3_clean[k] for k in E3_FLAG_COLUMNS}
        raw_rows.append({"shot_id": shot_id, "offset_ms": offset} | raw)
        rows.setdefault(shot_id, {}).update(si.offset_row(offset, raw, clean, kp_confidence))
    replayed = pd.DataFrame(raw_rows).set_index(["shot_id", "offset_ms"]).sort_index()
    cached = cached_features.set_index(["shot_id", "offset_ms"]).sort_index()
    cached = cached[cached.index.get_level_values("offset_ms").isin(si.OFFSETS)]
    if not cached.index.equals(replayed.index):
        raise ValueError("the replayed E3 rows are not the cached rows")
    if n := int(ex.feature_differences(cached, replayed, COLUMNS, 0.0).any(axis=1).sum()):
        raise ValueError(f"the replayed E3 features differ from the cache in {n} rows")
    table = pd.DataFrame.from_dict(rows, orient="index")[si.E3_COLUMNS]
    table.index.name = "shot_id"
    si.check_missing_groups(table)
    changed = {str(o): int((table[si.col("angle", o)] > np.pi).sum()) for o in si.OFFSETS}
    for offset in si.OFFSETS:
        table[si.col("angle", offset)] = si.wrap_angle(table[si.col("angle", offset)].to_numpy())
    return table.sort_index(), changed


def build_population(rule: loc.LocationRule = loc.E3, index: Path = INDEX, oof: Path = si.OOF, e3: Path = si.E3,
                     labels_root: Path = LABELS_ROOT, plan: pd.DataFrame | None = None) -> si.Population:
    """`inputs.build_population` with the E3 table of cache `e3` under `rule`; `inputs` also names the rule."""
    plan = a1.verify_plan(index) if plan is None else plan
    inputs = {"index": {"path": str(index), "sha256": sha256_file(index)},
              "folds": {"path": str(si.FOLDS.relative_to(si.EXPERIMENT)), "sha256": sha256_file(si.FOLDS)},
              "plan": {"path": str((a1.PLAN_DIR / "plan.csv").relative_to(si.EXPERIMENT)), "sha256": a1.PLAN_SHA256},
              "oof": {"path": str(oof.relative_to(si.EXPERIMENT)), "sha256": records.verified_file(oof)},
              "location_rule": rule.name}
    for name in ("frames.parquet", "features.parquet"):
        inputs[f"e3_{name.split('.')[0]}"] = {"path": str((e3 / name).relative_to(si.EXPERIMENT)),
                                              "sha256": records.verified_file(e3 / name)}
    shots = si.training_shots(index, oof, plan)
    ids = sorted(shots.index)
    frames = pd.read_parquet(e3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(e3 / "features.parquet", filters=[("shot_id", "in", ids)])
    detector, wrapped = e3_table(frames, features, rule)
    if not detector.index.equals(shots.index):
        raise ValueError("E3 does not cover the population exactly")
    games = sorted(shots.game.unique())
    annotations = {g: si.read_annotations(g, labels_root) for g in games}
    inputs["labels_v2"] = {"root": str(labels_root), "files": len(games),
                           "digest": canonical_digest({g: sha256_file(si.labels_file(g, labels_root)) for g in games})}
    table = pd.concat([shots, detector, si.flags_table(shots, annotations)], axis=1)
    counts = {"shots": len(table), "matches": int(table.game.nunique()), "folds": int(table.fold.nunique()),
              "shots_per_fold": {str(k): int(v) for k, v in table.fold.value_counts().sort_index().items()},
              "matches_per_fold": {str(k): int(v) for k, v in table.groupby("fold").game.nunique().items()},
              "flags": {f: int(table[f].sum()) for f in si.S_FLAGS},
              "e3_t0": {"shot_missing": int(table[si.col("shot_missing", 0)].sum()),
                        "frame_missing": int(table[si.col("frame_missing", 0)].sum()),
                        "kp_conf_missing": int(table[si.col("kp_conf_missing", 0)].sum())},
              "angles_wrapped": wrapped}
    if {k: counts[k] for k in si.EXPECTED} != si.EXPECTED:
        raise ValueError(f"population {counts} is not C5's {si.EXPECTED}")
    return si.Population(table, inputs, counts)


def heldout_table(shots: pd.DataFrame, z: pd.Series, e3: Path, annotations: dict[str, list[dict]],
                  rule: loc.LocationRule = loc.E3) -> tuple[pd.DataFrame, dict]:
    """`testset.heldout_table` with the E3 table of cache `e3` under `rule`."""
    ids = sorted(shots.index)
    frames = pd.read_parquet(e3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(e3 / "features.parquet", filters=[("shot_id", "in", ids)])
    detector, wrapped = e3_table(frames, features, rule)
    if not detector.index.equals(shots.index):
        raise ValueError("E3 does not cover the test population exactly")
    if not z.index.equals(shots.index):
        raise ValueError("the test z is not in the population's shot order")
    table = pd.concat([shots, z.rename(sf.Z), detector, si.flags_table(shots, annotations)], axis=1)
    return table, wrapped
