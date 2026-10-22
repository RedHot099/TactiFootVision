"""Label-free checks of the shot-location rules (`xgpoc.detector.location`, `xgpoc.stack_location.inputs`; frozen rule
`research/feet_location_rule.md`). CPU, no outcome read (inside the held-out label seal and the StatsBomb guard).

    ./run.sh --no-sync python scripts/location_check.py [--out results/feet_location_check]

1. Fidelity of the copies. Under `location.E3`, on every shot of the frozen E3 caches (xgd_full3, the old detector;
   xgd_full4, the retrained dd2c): `location.feature_rows` equals the cached `features.parquet`, `stack_location.
   e3_table` equals `stack.inputs.e3_table`, and `stack_location.build_population` equals `stack.inputs.
   build_population` (table and counts).
2. The feet_t0 cache (`scripts/feet_e3.py`): its record's hashes hold, `location.feature_rows(frames, FEET_T0)`
   reproduces it, and its features equal the development cache the monkey-patch built (cache/xgd_full5_feet_t0/E3,
   2026-10-22, while it exists).
3. The feet_t0 tables: the training population (328 matches) and the held-out table (172 matches; z left at 0, only the
   E3 columns are compared) under `FEET_T0` from the feet_t0 cache, as table digests (`table_digest`) against the
   digests of the patched tables behind results/xgs_stack_feet_t0_flags and results/sweep_feet_2026_10, recorded in
   `research/feet_location_rule.json` (`evidence.patched_tables`).

Writes <out>/complete.json (refuses an existing folder) and exits non-zero unless every check passes.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import extract as ex
from xgpoc.detector import location as loc
from xgpoc.detector import records, seal
from xgpoc.detector.features import COLUMNS
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX
from xgpoc.stack import inputs as si
from xgpoc.stack_heldout import testset as ts
from xgpoc.stack_location import inputs as sl

ROOT = Path(__file__).resolve().parents[1]
CACHES = {"xgd_full3": ROOT / "cache/xgd_full3/E3", "xgd_full4": ROOT / "cache/xgd_full4/E3"}
FEET_T0_CACHE = ROOT / "cache/xgd_full4_feet_t0/E3"
PATCHED_CACHE = ROOT / "cache/xgd_full5_feet_t0/E3"
RULE_JSON = ROOT / "research/feet_location_rule.json"


def table_digest(table: pd.DataFrame) -> str:
    """sha256 of a table's CSV (index, columns, every float at 17 significant digits): equal digests, equal tables."""
    return hashlib.sha256(table.to_csv(float_format="%.17g").encode()).hexdigest()


def same_features(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Rows (shot_id, offset_ms at t0 and -400 ms) and `features.COLUMNS` of two feature tables: equal exactly?"""
    a, b = (t.set_index(["shot_id", "offset_ms"]).sort_index() for t in (a, b))
    a, b = (t[t.index.get_level_values("offset_ms").isin(si.OFFSETS)] for t in (a, b))
    if not a.index.equals(b.index):
        return {"rows": [len(a), len(b)], "equal": False}
    differing = int(ex.feature_differences(a, b, COLUMNS, 0.0).any(axis=1).sum())
    return {"rows": len(a), "differing_rows": differing, "equal": differing == 0}


def fidelity(name: str, e3: Path, plan: pd.DataFrame) -> dict:
    frames, cached = pd.read_parquet(e3 / "frames.parquet"), pd.read_parquet(e3 / "features.parquet")
    out = {"feature_rows": same_features(loc.feature_rows(frames, loc.E3), cached)}
    frozen, _ = si.e3_table(frames, cached)
    copy, _ = sl.e3_table(frames, cached, loc.E3)
    out["e3_table"] = {"shots": len(frozen), "equal": bool(frozen.equals(copy)), "digest": table_digest(frozen)}
    reference, population = si.build_population(e3=e3, plan=plan), sl.build_population(loc.E3, e3=e3, plan=plan)
    out["population"] = {"equal": bool(reference.table.equals(population.table) and reference.counts == population.counts),
                         "digest": table_digest(reference.table)}
    print(name, json.dumps(out), flush=True)
    return out


def feet_tables(plan: pd.DataFrame) -> dict:
    population = sl.build_population(loc.FEET_T0, e3=FEET_T0_CACHE, plan=plan)
    shots = ts.heldout_shots(INDEX, plan)
    annotations = {g: si.read_annotations(g) for g in sorted(shots.game.unique())}
    heldout, _ = sl.heldout_table(shots, pd.Series(0.0, index=shots.index), FEET_T0_CACHE, annotations, loc.FEET_T0)
    return {"population": table_digest(population.table[si.E3_COLUMNS]), "heldout": table_digest(heldout[si.E3_COLUMNS]),
            "population_shots": len(population.table), "heldout_shots": len(heldout)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results/feet_location_check")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    checks: dict = {}
    with no_statsbomb_reads(), seal.development_labels_only(a1.PLAN_DIR / "plan.csv", INDEX):
        plan = a1.verify_plan(INDEX)
        checks["fidelity"] = {name: fidelity(name, e3, plan) for name, e3 in CACHES.items()}

        frames = pd.read_parquet(FEET_T0_CACHE / "frames.parquet")
        features = pd.read_parquet(FEET_T0_CACHE / "features.parquet")
        record = {name: records.verified_file(FEET_T0_CACHE / name) for name in ("frames.parquet", "features.parquet")}
        cache = {"record": record, "rebuilt": same_features(loc.feature_rows(frames, loc.FEET_T0), features)}
        if PATCHED_CACHE.exists():
            cache["patched"] = same_features(pd.read_parquet(PATCHED_CACHE / "features.parquet"), features) | {
                "features_sha256": [sha256_file(PATCHED_CACHE / "features.parquet"), record["features.parquet"]]}
        checks["feet_t0_cache"] = cache
        print("feet_t0 cache", json.dumps(cache), flush=True)

        expected = json.loads(RULE_JSON.read_text())["evidence"]["patched_tables"]
        tables = feet_tables(plan)
        checks["feet_t0_tables"] = tables | {"expected": expected,
                                             "equal": all(tables[k] == expected[k] for k in expected)}
        print("feet_t0 tables", json.dumps(checks["feet_t0_tables"]), flush=True)

    passed = (all(v["feature_rows"]["equal"] and v["e3_table"]["equal"] and v["population"]["equal"]
                  for v in checks["fidelity"].values())
              and cache["rebuilt"]["equal"] and cache.get("patched", {"equal": True})["equal"]
              and checks["feet_t0_tables"]["equal"])
    args.out.mkdir(parents=True)
    complete = {"all_passed": passed, "checks": checks, "rule": {"path": str(RULE_JSON.relative_to(ROOT)),
                                                                  "sha256": sha256_file(RULE_JSON)},
                "code": records.code_hashes(Path(__file__)) | {"stack_location": {
                    str(f.relative_to(ROOT)): sha256_file(f) for f in sorted((ROOT / "xgpoc/stack_location").glob("*.py"))}},
                "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"),
                        "finished": datetime.now(UTC).isoformat(timespec="seconds")}}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1, default=float) + "\n")
    print("all passed" if passed else "FAILED", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
