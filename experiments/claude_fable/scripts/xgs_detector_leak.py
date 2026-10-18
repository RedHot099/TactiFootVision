"""Leak check of the detector re-run: xg-stack's read-out restricted to shots whose own frames the retrained detector
never saw in training (development only, CPU).

    ./run.sh --no-sync python scripts/xgs_detector_leak.py [--window-ms 3000] [--out results/xgs_stack_dd2c_leak]

Why: detector-data's pool holds SoccerNet-v3 frames of the xG matches. v3's action images sit at Labels-v2 actions
(Goal, Shots on/off target, ...) and its replay images show the same actions, so the retrained detector was trained on
the t0 frames of many population shots, more often for goals than for other shots. A shot is "seen" when a v3 pool
image (action or replay; v3 metadata `half` and `position`, the action's time) of its match lies within
`--window-ms` of the shot's annotated time. Subsets (each fitted and scored on its own, same folds, same frozen C6
spec; `analysis.analyse` without the GBM): `unseen` (no such image), `no_v3_game` (matches without any v3 image).
For each subset and each E3 cache (frozen `cache/xgd_full3/E3`, retrained `cache/xgd_full4/E3`) the arms' deltas
against R, and the paired new-minus-frozen delta per arm. The pool index and the v3 label files are read for game,
half, position and image type only; the outcome is the population's (held-out seal), never a selection input.

Writes <out>/seen.csv (shot_id, seen), summary.json and complete.json, last.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgs_detector_rerun import FROZEN_REAL, paired
from xgs_run import PROTOCOL_SHA256, write_json
from xgpoc.bundle import sha256_file
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import analysis as sa
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack import records
from xgpoc.stack import stats as ss

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_DATA = Path("/mnt/storage/tactifoot_research/detector_data")
POOL = DETECTOR_DATA / "pool_index.parquet"
V3_RAW = DETECTOR_DATA / "raw/soccernet_v3"
NEW_E3 = ROOT / "cache/xgd_full4/E3"


def v3_times(pool: Path = POOL, raw: Path = V3_RAW) -> pd.DataFrame:
    """game, half, position (ms) and image type of every SoccerNet-v3 image in the detector pool."""
    pool_v3 = pd.read_parquet(pool, columns=["source", "game", "source_image", "image_type"])
    v3 = pool_v3[pool_v3.source == "soccernet_v3"].copy()
    v3["key"] = v3.source_image.str.rsplit("__", n=1).str[1].str.replace(".jpg", ".png")
    rows = []
    for game in sorted(v3.game.unique()):
        meta = json.loads((raw / game / "Labels-v3.json").read_text())
        for kind in ("actions", "replays"):
            for name, m in meta[kind].items():
                rows.append({"game": game, "key": name, "half": int(m["imageMetadata"]["half"]),
                             "position": int(m["imageMetadata"]["position"])})
    out = v3.merge(pd.DataFrame(rows), on=["game", "key"], how="left", validate="one_to_one")
    if out.position.isna().any():
        raise SystemExit(f"{int(out.position.isna().sum())} v3 pool images have no Labels-v3 entry")
    return out[["game", "half", "position", "image_type"]]


def seen_shots(table: pd.DataFrame, v3: pd.DataFrame, window_ms: int) -> pd.Series:
    """Per population shot: a v3 pool image of its match and half within `window_ms` of its time."""
    m = table[["game", "half", "position_ms"]].reset_index().merge(v3, on=["game", "half"], how="inner")
    hit = m[(m.position - m.position_ms).abs() <= window_ms].shot_id.unique()
    return pd.Series(table.index.isin(hit), index=table.index, name="seen")


def read_out(population: si.Population, y: np.ndarray, frozen: dict) -> tuple[dict, dict]:
    """The frozen read-out on a subset: G and GS must keep the protocol's columns; columns another arm loses because
    they are constant in the subset are returned (dropped by `fit.spec`, as on the full population)."""
    spec = sf.spec(frozen["design"]["chosen"], population.table)
    arms = json.loads(json.dumps(pr.design(spec)))["arms"]
    for name in ("R", "G", "GS"):
        if arms[name] != frozen["design"]["arms"][name]:
            raise SystemExit(f"arm {name} differs from the protocol's on this subset")
    lost = {name: sorted(set(frozen["design"]["arms"][name]["columns"]) - set(arms[name]["columns"]))
            for name in arms if arms[name] != frozen["design"]["arms"][name]}
    return sa.analyse(population.table, population.folds, y, ss.bootstrap_weights(population.games), spec,
                      with_gbm=False), lost


def subset(population: si.Population, mask: np.ndarray) -> si.Population:
    return si.Population(population.table[mask], population.inputs, population.counts)


def compact(result: dict) -> dict:
    return {"scores": {k: {m: v[m] for m in ("n", "goals", "log_loss", "auc")} for k, v in result["scores"].items()},
            "deltas_vs_R": {k: {m: v[m] for m in ("delta_ll", "lo95", "hi95")} for k, v in result["deltas"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--window-ms", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgs_stack_dd2c_leak")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    frozen = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    v3 = v3_times()

    with no_statsbomb_reads():
        old, new = si.build_population(), si.build_population(e3=NEW_E3)
        y_all = si.load_outcomes(old).aligned(old)
        seen = seen_shots(old.table, v3, args.window_ms)
        masks = {"unseen": ~seen.to_numpy(), "no_v3_game": ~old.table.game.isin(set(v3.game)).to_numpy()}
        summary = {"window_ms": args.window_ms, "seen_shots": int(seen.sum()), "shots": len(seen),
                   "seen_share_by_outcome": {str(int(k)): float(v) for k, v in seen.groupby(y_all).mean().items()},
                   "subsets": {}}
        for name, mask in masks.items():
            y = y_all[mask]
            out = {"shots": int(mask.sum()), "goals": int(y.sum()), "matches": int(old.table.game[mask].nunique())}
            reads = {}
            for label, population in (("frozen", old), ("retrained", new)):
                part = subset(population, mask)
                reads[label], lost = read_out(part, y, frozen)
                out[label] = compact(reads[label]) | {"columns_lost": lost}
            out["retrained_minus_frozen"] = paired(reads["retrained"]["probabilities"], reads["frozen"]["probabilities"],
                                                   y, old.table.game[mask].to_numpy())
            summary["subsets"][name] = out
            print(name, json.dumps({k: out[k] for k in ("shots", "goals", "matches")}),
                  {lab: {a: round(d["delta_ll"], 5) for a, d in out[lab]["deltas_vs_R"].items()} for lab in ("frozen", "retrained")},
                  flush=True)
        args.out.mkdir(parents=True)
        seen.rename("seen").to_frame().to_csv(args.out / "seen.csv")
        write_json(args.out / "summary.json", summary)

    complete = {"protocol_sha256": PROTOCOL_SHA256, "mode": "detector re-run leak check (development)",
                "inputs": {"pool_index": {"path": str(POOL), "sha256": sha256_file(POOL)}, "frozen_real": str(FROZEN_REAL),
                           "population_frozen": old.inputs, "population_retrained": new.inputs},
                "code": records.code_hashes(Path(__file__)), "git": detector_records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": {str(p.relative_to(args.out)): sha256_file(p) for p in sorted(args.out.rglob("*")) if p.is_file()}}
    write_json(args.out / "complete.json", complete)


if __name__ == "__main__":
    main()
