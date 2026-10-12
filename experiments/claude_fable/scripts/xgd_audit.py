"""Audit C1-C4 against StatsBomb freeze frames, choose the configuration and apply the gate (xg-detector ticket 02).

    ./run.sh python scripts/xgd_audit.py [--extract cache/xgd_extract] [--out results/xgd_audit] [--n-boot 5000]

Diagnostic only (`xgpoc.detector.audit`): the one detector script allowed to read StatsBomb data and the pairing,
each with its columns named (no goal or outcome column). For every configuration it checks the cache against its
completion record, replays the t0 features from the cached frames (they must equal the cached feature table, and add
the goalkeeper position), compares them with the StatsBomb positions of the paired non-penalty shots, and writes:

    per_shot.parquet     one row per configuration and shot: detected and StatsBomb positions, errors, flags
    metrics.csv          configuration x quantity x subset x statistic, with 95% match-bootstrap intervals
    cone_confusion.csv   defenders-in-cone counts, detected x StatsBomb (>= 1, >= 2, exact)
    timing.csv           median errors by |timing residual|, descriptive
    choice.json          the config choice and the gate (PASS / STOP)
    complete.json        completion record (inputs, code, parameters, replay check, outputs), written last
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import audit, records
from xgpoc.detector import extract as ex
from xgpoc.paths import STATSBOMB

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results/xgv_pairing/pairs.parquet"
SHOTS = STATSBOMB / "tables/shots.parquet"


def verified_cache(folder: Path) -> dict:
    """The configuration's completion record, after checking every listed output file against it."""
    record = json.loads((folder / "complete.json").read_text())
    for name, digest in record["files"].items():
        if sha256_file(folder / name) != digest:
            raise SystemExit(f"{folder / name} does not match its completion record")
    return record


def json_default(value):
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    raise TypeError(f"{type(value)} is not JSON serialisable")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--extract", type=Path, default=ROOT / "cache/xgd_extract")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_audit")
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--shots", type=Path, default=SHOTS)
    parser.add_argument("--configs", nargs="+", default=list(ex.CONFIGS), choices=list(ex.CONFIGS))
    parser.add_argument("--n-boot", type=int, default=audit.N_BOOT)
    parser.add_argument("--seed", type=int, default=audit.SEED)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started = datetime.now(UTC)

    pop = audit.read_population(args.pairs)
    sb = audit.truth(pop, audit.read_freeze_frames(args.shots, pop.sb_event_id))
    tables, caches, replay = [], {}, {}
    for name in args.configs:
        folder = args.extract / name
        record = verified_cache(folder)
        if record["config"]["name"] != name:
            raise SystemExit(f"{folder} holds {record['config']['name']}, not {name}")
        frames = pd.read_parquet(folder / "frames.parquet", columns=audit.FRAME_READ_COLUMNS)
        cached = pd.read_parquet(folder / "features.parquet")
        replayed = ex.replay_features(frames, ex.CONFIGS[name])
        replay[name] = audit.check_replay(cached, replayed)
        if replay[name]:
            raise SystemExit(f"{name}: replayed features differ from the cache: {replay[name]}")
        tables.append(audit.compare(pop, sb, audit.detected(replayed, frames), name))
        caches[name] = {"complete_json_sha256": sha256_file(folder / "complete.json"), "files": record["files"],
                        "code_digest": record["identity"]["code"]["digest"], "git": record["identity"]["git"]}
        print(f"{name}: replay identical on {len(cached)} shots", flush=True)
    per_shot = pd.concat(tables, ignore_index=True)

    draws = audit.draws(pop.game.to_numpy(), args.n_boot, args.seed)
    metrics = audit.metrics(per_shot, draws)
    decision = audit.decide(metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    per_shot.to_parquet(args.out / "per_shot.parquet", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.6g")
    audit.cone_confusion(per_shot).to_csv(args.out / "cone_confusion.csv", index=False)
    audit.timing_table(per_shot).to_csv(args.out / "timing.csv", index=False, float_format="%.6g")
    write_json(args.out / "choice.json", decision)
    population = {"rule": "pairs.paired and not pairs.is_penalty", "shots": len(pop), "matches": int(pop.game.nunique()),
                  "with_freeze_frame": int(sb.sb_has_ff.sum()), "with_statsbomb_goalkeeper": int(sb.sb_gk_x.notna().sum())}
    record = {"inputs": {"pairs": {"path": str(args.pairs), "sha256": sha256_file(args.pairs), "columns": audit.PAIR_COLUMNS},
                         "statsbomb_shots": {"path": str(args.shots), "sha256": sha256_file(args.shots),
                                             "columns": audit.SHOT_COLUMNS},
                         "caches": caches},
              "population": population, "parameters": {"n_boot": args.n_boot, "seed": args.seed, "thresholds_m": list(audit.THRESHOLDS_M),
                                                       "kp_strict": audit.KP_STRICT, "coverage_min": audit.COVERAGE_MIN,
                                                       "median_max_m": audit.MEDIAN_MAX_M, "gk_coverage_min": audit.GK_COVERAGE_MIN},
              "replay_check": {name: "identical" for name in replay}, "decision": decision,
              "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
              "run": {"started": started.isoformat(timespec="seconds"),
                      "finished": datetime.now(UTC).isoformat(timespec="seconds")},
              "files": records.output_hashes(args.out)}
    write_json(args.out / "complete.json", record)
    gate = decision["gate"]
    print(f"chosen {decision['chosen']} ({decision['chosen_from']}); gate {gate['result']}: "
          + ", ".join(f"{k} {v['value']:.4f}" for k, v in gate["checks"].items()))


if __name__ == "__main__":
    main()
