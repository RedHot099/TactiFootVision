"""Audit the declared configurations against StatsBomb freeze frames, choose one and apply the gate (xg-detector
tickets 02 and 09, pitch-keypoints 04).

    audit 3 (pitch-keypoints 04, the defaults): E1-E3 from cache/xgd_extract3, audit 2's D3 from cache/xgd_extract2 as
    a reference: ./run.sh python scripts/xgd_audit.py [--out results/xgd_audit3] [--n-boot 5000]

    audit 2 (ticket 09): ./run.sh python scripts/xgd_audit.py --extract cache/xgd_extract2 --configs D1 D2 D3 D4 \
        --reference-configs C4 --reference-extract cache/xgd_extract --reference-metrics results/xgd_audit/metrics.csv \
        --out results/xgd_audit2

    audit 1 (ticket 02): ./run.sh python scripts/xgd_audit.py --extract cache/xgd_extract --configs C1 C2 C3 C4 \
        --reference-configs --out results/xgd_audit

Diagnostic only (`xgpoc.detector.audit`): the one detector script allowed to read StatsBomb data and the pairing,
each with its columns named (no goal or outcome column). Auditing any of D1-D4 or E1-E3 requires that audit's committed
declaration (`research/xg_detector_audit2_configs.json`, `research/pitchkp_audit3_configs.json`) to have its committed
sha256; one run audits the candidates of one declaration. For every configuration, candidate or
reference, it checks the cache against its completion record, replays the t0 features from the cached frames (they
must equal the cached feature table, and add the goalkeeper position) and compares them with the StatsBomb positions of
the paired non-penalty shots. The choice and the gate see the candidates (`--configs`) only. The reference
configurations' metrics must equal those of the earlier audit (`--reference-metrics`) row by row. It writes:

    per_shot.parquet     one row per configuration and shot: detected and StatsBomb positions, errors, flags
    metrics.csv          configuration x quantity x subset x statistic, with 95% match-bootstrap intervals
    cone_confusion.csv   defenders-in-cone counts, detected x StatsBomb (>= 1, >= 2, exact)
    timing.csv           median errors by |timing residual|, descriptive
    choice.json          the config choice among the candidates and the gate (PASS / STOP)
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
from xgpoc.detector import extract3 as e3

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ex.CONFIGS | e3.CONFIGS
# Declared candidate sets: configurations, the committed declaration file and its sha256.
DECLARATIONS = {"D1-D4": (ex.AUDIT_2, ex.AUDIT_2_DECLARATION, ex.AUDIT_2_DECLARATION_SHA256),
                "E1-E3": (e3.AUDIT_3, e3.AUDIT_3_DECLARATION, e3.AUDIT_3_DECLARATION_SHA256)}


def replayed_features(frames: pd.DataFrame, name: str) -> pd.DataFrame:
    """The configuration's t0 features recomputed from its cached frames."""
    config = CONFIGS[name]
    return e3.replay_features(frames, config) if isinstance(config, e3.SourceConfig) else ex.replay_features(frames, config)


def checked_declaration(configs: list[str], path: Path | None) -> dict | None:
    """The committed declaration of the candidates (path and sha256), or None for undeclared audit-1 candidates;
    `SystemExit` when the file is not the committed one or the candidates mix declarations."""
    sets = [label for label, (names, _, _) in DECLARATIONS.items() if set(configs) & set(names)]
    if not sets:
        return None
    if len(sets) > 1:
        raise SystemExit(f"one audit audits the candidates of one declaration, got {' and '.join(sets)}")
    _, default, expected = DECLARATIONS[sets[0]]
    path = path or default
    digest = sha256_file(path) if path.is_file() else None
    if expected is None or digest != expected:
        raise SystemExit(f"{path} is not the committed {sets[0]} declaration (sha256 {expected})")
    return {"path": str(path), "sha256": digest}


def verified_cache(folder: Path) -> dict:
    """The configuration's completion record, after checking every listed output file against it."""
    record = json.loads((folder / "complete.json").read_text())
    try:
        for name in record["files"]:
            records.verified_file(folder / name)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return record


def json_default(value):
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    raise TypeError(f"{type(value)} is not JSON serialisable")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--extract", type=Path, default=ROOT / "cache/xgd_extract3")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_audit3")
    parser.add_argument("--pairs", type=Path, default=audit.PAIRS)
    parser.add_argument("--shots", type=Path, default=audit.SHOTS)
    parser.add_argument("--configs", nargs="+", default=list(e3.AUDIT_3), choices=list(CONFIGS),
                        help="the candidates of the choice")
    parser.add_argument("--reference-configs", nargs="*", default=["D3"], choices=list(CONFIGS),
                        help="audited next to the candidates, never chosen")
    parser.add_argument("--reference-extract", type=Path, default=ROOT / "cache/xgd_extract2")
    parser.add_argument("--reference-metrics", type=Path, default=ROOT / "results/xgd_audit2/metrics.csv",
                        help="the earlier audit whose rows the reference configurations must reproduce")
    parser.add_argument("--declaration", type=Path, default=None, help="default: the candidates' committed declaration")
    parser.add_argument("--n-boot", type=int, default=audit.N_BOOT)
    parser.add_argument("--seed", type=int, default=audit.SEED)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    if set(args.configs) & set(args.reference_configs):
        raise SystemExit("a configuration cannot be both a candidate and a reference")
    declaration = checked_declaration(args.configs, args.declaration)
    started = datetime.now(UTC)

    pop = audit.read_population(args.pairs)
    sb = audit.truth(pop, audit.read_freeze_frames(args.shots, pop.sb_event_id))
    tables, caches, replay = [], {}, {}
    sources = [(name, args.extract) for name in args.configs] + [(name, args.reference_extract) for name in args.reference_configs]
    for name, extract in sources:
        folder = extract / name
        record = verified_cache(folder)
        if record["config"]["name"] != name:
            raise SystemExit(f"{folder} holds {record['config']['name']}, not {name}")
        frames = pd.read_parquet(folder / "frames.parquet", columns=audit.FRAME_READ_COLUMNS)
        cached = pd.read_parquet(folder / "features.parquet")
        replayed = replayed_features(frames, name)
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
    decision = audit.decide(metrics, args.configs) | {"reference_configs": args.reference_configs, "declaration": declaration}

    args.out.mkdir(parents=True, exist_ok=True)
    per_shot.to_parquet(args.out / "per_shot.parquet", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.6g")
    audit.cone_confusion(per_shot).to_csv(args.out / "cone_confusion.csv", index=False)
    audit.timing_table(per_shot).to_csv(args.out / "timing.csv", index=False, float_format="%.6g")
    reference = None
    if args.reference_configs:
        reference = {"metrics": str(args.reference_metrics), "sha256": sha256_file(args.reference_metrics)} | audit.metric_differences(
            pd.read_csv(args.out / "metrics.csv"), pd.read_csv(args.reference_metrics), args.reference_configs)
        if reference["missing"] or reference["differing"] or not reference["rows"]:
            raise SystemExit(f"the reference configurations do not reproduce {args.reference_metrics}: {reference}")
    write_json(args.out / "choice.json", decision)
    population = {"rule": "pairs.paired and not pairs.is_penalty", "shots": len(pop), "matches": int(pop.game.nunique()),
                  "with_freeze_frame": int(sb.sb_has_ff.sum()), "with_statsbomb_goalkeeper": int(sb.sb_gk_x.notna().sum())}
    record = {"inputs": {"pairs": {"path": str(args.pairs), "sha256": sha256_file(args.pairs), "columns": audit.PAIR_COLUMNS},
                         "statsbomb_shots": {"path": str(args.shots), "sha256": sha256_file(args.shots),
                                             "columns": audit.SHOT_COLUMNS},
                         "caches": caches, "declaration": declaration},
              "population": population, "parameters": {"n_boot": args.n_boot, "seed": args.seed, "thresholds_m": list(audit.THRESHOLDS_M),
                                                       "kp_strict": audit.KP_STRICT, "coverage_min": audit.COVERAGE_MIN,
                                                       "median_max_m": audit.MEDIAN_MAX_M, "gk_coverage_min": audit.GK_COVERAGE_MIN},
              "replay_check": {name: "identical" for name in replay}, "reference_check": reference, "decision": decision,
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
