"""H3 decoding probes on an evaluation table (xg-vendor ticket 09; `xgpoc.vendor.probes`).

Opens and validates the table (`evaltable.load_table`; a confirmatory table with labels needs `--prereg`), derives the
probe targets from the StatsBomb shots table (`probe_targets.probe_targets`, by `statsbomb_event_id`), checks every
embedding file against the sidecar's sha256 (relative files are read from the table directory), runs the probes of
every embedding set the table references on the primary rows that are not penalties (StatsBomb shot type "Penalty", a
covariate, so no label is read; the secondary rows' frames sit at StatsBomb time and are not probed) and writes:

    probes.csv               metric per encoder x subset x coverage x target, match-bootstrap interval
    probe_differences.csv    G-combo minus frozen DINOv2 on the shots both cover, with the interval
    probe_predictions.parquet  out-of-fold predictions (long format)
    targets_check.csv        (--prototype) derived targets against the prototype's sb_features.parquet
    prototype_comparison.csv (--prototype) against the prototype's probes.csv
    complete.json            completion record: inputs and code hashes, parameters, key numbers

    ./run.sh python scripts/vendor_probes.py --table DIR --name NAME --prereg FILE --out DIR [--shots PATH]
    ./run.sh python scripts/vendor_probes.py --table results/xgv_eval_dev --name table --out results/xgv_probes_dev \\
        [--shots PATH] [--n-boot 5000] [--seed 0] [--by-split] [--thresholds JSON] [--prototype DIR]      (development tables only)

A dry-run or confirmatory table needs `--prereg`: the number of draws, the bootstrap seed and the H3 thresholds come from
that record (at least `ladder.MIN_CONFIRMATORY_BOOT` draws) and cannot be given on the command line; the development
options are refused (`probe_parameters`). The shuffled-target control is a hard gate (`probes.control_gate`): when an
encoder's control decodes above R^2 = 0, the outputs and the completion record are written with the gate failed and the
script exits with status `probes.CONTROL_FAILED_EXIT`.

A development table gives development numbers and a dry-run table dry-run numbers, labelled so in complete.json; neither
are results.
"""

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import code_version, sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import probe_targets as pt
from xgpoc.vendor import probes
from xgpoc.vendor.draws import N_BOOT, SEED, split_subsets, table_draws
from xgpoc.vendor.ladder import MIN_CONFIRMATORY_BOOT

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
SHOTS = Path("/home/kuba/projects/football_data/statsbomb_open_v1/tables/shots.parquet")
SOURCE_FILES = (Path(__file__).resolve(), *(ROOT / "xgpoc/vendor" / f for f in ("probes.py", "probe_targets.py", "draws.py", "evaltable.py",
                                                                              "ladder.py")),
                ROOT / "xgpoc/probe/readability.py")
PENALTY_SHOT_TYPE = "Penalty"
PROTOTYPE_ENCODERS = {"emb_g_fold0": "g_combo", "emb_g_fold1": "g_combo", "emb_dinov2": "dinov2_frozen"}


def probe_parameters(kind: str, prereg: et.PreRegistration | None, args: argparse.Namespace) -> dict:
    """Draws, bootstrap seed and H3 thresholds of the run, with their source. Raises `ValueError` for a combination the
    contract forbids: a dry-run or confirmatory table takes them only from the pre-registration record (at least
    `MIN_CONFIRMATORY_BOOT` draws, no development options); a development table takes them from the options."""
    given = {k: getattr(args, k) for k in ("n_boot", "seed", "thresholds") if getattr(args, k) is not None}
    if prereg is not None:
        if given:
            raise ValueError(f"the pre-registration record fixes the parameters; drop {sorted(given)}")
        params = {"n_boot": prereg.n_boot, "seed": prereg.boot_seed, "thresholds": dict(prereg.h3_thresholds),
                  "confirmatory_targets": list(prereg.h3_confirmatory_targets), "source": "pre-registration record"}
    elif kind != "development":
        raise ValueError(f"a {kind} table needs --prereg; its parameters come from the pre-registration record")
    else:
        params = {"n_boot": N_BOOT if args.n_boot is None else args.n_boot, "seed": SEED if args.seed is None else args.seed,
                  "thresholds": json.loads(args.thresholds.read_text()) if args.thresholds else None, "confirmatory_targets": None,
                  "source": "command line"}
    if kind != "development" and (params["n_boot"] < MIN_CONFIRMATORY_BOOT or args.by_split or args.prototype):
        raise ValueError(f"a {kind} run needs at least {MIN_CONFIRMATORY_BOOT} draws and no development options")
    return params


def load_all_embeddings(table: pd.DataFrame, meta: dict, table_dir: Path) -> tuple[dict, dict]:
    """Embeddings of every set with at least one reference, after checking each file's sha256 once. A relative file
    name in the sidecar is relative to the table directory."""
    hashes, embeddings = {}, {}
    for column in et.EMBEDDING_COLUMNS:
        if not (table[column] >= 0).any():
            continue
        entry = meta["embedding_sets"][column]
        if entry["file"] not in hashes:
            hashes[entry["file"]] = sha256_file(Path(table_dir) / entry["file"])   # an absolute name stays absolute
        if hashes[entry["file"]] != entry["sha256"]:
            raise et.EvalTableError(f"{entry['file']} does not match the sidecar's sha256")
        embeddings[column] = et.load_embeddings(table, meta, column, root=table_dir)
    return embeddings, hashes


def probe_subsets(table: pd.DataFrame, by_split: bool) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """The reported row subsets restricted to primary rows that are not penalties (StatsBomb shot type "Penalty"), and
    the numbers of rows left out."""
    penalty = table.sb_shot_type.eq(PENALTY_SHOT_TYPE).to_numpy()
    primary = table.population.eq("primary").to_numpy()
    keep = primary & ~penalty
    left_out = {"penalties": int((primary & penalty).sum()), "secondary_rows": int((~primary).sum())}
    return {name: mask & keep for name, mask in split_subsets(table, by_split).items()}, left_out


def prototype_comparison(metrics: pd.DataFrame, prototype: pd.DataFrame) -> pd.DataFrame:
    """Own-coverage rows of this run next to the prototype's probes.csv (encoder g_combo / dinov2_frozen, population
    trainvalid / all). The prototype's G-combo rows mix the five fold models over all train/valid shots, so a fold model
    here is compared on a subset of those shots."""
    proto = prototype.set_index(["encoder", "population", "target"])
    rows = []
    for r in metrics[metrics.coverage.eq("own")].itertuples():
        population = {"all": "all", "trainvalid": "trainvalid"}.get(r.subset)
        if population and r.encoder != "emb_dinov2":
            population = "trainvalid"   # the prototype had G-combo embeddings for train/valid shots only
        key = (PROTOTYPE_ENCODERS[r.encoder], population, r.target)
        if population is None or key not in proto.index:
            continue
        p = proto.loc[key]
        rows.append({"encoder": r.encoder, "subset": r.subset, "target": r.target, "metric": r.metric, "n": r.n, "value": r.value,
                     "lo": r.lo, "hi": r.hi, "prototype_encoder": key[0], "prototype_population": key[1], "prototype_n": int(p.n),
                     "prototype_value": float(p.value), "difference": r.value - float(p.value), "same_shots": int(p.n) == r.n})
    return pd.DataFrame(rows)


def rounded(*values) -> list:
    """JSON-safe summary numbers: 4 decimals, None for NaN."""
    return [round(float(v), 4) if np.isfinite(v) else None for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, required=True, help="directory of the evaluation table")
    parser.add_argument("--name", default="table", help="table name inside --table (label-free or joined)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=Path, default=SHOTS, help="derived StatsBomb shots table (freeze frames)")
    parser.add_argument("--prereg", type=Path, help="pre-registration record (required for a dry-run or confirmatory table)")
    parser.add_argument("--n-boot", type=int, help=f"development tables only (default {N_BOOT})")
    parser.add_argument("--seed", type=int, help=f"development tables only (default {SEED})")
    parser.add_argument("--by-split", action="store_true", help="development tables only: also report the test and train/valid matches")
    parser.add_argument("--thresholds", type=Path, help="development tables only: JSON {target: threshold}")
    parser.add_argument("--prototype", type=Path, help="development tables only: prototype directory (probes.csv, sb_features.parquet)")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        sys.exit(f"{args.out} is complete; delete it to rerun")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} holds files without a completion record; delete it to rerun")
    started = time.time()

    prereg = et.load_preregistration(args.prereg) if args.prereg else None
    table, meta = et.load_table(args.table, args.name, prereg)
    kind = meta["table_kind"]
    try:
        params = probe_parameters(kind, prereg, args)
    except ValueError as refusal:
        sys.exit(str(refusal))
    n_boot, seed, thresholds = params["n_boot"], params["seed"], params["thresholds"]
    shots = pd.read_parquet(args.shots, columns=pt.SHOT_INPUT_COLUMNS)
    derived = pt.probe_targets(shots)
    targets = pt.targets_for_table(table, derived)
    embeddings, embedding_hashes = load_all_embeddings(table, meta, args.table)
    draws = table_draws(table, n_boot, seed)
    subsets, left_out = probe_subsets(table, args.by_split)
    print(f"{kind} table: {len(table)} rows, {len(draws.keys)} matches; encoders "
          f"{ {c: len(p) for c, (p, _) in embeddings.items()} }; {n_boot} draws", flush=True)

    result = probes.probe_table(table, targets, embeddings, draws, subsets)
    gate = probes.control_gate(result.metrics)
    metrics = probes.apply_thresholds(result.metrics, thresholds) if thresholds else result.metrics
    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "probes.csv", index=False)
    result.differences.to_csv(args.out / "probe_differences.csv", index=False)
    result.predictions.to_parquet(args.out / "probe_predictions.parquet", index=False)

    inputs = {"table": {"directory": str(args.table), "name": args.name, "parquet_sha256": meta["table_sha256"], "labels_state": meta["labels_state"]},
              "shots": {"file": str(args.shots), "sha256": sha256_file(args.shots)}, "embeddings": embedding_hashes,
              "prereg": None if prereg is None else {"file": str(args.prereg), "sha256": prereg.sha256},
              "thresholds": None if thresholds is None else {"source": params["source"], "values": thresholds,
                                                             "file": str(args.thresholds) if args.thresholds else None,
                                                             "sha256": sha256_file(args.thresholds) if args.thresholds else None}}
    checks = {}
    if args.prototype:
        check = pt.compare_with_features(derived, pd.read_parquet(args.prototype / "sb_features.parquet"))
        check.to_csv(args.out / "targets_check.csv", index=False)
        comparison = prototype_comparison(metrics, pd.read_csv(args.prototype / "probes.csv"))
        comparison.to_csv(args.out / "prototype_comparison.csv", index=False)
        inputs["prototype"] = {name: sha256_file(args.prototype / name) for name in ("sb_features.parquet", "probes.csv")}
        same = comparison[comparison.same_shots]
        checks = {"targets_max_abs_diff": rounded(check.max_abs_diff.max())[0], "targets_nan_mismatch": int(check.nan_mismatch.sum()),
                  "targets_compared_shots": int(check.shots.iloc[0]),
                  "prototype_same_shots_max_abs_diff": rounded(same.difference.abs().max())[0] if len(same) else None}
        print(check.to_string(index=False))
        print(comparison.round(4).to_string(index=False))

    headline = metrics[metrics.subset.eq("all") & metrics.coverage.eq("own")]
    summary = {"table_kind": kind, "development": kind == "development", "rows": len(table), "left_out": left_out,
               "embedding_references": {c: len(p) for c, (p, _) in embeddings.items()},
               "metrics_all_own": {f"{r.encoder}|{r.target}": [*rounded(r.value, r.lo, r.hi), int(r.n)] for r in headline.itertuples()},
               "differences_all": {f"{r.encoder}|{r.target}": [*rounded(r.difference, r.lo, r.hi), int(r.n)]
                                   for r in result.differences[result.differences.subset.eq("all")].itertuples()},
               "negative_control_max": rounded(metrics.loc[metrics.negative_control, "value"].max())[0], "control_gate": gate,
               "checks": checks,
               "runtime_s": round(time.time() - started, 1)}
    print(pd.DataFrame(result.differences).round(4).to_string(index=False))
    names = sorted(p.name for p in args.out.iterdir() if p.is_file())
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "label": et.RUN_LABELS[kind],
              "identity": {"analysis": "xg-vendor H3 probes", "table_kind": kind, "inputs": inputs,
                           "parameters": {"n_boot": n_boot, "seed": seed, "source": params["source"],
                                          "confirmatory_targets": params["confirmatory_targets"], "subsets": list(subsets),
                                          "alphas": list(probes.ALPHAS),
                                          "n_outer": probes.N_OUTER, "n_inner": probes.N_INNER, "control": probes.CONTROL,
                                          "control_seed": probes.CONTROL_SEED, "reference": probes.REFERENCE},
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": summary,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "files": {name: sha256_file(args.out / name) for name in names}}
    (args.out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out} ({summary['runtime_s']} s)")
    if not gate["passed"]:
        failed = sorted(e for e, g in gate["encoders"].items() if not g["passed"])
        print(f"H3 CONTROL GATE FAILED: the shuffled-target control decodes above R^2 = {gate['threshold']} for {failed}; "
              "every H3 reading of these encoders is invalid", file=sys.stderr)
        sys.exit(probes.CONTROL_FAILED_EXIT)


if __name__ == "__main__":
    main()
