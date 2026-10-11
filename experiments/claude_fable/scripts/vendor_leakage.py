"""H4 leakage diagnostic on an evaluation table with labels joined (xg-vendor ticket 09; `xgpoc.vendor.leakage`).

Opens and validates the table (`evaltable.load_table`), reads the ticket-04 timing verdict record when given
(`xgpoc.vendor.timing_verdict`), accepting it only under a pinned rule hash, and writes:

    leakage.csv               every coefficient of every spec and subset, with the match-bootstrap interval
    prototype_comparison.csv  (--prototype-report) against the prototype's `ladder_report.json` leakage block
    complete.json             completion record: inputs and code hashes, parameters, counts, residual-arm status

The timing-residual arm runs only when the verdict record says "independent"; `--development-override` runs it on a
development table without that verdict (refused on a confirmatory table).

    ./run.sh python scripts/vendor_leakage.py --table DIR --name NAME --prereg FILE --timing-verdict PATH --out DIR
    ./run.sh python scripts/vendor_leakage.py --table results/xgv_eval_dev --name table --out results/xgv_leakage_dev \\
        [--offset-logits calibrated|raw] [--timing-verdict PATH --expected-rule-sha256 HEX] [--development-override] \\
        [--n-boot 5000] [--seed 0] [--by-split] [--prototype-report PATH]                         (development tables only)

A dry-run or confirmatory table needs `--prereg` and `--timing-verdict`: the number of draws, the bootstrap seed, the
offset logits and the rule hash the verdict must carry come from the record (at least `ladder.MIN_CONFIRMATORY_BOOT`
draws) and cannot be given on the command line; the development options are refused (`leakage_parameters`). A
development run that reads a verdict needs `--expected-rule-sha256`.

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
from xgpoc.vendor import leakage as lk
from xgpoc.vendor.draws import N_BOOT, SEED, split_subsets, table_draws
from xgpoc.vendor.ladder import MIN_CONFIRMATORY_BOOT
from xgpoc.vendor.timing_verdict import load_timing_verdict

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
SOURCE_FILES = (Path(__file__).resolve(), *(ROOT / "xgpoc/vendor" / f for f in ("leakage.py", "timing_verdict.py", "draws.py", "evaltable.py",
                                                                              "ladder.py")),
                ROOT / "xgpoc/probe/readability.py", ROOT / "xgpoc/eval/calibration.py")
DEFAULT_OFFSET_LOGITS = "calibrated"
# this run's unadjusted specs and the prototype's leakage keys ("<population>|<outcome>|<regressors>", coefficient name)
PROTOTYPE_KEYS = {("gap_on_target_unadjusted", "on_target"): ("G-vendor|on_target_sn", "on_target_sn"),
                  ("offset_on_target_unadjusted", "on_target"): ("Glogit(0)-Glogit(-400)|on_target_sn", "on_target_sn"),
                  ("gap_residual_slope_unadjusted", "residual_s"): ("G-vendor|res_s", "res_s"),
                  ("gap_early_late_unadjusted", "early"): ("G-vendor|late+early", "early"),
                  ("gap_early_late_unadjusted", "late"): ("G-vendor|late+early", "late")}


def leakage_parameters(kind: str, prereg: et.PreRegistration | None, args: argparse.Namespace) -> dict:
    """Draws, bootstrap seed, offset logits and the rule hash a timing verdict must carry, with their source. Raises
    `ValueError` for a combination the contract forbids: a dry-run or confirmatory table takes them only from the
    pre-registration record (at least `MIN_CONFIRMATORY_BOOT` draws, a timing verdict, no development options); a
    development table takes them from the options, and a verdict there needs `--expected-rule-sha256`."""
    given = {k: getattr(args, k) for k in ("n_boot", "seed", "offset_logits", "expected_rule_sha256") if getattr(args, k) is not None}
    if prereg is not None:
        if given:
            raise ValueError(f"the pre-registration record fixes the parameters; drop {sorted(given)}")
        params = {"n_boot": prereg.n_boot, "seed": prereg.boot_seed, "offset_logits": prereg.offset_logits,
                  "expected_rule_sha256": prereg.timing_rule_sha256, "source": "pre-registration record"}
    elif kind != "development":
        raise ValueError(f"a {kind} table needs --prereg; its parameters come from the pre-registration record")
    else:
        params = {"n_boot": N_BOOT if args.n_boot is None else args.n_boot, "seed": SEED if args.seed is None else args.seed,
                  "offset_logits": args.offset_logits or DEFAULT_OFFSET_LOGITS, "expected_rule_sha256": args.expected_rule_sha256,
                  "source": "command line"}
    if kind != "development":
        if params["n_boot"] < MIN_CONFIRMATORY_BOOT or args.by_split or args.prototype_report or args.development_override:
            raise ValueError(f"a {kind} run needs at least {MIN_CONFIRMATORY_BOOT} draws and no development options")
        if args.timing_verdict is None:
            raise ValueError(f"a {kind} run needs --timing-verdict: the residual arm follows the ticket-04 verdict")
    if args.timing_verdict is not None and params["expected_rule_sha256"] is None:
        raise ValueError("--timing-verdict needs --expected-rule-sha256: a verdict is accepted only under a pinned rule")
    return params


def prototype_comparison(coefficients: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Unadjusted specs next to the prototype's numbers. The prototype dropped StatsBomb goals as well as SoccerNet goals
    (2,328 shots); this run drops SoccerNet goals only (decision D4). Its restricted residual rows are `all_res200`."""
    rows = []
    for r in coefficients.itertuples():
        spec = r.spec.removesuffix("_within_200ms")
        if (spec, r.regressor) not in PROTOTYPE_KEYS:
            continue
        key, name = PROTOTYPE_KEYS[spec, r.regressor]
        population = "all_res200" if r.restriction != "none" else r.subset
        entry = report.get(f"{population}|{key}")
        if entry is None or (r.restriction != "none" and r.subset != "all"):
            continue
        coef, lo, hi = entry[name]
        rows.append({"spec": r.spec, "regressor": r.regressor, "subset": r.subset, "n": r.n, "coef": r.coef, "lo": r.lo, "hi": r.hi,
                     "prototype_key": f"{population}|{key}", "prototype_n": entry["n"], "prototype_coef": coef, "prototype_lo": lo,
                     "prototype_hi": hi, "difference": r.coef - coef})
    return pd.DataFrame(rows)


def rounded(*values) -> list:
    """JSON-safe summary numbers: 4 decimals, None for NaN."""
    return [round(float(v), 4) if np.isfinite(v) else None for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, required=True, help="directory of the evaluation table")
    parser.add_argument("--name", default="table", help="table name inside --table (labels joined)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offset-logits", choices=sorted(lk.OFFSET_LOGITS), help=f"development tables only (default {DEFAULT_OFFSET_LOGITS})")
    parser.add_argument("--timing-verdict", type=Path, help="ticket-04 verdict record (JSON); required for a dry-run or confirmatory table")
    parser.add_argument("--expected-rule-sha256", help="development tables only, required with --timing-verdict: the rule the verdict must carry")
    parser.add_argument("--development-override", action="store_true", help="run the residual arm without an 'independent' verdict (development tables only)")
    parser.add_argument("--prereg", type=Path, help="pre-registration record (required for a dry-run or confirmatory table)")
    parser.add_argument("--n-boot", type=int, help=f"development tables only (default {N_BOOT})")
    parser.add_argument("--seed", type=int, help=f"development tables only (default {SEED})")
    parser.add_argument("--by-split", action="store_true", help="development tables only: also report the test and train/valid matches")
    parser.add_argument("--prototype-report", type=Path, help="development tables only: prototype ladder_report.json")
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
        params = leakage_parameters(kind, prereg, args)
    except ValueError as refusal:
        sys.exit(str(refusal))
    n_boot, seed, offset_logits = params["n_boot"], params["seed"], params["offset_logits"]
    verdict = load_timing_verdict(args.timing_verdict, params["expected_rule_sha256"]) if args.timing_verdict else None
    draws = table_draws(table, n_boot, seed)
    subsets = split_subsets(table, args.by_split)
    result = lk.leakage_table(table, draws, offset_logits=offset_logits, subsets=subsets, verdict=verdict,
                              development_override=args.development_override)
    args.out.mkdir(parents=True, exist_ok=True)
    result.coefficients.to_csv(args.out / "leakage.csv", index=False)
    of_interest = result.coefficients[result.coefficients.of_interest]
    print(f"{kind} table, {offset_logits} offset logits; residual arm: {result.residual_arm}")
    print(of_interest.drop(columns=["outcome", "of_interest", "skipped_draws"]).round(4).to_string(index=False))

    inputs = {"table": {"directory": str(args.table), "name": args.name, "parquet_sha256": meta["table_sha256"], "labels_state": meta["labels_state"]},
              "prereg": None if prereg is None else {"file": str(args.prereg), "sha256": prereg.sha256},
              "timing_verdict": None if verdict is None else {"file": str(args.timing_verdict), "sha256": sha256_file(args.timing_verdict),
                                                              "verdict": verdict.verdict, "rule_sha256": verdict.rule_sha256}}
    checks = {}
    if args.prototype_report:
        comparison = prototype_comparison(result.coefficients, json.loads(args.prototype_report.read_text())["leakage"])
        comparison.to_csv(args.out / "prototype_comparison.csv", index=False)
        inputs["prototype_report"] = {"file": str(args.prototype_report), "sha256": sha256_file(args.prototype_report)}
        checks = {"prototype_max_abs_diff": rounded(comparison.difference.abs().max())[0], "prototype_rows": len(comparison)}
        print(comparison.round(4).to_string(index=False))

    summary = {"table_kind": kind, "development": kind == "development", "counts": result.counts, "residual_arm": result.residual_arm,
               "of_interest_all": {f"{r.spec}|{r.regressor}": [*rounded(r.coef, r.lo, r.hi), int(r.n)]
                                   for r in of_interest[of_interest.subset.eq("all")].itertuples()},
               "checks": checks, "runtime_s": round(time.time() - started, 1)}
    names = sorted(p.name for p in args.out.iterdir() if p.is_file())
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "label": et.RUN_LABELS[kind],
              "identity": {"analysis": "xg-vendor H4 leakage diagnostic", "table_kind": kind, "inputs": inputs,
                           "parameters": {"n_boot": n_boot, "seed": seed, "source": params["source"], "subsets": list(subsets),
                                          "offset_logits": offset_logits, "development_override": args.development_override,
                                          "expected_rule_sha256": params["expected_rule_sha256"],
                                          "residual_clip_ms": lk.RESIDUAL_CLIP_MS, "within_ms": lk.WITHIN_MS, "early_late_ms": lk.EARLY_LATE_MS,
                                          "controls": list(lk.CONTROLS)},
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": summary,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "files": {name: sha256_file(args.out / name) for name in names}}
    (args.out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out} ({summary['runtime_s']} s)")


if __name__ == "__main__":
    main()
