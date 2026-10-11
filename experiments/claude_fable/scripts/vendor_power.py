"""Power simulation of the H2 equivalence margin on the DEVELOPMENT evaluation table (xg-vendor ticket 08, decision D5).

    ./run.sh python scripts/vendor_power.py [--table-dir results/xgv_eval_dev] [--table-name table] [--n-sims 1000]
        [--n-boot 5000] [--out results/xgv_power]

Checks the rule note (`research/xg_vendor_delta_rule.md`) against the hash pinned in `xgpoc.vendor.power`, fits the two
truths on the table's primary population, calibrates the weaker-frame noise, runs the eight scenarios, applies the
rule and writes:
    power.csv          scenario x margin: TOST power, Monte Carlo SE, superiority power, H2 difference mean and SDs
    studies.parquet    one row per simulated study
    decision.json      the rule note's sha256, decision power per margin, the chosen margin, truths, noise levels and
                       the design check (H2 bootstrap SD on the real development labels next to the simulated SDs)
    complete.json      completion record
Only a development table is accepted: the power simulation must run before any confirmatory label exists.
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import code_version, sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import power as pw

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/power.py", ROOT / "xgpoc/vendor/ladder.py", ROOT / "xgpoc/vendor/evaltable.py",
                ROOT / "xgpoc/vendor/draws.py", ROOT / "xgpoc/probe/readability.py", ROOT / "xgpoc/eval/calibration.py")
SIGMA_STREAM = len(pw.SCENARIOS)   # generator stream of the noise calibration, after the scenario streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table-dir", type=Path, default=ROOT / "results/xgv_eval_dev")
    parser.add_argument("--table-name", default="table")
    parser.add_argument("--n-sims", type=int, default=pw.N_SIMS)
    parser.add_argument("--n-boot", type=int, default=ld.MIN_CONFIRMATORY_BOOT, help="draws per simulated study (the rule note fixes 5,000)")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgv_power")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty; delete it to rerun")
    rule_sha = pw.rule_note_sha256(ROOT)
    started = datetime.now(UTC)

    table, meta = et.load_table(args.table_dir, args.table_name)
    if meta["table_kind"] != "development":
        sys.exit("the power simulation runs on the development table only")
    basis = pw.basis_from_table(table)
    truths = {t: pw.truth_probabilities(basis, t) for t in pw.TRUTHS}
    rng = np.random.default_rng([pw.SIM_SEED, SIGMA_STREAM])
    sigmas = {t: pw.weaker_frame_sigma(truths[t][0], basis.frame_logit, pw.WEAKER_FRAME_LOG_LOSS, rng) for t in pw.TRUTHS}
    print(f"basis {len(basis.y)} shots, {len(np.unique(basis.matches))} matches; truths {({t: np.round(c, 3).tolist() for t, (_, c) in truths.items()})}; "
          f"weaker-frame sigma {sigmas}", flush=True)

    studies = []
    for scenario in pw.SCENARIOS:
        q = truths[scenario.truth][0]
        studies.append(pw.run_scenario(basis, scenario, q, sigmas[scenario.truth], args.n_sims, args.n_boot))   # sigma used by "weaker" only
        last = pw.power_table(studies[-1])
        print(f"{scenario.name}: " + ", ".join(f"{m}: {p:.3f}" for m, p in zip(last.margin, last.power_tost, strict=True))
              + f"; mean diff {last.mean_diff.iloc[0]:+.5f}, sd {last.sd_diff.iloc[0]:.5f}", flush=True)
    studies = pd.concat(studies, ignore_index=True)
    power = pw.power_table(studies)
    decision = pw.choose_margin(power)

    observed_point, observed_draws = ld.h2_draws(basis.y, basis.positions, basis.frame_logit, basis.matches, ld.bootstrap(basis.matches, args.n_boot))
    reference = power[power.scenario.eq(pw.DECISION_SCENARIOS[0].name)].iloc[0]
    design_check = {"observed_h2": observed_point, "observed_boot_sd": float(np.std(observed_draws, ddof=1)),
                    "simulated_sd_diff": float(reference.sd_diff), "simulated_mean_boot_sd": float(reference.mean_boot_sd),
                    "scenario": pw.DECISION_SCENARIOS[0].name}

    args.out.mkdir(parents=True, exist_ok=True)
    power.assign(table_kind="development").to_csv(args.out / "power.csv", index=False)
    studies.to_parquet(args.out / "studies.parquet", index=False)
    result = {"table_kind": "development", "rule_note": str(pw.RULE_NOTE), "rule_note_sha256": rule_sha, "margins": list(pw.MARGINS),
              **decision, "truths": {t: {"coefficients": c, "mean_q": float(q.mean())} for t, (q, c) in truths.items()},
              "truth_coefficient_order": {"development": ["intercept", "frame_logit", "logit_positions"], "null": ["intercept", "logit_positions"]},
              "weaker_frame_sigma": sigmas, "weaker_frame_log_loss": pw.WEAKER_FRAME_LOG_LOSS,
              "combined_extra_shots": pw.EXTRA_SHOTS, "combined_extra_rate": pw.EXTRA_SHOTS / len(basis.y), "design_check": design_check}
    (args.out / "decision.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("decision_power", "chosen_margin", "design_check")}, indent=1))

    names = sorted(p.name for p in args.out.iterdir() if p.is_file())
    record = {"started_at": started.strftime("%Y-%m-%d %H:%M:%S"), "completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "identity": {"table_kind": "development", "table": {"dir": str(args.table_dir), "name": args.table_name, "sha256": meta["table_sha256"]},
                           "rule_note_sha256": rule_sha,
                           "parameters": {"n_sims": args.n_sims, "n_boot": args.n_boot, "sim_seed": pw.SIM_SEED, "fold_seed": ld.FOLD_SEED,
                                          "boot_seed": ld.BOOT_SEED, "margins": list(pw.MARGINS), "power_target": pw.POWER_TARGET,
                                          "noise_draws": pw.NOISE_DRAWS, "scenarios": [s.name for s in pw.SCENARIOS]},
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": {"chosen_margin": decision["chosen_margin"], "decision_power": decision["decision_power"], "design_check": design_check},
              "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
              "files": {name: sha256_file(args.out / name) for name in names}}
    (args.out / "complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
