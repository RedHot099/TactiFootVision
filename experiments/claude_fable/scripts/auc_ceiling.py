"""AUC ceiling if StatsBomb's xG were the true goal probability (2026-10-20 chat result, persisted 2026-10-22).
Descriptive. Data: StatsBomb. CPU.

    ./run.sh --no-sync python scripts/auc_ceiling.py [--out results/auc_ceiling_2026_10]

Shots: the StatsBomb-paired primary shots (`results/xgc_compare/table/shots.parquet`, population "primary") among the
3,819 shots of the 172 held-out matches (`results/sweep_2026_10/predictions.csv`), in the StatsBomb table's order:
2,721 shots. p = sigmoid(z_vendor). Each of 2,000 draws (numpy default_rng, seed 0) samples outcomes y ~ Bernoulli(p)
and computes AUC(y, p): the AUC a perfect model of these shots would reach if vendor xG were the truth. The observed
vendor AUC uses the index `goal` of the same shots (`xgs_test.read_test_labels`, as `feet_vendor.py`; the 172 matches
are spent).

Writes <out>/simulated_auc.csv (one AUC per draw), summary.json and complete.json (input and output hashes), last.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.paths import INDEX
from xgs_test import read_test_labels

ROOT = Path(__file__).resolve().parents[1]
SB_TABLE = ROOT / "results/xgc_compare/table/shots.parquet"
HELDOUT = ROOT / "results/sweep_2026_10/predictions.csv"
OUT = ROOT / "results/auc_ceiling_2026_10"
DRAWS, SEED = 2000, 0


def paired_vendor() -> pd.Series:
    """Vendor probability of the paired primary held-out shots, index soccernet_shot_id, in the StatsBomb table's order."""
    held_out = pd.read_csv(HELDOUT, usecols=["shot_id"]).shot_id
    sb = pd.read_parquet(SB_TABLE, columns=["soccernet_shot_id", "population", "z_vendor"])
    sb = sb[sb.population == "primary"].set_index("soccernet_shot_id")
    sb = sb[sb.index.isin(held_out)]
    return 1 / (1 + np.exp(-sb.z_vendor.astype(float)))


def simulate(p: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """AUC(y, p) for `draws` outcome vectors y ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    return np.array([roc_auc_score(rng.random(len(p)) < p, p) for _ in range(draws)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    vendor = paired_vendor()
    p = vendor.to_numpy()
    auc = simulate(p)
    y = read_test_labels(pd.Series(vendor.index))
    observed = float(roc_auc_score(y, p))
    lo, hi = np.percentile(auc, [2.5, 97.5])
    summary = {"label": "descriptive; Data: StatsBomb; the 172 held-out matches are spent",
               "shots": len(p), "goals": int(y.sum()), "mean_vendor_xg": float(p.mean()),
               "vendor_xg_percentiles": dict(zip(["p10", "p50", "p90", "p99"], np.percentile(p, [10, 50, 90, 99]).tolist(),
                                                 strict=True)),
               "share_p_above_0_5": float((p > 0.5).mean()), "share_p_below_0_1": float((p < 0.1).mean()),
               "simulated_auc": {"draws": DRAWS, "seed": SEED, "mean": float(auc.mean()), "lo95": float(lo),
                                 "hi95": float(hi), "sd": float(auc.std(ddof=1))},
               "observed_vendor_auc": observed,
               "share_of_draws_at_or_below_observed": float((auc <= observed).mean()),
               "chat_2026_10_20": {"mean": 0.80, "ci95": [0.77, 0.83], "observed": 0.793}}
    args.out.mkdir(parents=True)
    pd.DataFrame({"draw": np.arange(DRAWS), "auc": auc}).to_csv(args.out / "simulated_auc.csv", index=False,
                                                                 float_format="%.6f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    script = Path(__file__).resolve()
    complete = {"inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                           for name, path in (("statsbomb_table", SB_TABLE), ("heldout_shots", HELDOUT), ("index", INDEX))},
                "code": {str(script.relative_to(ROOT)): sha256_file(script),
                         "scripts/xgs_test.py": sha256_file(ROOT / "scripts/xgs_test.py")},
                "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"),
                        "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("shots", "goals", "simulated_auc", "observed_vendor_auc",
                                              "share_of_draws_at_or_below_observed")}, indent=1))


if __name__ == "__main__":
    main()
