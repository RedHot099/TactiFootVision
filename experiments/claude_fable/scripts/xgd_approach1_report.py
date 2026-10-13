"""Development comparison of approach 1 against G-combo seed 0, out-of-fold (xg-detector ticket 03; CPU, seconds).

    ./run.sh python scripts/xgd_approach1_report.py                       # baseline and noise floor only
    ./run.sh python scripts/xgd_approach1_report.py --variants A1 A2      # plus the variants (finished, replay passed)

The development baseline is G-combo's stored seed-0 out-of-fold predictions (`results/xgv_g_combo/fold<k>/seed0/
<name>_valid_predictions.csv`, no retraining), checked against the files: every training-pool shot exactly once, in its
plan fold, with the index label, each file matching its completion record, and p = sigmoid(logit / T) with the
member's temperature of ensemble.json. Seed 1's files, checked the same way, give the noise floor.

Population: the non-penalty shots of the plan's 328 training matches (`goal` labels, `is_penalty` excluded), the
same shots for every model. The probabilities are each fold model's temperature-scaled p, as stored. Differences are
variant - G-combo seed 0 in log-loss, with a paired match bootstrap over the 328 matches (5,000 draws, seed 0,
`xgpoc.probe.readability.match_draw_counts`; the same draws for every model). The noise floor is G-combo seed 1 -
seed 0 on the same shots and draws; council C1's figure (seed 0 - seed 1 on all 7,166 pool shots, penalties included)
is recomputed next to it. The A3 trigger (ticket 04) is a point difference <= -0.002 by A1 or A2; a crossing inside the
council's noise band [-0.0032; +0.0023] is flagged noise-compatible. The development-selected variant for the test
(ticket 06) is the one with the lower out-of-fold log-loss.

Writes <out>/{baseline.csv, comparison.csv, summary.json, complete.json}; runs inside `no_statsbomb_reads()`.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.eval.calibration import PlattScaler, logit
from xgpoc.eval.metrics import evaluate
from xgpoc.paths import INDEX
from xgpoc.probe.readability import match_draw_counts

ROOT = Path(__file__).resolve().parents[1]
G_COMBO = ROOT / "results/xgv_g_combo"
G_NAME = "g_finetune_dinov2_l_b4_overlay_combo"
N_BOOT, SEED = 5000, 0
TRIGGER = -0.002
COUNCIL_BAND = (-0.0032, 0.0023)
FOLDS = range(5)


def out_of_fold(root: Path, name: str, seed: int) -> pd.DataFrame:
    """shot_id, goal, logit, p, fold of the five held-out folds of one seed; each file checked against its record."""
    parts = []
    for k in FOLDS:
        path = root / f"fold{k}" / f"seed{seed}" / f"{name}_valid_predictions.csv"
        records.verified_file(path)
        parts.append(pd.read_csv(path).assign(fold=k))
    return pd.concat(parts, ignore_index=True)


def check_out_of_fold(oof: pd.DataFrame, pool: pd.DataFrame, temperatures: dict[int, float] | None = None) -> None:
    """Raise ValueError unless every pool shot appears once, in its plan fold, with the index label, and p is the
    temperature-scaled logit (when `temperatures` per fold are given)."""
    if oof.shot_id.duplicated().any() or set(oof.shot_id) != set(pool.shot_id):
        raise ValueError(f"out-of-fold predictions cover {oof.shot_id.nunique()} shots, the pool has {len(pool)}")
    joined = oof.merge(pool[["shot_id", "goal", "fold"]], on="shot_id", suffixes=("", "_plan"))
    if (joined.fold != joined.fold_plan).any():
        raise ValueError(f"{int((joined.fold != joined.fold_plan).sum())} shots are scored outside their plan fold")
    if (joined.goal.astype(int) != joined.goal_plan.astype(int)).any():
        raise ValueError("stored goal labels differ from the index")
    if temperatures is not None:
        t = joined.fold.map(temperatures)
        if (gap := float(np.max(np.abs(1 / (1 + np.exp(-joined.logit / t)) - joined.p)))) > 1e-6:
            raise ValueError(f"stored p is not sigmoid(logit / T) (max gap {gap:.2e})")


def g_temperatures(seed: int) -> dict[int, float]:
    members = json.loads((G_COMBO / "ensemble.json").read_text())["members"]
    return {m["fold"]: m["temperature"] for m in members if m["seed"] == seed}


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Log-loss, AUC, Brier and the calibration summary: ECE (10 bins), mean p - goal rate, and the logistic
    recalibration slope and intercept of logit(p)."""
    m = evaluate(y, p)
    platt = PlattScaler().fit(logit(np.clip(p, 1e-6, 1 - 1e-6)), y)
    return {"n": m["n"], "goals": int(y.sum()), "log_loss": m["log_loss"], "auc": m["auc"], "brier": m["brier"],
            "ece": m["ece"], "mean_p_minus_rate": float(np.mean(p) - np.mean(y)), "cal_slope": platt.a, "cal_intercept": platt.b}


def shot_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_delta(y: np.ndarray, p: np.ndarray, p_base: np.ndarray, weights: np.ndarray) -> dict:
    """Log-loss difference p - p_base: the point and the 95% percentile interval of the match bootstrap (`weights` (B, n),
    each shot weighted by its match's draw count)."""
    d = shot_log_loss(y, p) - shot_log_loss(y, p_base)
    boot = (weights * d).sum(axis=1) / weights.sum(axis=1)
    return {"delta_ll": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


def trigger(delta: float) -> dict:
    crosses = bool(delta <= TRIGGER)
    return {"crosses": crosses, "noise_compatible": bool(crosses and COUNCIL_BAND[0] <= delta <= COUNCIL_BAND[1])}


def variant_predictions(variant: str) -> tuple[pd.DataFrame, dict[int, float]]:
    """A finished variant's out-of-fold predictions and each fold model's temperature (its training.json); refused
    unless every fold has its record and the replay passed."""
    root = a1.results_dir(variant)
    replay = json.loads((root / "replay.json").read_text())
    if not replay["passed"] or sorted((r["fold"], r["seed"]) for r in replay["runs"]) != [(k, 0) for k in FOLDS]:
        raise ValueError(f"{root}: the replay did not pass on all five seed-0 folds")
    temperatures = {}
    for k in FOLDS:
        training = root / f"fold{k}" / "seed0" / "training.json"
        records.verified_file(training)
        temperatures[k] = json.loads(training.read_text())["temperature"]
    return out_of_fold(root, a1.run_name(variant), 0), temperatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="*", default=[], choices=a1.VARIANTS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_approach1")
    args = parser.parse_args()

    with no_statsbomb_reads():
        plan = a1.verify_plan(args.index)
        shots = pd.read_parquet(args.index, columns=["shot_id", "game", "goal", "is_penalty"])
        pool = a1.training_pool(shots, plan)
        g = {}
        for seed in (0, 1):
            g[seed] = out_of_fold(G_COMBO, G_NAME, seed)
            check_out_of_fold(g[seed], pool, g_temperatures(seed))
        population = pool[~pool.is_penalty].sort_values("shot_id").reset_index(drop=True)
        y = population.goal.astype(float).to_numpy()
        keys, counts = match_draw_counts(population.game.to_numpy(), N_BOOT, SEED)
        weights = counts[:, np.searchsorted(keys, population.game.to_numpy())].astype(float)

        def aligned(oof: pd.DataFrame) -> np.ndarray:
            return population[["shot_id"]].merge(oof, on="shot_id", how="left", validate="one_to_one").p.to_numpy()

        base = aligned(g[0])
        rows = [{"model": "G-combo seed 0", **scores(y, base), "delta_ll": 0.0, "lo": np.nan, "hi": np.nan}]
        rows.append({"model": "noise floor: G-combo seed 1", **scores(y, aligned(g[1])),
                     **paired_delta(y, aligned(g[1]), base, weights)})
        for variant in args.variants:
            oof, temperatures = variant_predictions(variant)
            check_out_of_fold(oof, pool, temperatures)
            rows.append({"model": variant, **scores(y, aligned(oof)), **paired_delta(y, aligned(oof), base, weights)})
        comparison = pd.DataFrame(rows)

        everything = pool.sort_values("shot_id").reset_index(drop=True)
        all_keys, all_counts = match_draw_counts(everything.game.to_numpy(), N_BOOT, SEED)
        all_weights = all_counts[:, np.searchsorted(all_keys, everything.game.to_numpy())].astype(float)
        by_id = {s: everything[["shot_id"]].merge(g[s], on="shot_id", validate="one_to_one").p.to_numpy() for s in (0, 1)}
        council = paired_delta(everything.goal.astype(float).to_numpy(), by_id[0], by_id[1], all_weights)

        baseline = []
        for k in FOLDS:
            in_fold = (population.shot_id.map(g[0].set_index("shot_id").fold) == k).to_numpy()
            baseline.append({"fold": k, "matches": int(population.game[in_fold].nunique()), **scores(y[in_fold], base[in_fold])})
        baseline.append({"fold": "all", "matches": int(population.game.nunique()), **scores(y, base)})
        baseline = pd.DataFrame(baseline)

        variants = comparison[comparison.model.isin(a1.VARIANTS)]
        triggers = {r.model: trigger(r.delta_ll) for r in variants.itertuples()}
        selected = variants.sort_values("log_loss").iloc[0] if len(variants) else None
        summary = {
            "population": {"shots": len(population), "matches": int(population.game.nunique()), "goals": int(y.sum()),
                           "rule": "training-pool shots (plan role training) without penalties"},
            "bootstrap": {"draws": N_BOOT, "seed": SEED, "unit": "match", "interval": "95% percentile"},
            "noise_floor": {"rule": "G-combo seed 1 - seed 0 on the population (the variant - baseline convention)",
                            **comparison.iloc[1][["delta_ll", "lo", "hi"]].to_dict()},
            "council_noise_figure": {"rule": "G-combo seed 0 - seed 1, all pool shots incl. penalties (the council's sign; "
                                             "its band is applied to variant - seed 0 as the council wrote it)",
                                     "shots": len(everything), **council, "council_band": list(COUNCIL_BAND)},
            "a3_trigger": {"threshold": TRIGGER, "variants": triggers,
                           "runs_ticket_04": bool(any(t["crosses"] for t in triggers.values())) if triggers else None},
            "development_selected": None if selected is None else {
                "variant": selected.model, "log_loss": float(selected.log_loss), "delta_ll": float(selected.delta_ll),
                "beats_g_combo_seed0_point": bool(selected.delta_ll < 0),
                "neither_beats_g_combo_seed0_point": bool((variants.delta_ll >= 0).all())},
        }
        args.out.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.out / "comparison.csv", index=False, float_format="%.6f")
        baseline.to_csv(args.out / "baseline.csv", index=False, float_format="%.6f")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        inputs = {"index": sha256_file(args.index), "plan": a1.PLAN_SHA256, "g_combo_ensemble": sha256_file(G_COMBO / "ensemble.json")}
        inputs |= {v: sha256_file(a1.results_dir(v) / "replay.json") for v in args.variants}
        record = {"inputs": inputs, "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(baseline.round(4).to_string(index=False))
        print(comparison.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
