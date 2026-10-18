"""The owner-ordered method sweep on the 172 held-out matches (2026-10-18), under `research/sweep_2026_10_rule.md`.
Descriptive, not a confirmatory test (see the rule). CPU.

    ./run.sh --no-sync python scripts/sweep_test.py --no-labels --out /tmp/sweep_dry   # label-free dry run
    ./run.sh --no-sync python scripts/sweep_test.py            # predictions, checks, then the labels once
    ./run.sh --no-sync python scripts/sweep_test.py --statsbomb   # StatsBomb reference, after the sweep's result

Default mode: inside `testset.no_vendor_reads()` and the held-out label seal, every model of the rule is fitted on the
7,062 training shots (`final.final_fit`; xg-stack arms from the C6 spec of each E3 cache, position-only arms with every
coefficient penalised), the test tables of both E3 caches are built (`testset.heldout_table`) and every method's test
probability is written to predictions.csv (no label). stack_R/G/GS_old must equal `results/xgs_test_prep/
predictions.csv`. Then, outside the seal (still without a StatsBomb path), the index `goal` of the 3,819 test shots is
read once and every method is scored on the rule's subsets. Writes results/sweep_2026_10/{predictions.csv, fits.json,
seen.csv, scores.csv, deltas.csv, complete.json}.

--statsbomb: refuses unless the sweep's complete.json exists; scores vendor xG and the StatsBomb comparator rungs
beside every method on the paired primary shots (`results/xgc_compare/table/shots.parquet`). Writes
results/sweep_2026_10/statsbomb/{scores.csv, deltas.csv, complete.json}. Data: StatsBomb.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal
from xgpoc.eval.calibration import sigmoid
from xgpoc.paths import INDEX
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack_heldout import final
from xgpoc.stack_heldout import rule as tr
from xgpoc.stack_heldout import testset as ts
from xgs_detector_leak import v3_times
from xgs_run import PROTOCOL_SHA256
from xgs_test import deployed_probability, g_combo_ensemble, read_test_labels

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "research/sweep_2026_10_rule.md"
OUT = ROOT / "results/sweep_2026_10"
NEW_E3 = ROOT / "cache/xgd_full4/E3"
OLD_PREP = ROOT / "results/xgs_test_prep/predictions.csv"
A1_TEST = ROOT / "results/xgd_test/predictions.csv"
SB_TABLE = ROOT / "results/xgc_compare/table/shots.parquet"
POSITIONS_ONLY = sf.Model("inner_cv_all_penalised", sf.C_GRID, False)
WINDOW_MS = 3000
CLIP = 1e-6
STACK_ARMS = {"old": ("R", "G", "GS", "ALL"), "new": ("G", "GS", "Q", "ALL")}


def positions_only(arm: sf.Arm, name: str) -> sf.Arm:
    """`arm` without the G-combo logit (no pixels)."""
    return sf.Arm(name, "descriptive", tuple(c for c in arm.linear if c != sf.Z), arm.spline, arm.df)


def fitted_methods(plan: pd.DataFrame) -> tuple[dict, dict, dict, pd.DataFrame]:
    """(final fits by method, the test tables by cache, training goal rate, the test shots). Inside the seal."""
    protocol = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    shots = ts.heldout_shots(INDEX, plan)
    ensemble, _ = g_combo_ensemble()
    p_deployed = deployed_probability(shots)
    z = pd.Series(ts.invert_platt(p_deployed.to_numpy(), ensemble["calibration"]), index=shots.index)
    annotations = {g: si.read_annotations(g) for g in sorted(shots.game.unique())}
    fits, tables = {}, {}
    rate = None
    for cache, e3 in (("old", si.E3), ("new", NEW_E3)):
        population = si.build_population(e3=e3, plan=plan)
        spec = sf.spec(protocol["design"]["chosen"], population.table)
        y = si.load_outcomes(population).aligned(population)
        rate = float(np.mean(y))
        for arm in STACK_ARMS[cache]:
            if arm != "R" or cache == "old":
                name = "stack_R" if arm == "R" else f"stack_{arm}_{cache}"
                fits[name] = final.final_fit(population.table, population.folds, y, spec.arms[arm], spec.model)
        da = positions_only(spec.arms["G"], f"pos_DA_{cache}")
        fits[da.name] = final.final_fit(population.table, population.folds, y, da, POSITIONS_ONLY)
        if cache == "new":
            full = positions_only(spec.arms["ALL"], "pos_ALL_new")
            fits[full.name] = final.final_fit(population.table, population.folds, y, full, POSITIONS_ONLY)
        tables[cache], _ = ts.heldout_table(shots, z, e3, annotations)
        print(f"{cache} E3: fitted", flush=True)
    return fits, tables, {"rate": rate, "p_g_combo": p_deployed}, shots


def predictions_table(fits: dict, tables: dict, extra: dict, shots: pd.DataFrame) -> pd.DataFrame:
    a1_test = pd.read_csv(A1_TEST).set_index("shot_id").reindex(shots.index)
    if a1_test.p_a1.isna().any() or not np.allclose(a1_test.p_g_combo, extra["p_g_combo"], atol=1e-12, rtol=0):
        raise SystemExit("A1's test predictions do not cover the shots or disagree on G-combo")
    out = pd.DataFrame({"game": shots.game, "role": tables["old"].role, "p_const": extra["rate"],
                        "p_g_combo": extra["p_g_combo"], "p_a1": a1_test.p_a1}, index=shots.index)
    for name, fit in fits.items():
        out[f"p_{name}"] = fit.predict(tables["new" if name.endswith("_new") else "old"])
    old = pd.read_csv(OLD_PREP).set_index("shot_id").reindex(shots.index)
    for arm in ("R", "G", "GS"):
        name = "stack_R" if arm == "R" else f"stack_{arm}_old"
        if not np.array_equal(old[f"p_{arm}"].to_numpy(), out[f"p_{name}"].to_numpy()):
            gap = float(np.abs(old[f"p_{arm}"] - out[f"p_{name}"]).max())
            if gap > 1e-12:
                raise SystemExit(f"{name} does not reproduce the xg-stack test predictions (max |dp| {gap:g})")
    return out


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    q = np.clip(p, CLIP, 1 - CLIP)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    logit = np.log(q / (1 - q))
    slope = sf.logistic_fit(logit[:, None], y, np.zeros(1))[1] if len(np.unique(y)) == 2 else np.nan
    return {"log_loss": float(ll.mean()), "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(np.mean((p - y) ** 2)), "cal_slope": float(slope), "mean_p": float(p.mean())}


def paired_delta(y: np.ndarray, p: np.ndarray, base: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict:
    """ΔLL(p - base) on the shots of `mask` with the rule's paired match bootstrap (`rule.delta`'s replicate); a draw
    that holds none of the subset's matches is left out and counted (small subsets only)."""
    d = tr.shot_log_loss(y, p, CLIP) - tr.shot_log_loss(y, base, CLIP)
    w = weights[:, mask]
    total = w.sum(axis=1)
    boot = (w[total > 0] @ d[mask]) / total[total > 0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_ll": float(d[mask].mean()), "lo95": float(lo), "hi95": float(hi), "draws_used": int(len(boot))}


def score_all(pred: pd.DataFrame, y: np.ndarray, subsets: dict, methods: list[str], base: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule = tr.load_rule()
    weights = tr.bootstrap_weights(pred.game.to_numpy(), rule)
    rows, deltas = [], []
    for subset, mask in subsets.items():
        for m in methods:
            p = pred[f"p_{m}"].to_numpy()
            rows.append({"subset": subset, "method": m, "shots": int(mask.sum()), "goals": int(y[mask].sum()),
                         "matches": int(pred.game[mask].nunique())} | scores(y[mask], p[mask]))
            if m != base:
                deltas.append({"subset": subset, "method": m, "base": base}
                              | paired_delta(y, p, pred[f"p_{base}"].to_numpy(), weights, mask))
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def run_sweep(out: Path = OUT, labels: bool = True) -> None:
    if out.exists():
        raise SystemExit(f"{OUT} exists: the sweep reads its labels once")
    started = datetime.now(UTC)
    v3 = v3_times()
    with ts.no_vendor_reads():
        with seal.development_labels_only(a1.PLAN_DIR / "plan.csv", INDEX):
            plan = a1.verify_plan(INDEX)
            fits, tables, extra, shots = fitted_methods(plan)
            pred = predictions_table(fits, tables, extra, shots)
            seen = pd.DataFrame({"seen": si_seen(tables["old"], v3), "game_in_v3": shots.game.isin(set(v3.game))})
            out.mkdir(parents=True)
            pred.to_csv(out / "predictions.csv", float_format="%.17g")
            seen.to_csv(out / "seen.csv")
            write(out / "fits.json", {name: fit.record() for name, fit in fits.items()})
            print(f"predictions written for {len(pred)} shots; no test label read yet", flush=True)
            if not labels:
                return
        y = read_test_labels(pd.Series(pred.index))
        subsets = {"all": np.ones(len(pred), bool), "evaluation": (pred.role == "evaluation").to_numpy(),
                   "gate": (pred.role == "gate").to_numpy(), "unseen": ~seen.seen.to_numpy(),
                   "no_v3_game": ~seen.game_in_v3.to_numpy()}
        methods = [c[2:] for c in pred.columns if c.startswith("p_")]
        table, deltas = score_all(pred, y, subsets, methods, "g_combo")
    table.to_csv(out / "scores.csv", index=False, float_format="%.6g")
    deltas.to_csv(out / "deltas.csv", index=False, float_format="%.6g")
    write(out / "complete.json", {"rule": {"path": str(RULE.relative_to(ROOT)), "sha256": sha256_file(RULE)},
                                  "inputs": {"new_e3": str(NEW_E3), "old_prep": sha256_file(OLD_PREP), "a1": sha256_file(A1_TEST)},
                                  "git": records.git_state(), "started": started.isoformat(timespec="seconds"),
                                  "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                                  "files": {p.name: sha256_file(p) for p in sorted(out.glob("*")) if p.is_file()}})
    show(table, deltas)


def si_seen(table: pd.DataFrame, v3: pd.DataFrame) -> pd.Series:
    from xgs_detector_leak import seen_shots
    return seen_shots(table, v3, WINDOW_MS)


def run_statsbomb() -> None:
    if not (OUT / "complete.json").exists():
        raise SystemExit("run the sweep first")
    out = OUT / "statsbomb"
    if out.exists():
        raise SystemExit(f"{out} exists")
    pred = pd.read_csv(OUT / "predictions.csv").set_index("shot_id")
    sb = pd.read_parquet(SB_TABLE, columns=["soccernet_shot_id", "population", "z_vendor", "z_positions", "z_events",
                                           "z_dist_angle"])
    sb = sb[sb.population == "primary"].set_index("soccernet_shot_id")
    sb = sb[sb.index.isin(pred.index)]
    pred = pred.loc[sb.index].copy()
    for rung in ("vendor", "positions", "events", "dist_angle"):
        pred[f"p_sb_{rung}"] = sigmoid(sb[f"z_{rung}"].to_numpy(float))
    y = read_test_labels(pd.Series(pred.index))
    methods = [c[2:] for c in pred.columns if c.startswith("p_")]
    table, deltas = score_all(pred, y, {"paired_primary": np.ones(len(pred), bool)}, methods, "g_combo")
    out.mkdir()
    table.to_csv(out / "scores.csv", index=False, float_format="%.6g")
    deltas.to_csv(out / "deltas.csv", index=False, float_format="%.6g")
    write(out / "complete.json", {"data": "Data: StatsBomb.", "table": {"path": str(SB_TABLE), "sha256": sha256_file(SB_TABLE)},
                                  "git": records.git_state(), "files": {p.name: sha256_file(p) for p in sorted(out.glob("*"))}})
    show(table, deltas)


def show(table: pd.DataFrame, deltas: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    merged = table.merge(deltas, on=["subset", "method"], how="left")
    print(merged[["subset", "method", "shots", "goals", "log_loss", "auc", "cal_slope", "delta_ll", "lo95", "hi95"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--statsbomb", action="store_true")
    parser.add_argument("--no-labels", action="store_true", help="label-free dry run: predictions and checks only")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run_statsbomb() if args.statsbomb else run_sweep(args.out, labels=not args.no_labels)


if __name__ == "__main__":
    sys.exit(main())
