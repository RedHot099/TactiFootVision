"""The 10-model A1 ensemble's calibration and its out-of-fold comparison with the 10-model G-combo (xg-detector 06; CPU).

    ./run.sh python scripts/xgd_a1_ensemble.py       # -> results/xgd_a1_ensemble/{ensemble.json, oof.csv, comparison.json, complete.json}

Members: the ten A1 runs of results/xgd_a1 (folds 0-4 x seeds 0, 1), each checked against its completion record, all
with the same recipe, inputs and constants, and refused unless `replay_ensemble.json` passed on all ten.

Calibration, exactly as G-combo's `ensemble.json` was made (`scripts/vendor_verify_g.py`): each pool shot's out-of-fold
ensemble logit is the mean of logit(stored p) of the two models that held it out; Platt is fitted on all 7,166 pool
shots (penalties included), the prior correction goes from the out-of-fold goal rate to the pool's, and penalties are
routed to the pool's penalty rate (`g_bundle.fit_ensemble_calibration`). The same procedure refitted on G-combo's stored
out-of-fold files must give G-combo's own Platt coefficients (checked). `check_max_abs_dp_vs_oof_predictions` compares
`g_bundle.ensemble_probability` on the raw member logits with the calibrated stored-p path.

Comparison (descriptive, development): on the 7,062 non-penalty pool shots, both arms' calibrated out-of-fold ensemble
probabilities, ΔLL = A1 - G-combo with the paired match bootstrap (5,000 draws, seed 0), AUC and the calibration summary.
Platt is fitted on the same out-of-fold predictions it scores, for both arms alike. The standard error of this Δ, scaled
to the test population as the rule's power is, gives a sensitivity figure for the power. Runs inside the held-out label
seal and `no_statsbomb_reads()`.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgd_approach1_report as dev
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal, testrule
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.eval.calibration import logit, penalty_rate, sigmoid
from xgpoc.models.g_bundle import ensemble_probability, fit_ensemble_calibration
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = [(k, s) for k in range(5) for s in (0, 1)]
REPLAY = "replay_ensemble.json"
POWER = ROOT / "results/xgd_test_power"
SHARED_IDENTITY = ("index", "folds", "plan", "overlay_sha256", "frames", "sanitised", "variant", "name", "backbone", "cfg",
                   "rrc", "backbone_lr_mult", "constants", "smoke")
PLATT_TOLERANCE = 1e-6


def member_records(root: Path) -> dict[tuple[int, int], dict]:
    """(fold, seed) -> completion record of each A1 run, every file checked, the shared identity equal across runs
    (the seed aside), and the replay of all ten passed."""
    out = {}
    for k, s in MEMBERS:
        run = root / f"fold{k}" / f"seed{s}"
        record = json.loads((run / "complete.json").read_text())
        for name, digest in record["files"].items():
            if sha256_file(run / name) != digest:
                raise ValueError(f"{run / name} does not match its completion record")
        ident = record["identity"]
        if (ident["fold"], ident["seed"], ident["variant"]) != (k, s, "A1"):
            raise ValueError(f"{run} holds fold {ident['fold']} seed {ident['seed']} {ident['variant']}")
        out[k, s] = record
    cfgs = [{**r["identity"]["cfg"], "seed": None} for r in out.values()]
    for key in SHARED_IDENTITY:
        values = cfgs if key == "cfg" else [r["identity"][key] for r in out.values()]
        if any(v != values[0] for v in values):
            raise ValueError(f"the A1 runs differ in {key}")
    replay = json.loads((root / REPLAY).read_text())
    ran = {(r["fold"], r["seed"]): r["bundle_sha256"] for r in replay["runs"]}
    if not replay["passed"] or ran != {key: r["bundle"]["sha256"] for key, r in out.items()}:
        raise ValueError(f"{root / REPLAY} did not pass on exactly the ten bundles")
    return out


def stored_out_of_fold(root: Path, name: str, temperatures: dict[tuple[int, int], float]) -> dict[tuple[int, int], pd.DataFrame]:
    """(fold, seed) -> held-out predictions (shot_id, goal, logit, p, z = logit(p)), checked against their records and
    p = sigmoid(logit / T)."""
    runs = {}
    for (k, s), t in temperatures.items():
        path = root / f"fold{k}" / f"seed{s}" / f"{name}_valid_predictions.csv"
        records.verified_file(path)
        frame = pd.read_csv(path)
        if (gap := float(np.max(np.abs(sigmoid(frame.logit / t) - frame.p)))) > 1e-6:
            raise ValueError(f"{path}: p is not sigmoid(logit / T) (max gap {gap:.2e})")
        runs[k, s] = frame.assign(z=logit(frame.p.to_numpy()), fold=k)
    return runs


def ensemble_out_of_fold(runs: dict, pool: pd.DataFrame) -> pd.DataFrame:
    """shot_id -> mean z of the models that held the shot out (one per seed), refused unless every pool shot is held out
    by exactly one model per seed, in its plan fold."""
    frame = pd.concat(list(runs.values()), ignore_index=True)
    per_shot = frame.groupby("shot_id").agg(z=("z", "mean"), models=("z", "size"), fold=("fold", "first"))
    seeds = len({s for _, s in runs})
    if set(per_shot.index) != set(pool.shot_id) or (per_shot.models != seeds).any():
        raise ValueError("the out-of-fold predictions do not cover the pool once per seed")
    if (per_shot.fold != pool.set_index("shot_id").fold.loc[per_shot.index]).any():
        raise ValueError("a shot is scored outside its plan fold")
    return per_shot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_a1_ensemble")
    args = parser.parse_args()

    with no_statsbomb_reads(), seal.development_labels_only(a1.PLAN_DIR / "plan.csv", args.index):
        rule = testrule.load_rule()
        plan = a1.verify_plan(args.index)
        shots = pd.read_parquet(args.index, columns=["shot_id", "game", "goal", "is_penalty"])   # training rows only (seal)
        pool = a1.training_pool(shots, plan).sort_values("shot_id").reset_index(drop=True)
        root = a1.results_dir("A1")
        member = member_records(root)
        temperatures = {key: float(json.loads((root / f"fold{key[0]}" / f"seed{key[1]}" / "training.json").read_text())["temperature"])
                        for key in MEMBERS}
        runs = stored_out_of_fold(root, a1.run_name("A1"), temperatures)
        oof = ensemble_out_of_fold(runs, pool)
        labels = pool.set_index("shot_id").goal.astype(int)
        pens = pool[pool.is_penalty]
        p_pen, pooled_rate = penalty_rate(int(pens.goal.sum()), len(pens)), float(pool.goal.mean())
        constants = member[0, 0]["identity"]["constants"]
        if not (np.isclose(constants["penalty_rate"], p_pen) and np.isclose(constants["pooled_goal_rate"], pooled_rate)):
            raise ValueError(f"the runs' constants {constants} differ from the pool's ({p_pen}, {pooled_rate})")
        calibration = fit_ensemble_calibration(oof.z.to_numpy(), labels.loc[oof.index].to_numpy(), pooled_rate, p_pen)
        is_pen = pool.set_index("shot_id").is_penalty.astype(bool).loc[oof.index].to_numpy()

        def calibrated(z: np.ndarray, cal: dict, penalty: np.ndarray) -> np.ndarray:
            """The deployed path on an already averaged logit (one member, temperature 1)."""
            return ensemble_probability([z], [1.0], cal, penalty)

        p_a1 = calibrated(oof.z.to_numpy(), calibration, is_pen)
        by_shot = {key: runs[key].set_index("shot_id") for key in MEMBERS}
        penalty_of, p_of = pool.set_index("shot_id").is_penalty.astype(bool), pd.Series(p_a1, index=oof.index)
        gaps = []
        for k in range(5):                           # the deployed path on the raw member logits of each held-out fold
            ids = by_shot[k, 0].index
            q = ensemble_probability([by_shot[k, s].logit.loc[ids].to_numpy() for s in (0, 1)], [temperatures[k, s] for s in (0, 1)],
                                     calibration, penalty_of.loc[ids].to_numpy())
            gaps.append(float(np.max(np.abs(q - p_of.loc[ids].to_numpy()))))
        calibration["check_max_abs_dp_vs_oof_predictions"] = max(gaps)

        g_ensemble = json.loads((dev.G_COMBO / "ensemble.json").read_text())
        g_temps = {(m["fold"], m["seed"]): m["temperature"] for m in g_ensemble["members"]}
        g_runs = stored_out_of_fold(dev.G_COMBO, dev.G_NAME, g_temps)
        g_oof = ensemble_out_of_fold(g_runs, pool)
        g_refit = fit_ensemble_calibration(g_oof.z.to_numpy(), labels.loc[g_oof.index].to_numpy(), pooled_rate, p_pen)
        gap = max(abs(g_refit["platt"][c] - g_ensemble["calibration"]["platt"][c]) for c in ("a", "b"))
        if gap > PLATT_TOLERANCE or g_refit["n_oof"] != g_ensemble["calibration"]["n_oof"]:
            raise ValueError(f"the procedure refitted on G-combo's out-of-fold files does not give its ensemble.json (gap {gap:.2e})")
        p_g = calibrated(g_oof.z.loc[oof.index].to_numpy(), g_ensemble["calibration"], is_pen)

        members = [{"fold": k, "seed": s, "bundle": str((root / f"fold{k}" / f"seed{s}" / "bundle.pt").relative_to(ROOT)),
                    "temperature": temperatures[k, s], "sha256": member[k, s]["bundle"]["sha256"]} for k, s in MEMBERS]
        ensemble = {"name": a1.run_name("A1"), "variant": "A1", "plan": {"sha256": a1.PLAN_SHA256}, "members": members,
                    "calibration": calibration,
                    "procedure": "as results/xgv_g_combo/ensemble.json (scripts/vendor_verify_g.py): Platt on the mean logit(stored p) "
                                 "of the pool's out-of-fold predictions, prior correction, penalty routing"}

        non_pen = ~is_pen
        y = labels.loc[oof.index].to_numpy()[non_pen]
        games = pool.set_index("shot_id").game.loc[oof.index].to_numpy()[non_pen]
        dev_rule = {**rule, "bootstrap": {**rule["bootstrap"], "seed": 0}}
        delta = testrule.paired_delta(y, p_a1[non_pen], p_g[non_pen], games, dev_rule)
        power = json.loads((POWER / "power.json").read_text())
        se_test = delta["boot_sd"] * power["scaling"]["shots"]
        comparison = {
            "population": {"shots": int(non_pen.sum()), "matches": int(len(np.unique(games))), "goals": int(y.sum()),
                           "rule": "non-penalty pool shots, out-of-fold, each arm's calibrated 10-model ensemble"},
            "a1_10": dev.scores(y, p_a1[non_pen]), "g_combo_10": dev.scores(y, p_g[non_pen]), "delta": delta,
            "g_combo_platt_refit_gap": gap,
            "sensitivity_power": {"rule": "this Δ's bootstrap SE scaled to the test population as the rule's power",
                                  "se_test": float(se_test),
                                  "at": [testrule.expected_power(x, float(se_test), rule) for x in (-0.0007, -0.002)]},
            "note": "descriptive development figures; Platt is fitted on the out-of-fold predictions it scores, for both arms",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ensemble.json").write_text(json.dumps(ensemble, indent=2) + "\n")
        pd.DataFrame({"shot_id": oof.index, "fold": oof.fold.to_numpy(), "z_a1": oof.z.to_numpy(), "p_a1": p_a1,
                      "z_g_combo": g_oof.z.loc[oof.index].to_numpy(), "p_g_combo": p_g, "is_penalty": is_pen}).to_csv(
            args.out / "oof.csv", index=False, float_format="%.9g")
        (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float) + "\n")
        inputs = {"index": sha256_file(args.index), "plan": a1.PLAN_SHA256, "rule_parameters": testrule.parameters_digest(rule),
                  "a1_replay": sha256_file(root / REPLAY), "g_combo_ensemble": sha256_file(dev.G_COMBO / "ensemble.json"),
                  "power": sha256_file(POWER / "power.json"),
                  "a1_records": {f"f{k}s{s}": sha256_file(root / f"fold{k}" / f"seed{s}" / "complete.json") for k, s in MEMBERS}}
        record = {"inputs": inputs, "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(ensemble["calibration"], indent=2))
    print(json.dumps(comparison, indent=2, default=float))


if __name__ == "__main__":
    main()
