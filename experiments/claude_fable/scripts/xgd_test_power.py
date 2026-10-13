"""Expected power of the frozen test of A1 (xg-detector ticket 06, council C2; CPU, seconds). Development data only.

    ./run.sh python scripts/xgd_test_power.py        # -> results/xgd_test_power/{power.json, complete.json}

The variance is ticket 03's: the per-shot log-loss differences A1 seed 0 - G-combo seed 0, out-of-fold on the 7,062
non-penalty pool shots (328 matches), read from the stored out-of-fold files and checked as ticket 03's report checks
them. Its match-bootstrap standard error (5,000 draws, seed 0, ticket 03's draws) and its analytic cluster standard
error, sqrt(sum over matches of (D_m - n_m d)^2) / N with D_m the match's sum of differences, are both reported; the
bootstrap one is scaled to the test population by sqrt(N_dev / N_test), N the non-penalty shots (the match-count
scaling is reported next to it). The test population's size comes from the index's shot_id, game and is_penalty and
the plan's roles: no label of a test shot is read (everything runs inside the held-out label seal and
`no_statsbomb_reads()`). The probabilities of each reading at Δ = -0.0007 (ticket 03's point) and Δ = -0.002 (the A3
trigger) use the normal approximation of `testrule.expected_power` with the rule's levels.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgd_approach1_report as dev
from xgd_gcombo_test import held_out_shots
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal, testrule
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
DELTAS = (-0.0007, -0.002)
DEV_SEED = 0                      # ticket 03's bootstrap seed


def cluster_se(d: np.ndarray, games: np.ndarray) -> float:
    """Analytic standard error of the shot-weighted mean of `d` with matches as clusters."""
    frame = pd.DataFrame({"d": d, "game": games})
    sums = frame.groupby("game").d.agg(["sum", "size"])
    resid = sums["sum"] - sums["size"] * d.mean()
    return float(np.sqrt((resid**2).sum()) / len(d))


def test_population_size(index: Path, plan: pd.DataFrame, roles: list[str]) -> dict:
    """Non-penalty shots and matches of the plan's test roles, from label-free index columns."""
    shots = held_out_shots(index, plan)
    test = shots[shots.role.isin(roles) & ~shots.is_penalty]
    return {"shots": len(test), "matches": int(test.game.nunique()),
            "shots_by_role": {r: int((test.role == r).sum()) for r in roles}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_test_power")
    args = parser.parse_args()

    with no_statsbomb_reads(), seal.development_labels_only(a1.PLAN_DIR / "plan.csv", args.index):
        rule = testrule.load_rule()
        plan = a1.verify_plan(args.index)
        size = test_population_size(args.index, plan, rule["population"]["roles"])
        if size["shots"] != rule["population"]["expected_shots"] or size["matches"] != rule["population"]["matches"]:
            raise SystemExit(f"test population {size} differs from the rule's")

        shots = pd.read_parquet(args.index, columns=["shot_id", "game", "goal", "is_penalty"])   # training rows only (seal)
        pool = a1.training_pool(shots, plan)
        g0 = dev.out_of_fold(dev.G_COMBO, dev.G_NAME, 0)
        dev.check_out_of_fold(g0, pool, dev.g_temperatures(0))
        a1_oof, temperatures = dev.variant_predictions("A1")
        dev.check_out_of_fold(a1_oof, pool, temperatures)
        population = pool[~pool.is_penalty].sort_values("shot_id").reset_index(drop=True)
        y = population.goal.astype(float).to_numpy()
        games = population.game.to_numpy()

        def aligned(oof: pd.DataFrame) -> np.ndarray:
            return population[["shot_id"]].merge(oof, on="shot_id", how="left", validate="one_to_one").p.to_numpy()

        dev_rule = {**rule, "bootstrap": {**rule["bootstrap"], "seed": DEV_SEED}}
        delta = testrule.paired_delta(y, aligned(a1_oof), aligned(g0), games, dev_rule)
        d = testrule.shot_log_loss(y, aligned(a1_oof), rule["metric"]["clip"]) - testrule.shot_log_loss(y, aligned(g0), rule["metric"]["clip"])
        quoted = pd.read_csv(ROOT / "results/xgd_approach1/comparison.csv").set_index("model").loc["A1"]
        if max(abs(quoted.delta_ll - delta["delta_ll"]), abs(quoted.lo - delta["lo95"]), abs(quoted.hi - delta["hi95"])) > 1e-6:
            raise SystemExit(f"recomputed development comparison {delta} differs from ticket 03's {quoted.to_dict()}")

        n_dev, m_dev = len(population), int(population.game.nunique())
        shot_scale, match_scale = np.sqrt(n_dev / size["shots"]), np.sqrt(m_dev / size["matches"])
        se_test = delta["boot_sd"] * shot_scale
        result = {
            "rule": "the development out-of-fold paired match variance of ticket 03 (A1 seed 0 - G-combo seed 0), scaled to the "
                    "test population by sqrt(N_dev / N_test) shots; normal approximation with the rule's levels",
            "development": {"shots": n_dev, "matches": m_dev, "delta_ll": delta["delta_ll"], "lo95": delta["lo95"],
                            "hi95": delta["hi95"], "se_bootstrap": delta["boot_sd"], "se_cluster": cluster_se(d, games),
                            "bootstrap": {"draws": rule["bootstrap"]["draws"], "seed": DEV_SEED}},
            "test_population": size,
            "scaling": {"shots": float(shot_scale), "matches": float(match_scale)},
            "se_test": float(se_test), "se_test_match_scaling": float(delta["boot_sd"] * match_scale),
            "at": [testrule.expected_power(x, float(se_test), rule) for x in DELTAS],
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "power.json").write_text(json.dumps(result, indent=2) + "\n")
        inputs = {"index": sha256_file(args.index), "plan": a1.PLAN_SHA256, "rule_parameters": testrule.parameters_digest(rule),
                  "ticket03_comparison": sha256_file(ROOT / "results/xgd_approach1/comparison.csv"),
                  "a1_replay": sha256_file(a1.results_dir("A1") / "replay.json"),
                  "g_combo_ensemble": sha256_file(dev.G_COMBO / "ensemble.json")}
        record = {"inputs": inputs, "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
