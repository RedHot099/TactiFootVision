"""The retrain runner's split and GPU guard, and the gate report (xg-vendor ticket 10); CPU only, no model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vendor_retrain_g import foreign_gpu_jobs, plan_split, smoke_plan  # noqa: E402
from xgpoc.vendor.retrain_gate import gate_report  # noqa: E402
from xgpoc.vendor.split_check import gate_passes  # noqa: E402


@pytest.fixture
def plan():
    return pd.DataFrame({"game": ["a", "b", "c", "d", "e", "f"],
                         "role": ["training", "training", "training", "gate", "evaluation", "gate"],
                         "fold": [0, 1, 1, -1, -1, -1]})


def shots(games):
    return pd.DataFrame({"shot_id": [f"{g}{i}" for g in games for i in range(2)], "game": [g for g in games for _ in range(2)]})


def test_pool_is_the_training_matches_with_their_fold_and_gate_the_gate_matches(plan):
    pool, gate = plan_split(shots(plan.game), plan)
    assert sorted(pool.game.unique()) == ["a", "b", "c"] and pool.groupby("game").fold.first().to_dict() == {"a": 0, "b": 1, "c": 1}
    assert sorted(gate.game.unique()) == ["d", "f"] and gate.index.tolist() == list(range(len(gate)))
    assert not {"e"} & (set(pool.game) | set(gate.game))  # an evaluation shot is in neither


def test_a_match_the_plan_does_not_list_is_refused(plan):
    with pytest.raises(ValueError, match="not in the plan"):
        plan_split(shots([*plan.game, "zz"]), plan)


def test_smoke_plan_has_gate_and_five_training_folds():
    index = pd.DataFrame({"game": [f"g{i:02d}" for i in range(40)], "split": ["test" if i % 4 == 0 else "train" for i in range(40)]})
    out = smoke_plan(index)
    assert (out.role == "gate").sum() == 10 and set(out.fold[out.role == "training"]) == {0, 1, 2, 3, 4}
    assert (out.fold[out.role == "gate"] == -1).all()


def test_only_a_large_foreign_compute_process_counts_as_a_job():
    csv = "100, 210\n200, 6000\n300, 150\n999, 8000\nnot-a-pid, 9999"
    assert foreign_gpu_jobs(csv, own_pid=999) == ["200, 6000"]
    assert foreign_gpu_jobs("", own_pid=1) == []


def planted(n_matches=40, per_match=20, seed=0):
    rng = np.random.default_rng(seed)
    matches = np.repeat(np.arange(n_matches), per_match)
    q = rng.uniform(0.02, 0.5, len(matches))
    y = (rng.random(len(matches)) < q).astype(int)
    return y, q, matches, rng


def test_identical_models_pass_with_a_zero_interval():
    y, q, matches, _ = planted()
    r = gate_report(y, q, q, matches, n_boot=200)
    assert r["passed"] and r["n_matches"] == 40
    for m in r["metrics"].values():
        assert m["delta"] == 0 and m["bootstrap"]["lo95"] == 0 == m["bootstrap"]["hi95"]


def test_a_broken_retrain_fails_and_the_interval_excludes_zero():
    y, q, matches, rng = planted()
    broken = rng.uniform(0.02, 0.5, len(y))  # noise instead of the signal
    r = gate_report(y, broken, q, matches, n_boot=300)
    assert not r["passed"] and r["decision"]["log_loss"] is False and r["decision"]["auc"] is False
    assert r["metrics"]["log_loss"]["bootstrap"]["lo95"] > 0 and r["metrics"]["auc"]["bootstrap"]["hi95"] < 0


def test_the_decision_uses_point_estimates_and_the_plan_tolerance():
    y, q, matches, _ = planted()
    r = gate_report(y, np.clip(q * 1.02, 0, 1), q, matches, n_boot=100)
    d = r["metrics"]
    assert r["decision"] == gate_passes(d["log_loss"]["delta"], d["auc"]["delta"])
