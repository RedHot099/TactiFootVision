"""The ticket 15 masking read-out (scripts/xgdino_info_mask.py) on synthetic masked scores: absent shots keep the
unmasked z, a z-destroying mask shows as degradation against the unmasked and the placebo reference, a placebo-sized
nuisance does not, the size bins split the test shots in terciles, and the loader refuses a changed reference."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgdino_info_mask as script  # noqa: E402
from test_xgdino_info_protocol import population  # noqa: E402
from xgpoc.dino import ablation as ab  # noqa: E402
from xgpoc.dino import info_protocol as ip  # noqa: E402
from xgpoc.stack import stats as ss  # noqa: E402


@pytest.fixture(scope="module")
def world():
    table, _, y = population(seed=60, n_matches=80, per_match=25)
    arms, _ = ab.arm_set(table, ("A", "C"))
    estimators = {a: script.Estimator(table, arms[a], ab.model_for(arms[a]), y) for a in script.ARMS}
    weights = ss.bootstrap_weights(table.game.to_numpy(), 200, 0)
    return table, y, estimators, weights


def variants(table, rng):
    """Region 'shooter' present on shots 0.. 3n/4 (placebo-feasible on the first half), killing z on its shots; the
    placebos shift z by noise of the same size on a random other subset."""
    n = len(table)
    z0 = table[ab.Z].to_numpy(float)
    present = np.arange(n) < 3 * n // 4
    feasible = np.arange(n) < n // 2
    z = {}
    for fill in ip.FILLS:
        v = z0.copy()
        v[present] = z0.mean()                      # the region carries all the signal
        z[("shooter", fill)] = v
        for k in range(ip.N_PLACEBO):
            p = z0.copy()
            p[feasible] += rng.normal(0, 0.05, feasible.sum())
            z[(f"shooter|placebo{k}", fill)] = p
    facts = {"present": present, "feasible": feasible, "n_patches": np.where(present, 10 + np.arange(n) % 30, np.nan)}
    return z0, z, facts


def test_masked_signal_degrades_against_unmasked_and_placebo(world):
    table, y, estimators, weights = world
    z0, z, facts = variants(table, np.random.default_rng(0))
    losses, ref = script.all_losses(estimators, z0, z, lambda s: None)
    allfacts = {r: facts for r in script.REGIONS}
    losses |= {(a, e, f, r): losses[(a, e, f, "shooter")] for a in script.ARMS for e in ip.ESTIMANDS for f in ip.FILLS
               for r in script.REGIONS if r != "shooter"}
    for r in script.FAMILY:
        for k in range(ip.N_PLACEBO):
            for a in script.ARMS:
                for e in ip.ESTIMANDS:
                    for f in ip.FILLS:
                        losses[(a, e, f, f"{r}|placebo{k}")] = losses[(a, e, f, f"shooter|placebo{k}")]
    rows, out = script.region_rows(losses, ref, allfacts, weights, "A", "frozen", "mean")
    shooter = next(r for r in rows if r["region"] == "shooter")
    assert shooter["test_shots"] == int((facts["present"] & facts["feasible"]).sum())
    assert shooter["vs_unmasked_lo95"] > 0 and shooter["vs_placebo_lo95"] > 0
    assert shooter["p"] == max(shooter["p_vs_unmasked"], shooter["p_vs_placebo"])
    assert out["holm"]["shooter"]


def test_placebo_loss_is_the_mean_over_draws():
    n = 4
    losses = {("C", "refit", "mean", f"x|placebo{k}"): np.full(n, float(k)) for k in range(ip.N_PLACEBO)}
    assert np.allclose(script.placebo_loss(losses, "C", "refit", "mean", "x"), np.mean(range(ip.N_PLACEBO)))


def test_size_bins_split_the_test_shots_in_terciles(world):
    table, y, estimators, weights = world
    z0, z, facts = variants(table, np.random.default_rng(1))
    losses, ref = script.all_losses(estimators, z0, z, lambda s: None)
    for r in script.FAMILY:
        losses |= {(a, e, f, r): losses[(a, e, f, "shooter")] for a in script.ARMS for e in ip.ESTIMANDS for f in ip.FILLS}
        losses |= {(a, e, f, f"{r}|placebo{k}"): losses[(a, e, f, f"shooter|placebo{k}")] for a in script.ARMS
                   for e in ip.ESTIMANDS for f in ip.FILLS for k in range(ip.N_PLACEBO)}
    rows = script.size_bins(losses, ref, {r: facts for r in script.FAMILY}, weights, "C", "refit", "mean")
    shooter = [r for r in rows if r["region"] == "shooter"]
    assert [r["bin"] for r in shooter] == list(script.BINS)
    assert sum(r["shots"] for r in shooter) == int((facts["present"] & facts["feasible"]).sum())
    assert shooter[0]["patches_max"] <= shooter[1]["patches_min"] <= shooter[2]["patches_min"]


def test_loader_keeps_absent_shots_at_the_unmasked_z(tmp_path, monkeypatch):
    index = pd.Index(["a", "b", "c"])
    base = {"fold": 0, "draw": 0, "dx": 0, "dy": 0, "parts": "", "n_patches": 5, "phase": "x", "row": 0, "chunk": 0}
    rows = [{"shot_id": s, "variant": "unmasked", "kind": "unmasked", "region": "", "fill": "none", "z": z} | base
            for s, z in zip(index, (1.0, 2.0, 3.0), strict=True)]
    rows += [{"shot_id": "a", "variant": "shooter", "kind": "region", "region": "shooter", "fill": f, "z": 9.0} | base
             for f in ip.FILLS]
    pd.DataFrame(rows).to_parquet(tmp_path / "scores.parquet")
    pd.DataFrame({"shot_id": index, "z": [1.0, 2.0, 3.0]}).to_parquet(tmp_path / "unmasked.parquet")
    monkeypatch.setattr(script, "MASKED", tmp_path)
    z0, z, facts = script.load_scores(index)
    assert z[("shooter", "mean")].tolist() == [9.0, 2.0, 3.0]
    assert facts["shooter"]["present"].tolist() == [True, False, False]
    pd.DataFrame({"shot_id": index, "z": [1.0, 2.0, 3.5]}).to_parquet(tmp_path / "unmasked.parquet")
    with pytest.raises(SystemExit):
        script.load_scores(index)
