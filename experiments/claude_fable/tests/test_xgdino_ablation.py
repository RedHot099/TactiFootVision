"""The xg-dino A/B/C ablation (ticket 01, council C1) at its entry points on synthetic populations: geometry carrying
all signal, permuted labels, shared folds and recalibration, penalties by column name, the geometry allowlist, the
decision table, the leak verdict on a planted heterogeneity, the label-free controls, the StatsBomb guard and the run
script's gates."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgdino_ablation  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.compare.stats import permute_within_matches  # noqa: E402
from xgpoc.detector.guard import StatsBombRead  # noqa: E402
from xgpoc.dino import ablation as ab  # noqa: E402
from xgpoc.dino import controls as dc  # noqa: E402
from xgpoc.dino import records  # noqa: E402
from xgpoc.eval.calibration import sigmoid  # noqa: E402
from xgpoc.stack import fit as sf  # noqa: E402
from xgpoc.stack import inputs as si  # noqa: E402
from xgpoc.stack import provenance as sp  # noqa: E402
from xgs_fixtures import outcomes, standardised_distance, synthetic_table  # noqa: E402

DRAWS = 200


def inputs_of(table: pd.DataFrame, seed: int = 0) -> ab.Inputs:
    rng = np.random.default_rng(seed)
    games = table.game.to_numpy()
    first = np.isin(games, np.unique(games)[: len(np.unique(games)) // 4])
    return ab.Inputs(si.Population(table, {}, {}), None, {"no_v3": first, "unseen": rng.random(len(table)) < 0.5})


def geometry_outcomes(table: pd.DataFrame, beta: float = 1.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(len(table)) < sigmoid(-1.8 + beta * standardised_distance(table))).astype(int)


@pytest.fixture(scope="module")
def geometry_run():
    table = synthetic_table(n_matches=100, per_match=30, seed=40)
    inputs = inputs_of(table)
    prepared = ab.prepare(inputs, draws=DRAWS)
    return table, inputs, prepared, ab.read_out(prepared, geometry_outcomes(table))


def test_geometry_carrying_all_signal_gives_c_minus_b_near_0_and_b_minus_a_below_0(geometry_run):
    primary = geometry_run[3]["summary"]["blocks"]["primary"]
    c_b, b_a, c_a = (primary["contrasts"][k]["log_loss"] for k in ("C-B", "B-A", "C-A"))
    assert abs(c_b["delta"]) < 0.002 and c_b["lo95"] < 0.001 and c_b["hi95"] > -0.001
    assert b_a["hi95"] < 0 and b_a["p_two_sided"] < 0.01 and c_a["hi95"] < 0
    assert primary["holm_superiority"] == {"C-B": False, "C-A": True}
    decision = geometry_run[3]["summary"]["decision"]
    assert decision["C-A"] == "geometry adds over the image" and decision["C-B"] != "DINOv2 adds"


def test_permuted_labels_give_no_holm_significant_contrast():
    table = synthetic_table(n_matches=80, per_match=25, seed=41)
    y = outcomes(table, beta=1.0, seed=42)
    permuted = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), 0).goal.to_numpy()
    result = ab.read_out(ab.prepare(inputs_of(table), draws=DRAWS), permuted)
    verdict = ab.dry_run_check(result["summary"])
    assert verdict["no_holm_rejection"] and verdict["holm_rejected"] == [] and not verdict["b_minus_a_95_excludes_0"]


def test_every_arm_shares_the_outer_and_inner_folds_and_a_is_platt_per_fold(geometry_run):
    table, _, prepared, result = geometry_run
    block = prepared.blocks["primary"]
    folds = ab.fold_record(block)
    outer = {n: [{k: f[k] for k in ("fold", "train", "test")} for f in v] for n, v in folds.items()}
    assert all(v == outer["A"] for v in outer.values())
    inner = [[f["inner"] for f in v] for n, v in folds.items() if n != "A"]
    assert all(i == inner[0] for i in inner) and len(inner[0][0]) == 4
    y, z, fold = geometry_outcomes(table), table[sf.Z].to_numpy(), table.fold.to_numpy()
    platt = np.zeros(len(y))
    for k in np.unique(fold):
        model = LogisticRegression(C=1e10, tol=1e-12, max_iter=1000).fit(z[fold != k, None], y[fold != k])
        platt[fold == k] = model.predict_proba(z[fold == k, None])[:, 1]
    np.testing.assert_allclose(result["probabilities"]["primary"].p_A, platt, atol=1e-6)
    assert all(m.grid == ab.C_GRID for m in block.models.values()) and ab.C_GRID[-4:] == (3.0, 10.0, 30.0, 100.0)
    edges = result["summary"]["blocks"]["primary"]["grid_edge"]
    assert edges["A"] == [None] * 5 and all(isinstance(e, bool) for e in edges["C"])


def test_penalties_follow_the_column_names():
    table = synthetic_table(n_matches=20, per_match=10, seed=43)
    arms, _ = ab.arm_set(table)
    for name in ("B_DA", "B", "B_legacy"):
        assert (ab.model_for(arms[name]).penalties(len(arms[name].columns), 0.1) == 10.0).all()
    lam = ab.model_for(arms["C"]).penalties(len(arms["C"].columns), 0.1)
    assert lam[arms["C"].columns.index(sf.Z)] == 0.0 and (np.delete(lam, arms["C"].columns.index(sf.Z)) == 10.0).all()
    moved = sf.Arm("moved", "primary", ("angle@0", "shot_missing@0", sf.Z), ("dist@0",), 3)
    assert ab.model_for(moved).penalties(len(moved.columns), 1.0).tolist() == [1, 1, 0, 1, 1, 1]
    assert not ab.model_for(arms["A"]).penalised
    with pytest.raises(ValueError):
        ab.model_for(moved).penalties(3, 1.0)


def test_b_and_c_are_geometry_only_and_no_outcome_annotation_changes_an_input():
    table = synthetic_table(n_matches=30, per_match=10, seed=44)
    arms, _ = ab.arm_set(table)
    for name in ("B_DA", "B", "C"):
        assert not set(arms[name].inputs) & {*si.S_FLAGS, *ab.QUALITY}
        assert set(arms[name].inputs) - {sf.Z} <= set(ab.GEOMETRY)
    assert {"ball_none@0", "kp_instance_conf@0", "n_players@-400", "shooter_by_net@0", "shot_from_ball@0"} <= set(ab.QUALITY)
    assert {"gk_depth@-400", "ff_opp_3m@0", "shot_x@0"} <= set(arms["B"].inputs)
    assert set(ab.QUALITY) - set(arms["BQ"].inputs) <= set(sf.constant_columns(table, ab.QUALITY))
    assert set(si.S_FLAGS) <= set(arms["C_legacy"].inputs) and sf.Z in arms["CQ"].inputs
    shuffled = table.copy()
    shuffled[si.S_FLAGS] = 1.0 - shuffled[si.S_FLAGS]
    shuffled[list(ab.QUALITY)] = shuffled[list(ab.QUALITY)].sample(frac=1.0, random_state=0).to_numpy()
    for name in ("B_DA", "B", "C"):
        a = sf.fold_designs(table, table.fold.to_numpy(), arms[name])
        b = sf.fold_designs(shuffled, table.fold.to_numpy(), arms[name])
        assert all(np.array_equal(x.x_test, w.x_test) for x, w in zip(a, b, strict=True))
    shots = pd.DataFrame({"game": "g", "half": [1, 1, 2], "position_ms": [10000, 50000, 30000]})
    raw = [{"gameTime": "1 - 00:10", "label": "Shots on target", "position": "10000", "team": "home"},
           {"gameTime": "1 - 00:40", "label": "Corner", "position": "40000", "team": "home"},
           {"gameTime": "1 - 00:50", "label": "Shots off target", "position": "50000", "team": "home"},
           {"gameTime": "2 - 00:30", "label": "Shots off target", "position": "30000", "team": "away"}]
    sp.check_flag_code(shots, {"g": raw})       # adds Goal annotations, swaps shot types: flags unchanged
    with pytest.raises(sp.ProvenanceError):
        ab.prepare(inputs_of(table.assign(goal=0)), draws=10)


def test_subsets_are_read_refitted_and_restricted_with_goal_and_non_goal_splits(geometry_run):
    _, inputs, prepared, result = geometry_run
    blocks = result["summary"]["blocks"]
    for name in ab.SUBSETS:
        refit, restricted = blocks[f"refit:{name}"], blocks[f"restricted:{name}"]
        assert refit["shots"] == restricted["shots"] == int(inputs.subsets[name].sum())
        for read in (refit, restricted):
            assert set(ab.CONTRASTS) <= set(read["contrasts"]) and set(read["holm_superiority"]) == set(ab.SUPERIORITY)
            for c in ab.CONTRASTS:
                ll = read["contrasts"][c]["log_loss"]
                assert {"brier", "auc"} <= set(read["contrasts"][c]) and {"tost", "goals", "non_goals", "standardised"} <= set(ll)
        np.testing.assert_array_equal(result["probabilities"][f"refit:{name}"].shot_id,
                                      prepared.shot_id[inputs.subsets[name]])
    assert {"restricted:v3", "restricted:seen", "restricted:no_v3", "restricted:unseen"} <= set(blocks)
    assert "BQ-B" in blocks["restricted:no_v3"]["contrasts"]          # the quality-leak diagnostic
    summary = result["summary"]
    assert set(summary["heterogeneity"]) == set(ab.HETEROGENEITY) and summary["leak"]["verdict"] in ("decidable", "not decidable")
    # restricted = the primary probabilities on the subset's shots: the goal-rate-standardised ΔLL recombines the split
    ll = blocks["restricted:unseen"]["contrasts"]["C-B"]["log_loss"]
    rate = summary["goal_rate"]
    assert ll["standardised"]["delta"] == pytest.approx(rate * ll["goals"]["delta"] + (1 - rate) * ll["non_goals"]["delta"])


def test_the_leak_verdict_refuses_a_planted_heterogeneity():
    table = synthetic_table(n_matches=100, per_match=30, seed=46)
    inputs = inputs_of(table, seed=1)
    seen = ~inputs.subsets["unseen"]
    rng = np.random.default_rng(2)
    z = table[sf.Z].to_numpy()
    logit = -1.8 + 1.5 * standardised_distance(table) + np.where(seen, 2.0 * (z - z.mean()), 0.0)
    y = (rng.random(len(z)) < sigmoid(logit)).astype(int)
    summary = ab.read_out(ab.prepare(inputs, draws=DRAWS), y)["summary"]
    assert summary["heterogeneity"]["seen-unseen"]["hi95"] < 0
    assert summary["leak"]["verdict"] == "not decidable" and "seen-unseen" in summary["leak"]["heterogeneity_excluding_0"]
    clean = {"seen-unseen": {"lo95": -0.001, "hi95": 0.001}, "v3-no_v3": {"lo95": -0.002, "hi95": 0.0005}}
    assert ab.leak_verdict(-0.001, {"lo95": -0.004, "hi95": 0.0}, clean)["verdict"] == "decidable"
    assert ab.leak_verdict(-0.001, {"lo95": -0.0005, "hi95": 0.002}, clean)["verdict"] == "not decidable"


def test_the_decision_table_follows_c1():
    def ll(upper, equivalent, margin=0.0031):
        return {"upper_one_sided_95": upper, "tost": {"equivalent": equivalent, "smallest_margin": margin}}
    adds = ab.decision(ll(-0.001, False), ll(-0.01, False), {"C-B": True, "C-A": True})
    assert (adds["C-B"], adds["C-A"]) == ("DINOv2 adds", "geometry adds over the image")
    assert adds["flags"]["C-B"] == {"upper_below_0": True, "holm_rejected": True, "tost_pass": False}
    raw = ab.decision(ll(-0.001, True), ll(0.0, False), {"C-B": False, "C-A": False})
    assert raw["C-B"].startswith("equivalent") and raw["flags"]["C-B"] == {"upper_below_0": True, "holm_rejected": False,
                                                                           "tost_pass": True}
    assert ab.decision(ll(0.001, True), ll(0.0, False), {"C-B": False, "C-A": False})["C-B"] == \
        "equivalent for this stacker at δ = 0.002"
    wording = ab.decision(ll(-0.001, False), ll(0.0, False), {"C-B": False, "C-A": False})
    assert wording["C-B"] == "inconclusive at δ = 0.002; smallest margin shown = 0.0031"
    assert wording["C-A"] == "no detectable gain of geometry over the image"


def test_the_controls_build_label_free_truths_and_replicate_deterministically():
    table = synthetic_table(n_matches=60, per_match=25, seed=47)
    prepared = ab.prepare(inputs_of(table), draws=100, primary_arms=dc.ARMS, full=False)
    truths = dc.Truths.build(prepared.blocks["primary"], table)
    assert dc.marginal(truths.g).mean() == pytest.approx(dc.GOAL_RATE)
    assert abs(truths.oracle(0.0)) < 1e-9 and np.abs(truths.x_b.T @ truths.r / len(truths.r)).max() < 1e-8
    beta = truths.beta()
    # calibrated under the clustered generator (C2): the marginal over the match effect, not sigmoid(g + βr)
    assert beta > 0 and truths.oracle(beta) == pytest.approx(dc.PLANT_TARGET, abs=1e-8)
    assert abs(truths.oracle(beta, clustered=False) - dc.PLANT_TARGET) > 1e-7
    rng = np.random.default_rng(0)
    eta = truths.g[:50]
    mc = sigmoid(eta[:, None] + rng.normal(0, dc.CLUSTER_SD, (1, 200_000))).mean(1)
    np.testing.assert_allclose(dc.marginal(eta), mc, atol=2e-4)
    block = prepared.blocks["primary"]
    serial = dc.simulate(block, truths.clustered(beta), 2, 7, jobs=1)
    parallel = dc.simulate(block, truths.clustered(beta), 2, 7, jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)
    pd.testing.assert_frame_equal(dc.simulate(block, truths.clustered(beta), 2, 7, start=1).reset_index(drop=True),
                                  serial.iloc[1:].reset_index(drop=True))
    report = dc.summary(serial, truths.oracle(beta))
    assert report["replicates"] == 2 and {"rate_DINOv2", "rate_equivalent", "rate_inconclusive"} <= set(report)
    assert set(report["raw"]) == {"upper_below_0", "holm_rejected", "tost_pass"}
    assert set(serial["wording_C-B"]) <= {"DINOv2", "equivalent", "inconclusive"}
    with pytest.raises(ValueError, match="exactly once"):
        dc.finish({"replicates": 3, "oracle_plant_clustered": -0.002, "beta_plant": beta},
                  pd.concat([serial.assign(scenario=n) for n in dc.SCENARIOS]))


def test_wilson_intervals_and_the_gates():
    lo, hi = dc.wilson(10, 2000)
    assert lo == pytest.approx(0.002718, abs=1e-5) and hi == pytest.approx(0.009180, abs=1e-5)
    assert dc.wilson(0, 100)[0] == pytest.approx(0.0, abs=1e-15)
    passing = {"null": {"rate_DINOv2": {"rate": 0.05}}, "plant": {"rate_equivalent": {"rate": 0.01}}}
    failing = {"null": {"rate_DINOv2": {"rate": 0.01}}, "plant": {"rate_equivalent": {"rate": 0.051}}}
    assert dc.gates(passing)["passed"] and not dc.gates(failing)["passed"]


def test_a_statsbomb_read_is_refused_in_the_inputs_and_in_the_read_out(tmp_path, monkeypatch):
    planted = tmp_path / "statsbomb" / "seen.csv"
    planted.parent.mkdir()
    planted.write_text("shot_id,seen\n")
    with pytest.raises(StatsBombRead):
        ab.load_inputs(ab.Paths(seen=planted))
    table = synthetic_table(n_matches=20, per_match=10, seed=45)
    prepared = ab.prepare(inputs_of(table), draws=10)
    original = sf.out_of_fold

    def leaky(*args, **kwargs):
        pd.read_parquet(tmp_path / "xgv_statsbomb_data" / "shots.parquet")
        return original(*args, **kwargs)
    monkeypatch.setattr(sf, "out_of_fold", leaky)
    with pytest.raises(StatsBombRead):
        ab.read_out(prepared, outcomes(table))


def unfilled(obj, path="") -> list[str]:
    """The paths of every None or TODO value."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in unfilled(v, f"{path}.{k}")]
    if isinstance(obj, list | tuple):
        return [p for i, v in enumerate(obj) for p in unfilled(v, f"{path}[{i}]")]
    return [path] if obj is None or (isinstance(obj, str) and "TODO" in obj) else []


def fake_controls(folder: Path, digest: str, passed: bool = True) -> None:
    folder.mkdir(parents=True)
    scenario = {"mean_half_width_90_C-B": 0.0005, "mean_C-B": 0.0001}
    record = {"files": {}, "code": {"package_digest": digest},
              "controls": {"seeds": dc.SEEDS, "truth": {"beta_plant": 0.2}, "scenarios": {"null": scenario, "plant": scenario},
                           "gates": {"passed": passed}}}
    (folder / "complete.json").write_text(json.dumps(record))


def test_the_pinned_protocol_is_accepted_and_any_edit_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(xgdino_ablation, "ROOT", tmp_path)
    table = synthetic_table(n_matches=20, per_match=10, seed=48)
    inputs = inputs_of(table)
    code = records.package_hashes()
    fake_controls(tmp_path / "controls", code["digest"])
    record = {"frozen": xgdino_ablation.frozen_part(inputs, tmp_path / "controls", code["digest"]), "inputs": inputs.hashes,
              "counts": inputs.counts, "code": code}
    assert unfilled(record["frozen"]) == []
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(record, indent=2, default=float))
    pinned = sha256_file(path)
    assert xgdino_ablation.checked_protocol(path, inputs, pinned, need_pin=True)[1] == pinned
    path.write_text(path.read_text().replace('"delta": 0.002', '"delta": 0.003', 1))
    with pytest.raises(SystemExit, match="not pinned or does not match"):
        xgdino_ablation.checked_protocol(path, inputs, pinned, need_pin=True)
    with pytest.raises(SystemExit, match="frozen part differs"):
        xgdino_ablation.checked_protocol(path, inputs, sha256_file(path), need_pin=True)
    fake_controls(tmp_path / "failed", code["digest"], passed=False)
    with pytest.raises(SystemExit, match="gate failed"):
        xgdino_ablation.frozen_part(inputs, tmp_path / "failed", code["digest"])
    with pytest.raises(SystemExit, match="other code"):
        xgdino_ablation.frozen_part(inputs, tmp_path / "controls", "0" * 64)


def test_the_real_run_refuses_without_the_pinned_protocol(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    assert len(xgdino_ablation.PROTOCOL_SHA256) == 64          # pinned in ticket 01b
    for pin in (None, "0" * 64):
        with pytest.raises(SystemExit, match="not pinned"):
            xgdino_ablation.checked_protocol(protocol, None, pin, need_pin=True)
    with pytest.raises(SystemExit, match="does not exist"):
        xgdino_ablation.checked_protocol(tmp_path / "missing.json", None, None, need_pin=False)


def test_the_package_digest_covers_the_import_closure_and_ignores_the_pin(tmp_path):
    hashes = records.package_hashes()["files"]
    for name in ("xgpoc/dino/ablation.py", "xgpoc/stack/fit.py", "xgpoc/stack_location/inputs.py",
                 "xgpoc/detector/guard.py", "xgpoc/vendor/ladder.py", "xgpoc/compare/stats.py"):
        assert name in hashes
    assert "xgpoc/dino/layers.py" not in hashes and "xgpoc/dino/controls.py" in hashes
    script = Path(records.EXPERIMENT / records.RUN_SCRIPT).read_text()
    pinned = tmp_path / "pinned.py"
    pinned.write_text(script.replace("PROTOCOL_SHA256: str | None = None", 'PROTOCOL_SHA256: str | None = "ab"'))
    edited = tmp_path / "edited.py"
    edited.write_text(script + "\n# edit\n")
    original = records.run_script_sha256(records.EXPERIMENT / records.RUN_SCRIPT)
    assert records.run_script_sha256(pinned) == original != records.run_script_sha256(edited)
