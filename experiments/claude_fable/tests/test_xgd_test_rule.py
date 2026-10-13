"""The frozen test rule (xg-detector 06): its arithmetic, the power computation, the A1 ensemble and G-combo test
preparation helpers, and the test script's refusals and label order."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import testrule
from xgpoc.detector.seal import HeldOutLabelRead, held_out_labels_sealed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xgd_a1_ensemble as ens  # noqa: E402
import xgd_gcombo_test as gtest  # noqa: E402
import xgd_test  # noqa: E402
import xgd_test_power as power  # noqa: E402

RULE = testrule.load_rule()


# ------------------------------------------------------------------ the rule's parameters
def test_the_committed_rule_holds_council_c2s_values_and_the_power(tmp_path):
    assert RULE["variant"] == "A1" and RULE["bootstrap"] == {**RULE["bootstrap"], "draws": 5000, "seed": 0}
    assert (RULE["success"]["interval"], RULE["equivalence"]["interval"], RULE["equivalence"]["margin"]) == (0.95, 0.90, 0.005)
    assert RULE["population"]["roles"] == ["evaluation", "gate"] and RULE["population"]["matches"] == 172
    assert RULE["metric"]["clip"] == 1e-6 and RULE["population"]["plan_sha256"] == a1.PLAN_SHA256
    assert RULE["scope"] == ("The result applies to the D3 detector and homography as audited in ticket 09, not to "
                             "detector-derived structure in general.")
    assert RULE["power"] is not None and [x["delta"] for x in RULE["power"]["at"]] == [-0.0007, -0.002]
    changed = tmp_path / "rule.json"
    changed.write_text(json.dumps({**RULE, "equivalence": {**RULE["equivalence"], "margin": 0.01}}))
    with pytest.raises(ValueError, match="equivalence.margin"):
        testrule.load_rule(changed)


@pytest.mark.skipif(not (ROOT / "results/xgd_test_power/power.json").exists(), reason="needs the power record")
def test_the_rule_quotes_the_power_record():
    record = json.loads((ROOT / "results/xgd_test_power/power.json").read_text())
    assert RULE["power"]["at"] == record["at"] and RULE["power"]["se_test"] == record["se_test"]


# ------------------------------------------------------------------ arithmetic
def test_the_shot_log_loss_clips_at_the_rules_clip():
    out = testrule.shot_log_loss(np.array([1, 0, 1]), np.array([0.0, 0.0, 0.5]), 1e-6)
    assert out == pytest.approx([-np.log(1e-6), -np.log(1 - 1e-6), np.log(2)])
    with pytest.raises(ValueError):
        testrule.shot_log_loss(np.array([0.5]), np.array([0.5]), 1e-6)


def test_the_paired_delta_is_shot_weighted_and_its_intervals_nest():
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(40), rng.integers(5, 30, 40)).astype(str)
    y = (rng.random(len(games)) < 0.15).astype(float)
    base = np.clip(rng.beta(2, 10, len(games)), 0.01, 0.99)
    better = np.where(y == 1, base + 0.02, base - 0.005).clip(0.01, 0.99)
    d = testrule.paired_delta(y, better, base, games, RULE)
    expected = (testrule.shot_log_loss(y, better, 1e-6) - testrule.shot_log_loss(y, base, 1e-6)).mean()
    assert d["delta_ll"] == pytest.approx(expected) and d["ll_variant"] - d["ll_base"] == pytest.approx(expected)
    assert d["lo95"] <= d["lo90"] <= d["delta_ll"] <= d["hi90"] <= d["hi95"] and d["matches"] == 40
    same = testrule.paired_delta(y, base, base, games, RULE)
    assert (same["delta_ll"], same["lo95"], same["hi95"], same["boot_sd"]) == (0.0, 0.0, 0.0, 0.0)
    weights = testrule.match_weights(games, 5000, 0)
    assert weights.shape == (5000, len(games)) and (weights[:, games == games[0]] == weights[:, [0]]).all()


@pytest.mark.parametrize(("lo95", "hi95", "lo90", "hi90", "label"), [
    (-0.004, -0.001, -0.0035, -0.0015, "success and equivalent"),
    (-0.009, -0.001, -0.008, -0.002, "success"),
    (0.001, 0.004, 0.0015, 0.0035, "worse"),
    (-0.003, 0.002, -0.0025, 0.0015, "equivalent within δ"),
    (-0.007, 0.002, -0.0055, 0.0015, "inconclusive"),
    (-0.002, 0.0055, -0.001, 0.005, "inconclusive"),          # the margin is exclusive
])
def test_the_reading_labels_follow_the_frozen_precedence(lo95, hi95, lo90, hi90, label):
    out = testrule.reading({"lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90}, RULE)
    assert out["label"] == label and out["label"] in testrule.LABELS
    assert out["smallest_equivalence_margin"] == pytest.approx(max(abs(lo90), abs(hi90)))
    assert out["success"] == (hi95 < 0) and out["worse"] == (lo95 > 0)


def test_the_expected_power_is_the_normal_approximation():
    at_zero = testrule.expected_power(0.0, 0.001, RULE)
    assert at_zero["p_success"] == pytest.approx(0.025, abs=1e-4) == pytest.approx(at_zero["p_worse"], abs=1e-4)
    small, large = (testrule.expected_power(x, 0.0017, RULE) for x in (-0.0007, -0.002))
    assert small["p_success"] < large["p_success"] < 0.5 and small["p_equivalent"] > large["p_equivalent"]
    assert small["half_width_95"] == pytest.approx(1.959964 * 0.0017, rel=1e-5)


def test_the_cluster_standard_error_counts_matches_as_clusters():
    d, games = np.array([1.0, 1.0, -1.0, -1.0]), np.array(["a", "a", "b", "b"])
    assert power.cluster_se(d, games) == pytest.approx(np.sqrt(2 * 2.0**2) / 4)


def test_the_test_population_size_reads_no_label(tmp_path):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": list("abcde"), "game": ["T", "E", "E", "G", "X"], "goal": [1, 1, 0, 0, 1],
                  "is_penalty": [False, False, True, False, False]}).to_parquet(index)
    plan = pd.DataFrame({"game": ["T", "E", "G", "X"], "role": ["training", "evaluation", "gate", "other"]})
    with held_out_labels_sealed(["T"], ["a"]):
        size = power.test_population_size(index, plan, ["evaluation", "gate"])
        with pytest.raises(HeldOutLabelRead):
            pd.read_csv(_csv(tmp_path, {"shot_id": ["b"], "goal": [1]}))
    assert size == {"shots": 2, "matches": 2, "shots_by_role": {"evaluation": 1, "gate": 1}}


def _csv(tmp_path: Path, columns: dict) -> Path:
    path = tmp_path / "t.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ the A1 ensemble and the G-combo preparation
def test_the_out_of_fold_ensemble_needs_one_model_per_seed_in_the_plan_fold():
    pool = pd.DataFrame({"shot_id": ["a", "b"], "fold": [0, 1]})
    runs = {(0, 0): pd.DataFrame({"shot_id": ["a"], "z": [1.0], "fold": [0]}), (0, 1): pd.DataFrame({"shot_id": ["a"], "z": [3.0], "fold": [0]}),
            (1, 0): pd.DataFrame({"shot_id": ["b"], "z": [0.0], "fold": [1]}), (1, 1): pd.DataFrame({"shot_id": ["b"], "z": [2.0], "fold": [1]})}
    assert ens.ensemble_out_of_fold(runs, pool).z.to_dict() == {"a": 2.0, "b": 1.0}
    with pytest.raises(ValueError, match="once per seed"):
        ens.ensemble_out_of_fold({k: v for k, v in runs.items() if k != (1, 1)}, pool)
    with pytest.raises(ValueError, match="outside its plan fold"):
        ens.ensemble_out_of_fold(runs, pool.assign(fold=[1, 1]))


def test_the_held_out_shots_are_read_without_labels(tmp_path):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": ["b", "a", "c"], "game": ["E", "G", "T"], "goal": [1, 0, 1], "label": ["x"] * 3,
                  "is_penalty": [False, True, False], "frame_path": ["/f"] * 3}).to_parquet(index)
    plan = pd.DataFrame({"game": ["E", "G", "T"], "role": ["evaluation", "gate", "training"]})
    with held_out_labels_sealed(["T"], ["c"]):
        shots = gtest.held_out_shots(index, plan)
    assert shots.shot_id.tolist() == ["a", "b"] and not {"goal", "label"} & set(shots.columns)


# ------------------------------------------------------------------ the test script: refusals
def _frozen(tmp_path: Path, pinned: Path) -> Path:
    hashlist = tmp_path / "hashes.json"
    hashlist.write_text(json.dumps({"files": {str(pinned): sha256_file(pinned)}, "code": {"x.py": "1"}, "frames": {}}))
    return hashlist


def test_the_test_refuses_a_hash_list_other_than_the_committed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(xgd_test, "variant_problems", lambda rule, population: [])
    pinned = tmp_path / "bundle.pt"
    pinned.write_bytes(b"weights")
    hashlist = _frozen(tmp_path, pinned)
    with pytest.raises(xgd_test.Refused, match="hashes to"):
        xgd_test.verify(hashlist, "0" * 64, None, None, check_git=False)
    state = xgd_test.verify(hashlist, sha256_file(hashlist), None, None, check_git=False)
    assert state["hashlist_sha256"] == sha256_file(hashlist)
    with pytest.raises(xgd_test.Refused, match="not the committed"):
        xgd_test.verify(hashlist, sha256_file(hashlist), None, None, check_git=True)   # an untracked file


def test_the_test_refuses_a_changed_pinned_file_or_code(tmp_path, monkeypatch):
    monkeypatch.setattr(xgd_test, "variant_problems", lambda rule, population: [])
    pinned = tmp_path / "bundle.pt"
    pinned.write_bytes(b"weights")
    hashlist = _frozen(tmp_path, pinned)
    with pytest.raises(xgd_test.Refused, match="loaded code is not the pinned closure"):
        xgd_test.verify(hashlist, sha256_file(hashlist), {"x.py": "2"}, None, check_git=False)
    pinned.write_bytes(b"retrained")
    with pytest.raises(xgd_test.Refused, match="bundle.pt changed"):
        xgd_test.verify(hashlist, sha256_file(hashlist), None, None, check_git=False)
    pinned.unlink()
    with pytest.raises(xgd_test.Refused, match="bundle.pt changed"):
        xgd_test.verify(hashlist, sha256_file(hashlist), None, None, check_git=False)


def test_the_test_refuses_incomplete_variant_records(tmp_path, monkeypatch):
    a1_root = tmp_path / "xgd_a1"
    for k, s in ens.MEMBERS[:-1]:                                    # fold 4 seed 1 is missing
        (a1_root / f"fold{k}" / f"seed{s}").mkdir(parents=True)
        (a1_root / f"fold{k}" / f"seed{s}" / "complete.json").write_text(json.dumps(
            {"identity": {"fold": k, "seed": s, "variant": "A1"}, "files": {}, "bundle": {"sha256": "x"}}))
    monkeypatch.setattr(a1, "results_dir", lambda variant: a1_root)
    monkeypatch.setattr(xgd_test.dev, "G_COMBO", tmp_path / "missing_g_combo")
    problems = xgd_test.variant_problems({**RULE, "power": None}, None)
    assert [p.split(":")[0] for p in problems] == ["A1 records incomplete", "G-combo records incomplete", "the rule states no power"]
    pinned = tmp_path / "rule.md"
    pinned.write_text("rule")
    hashlist = _frozen(tmp_path, pinned)
    with pytest.raises(xgd_test.Refused, match="A1 records incomplete"):
        xgd_test.verify(hashlist, sha256_file(hashlist), None, None, check_git=False)


def test_the_test_runs_once_and_needs_the_expected_hash(tmp_path, monkeypatch):
    def gpu():
        raise AssertionError("the refusal comes before the GPU check")
    monkeypatch.setattr(xgd_test, "check_gpu_free", gpu)
    monkeypatch.setattr(sys, "argv", ["xgd_test.py"])
    with pytest.raises(xgd_test.Refused, match="expect-hashlist"):
        xgd_test.main()
    monkeypatch.setattr(xgd_test, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["xgd_test.py", "--expect-hashlist", "0" * 64])
    with pytest.raises(xgd_test.Refused, match="runs once"):
        xgd_test.main()


@pytest.mark.skipif(not xgd_test.HASHLIST.exists(), reason="needs the frozen hash list")
def test_the_committed_hash_list_pins_every_input_the_ticket_names():
    frozen = json.loads(xgd_test.HASHLIST.read_text())
    names = set(frozen["files"])
    for needle in ("research/xg_detector_test_rule.md", "research/xg_detector_test_rule.json", "results/xgd_a1_ensemble/ensemble.json",
                   "results/xgv_g_combo/ensemble.json", "results/xgd_gcombo_test/predictions.csv",
                   "cache/xgd_sanitised/D3/sanitised.parquet", "results/xgv_split_plan/plan.csv", "results/xgd_a1/replay_ensemble.json"):
        assert any(n.endswith(needle) for n in names), needle
    assert sum(n.endswith("bundle.pt") for n in names) == 20
    assert "experiments/claude_fable/scripts/xgd_test.py" in frozen["code"]
    assert "experiments/claude_fable/xgpoc/detector/testrule.py" in frozen["code"]
    assert frozen["frames"]["count"] == RULE["population"]["expected_shots"]


# ------------------------------------------------------------------ the test script: labels only after the predictions
def test_the_test_reads_labels_once_after_writing_the_predictions_and_outside_the_seal():
    tree = ast.parse((ROOT / "scripts/xgd_test.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    label_strings = {n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in ("goal", "label", "on_target")}
    inside_reader = {id(n) for n in ast.walk(functions["read_test_labels"])}
    assert label_strings and all(id(n) in inside_reader for n in label_strings), "only read_test_labels names a label"
    main = functions["main"]
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    readers = [n for n in calls if ast.unparse(n.func) == "read_test_labels"]
    seal_blocks = [n for n in ast.walk(main) if isinstance(n, ast.With) and "development_labels_only" in ast.unparse(n.items[0].context_expr)]
    assert len(readers) == 1 and len(seal_blocks) == 1
    assert id(readers[0]) not in {id(n) for n in ast.walk(seal_blocks[0])}
    writes = [n for n in calls if ast.unparse(n.func).endswith((".to_csv", ".write_text")) and "predictions" in ast.unparse(n)]
    assert writes and all(w.lineno < readers[0].lineno for w in writes)
    scoring = [n for n in calls if ast.unparse(n.func) in ("a1_predictions", "g_combo_predictions")]
    assert scoring and all(id(n) in {id(m) for m in ast.walk(seal_blocks[0])} for n in scoring)
