"""The gates of the xg-stack test scripts (ticket 04, council C7): the replay and preparation checks, the hash-list
verification, the single label read, the rule's application on synthetic data, and the StatsBomb look's refusal
before the test result exists."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgs_test  # noqa: E402
import xgs_test_statsbomb  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.stack_heldout import rule as tr  # noqa: E402
from xgs_fixtures import synthetic_table  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def unstated() -> dict:
    """The committed rule without its before_scoring block (the preparation writes that block)."""
    return {k: v for k, v in tr.load_rule().items() if k != tr.BEFORE_SCORING}


def prepared(files: dict[str, str]) -> xgs_test.Prepared:
    return xgs_test.Prepared(None, {}, pd.DataFrame(), pd.DataFrame(), {}, files)


def prep_folder(tmp_path, rule: dict, *, passed=True, level=0.95, code=None) -> Path:
    folder = tmp_path / "prep"
    folder.mkdir()
    (folder / "power.json").write_text(json.dumps({"p": 1}))
    (folder / "gate.json").write_text(json.dumps({"level_in_force": level, "passed": True}))
    for name in xgs_test.REPLAYED:
        (folder / name).write_text(name)
    record = {"rule_parameters_sha256": tr.parameters_digest(rule), "all_passed": passed, "level_in_force": level,
              "code": code or {"x.py": "h"}, "files": {p.name: sha256_file(p) for p in folder.iterdir()}}
    (folder / "complete.json").write_text(json.dumps(record))
    return folder


def stated(rule: dict, level=0.95) -> dict:
    return rule | {tr.BEFORE_SCORING: {"power": {"p": 1}, "gate": {"level_in_force": level, "passed": True},
                                       "level_in_force": level}}


def test_the_replay_must_reproduce_the_preparations_files_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(xgs_test, "PREP", tmp_path)
    files = {name: f"{name}\n1,2\n" for name in xgs_test.REPLAYED}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    assert xgs_test.replay_problems(prepared(files)) == []
    changed = files | {"predictions.csv": "predictions.csv\n1,2.0000000001\n"}
    assert xgs_test.replay_problems(prepared(changed)) == ["predictions.csv differs from the preparation's"]
    (tmp_path / "checks.json").unlink()
    assert xgs_test.replay_problems(prepared(files)) == ["checks.json differs from the preparation's"]
    assert "not" in xgs_test.replay_problems(prepared({"checks.json": "{}"}))[0]


def test_the_preparation_must_have_passed_under_this_rule_code_and_level(tmp_path, monkeypatch):
    rule = unstated()
    folder = prep_folder(tmp_path, rule)
    monkeypatch.setattr(xgs_test, "PREP", folder)
    code = {"x.py": "h"}
    assert xgs_test.prep_problems(stated(rule), code) == []
    assert any("other code" in p for p in xgs_test.prep_problems(stated(rule), {"x.py": "g"}))
    assert any("before_scoring" in p for p in xgs_test.prep_problems(stated(rule, 0.99), code))
    assert any("other rule parameters" in p for p in xgs_test.prep_problems(stated(rule) | {"scope": "x"}, code))
    assert any("incomplete" in p for p in xgs_test.prep_problems(rule, code))
    (folder / "power.json").write_text(json.dumps({"p": 2}))
    assert any("does not match" in p for p in xgs_test.prep_problems(stated(rule), code))


def test_a_failed_preparation_is_refused(tmp_path, monkeypatch):
    rule = unstated()
    monkeypatch.setattr(xgs_test, "PREP", prep_folder(tmp_path, rule, passed=False))
    assert "the preparation did not pass" in xgs_test.prep_problems(stated(rule), {"x.py": "h"})


def test_verify_refuses_another_hash_list_a_changed_file_or_other_code(tmp_path, monkeypatch):
    rule = stated(unstated())
    monkeypatch.setattr(xgs_test, "PREP", prep_folder(tmp_path, rule))
    pinned = tmp_path / "input.csv"
    pinned.write_text("x\n1\n")
    hashlist = tmp_path / "hashes.json"
    hashlist.write_text(json.dumps({"files": {str(pinned): sha256_file(pinned)}, "code": {"x.py": "h"}}))
    monkeypatch.setattr(xgs_test, "HASHLIST", hashlist)
    good = sha256_file(hashlist)
    files = {name: name for name in xgs_test.REPLAYED}                    # the preparation's files, as prep_folder wrote them
    state = xgs_test.verify(good, rule, prepared(files), {"x.py": "h"}, check_git=False)
    assert state["hashlist_sha256"] == good
    with pytest.raises(xgs_test.Refused, match="hashes to"):
        xgs_test.verify("0" * 64, rule, prepared(files), {"x.py": "h"}, check_git=False)
    with pytest.raises(xgs_test.Refused, match="pinned closure"):
        xgs_test.verify(good, rule, prepared(files), {"x.py": "other"}, check_git=False)
    with pytest.raises(xgs_test.Refused, match="differs from the preparation"):
        xgs_test.verify(good, rule, prepared(files | {"checks.json": "other"}), {"x.py": "h"}, check_git=False)
    pinned.write_text("x\n2\n")
    with pytest.raises(xgs_test.Refused, match="input.csv changed"):
        xgs_test.verify(good, rule, prepared(files), {"x.py": "h"}, check_git=False)


def test_the_test_labels_are_read_once_in_the_populations_order(tmp_path, monkeypatch):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": ["a", "b", "c"], "goal": [True, False, True], "game": "g"}).to_parquet(index)
    monkeypatch.setattr(xgs_test, "INDEX", index)
    assert xgs_test.read_test_labels(pd.Series(["c", "a", "b"])).tolist() == [1, 1, 0]
    with pytest.raises(xgs_test.Refused, match="no label"):
        xgs_test.read_test_labels(pd.Series(["c", "z"]))


def test_the_rule_is_applied_at_the_level_in_force_with_its_descriptive_readings():
    table = synthetic_table(n_matches=40, per_match=20, seed=2)
    table["role"] = np.where(table.game < "g030", "evaluation", "gate")
    rng = np.random.default_rng(0)
    p_r = 1 / (1 + np.exp(-table.z_g_combo.to_numpy()))
    predictions = pd.DataFrame({"shot_id": table.index, "game": table.game.to_numpy(), "role": table.role.to_numpy(),
                                "p_R": p_r, "p_G": np.clip(p_r * 1.05, 0, 0.99), "p_GS": p_r, "p_g_combo": p_r})
    y = (rng.random(len(table)) < p_r).astype(int)
    rule = stated(unstated(), 0.99)
    result, categories = xgs_test.apply_rule(rule, y, predictions, table)
    primary = result["primary"]
    assert result["level_in_force"] == 0.99 and primary["reading"]["level"] == 0.99
    assert primary["delta"]["draws"] == 5000 and primary["sentence"].startswith("Test na 172 meczach")
    assert result["descriptive"]["GS_vs_R"]["delta_ll"] == 0.0
    by_role = result["descriptive"]["G_vs_R_by_role"]
    assert by_role["evaluation"]["shots"] + by_role["gate"]["shots"] == len(table)
    assert set(categories.arm) == {"R", "G", "GS"} and (categories.label == "hypothesis formed on these matches").all()


def test_check_and_test_need_the_hash_list_and_test_refuses_an_existing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["xgs_test.py", "--check"])
    with pytest.raises(xgs_test.Refused, match="expect-hashlist"):
        xgs_test.main()
    monkeypatch.setattr(xgs_test, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["xgs_test.py", "--test", "--expect-hashlist", "0" * 64])
    with pytest.raises(xgs_test.Refused, match="runs once"):
        xgs_test.main()


def test_the_statsbomb_look_refuses_before_the_test_result_exists(tmp_path):
    hashlist, test = tmp_path / "hashes.json", tmp_path / "xgs_test"
    predictions, code = tmp_path / "prep" / "predictions.csv", tmp_path / "code" / "vendor.py"
    for path, text in ((predictions, "shot_id\n"), (code, "x = 1\n")):
        path.parent.mkdir()
        path.write_text(text)
    hashlist.write_text(json.dumps({"files": {"prep/predictions.csv": sha256_file(predictions),
                                              "code/vendor.py": sha256_file(code)}}))
    sha = sha256_file(hashlist)

    def check(expected: str = sha) -> None:
        xgs_test_statsbomb.require_test_done(expected, hashlist, test, predictions, (code,), check_git=False)

    with pytest.raises(xgs_test_statsbomb.Refused, match="does not exist"):
        check()
    test.mkdir()
    (test / "result.json").write_text("{}")
    record = {"hashlist_sha256": sha, "files": {"result.json": sha256_file(test / "result.json")}}
    (test / "complete.json").write_text(json.dumps(record))
    check()
    with pytest.raises(xgs_test_statsbomb.Refused, match="does not hash"):
        check("0" * 64)
    (test / "result.json").write_text('{"changed": 1}')
    with pytest.raises(xgs_test_statsbomb.Refused, match="does not match"):
        check()
    (test / "result.json").write_text("{}")
    code.write_text("x = 2\n")
    with pytest.raises(xgs_test_statsbomb.Refused, match="vendor.py is not the file the hash list pins"):
        check()
    code.write_text("x = 1\n")
    predictions.write_text("shot_id\nother\n")
    with pytest.raises(xgs_test_statsbomb.Refused, match="predictions.csv is not the file the hash list pins"):
        check()


def test_a_pin_is_matched_by_its_whole_relative_name_not_a_suffix(tmp_path):
    path = tmp_path / "a" / "predictions.csv"
    path.parent.mkdir()
    path.write_text("x")
    digest = sha256_file(path)
    assert xgs_test_statsbomb.pinned({"a/predictions.csv": digest}, path)
    assert not xgs_test_statsbomb.pinned({"xa/predictions.csv": digest}, path)
    assert not xgs_test_statsbomb.pinned({"a/predictions.csv": "0" * 64}, path)


def test_the_before_scoring_block_is_written_once_and_never_replaced(tmp_path):
    path = tmp_path / "rule.json"
    rule, block = unstated(), {"power": {"p": 0.5}, "gate": {"level_in_force": 0.95}, "level_in_force": 0.95}
    assert xgs_test.write_before_scoring(rule, block, path)
    written = json.loads(path.read_text())
    assert written[tr.BEFORE_SCORING] == block and tr.parameters_digest(written) == tr.parameters_digest(rule)
    assert not xgs_test.write_before_scoring(written, block, path)
    with pytest.raises(xgs_test.Refused, match="another before_scoring"):
        xgs_test.write_before_scoring(written, block | {"level_in_force": 0.99}, path)


def g_combo_folders(tmp_path) -> tuple[Path, Path, dict]:
    """A G-combo folder with two members' gate predictions and a test folder, each with completion records."""
    g_combo, g_test = tmp_path / "g_combo", tmp_path / "g_test"
    members = [{"fold": 0, "seed": 0, "temperature": 1.1}, {"fold": 12, "seed": 1, "temperature": 1.2}]
    ensemble = {"members": members, "calibration": {}}
    for m in members:
        run = g_combo / f"fold{m['fold']}" / f"seed{m['seed']}"
        run.mkdir(parents=True)
        name = f"{xgs_test.G_NAME}_gate_predictions.csv"
        pd.DataFrame({"shot_id": ["g1", "g2"], "logit": [-1.0, -2.0], "p": 0.2, "goal": [1, 0]}).to_csv(run / name, index=False)
        (run / "complete.json").write_text(json.dumps({"files": {name: sha256_file(run / name)}}))
    (g_combo / "ensemble.json").write_text(json.dumps(ensemble))
    g_test.mkdir()
    pd.DataFrame({"shot_id": ["a", "b", "c"], "game": ["m1", "m1", "m2"], "is_penalty": [False, False, True],
                  "p": [0.1, 0.2, 0.7]}).to_csv(g_test / "predictions.csv", index=False)
    pd.DataFrame({"shot_id": ["u1"], "f0s0": [-1.5], "f12s1": [-1.4]}).to_csv(g_test / "unpaired_members.csv", index=False)
    record = {"inputs": {"ensemble_json": sha256_file(g_combo / "ensemble.json")},
              "files": {n: sha256_file(g_test / n) for n in ("predictions.csv", "unpaired_members.csv")}}
    (g_test / "complete.json").write_text(json.dumps(record))
    return g_combo, g_test, ensemble


def test_g_combos_inputs_are_checked_against_their_records(tmp_path):
    g_combo, g_test, ensemble = g_combo_folders(tmp_path)
    assert xgs_test.g_combo_ensemble(g_combo, g_test)[0] == ensemble
    shots = pd.DataFrame({"game": ["m1", "m1"]}, index=pd.Index(["b", "a"], name="shot_id"))
    assert xgs_test.deployed_probability(shots, g_test).tolist() == [0.2, 0.1]
    with pytest.raises(xgs_test.Refused, match="miss 1"):
        xgs_test.deployed_probability(pd.DataFrame({"game": ["m1"]}, index=["z"]), g_test)
    with pytest.raises(xgs_test.Refused, match="disagree"):
        xgs_test.deployed_probability(pd.DataFrame({"game": ["m2"]}, index=["c"]), g_test)
    logits, temperatures = xgs_test.member_logits(ensemble, g_combo, g_test)
    assert temperatures == {"f0s0": 1.1, "f12s1": 1.2} and "goal" not in logits
    assert logits.loc["u1"].tolist() == [-1.5, -1.4] and logits.loc["g2", "f12s1"] == -2.0
    (g_combo / "ensemble.json").write_text(json.dumps(ensemble | {"name": "other"}))
    with pytest.raises(xgs_test.Refused, match="not the one"):
        xgs_test.g_combo_ensemble(g_combo, g_test)
    (g_test / "predictions.csv").write_text("changed")
    with pytest.raises(ValueError, match="completion record"):
        xgs_test.deployed_probability(shots, g_test)


def test_the_dry_run_scores_synthetic_outcomes_without_a_stated_level():
    table = synthetic_table(n_matches=40, per_match=5, seed=3)
    table["role"] = np.where(table.game < "g020", "evaluation", "gate")
    p = 1 / (1 + np.exp(-table.z_g_combo.to_numpy()))
    predictions = pd.DataFrame({"shot_id": table.index, "game": table.game.to_numpy(), "role": table.role.to_numpy(), "p_R": p,
                                "p_G": p, "p_GS": p, "p_g_combo": p})
    xgs_test.dry_run(unstated(), xgs_test.Prepared(None, {}, table, predictions, {}, {}))


def test_the_statsbomb_frame_joins_every_primary_shot_to_an_evaluation_prediction(monkeypatch):
    monkeypatch.setattr(xgs_test_statsbomb, "PRIMARY_SHOTS", 2)
    shots = pd.DataFrame({"soccernet_shot_id": ["a", "b", None], "in_primary": [True, True, False], "p_vendor": 0.1,
                          "sb_x": 100.0, "sb_y": 40.0, "sb_dist": 20.0, "sb_angle": 0.3})
    predictions = pd.DataFrame({"shot_id": ["a", "b", "c"], "role": ["evaluation", "evaluation", "gate"], "p_R": 0.1,
                                "p_G": 0.1, "p_g_combo": 0.1})
    assert len(xgs_test_statsbomb.primary_frame(shots, predictions)) == 2
    with pytest.raises(xgs_test_statsbomb.Refused, match="no evaluation-match"):
        xgs_test_statsbomb.primary_frame(shots, predictions.assign(role="gate"))
