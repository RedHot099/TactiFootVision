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


def prepared(files: dict[str, str]) -> xgs_test.Prepared:
    return xgs_test.Prepared(None, {}, pd.DataFrame(), pd.DataFrame(), {}, files)


def prep_folder(tmp_path, rule: dict, *, passed=True, level=0.95, code=None) -> Path:
    folder = tmp_path / "prep"
    folder.mkdir()
    (folder / "power.json").write_text(json.dumps({"p": 1}))
    (folder / "gate.json").write_text(json.dumps({"level_in_force": level, "passed": True}))
    (folder / "predictions.csv").write_text("shot_id,p_G\na,0.1\n")
    record = {"rule_parameters_sha256": tr.parameters_digest(rule), "all_passed": passed, "level_in_force": level,
              "code": code or {"x.py": "h"}, "files": {p.name: sha256_file(p) for p in folder.iterdir()}}
    (folder / "complete.json").write_text(json.dumps(record))
    return folder


def stated(rule: dict, level=0.95) -> dict:
    return rule | {tr.BEFORE_SCORING: {"power": {"p": 1}, "gate": {"level_in_force": level, "passed": True},
                                       "level_in_force": level}}


def test_the_replay_must_reproduce_the_preparations_files_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(xgs_test, "PREP", tmp_path)
    (tmp_path / "predictions.csv").write_text("a,b\n1,2\n")
    assert xgs_test.replay_problems(prepared({"predictions.csv": "a,b\n1,2\n"})) == []
    assert xgs_test.replay_problems(prepared({"predictions.csv": "a,b\n1,2.0000000001\n"})) == ["predictions.csv differs from the preparation's"]
    assert xgs_test.replay_problems(prepared({"checks.json": "{}"})) == ["checks.json differs from the preparation's"]


def test_the_preparation_must_have_passed_under_this_rule_code_and_level(tmp_path, monkeypatch):
    rule = tr.load_rule()
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
    rule = tr.load_rule()
    monkeypatch.setattr(xgs_test, "PREP", prep_folder(tmp_path, rule, passed=False))
    assert "the preparation did not pass" in xgs_test.prep_problems(stated(rule), {"x.py": "h"})


def test_verify_refuses_another_hash_list_a_changed_file_or_other_code(tmp_path, monkeypatch):
    rule = stated(tr.load_rule())
    monkeypatch.setattr(xgs_test.tr, "load_rule", lambda: rule)
    monkeypatch.setattr(xgs_test, "PREP", prep_folder(tmp_path, rule))
    pinned = tmp_path / "input.csv"
    pinned.write_text("x\n1\n")
    hashlist = tmp_path / "hashes.json"
    hashlist.write_text(json.dumps({"files": {str(pinned): sha256_file(pinned)}, "code": {"x.py": "h"}}))
    monkeypatch.setattr(xgs_test, "HASHLIST", hashlist)
    good = sha256_file(hashlist)
    state = xgs_test.verify(good, prepared({}), {"x.py": "h"}, check_git=False)
    assert state["hashlist_sha256"] == good
    with pytest.raises(xgs_test.Refused, match="hashes to"):
        xgs_test.verify("0" * 64, prepared({}), {"x.py": "h"}, check_git=False)
    with pytest.raises(xgs_test.Refused, match="pinned closure"):
        xgs_test.verify(good, prepared({}), {"x.py": "other"}, check_git=False)
    pinned.write_text("x\n2\n")
    with pytest.raises(xgs_test.Refused, match="input.csv changed"):
        xgs_test.verify(good, prepared({}), {"x.py": "h"}, check_git=False)


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
    rule = stated(tr.load_rule(), 0.99)
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
    hashlist, test, predictions = tmp_path / "hashes.json", tmp_path / "xgs_test", tmp_path / "predictions.csv"
    predictions.write_text("shot_id\n")
    hashlist.write_text(json.dumps({"files": {"x/results/xgs_test_prep/predictions.csv": sha256_file(predictions)}}))
    sha = sha256_file(hashlist)
    with pytest.raises(xgs_test_statsbomb.Refused, match="does not exist"):
        xgs_test_statsbomb.require_test_done(sha, hashlist, test, predictions)
    test.mkdir()
    (test / "result.json").write_text("{}")
    record = {"hashlist_sha256": sha, "files": {"result.json": sha256_file(test / "result.json")}}
    (test / "complete.json").write_text(json.dumps(record))
    xgs_test_statsbomb.require_test_done(sha, hashlist, test, predictions)
    with pytest.raises(xgs_test_statsbomb.Refused, match="does not hash"):
        xgs_test_statsbomb.require_test_done("0" * 64, hashlist, test, predictions)
    (test / "result.json").write_text('{"changed": 1}')
    with pytest.raises(xgs_test_statsbomb.Refused, match="does not match"):
        xgs_test_statsbomb.require_test_done(sha, hashlist, test, predictions)
    (test / "result.json").write_text("{}")
    predictions.write_text("shot_id\nother\n")
    with pytest.raises(xgs_test_statsbomb.Refused, match="not the file the hash list pins"):
        xgs_test_statsbomb.require_test_done(sha, hashlist, test, predictions)


def test_the_statsbomb_frame_joins_every_primary_shot_to_an_evaluation_prediction(monkeypatch):
    monkeypatch.setattr(xgs_test_statsbomb, "PRIMARY_SHOTS", 2)
    shots = pd.DataFrame({"soccernet_shot_id": ["a", "b", None], "in_primary": [True, True, False], "p_vendor": 0.1,
                          "sb_x": 100.0, "sb_y": 40.0, "sb_dist": 20.0, "sb_angle": 0.3})
    predictions = pd.DataFrame({"shot_id": ["a", "b", "c"], "role": ["evaluation", "evaluation", "gate"], "p_R": 0.1,
                                "p_G": 0.1, "p_g_combo": 0.1})
    assert len(xgs_test_statsbomb.primary_frame(shots, predictions)) == 2
    with pytest.raises(xgs_test_statsbomb.Refused, match="no evaluation-match"):
        xgs_test_statsbomb.primary_frame(shots, predictions.assign(role="gate"))
