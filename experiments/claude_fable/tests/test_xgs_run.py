"""The xg-stack frozen files and the run script's gates (ticket 01, councils C5 and C6): `--real` refuses without
passed controls under the same protocol and code, the files refuse a wrong hash or drifted code, a re-freeze refuses a
changed rule, and the plan and design transcribe C6."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgs_protocol  # noqa: E402
import xgs_run  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.stack import controls as sc  # noqa: E402
from xgpoc.stack import fit as sf  # noqa: E402
from xgpoc.stack import protocol as pr  # noqa: E402
from xgpoc.stack import records  # noqa: E402
from xgs_fixtures import synthetic_table  # noqa: E402


def controls_folder(tmp_path, *, passed=True, protocol="p" * 64, digest="d" * 64) -> Path:
    folder = tmp_path / "controls"
    folder.mkdir()
    (folder / "controls.json").write_text("{}")
    summary = {name: {"passed": passed or name != "4_positive"} for name in xgs_run.CONTROLS}
    record = {"protocol_sha256": protocol, "code": {"package_digest": digest}, "controls": summary,
              "all_passed": all(c["passed"] for c in summary.values()),
              "files": {"controls.json": sha256_file(folder / "controls.json")}}
    (folder / "complete.json").write_text(json.dumps(record))
    return folder


def test_the_real_run_needs_passed_controls_under_the_same_protocol_and_code(tmp_path):
    folder = controls_folder(tmp_path)
    assert xgs_run.check_controls(folder, "p" * 64, "d" * 64)["all_passed"]
    with pytest.raises(SystemExit, match="another protocol"):
        xgs_run.check_controls(folder, "q" * 64, "d" * 64)
    with pytest.raises(SystemExit, match="other code"):
        xgs_run.check_controls(folder, "p" * 64, "e" * 64)
    (folder / "controls.json").write_text('{"edited": 1}')
    with pytest.raises(SystemExit, match="does not match"):
        xgs_run.check_controls(folder, "p" * 64, "d" * 64)


def test_the_real_run_refuses_a_failed_control_or_missing_controls(tmp_path):
    with pytest.raises(SystemExit, match="run --controls first"):
        xgs_run.check_controls(tmp_path / "nothing", "p" * 64, "d" * 64)
    folder = controls_folder(tmp_path, passed=False)
    with pytest.raises(SystemExit, match="not every control passed"):
        xgs_run.check_controls(folder, "p" * 64, "d" * 64)


def test_a_frozen_file_is_refused_without_a_pinned_hash_with_a_wrong_hash_or_with_other_code(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"code": records.code_hashes()}))
    with pytest.raises(SystemExit, match="no SHA-256 is pinned"):
        xgs_run.opened(path, None)
    with pytest.raises(SystemExit, match="SHA-256"):
        xgs_run.opened(path, "0" * 64)
    assert xgs_run.opened(path, sha256_file(path))["code"]["digest"] == records.code_hashes()["digest"]
    path.write_text(json.dumps({"code": {"digest": "0" * 64}}))
    with pytest.raises(SystemExit, match="code differs"):
        xgs_run.opened(path, sha256_file(path))


def plan_record(design_change=None) -> dict:
    specs = {name: sf.spec(name, synthetic_table()) for name in sf.SIMPLICITY}
    plan = json.loads(json.dumps(pr.plan(specs)))
    if design_change:
        design_change(plan)
    return {"plan": plan, "history": [{"sha256": "a" * 64}], "written": "t", "git": {"sha": "x"}}


def test_a_refreeze_keeps_the_history_and_refuses_a_changed_rule(tmp_path):
    path = tmp_path / "plan.json"
    new = plan_record()
    with pytest.raises(SystemExit, match="existing file"):
        xgs_protocol.refreeze_history(path, "code fix", new, "plan")
    path.write_text(json.dumps(new))
    with pytest.raises(SystemExit, match="--reason"):
        xgs_protocol.refreeze_history(path, None, new, "plan")
    history = xgs_protocol.refreeze_history(path, "code fix", new, "plan")
    assert [h["sha256"] for h in history] == ["a" * 64, sha256_file(path)] and history[-1]["replaced_because"] == "code fix"
    looser = plan_record(lambda p: p["common"]["final_controls"]["4_positive"].update(pass_="[-0.004, 0.0]"))
    with pytest.raises(SystemExit, match="may not change"):
        xgs_protocol.refreeze_history(path, "looser", looser, "plan")
    other_seed = plan_record(lambda p: p["selection"]["seeds"].update(dist=999))
    with pytest.raises(SystemExit, match="may not change"):
        xgs_protocol.refreeze_history(path, "seed", other_seed, "plan")


def test_the_plan_transcribes_c6():
    specs = {name: sf.spec(name, synthetic_table()) for name in sf.SIMPLICITY}
    plan = json.loads(json.dumps(pr.plan(specs), allow_nan=False))
    assert set(plan["candidates"]) == {"b", "c", "bc"}
    assert plan["candidates"]["b"]["model"]["c_grid"] == [1.0] and not plan["candidates"]["b"]["model"]["z_unpenalised"]
    assert plan["candidates"]["c"]["model"]["c_grid"] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    assert plan["candidates"]["c"]["model"]["z_unpenalised"] and plan["candidates"]["bc"]["model"]["z_unpenalised"]
    selection = plan["selection"]
    assert selection["plants"] == ["dist", "dist_angle"] and selection["target_true_delta"] == -0.002
    assert selection["predicted_recovery_b"]["mean_recovered_delta_G"] == [-0.0017, -0.0015]
    assert selection["seeds_distinct_from_final"] and selection["outcomes"] == "simulated only"
    final = plan["common"]["final_controls"]
    assert "[-0.003, -0.001]" in final["4_positive"]["pass"] and "0.09" in final["2_synthetic_nulls"]["pass"]
    assert final["power_curve"]["targets"] == [-0.002, -0.003, -0.004] and final["clustered_nulls"]["random_intercept_sd"] == 0.25
    assert "no further iteration" in plan["common"]["stop_rule"]
    assert any("pre-Platt" in d and "outcome-dependent" in d for d in plan["common"]["disclosures"])
    gbm = plan["common"]["gbm"]
    assert (gbm["max_depth"], gbm["learning_rate"], gbm["max_iter"], gbm["min_samples_leaf"]) == (2, 0.05, 300, 40)
    windows = plan["common"]["soccernet_flags"]["windows_ms"]
    assert {k: (v["from"], v["to"]) for k, v in windows.items()} == {
        "sn_direct_fk": (-3000, 300), "sn_indirect_fk_3s": (-3000, 0), "sn_corner_20s": (-20000, 0), "sn_prev_shot_5s": (-5000, -300)}
    assert plan["common"]["read_out"]["bootstrap"]["draws"] == 2000 and plan["common"]["c5"]["sha256"].startswith("4a00c90b")


def test_the_design_records_the_chosen_candidate():
    spec = sf.spec("bc", synthetic_table())
    design = json.loads(json.dumps(pr.design(spec), allow_nan=False))
    assert design["chosen"] == "bc" and design["arms"]["G"]["columns"] == spec.arms["G"].columns
    assert design["model"]["name"] == "inner_cv" and design["final_controls"]["seeds"] == pr.FINAL_SEEDS
    assert design["final_controls"]["replicates"] == sc.N_REPLICATES == 200


def test_the_code_hashes_cover_the_stack_package_and_the_modules_it_uses():
    files = records.code_hashes()["files"]
    assert {f"xgpoc/stack/{name}.py" for name in ("inputs", "fit", "stats", "controls", "provenance", "protocol", "analysis",
                                                  "selection")} <= set(files)
    assert "xgpoc/detector/sanitation.py" in files and "scripts/xgs_run.py" not in files
    assert "scripts/xgs_run.py" in records.code_hashes(Path(xgs_run.__file__))["files"]


def test_the_final_refreeze_freezes_the_chosen_arms(tmp_path):
    spec = sf.spec("b", synthetic_table())
    record = {"design": json.loads(json.dumps(pr.design(spec))), "history": [], "written": "t", "git": {}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(record))
    assert xgs_protocol.refreeze_history(path, "code fix", record, "final")[-1]["replaced_because"] == "code fix"
    changed = json.loads(json.dumps(record))
    changed["design"]["arms"]["G"]["columns"].pop()
    with pytest.raises(SystemExit, match="arms"):
        xgs_protocol.refreeze_history(path, "fewer columns", changed, "final")


def selection_folder(tmp_path, plan="p" * 64, digest=None) -> Path:
    folder = tmp_path / "selection"
    folder.mkdir(parents=True)
    (folder / "selection.json").write_text('{"chosen": "b"}')
    record = {"plan_sha256": plan, "code": {"package_digest": digest or records.code_hashes()["digest"]},
              "files": {"selection.json": sha256_file(folder / "selection.json")}}
    (folder / "complete.json").write_text(json.dumps(record))
    return folder


def test_the_final_freeze_needs_the_selection_under_the_plan_and_the_code(tmp_path):
    folder = selection_folder(tmp_path)
    assert xgs_protocol.selection_record("p" * 64, folder)["plan_sha256"] == "p" * 64
    with pytest.raises(SystemExit, match="another plan"):
        xgs_protocol.selection_record("q" * 64, folder)
    (folder / "selection.json").write_text('{"chosen": "c"}')
    with pytest.raises(SystemExit, match="does not match"):
        xgs_protocol.selection_record("p" * 64, folder)
    other = selection_folder(tmp_path / "x", digest="0" * 64)
    with pytest.raises(SystemExit, match="other code"):
        xgs_protocol.selection_record("p" * 64, other)
    with pytest.raises(SystemExit, match="--select first"):
        xgs_protocol.selection_record("p" * 64, tmp_path / "missing")


def test_the_controls_need_the_pinned_selection_and_the_frozen_population(tmp_path):
    folder = selection_folder(tmp_path)
    protocol = {"selection": {"complete_sha256": sha256_file(folder / "complete.json")}, "inputs": {"index": {"sha256": "a"}},
                "counts": {"shots": 3, "angles_wrapped": {"0": 1}}}
    xgs_run.check_selection(folder / "complete.json", protocol)
    with pytest.raises(SystemExit, match="selection"):
        xgs_run.check_selection(folder / "complete.json", {"selection": {"complete_sha256": "0" * 64}})
    population = xgs_run.si.Population(None, {"index": {"sha256": "a"}}, {"shots": 3, "angles_wrapped": {"0": 1}})
    xgs_run.check_population(population, protocol)
    with pytest.raises(SystemExit, match="differ"):
        xgs_run.check_population(xgs_run.si.Population(None, {"index": {"sha256": "b"}}, population.counts), protocol)
