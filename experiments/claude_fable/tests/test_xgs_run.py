"""The xg-stack protocol and the run script's gates (ticket 01): `--real` refuses without passed controls under the
same protocol and code, and the protocol refuses a wrong hash or a design that drifted from the code."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgs_protocol  # noqa: E402
import xgs_run  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.stack import fit as sf  # noqa: E402
from xgpoc.stack import protocol as pr  # noqa: E402
from xgpoc.stack import records  # noqa: E402


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


def test_the_protocol_is_refused_without_a_pinned_or_with_a_wrong_hash(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"design": pr.design(), "code": records.code_hashes()}))
    with pytest.raises(SystemExit, match="no protocol hash"):
        xgs_run.opened_protocol(path, None)
    with pytest.raises(SystemExit, match="SHA-256"):
        xgs_run.opened_protocol(path, "0" * 64)
    assert xgs_run.opened_protocol(path, sha256_file(path))["design"]["confirmatory"] == ["G", "GS"]


def test_the_protocol_is_refused_when_the_design_or_the_code_drifted(tmp_path):
    path = tmp_path / "protocol.json"
    design = pr.design()
    design["read_out"]["holm"]["alpha"] = 0.1
    path.write_text(json.dumps({"design": design, "code": records.code_hashes()}))
    with pytest.raises(SystemExit, match="design differs"):
        xgs_run.opened_protocol(path, sha256_file(path))
    path.write_text(json.dumps({"design": pr.design(), "code": {"digest": "0" * 64}}))
    with pytest.raises(SystemExit, match="code differs"):
        xgs_run.opened_protocol(path, sha256_file(path))


def test_the_design_transcribes_c5():
    design = json.loads(json.dumps(pr.design(), allow_nan=False))
    assert design["arms"]["G"]["columns"] == sf.ARMS["G"].columns and design["arms"]["G"]["status"] == "confirmatory"
    assert design["models"]["logistic"] | {} == {"class": "sklearn.linear_model.LogisticRegression", "penalty": "l2", "C": 1.0,
                                                 "solver": "lbfgs", "max_iter": 5000,
                                                 "design": "standardised; z_g_combo linear; no interactions"}
    gbm = design["models"]["gbm"]
    assert (gbm["max_depth"], gbm["learning_rate"], gbm["max_iter"], gbm["min_samples_leaf"], gbm["l2_regularization"],
            gbm["early_stopping"], gbm["random_state"]) == (2, 0.05, 300, 40, 1.0, False, 0)
    windows = design["soccernet_flags"]["windows_ms"]
    assert {k: (v["from"], v["to"]) for k, v in windows.items()} == {
        "sn_direct_fk": (-3000, 300), "sn_indirect_fk_3s": (-3000, 0), "sn_corner_20s": (-20000, 0), "sn_prev_shot_5s": (-5000, -300)}
    read = design["read_out"]
    assert (read["bootstrap"]["draws"], read["bootstrap"]["seed"], read["holm"]["alpha"]) == (2000, 0, 0.05)
    assert read["holm"]["arms"] == ["G", "GS"] and "-0.002" in read["practical_candidate"]
    controls = design["controls"]
    assert controls["2_synthetic_nulls"]["replicates"] == 200 and "0.09" in controls["2_synthetic_nulls"]["pass"]
    assert controls["4_positive"]["replicates"] == 200 and controls["4_positive"]["target_true_delta"] == -0.002
    assert "[-0.003, -0.001]" in controls["4_positive"]["pass"]
    assert design["population"] | {} and (design["population"]["shots"], design["population"]["matches"]) == (7062, 328)


def test_the_code_hashes_cover_the_stack_package_and_the_modules_it_uses():
    files = records.code_hashes()["files"]
    assert {f"xgpoc/stack/{name}.py" for name in ("inputs", "fit", "stats", "controls", "provenance", "protocol", "analysis")} <= set(files)
    assert "xgpoc/detector/sanitation.py" in files and "scripts/xgs_run.py" not in files
    assert "scripts/xgs_run.py" in records.code_hashes(Path(xgs_run.__file__))["files"]


def test_a_refreeze_keeps_the_history_and_refuses_a_changed_threshold(tmp_path):
    path = tmp_path / "protocol.json"
    with pytest.raises(SystemExit, match="existing protocol"):
        xgs_protocol.refreeze_history(path, "code fix")
    path.write_text(json.dumps({"design": pr.design(), "history": [{"sha256": "a" * 64}], "written": "t", "git": {"sha": "x"}}))
    with pytest.raises(SystemExit, match="--reason"):
        xgs_protocol.refreeze_history(path, None)
    history = xgs_protocol.refreeze_history(path, "code fix")
    assert [h["sha256"] for h in history] == ["a" * 64, sha256_file(path)] and history[-1]["replaced_because"] == "code fix"
    design = pr.design()
    design["controls"]["4_positive"]["pass"] = "mean recovered ΔLL of G in [-0.004, 0.0]"
    path.write_text(json.dumps({"design": design, "history": [], "written": "t", "git": {}}))
    with pytest.raises(SystemExit, match="may not change"):
        xgs_protocol.refreeze_history(path, "looser")
