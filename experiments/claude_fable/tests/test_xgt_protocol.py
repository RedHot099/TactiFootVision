"""The frozen protocol of xg-temporal (ticket 03) on synthetic inputs: the grid and its control rows, the frozen part,
the generator's refusals and drafts, and the whole chain label-free controls -> protocol -> permutation -> real run,
in which `--permutation` and `--real` refuse to start without the pinned protocol and the passed controls. No real
label, embedding or file of results/ is read: every input is synthetic and every path is under tmp_path."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgt_protocol  # noqa: E402
import xgt_run  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.temporal import controls as tco  # noqa: E402
from xgpoc.temporal import inputs as ti  # noqa: E402
from xgpoc.temporal import labels as tl  # noqa: E402
from xgpoc.temporal import protocol as tpr  # noqa: E402
from xgpoc.temporal import records as tr_records  # noqa: E402
from xgt_fixtures import synthetic_population  # noqa: E402

EXPERIMENT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the frozen part
def test_the_grid_has_14_windows_with_every_late_o_max_a_flagged_control_row():
    cells = tpr.grid_cells()
    assert len(cells) == 14 and all(c["start"] < c["o_max"] for c in cells)
    assert [c for c in cells if c["role"] == "main"] == [c for c in cells if (c["start"], c["o_max"]) == (-1600, -600)]
    controls = {(c["start"], c["o_max"]) for c in cells if c["role"] == "leak_control"}
    assert controls == {(-1600, -400), (-1200, -400), (-800, -400), (-1600, -200), (-1600, 0), (-1600, 400)}
    assert all(c["control_flag"] == (c["role"] == "leak_control") for c in cells)
    assert [c["candidate"] for c in cells].count(True) == 1
    assert {c["o_max"] for c in cells if c["ticket"] == "04"} == set(tpr.LEAK_CURVE) - {-1200}
    assert next(c for c in cells if (c["start"], c["o_max"]) == (-800, -600))["motion_features"] == ["pc01..pc08_vs_start"]


def test_the_frozen_part_is_deterministic_json_with_the_c8_grid_the_c9_rules_and_the_stop_of_ticket_02():
    frozen = tpr.frozen()
    assert json.loads(json.dumps(frozen)) == json.loads(json.dumps(tpr.frozen()))
    assert frozen["model"]["c_grid"][:5] == [1e-5, 3e-5, 1e-4, 3e-4, 0.001] and frozen["model"]["z_unpenalised"]
    assert frozen["o_max"]["value"] == -600 and "STOP" in frozen["o_max"]["ticket_02"]
    assert frozen["arms"]["k"] == 32 and frozen["controls"]["seeds"] == tco.RUN_SEEDS
    assert frozen["controls"]["attempt_seeds"] == {"first": tco.SEEDS, "remedy": tco.REMEDY_SEEDS}
    assert [frozen["controls"]["attempts"][a]["gates"]["positive"] for a in ("first", "remedy")] == [False, False]
    assert frozen["model"]["t_variant"]["name"] == frozen["arms"]["T"]["model"] == "inner_cv_c8_motion_block_c9q2"
    assert frozen["arms"]["T"]["design"] == "c10_vs_start_pc01_pc08" and frozen["arms"]["T"]["columns"] == 47
    assert frozen["arms"]["motion_k"] == 8 and frozen["preprocessing"]["motion_features"]["components"] == 8
    assert any("First attempt" in d and "-0.00032" in d and "0.11" in d and "ecbe44c2" in d for d in frozen["disclosures"])
    assert any("Second attempt" in d and "-0.00035" in d and "4bb1a757" in d for d in frozen["disclosures"])
    assert any("-0.0016" in d and "-0.0013" in d and "-0.0006" in d for d in frozen["disclosures"])
    assert "closes the phase" in frozen["miss_rule"] and "8 directions" in frozen["readings"]["scope_wording"]
    assert frozen["controls"]["c10"]["prediction"]["positive_c9_diffuse_-0.002_mean_recovered_T-S"] == -0.0006
    assert {c["motion_features"][0] for c in frozen["grid"]} == {"pc01..pc08_vs_start"}
    for name in ("S_xgc/T_xgc", "F_S/F_T", "k_sensitivity", "seed_sensitivity"):
        assert "T" in frozen["descriptive"][name]["inputs"]
    assert frozen["readings"]["holm"]["contrasts"] == ["T-R", "T-S"]
    assert set(frozen["descriptive"]) == {"L", "S_xgc/T_xgc", "F_S/F_T", "k_sensitivity", "seed_sensitivity", "N"}
    assert frozen["descriptive"]["k_sensitivity"]["k"] == [16, 64]
    assert frozen["controls"]["remedy"]["seeds"] == tco.REMEDY_SEEDS and "C9" in frozen["leak"]["jump_rule"]["source"]
    assert frozen["leak"]["jump_rule"]["trend"].endswith("[-1000, -800, -600] (start -1600)")
    assert any("+0.0012" in d for d in frozen["disclosures"])
    assert set(frozen["readings"]["categories"]) == {"e3_outside_box", "e3_angle_lt_15deg", "e3_dist_lt_6m",
                                                     "e3_gk_depth_ge_4yd"}


def test_load_refuses_an_unpinned_or_different_protocol(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="pinned"):
        tpr.load(path, None)
    with pytest.raises(ValueError, match="SHA-256"):
        tpr.load(path, "0" * 64)
    assert tpr.load(path, sha256_file(path)) == {}


def test_the_package_digest_covers_every_xgpoc_module_the_scripts_import():
    code = ("import sys; sys.path.insert(0, 'scripts'); import xgt_run, xgt_protocol; "
            "from xgpoc.temporal import records as r; "
            "loaded = {str(p.relative_to(r.EXPERIMENT)) for p in r.loaded_files()}; "
            "missing = sorted(loaded - set(r.package_hashes()['files'])); print(missing); assert not missing")
    subprocess.run([sys.executable, "-c", code], cwd=EXPERIMENT, check=True)


def test_the_package_digest_pins_the_run_script_except_its_protocol_pin(tmp_path):
    assert f"{tr_records.RUN_SCRIPT} (pin normalised)" in tr_records.package_hashes()["files"]
    text = (EXPERIMENT / tr_records.RUN_SCRIPT).read_text()
    line = next(x for x in text.splitlines() if x.startswith("PROTOCOL_SHA256"))
    script = tmp_path / "xgt_run.py"
    script.write_text(text)
    before = tr_records.run_script_sha256(script)
    script.write_text(text.replace(line, f'PROTOCOL_SHA256: str | None = "{"a" * 64}"'))
    assert tr_records.run_script_sha256(script) == before                     # pinning after the freeze
    script.write_text(text.replace("REAL_OUT = ", "REAL_OUT =  "))
    assert tr_records.run_script_sha256(script) != before                     # any other edit
    script.write_text(text.replace(line, ""))
    with pytest.raises(ValueError, match="exactly once"):
        tr_records.run_script_sha256(script)


# ------------------------------------------------------------------ the generator's arguments
def test_the_generator_refuses_a_draft_on_the_protocol_path_and_another_source_without_draft():
    for argv in (["--draft"], ["--draft", "--out", str(EXPERIMENT / tpr.PROTOCOL)], ["--source", "g_xgc"],
                 ["--refreeze"]):
        with pytest.raises(SystemExit):
            xgt_protocol.parse_args(argv)


def test_the_real_and_permutation_modes_refuse_settings_and_an_unpinned_protocol(monkeypatch):
    for argv in (["--real", "--source", "g_xgc"], ["--real", "--k", "16"], ["--permutation", "--o-max", "-800"],
                 ["--real", "--out", "x"]):
        with pytest.raises(SystemExit):
            xgt_run.parse_args(argv)
    monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", None)
    monkeypatch.setattr(xgt_run.a1, "verify_plan", lambda *_: pytest.fail("an input was read"))
    monkeypatch.setattr(xgt_run.tl, "load_labels", lambda *_: pytest.fail("a label was read"))
    for mode in ("--real", "--permutation"):
        with pytest.raises(SystemExit, match="pinned"):
            xgt_run.main([mode])


def test_a_trial_run_of_the_controls_is_refused_in_the_protocols_folder():
    args = xgt_run.parse_args(["--controls", "--c10"])
    assert args.protocol_controls and args.c10 and args.out == xgt_run.ROOT / tpr.LABEL_FREE
    assert str(tpr.LABEL_FREE).endswith("label_free_c10") and tpr.LABEL_FREE not in tpr.ATTEMPTS.values()
    for argv in (["--controls"], ["--controls", "--c10", "--replicates", "3"], ["--controls", "--c10", "--source", "g_xgc"],
                 ["--controls", "--c10", "--simulations", "plain_nulls", "rich_nulls", "positive_-0.002"],
                 ["--controls", "--c10", "--simulations",
                  *[s for s in tco.SIMULATIONS if s != "positive_c9_diffuse_-0.002"]],    # the C9 plant must run
                 ["--controls", "--c10", "--out", str(xgt_run.ROOT / tpr.FIRST_ATTEMPT)],
                 ["--controls", "--out", str(xgt_run.ROOT / tpr.REMEDY_ATTEMPT)], ["--controls", "--remedy"],
                 ["--real", "--c10"], ["--synthetic", "--source", "g_xgv", "--c10"]):
        with pytest.raises(SystemExit):
            xgt_run.parse_args(argv)
    args = xgt_run.parse_args(["--controls", "--replicates", "3", "--out", "/tmp/x"])
    assert not args.protocol_controls
    assert not xgt_run.parse_args(["--controls", "--out", "/tmp/x"]).protocol_controls        # without --c10


def test_the_common_shots_are_in_every_population_and_restriction_keeps_the_window_size():
    population, _, _ = synthetic_population(ti.Window())
    ids = population.table.index
    other = SimpleNamespace(table=pd.DataFrame(index=ids[: len(ids) - 3][::-1]))   # another order
    third = SimpleNamespace(table=pd.DataFrame(index=ids[2:]))
    common = ti.common_shots([population, other, third])
    assert list(common) == list(ids[2: len(ids) - 3])                         # in the first population's order
    once = ti.restrict(population, ids[1:])
    twice = ti.restrict(once, common)
    assert twice.counts == ti.restrict(population, common).counts
    assert twice.counts["window_shots"] == len(ids) and twice.counts["shots"] == len(common)


def git_commit(root: Path, path: Path) -> None:
    """Commit `path` in a repository at `root` (created when missing), as the protocol's freeze does."""
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    if not (root / ".git").exists():
        subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", str(path)], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "protocol"], check=True)


def test_only_the_descriptive_rich_null_may_be_left_out_of_the_protocols_controls():
    assert tco.DROPPABLE == ("rich_nulls_-0.002",) and tco.DROPPABLE[0] in tco.SIMULATIONS
    kept = [s for s in tco.SIMULATIONS if s not in tco.DROPPABLE]
    assert xgt_run.omitted_simulations(tco.SIMULATIONS) == [] and xgt_run.omitted_simulations(kept) == list(tco.DROPPABLE)
    assert xgt_run.parse_args(["--controls", "--c10", "--simulations", *kept]).protocol_controls
    for other in ([s for s in tco.SIMULATIONS if s != "clustered_nulls"], list(reversed(tco.SIMULATIONS))):
        assert xgt_run.omitted_simulations(other) is None
        with pytest.raises(SystemExit):
            xgt_run.parse_args(["--controls", "--c10", "--simulations", *other])


# ------------------------------------------------------------------ the chain on synthetic inputs
def write_attempt(folder: Path, recovered: float = -0.00032, passed: bool = False, remedy: bool = False,
                  seeds: dict | None = None) -> str:
    """A synthetic failed attempt (controls.json, complete.json naming it): the first by default, the C9 Q2 remedy
    with `remedy`; returns controls.json's SHA-256."""
    seeds = (tco.REMEDY_SEEDS if remedy else tco.SEEDS) if seeds is None else seeds
    folder.mkdir(parents=True)
    gates = {g: {"passed": True, "criterion": "c"} for g in tco.LABEL_FREE_GATES}
    gates["positive"] = {"passed": passed, "criterion": "c", "mean_recovered_delta_T-S": recovered, "power_T-S": 0.15,
                         "power_T-S_mc_se": 0.025, "oracle_power": 0.895, "grid_floor_share": {"S": 0.54, "T": 0.12}}
    gates["rich_nulls"] |= {"passed": False, "holm_rejection": {"T-S": {"rate": 0.11, "mc_se": 0.022}},
                            "mean_delta": {"T-S": 0.00014}}
    (folder / "controls.json").write_text(json.dumps({"gates": gates, "seeds": seeds, "run": {"seconds": 5580.6}}))
    (folder / "complete.json").write_text(json.dumps({
        "protocol_controls": True, "label_free_passed": False, "remedy": remedy, "seeds": seeds,
        "files": {"controls.json": sha256_file(folder / "controls.json")}}))
    return sha256_file(folder / "controls.json")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Every input of the scripts made synthetic, every output under tmp_path; the controls' sample size and
    criteria made small (a synthetic population of 160 shots cannot meet the real criteria)."""
    for module in (xgt_run, xgt_protocol):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "complete.json").write_text("{}")
    files = SimpleNamespace(record=lambda: {"source": "synthetic"}, source=SimpleNamespace(folder=folder))
    cache = {}

    def build(source, window, plan, files_=None, **kwargs):
        population, embeddings, _ = synthetic_population(window)
        cache["embeddings"] = embeddings
        return population, files

    _, _, plan = synthetic_population(ti.Window())
    monkeypatch.setattr(ti, "build_population", build)
    monkeypatch.setattr(ti, "verify_source", lambda source: files)
    monkeypatch.setattr(ti, "load_embeddings", lambda f, refs: (cache["embeddings"], refs))
    monkeypatch.setattr(xgt_run.a1, "verify_plan", lambda *_: plan)
    monkeypatch.setattr(xgt_protocol, "safe_margin", lambda: {"verdict": "stop", "sha256": tpr.SAFE_MARGIN_RESULT_SHA256})
    monkeypatch.setattr(xgt_protocol.records, "verified_file", lambda path: "e3-sha256")
    monkeypatch.setattr(tco, "N_REPLICATES", 2)
    monkeypatch.setattr(tco, "NULL_MAX_REJECTION_RATE", 1.0)
    monkeypatch.setattr(tco, "POSITIVE_WINDOW", (-1.0, 1.0))
    rng = np.random.default_rng(1)

    def labels(population, index, plan_csv):                      # the same labels on every read
        n, own = len(population.table), np.random.default_rng(1)
        return tl.Labels(population.table.index.to_numpy(), (own.random(n) < 0.15).astype(int),
                         (own.random(n) < 0.4).astype(int))
    monkeypatch.setattr(xgt_run.tl, "load_labels", labels)

    def e3(shot_ids, source=None):
        n = len(shot_ids)
        frame = pd.DataFrame({"shot_x@0": rng.uniform(70, 105, n), "angle@0": rng.uniform(0, 1.5, n),
                              "dist@0": rng.uniform(2, 30, n), "gk_depth@0": rng.uniform(0, 8, n)}, index=shot_ids)
        return frame, {"frames.parquet": {"sha256": "e3"}}
    monkeypatch.setattr(xgt_run.tc, "e3_inputs", e3)
    monkeypatch.setattr(tpr, "ATTEMPT_CONTROLS_SHA256", {
        "first": write_attempt(tmp_path / tpr.FIRST_ATTEMPT),
        "remedy": write_attempt(tmp_path / tpr.REMEDY_ATTEMPT, -0.00035, remedy=True)})
    return tmp_path


def test_the_draft_records_the_missing_parts_as_pending(lab):
    out = lab / "draft.json"
    xgt_protocol.main(["--draft", "--source", "g_xgc", "--out", str(out)])
    draft = json.loads(out.read_text())
    assert draft["draft"] and draft["data"]["label_free_controls"]["status"] == "pending"
    assert len(draft["data"]["cells"]) == 14 and draft["data"]["main"]["columns"]["T"][0] == "z_g_combo"
    assert set(draft["data"]["descriptive_sources"]) == {"g_xgc", "dinov2"}
    assert draft["data"]["e3"]["source"] == "cache/xgd_full3/E3" and not draft["data"]["e3"]["fallback_used"]
    assert draft["data"]["main"]["source_files"]["complete_json"]["sha256"] == sha256_file(lab / "source/complete.json")
    with pytest.raises(SystemExit, match="exists"):
        xgt_protocol.main(["--draft", "--source", "g_xgc", "--out", str(out)])
    with pytest.raises(SystemExit, match="label-free controls|--controls"):
        xgt_protocol.main(["--out", str(lab / "protocol.json")])
    trial = lab / "trial"
    xgt_run.main(["--controls", "--jobs", "1", "--replicates", "1", "--out", str(trial)])
    assert json.loads((trial / "complete.json").read_text())["seeds"] == tco.TRIAL_SEEDS      # a trial: other seeds
    xgt_protocol.main(["--draft", "--source", "g_xgc", "--controls", str(trial), "--out", str(lab / "draft2.json")])
    recorded = json.loads((lab / "draft2.json").read_text())["data"]["label_free_controls"]
    assert "not the protocol's controls (a trial run)" in recorded["problems"] and "source g_xgv" in recorded["problems"]
    assert "not council C10's run on RUN_SEEDS (two earlier attempts failed)" in recorded["problems"]
    with pytest.raises(SystemExit, match="cannot enter"):
        xgt_protocol.main(["--controls", str(trial), "--out", str(lab / "protocol.json")])


def run_protocol(lab: Path, controls: Path) -> None:
    xgt_protocol.main(["--controls", str(controls), "--out", str(lab / "protocol.json")])


def test_the_protocol_refuses_every_controls_record_but_c10s(lab):
    xgt_run.main(["--controls", "--jobs", "1", "--out", str(lab / "no_c10")])       # the protocol's settings, no --c10
    record = json.loads((lab / "no_c10" / "complete.json").read_text())
    assert not record["protocol_controls"] and not record["c10"] and record["seeds"] == tco.TRIAL_SEEDS
    with pytest.raises(SystemExit, match="not council C10's run on RUN_SEEDS"):
        run_protocol(lab, lab / "no_c10")
    assert not (lab / "protocol.json").exists()
    complete = lab / "no_c10" / "complete.json"
    for forged in ({"protocol_controls": True},                                    # no c10
                   {"protocol_controls": True, "c10": True},                       # c10, the trial seeds
                   {"protocol_controls": True, "c10": True, "seeds": tco.RUN_SEEDS, "t_design": "c9_96_columns"}):
        complete.write_text(json.dumps(record | forged))
        with pytest.raises(SystemExit, match="not council C10's run on RUN_SEEDS"):
            run_protocol(lab, lab / "no_c10")


@pytest.mark.parametrize("name", ["first", "remedy"])
def test_the_protocol_refuses_an_earlier_attempt_as_its_controls(lab, name):
    with pytest.raises(SystemExit, match="an earlier failed attempt"):
        run_protocol(lab, lab / tpr.ATTEMPTS[name])
    recorded = xgt_protocol.label_free_controls(lab / tpr.ATTEMPTS[name], {}, True)       # a draft lists why
    assert "an earlier failed attempt, not the C10 run" in recorded["problems"]


def test_the_attempt_trail_pins_both_failed_attempts(lab, monkeypatch):
    trail = xgt_protocol.attempts({n: lab / f for n, f in tpr.ATTEMPTS.items()}, False)
    first, remedy = trail["first"], trail["remedy"]
    assert first["controls_sha256"] == tpr.ATTEMPT_CONTROLS_SHA256["first"] and not first["label_free_passed"]
    assert first["positive"]["mean_recovered_T-S"] == -0.00032 and first["rich_nulls"]["holm_rejection_T-S"]["rate"] == 0.11
    assert first["gates"] == {"provenance": True, "plain_nulls": True, "rich_nulls": False, "positive": False}
    assert remedy["controls_sha256"] == tpr.ATTEMPT_CONTROLS_SHA256["remedy"] and remedy["seeds"] == tco.REMEDY_SEEDS
    assert remedy["positive"]["mean_recovered_T-S"] == -0.00035 and "C10" in remedy["consequence"]
    pending = xgt_protocol.attempts({"first": lab / "missing", "remedy": lab / tpr.REMEDY_ATTEMPT}, True)
    assert pending["first"]["status"] == "pending"
    with pytest.raises(SystemExit, match="does not exist"):
        xgt_protocol.attempts({"first": lab / tpr.FIRST_ATTEMPT, "remedy": lab / "missing"}, False)
    for folders in ({"first": lab / tpr.FIRST_ATTEMPT}, {"remedy": lab / tpr.REMEDY_ATTEMPT, "first": lab / tpr.FIRST_ATTEMPT}):
        with pytest.raises(SystemExit, match="trail"):
            xgt_protocol.attempts(folders, False)                                  # both, in run order
    with pytest.raises(SystemExit, match="pins"):
        xgt_protocol.attempt("first", lab / tpr.REMEDY_ATTEMPT, False)              # the other attempt's hash
    for name, kwargs, message in (("passed", {"passed": True}, "positive gate passed"),
                                  ("too_low", {"recovered": -0.0035}, "no trigger")):
        sha = write_attempt(lab / name, **kwargs)
        monkeypatch.setattr(tpr, "ATTEMPT_CONTROLS_SHA256", tpr.ATTEMPT_CONTROLS_SHA256 | {"first": sha})
        with pytest.raises(SystemExit, match=message):
            xgt_protocol.attempt("first", lab / name, False)
    sha = write_attempt(lab / "wrong_seeds", -0.00035, remedy=True, seeds=tco.SEEDS)
    monkeypatch.setattr(tpr, "ATTEMPT_CONTROLS_SHA256", tpr.ATTEMPT_CONTROLS_SHA256 | {"remedy": sha})
    with pytest.raises(SystemExit, match="REMEDY_SEEDS"):
        xgt_protocol.attempt("remedy", lab / "wrong_seeds", False)


def test_controls_then_protocol_then_permutation_then_the_real_run(lab, monkeypatch):
    xgt_run.main(["--controls", "--c10", "--jobs", "1"])
    label_free = lab / tpr.LABEL_FREE
    record = json.loads((label_free / "complete.json").read_text())
    assert record["protocol_controls"] and record["label_free_passed"] and set(record["gates"]) == set(tco.LABEL_FREE_GATES)
    assert record["c10"] and record["seeds"] == tco.RUN_SEEDS and record["t_model"] == tpr.tr.T_MODEL
    assert record["t_design"] == tpr.tf.T_DESIGN and len(record["columns"]["T"]) == len(record["columns"]["S"]) + 8
    controls_json = json.loads((label_free / "controls.json").read_text())
    assert controls_json["gates"]["positive"]["seed"] == 30505 and controls_json["gates"]["positive"]["w_seed"] == 30510
    assert "passed" not in controls_json["reports"]["positive_c9_diffuse_-0.002"]
    with pytest.raises(SystemExit, match="exists"):
        xgt_run.main(["--controls", "--c10", "--jobs", "1"])

    xgt_protocol.main([])
    path = lab / tpr.PROTOCOL
    protocol = json.loads(path.read_text())
    assert not protocol["draft"] and protocol["data"]["label_free_controls"]["complete_sha256"] == sha256_file(
        label_free / "complete.json")
    assert protocol["data"]["code"]["digest"] == tr_records.package_hashes()["digest"]
    assert protocol["data"]["label_free_controls"]["omitted_simulations"] == []
    assert protocol["data"]["label_free_controls"]["c10"] and protocol["data"]["label_free_controls"]["seeds"] == tco.RUN_SEEDS
    trail = protocol["data"]["attempts"]
    assert list(trail) == ["first", "remedy"]
    for name, folder in tpr.ATTEMPTS.items():
        assert trail[name]["controls_sha256"] == tpr.ATTEMPT_CONTROLS_SHA256[name] == sha256_file(lab / folder / "controls.json")
        assert trail[name]["complete_sha256"] == sha256_file(lab / folder / "complete.json") and not trail[name]["label_free_passed"]
    common = protocol["data"]["leak_common"]
    assert set(common["cells"]) == {str(o) for o in tpr.LEAK_CURVE}
    assert {c["shot_list_sha256"] for c in common["cells"].values()} == {common["shot_list_sha256"]}
    assert common["shots"] <= min(c["counts"]["shots"] for c in protocol["data"]["cells"] if c["start"] == -1600)
    with pytest.raises(SystemExit, match="code change"):
        xgt_protocol.main(["--refreeze", "--reason", "nothing changed"])

    monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        xgt_run.main(["--permutation"])
    monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", sha256_file(path))
    with pytest.raises(SystemExit, match="not committed"):
        xgt_run.main(["--permutation"])                             # pinned, but not in the HEAD commit
    git_commit(lab, path)
    with pytest.raises(SystemExit, match="does not exist"):
        xgt_run.main(["--real"])                                    # no permutation yet
    xgt_run.main(["--permutation"])
    controls = json.loads((lab / tpr.CONTROLS / "complete.json").read_text())
    assert set(controls["controls"]) == set(tco.GATES) and controls["protocol_sha256"] == sha256_file(path)
    assert not json.loads((lab / tpr.CONTROLS / "permutation.json").read_text()).keys() & {"y", "labels", "probabilities"}
    with pytest.raises(SystemExit, match="once"):
        xgt_run.main(["--permutation"])

    assert controls["all_passed"]                                   # the synthetic labels are seeded: deterministic
    extraction = lab / "source/complete.json"
    extraction.write_text('{"changed": true}')
    with pytest.raises(SystemExit, match="extraction record"):
        xgt_run.main(["--real"])
    assert not (lab / xgt_run.REAL_OUT / "summary.json").exists()
    extraction.write_text("{}")
    import shutil
    shutil.rmtree(lab / xgt_run.REAL_OUT)                          # the refused run made only its empty folder
    xgt_run.main(["--real"])
    summary = json.loads((lab / xgt_run.REAL_OUT / "summary.json").read_text())
    assert {"T-R", "T-S", "S-R"} == set(summary["deltas"]) and "readability_veto" in summary
    assert set(summary["leak"]) == {"method_positive", "trend", "jump", "leak_signal", "population", "shots"} and "motion_helps" in summary["claims"]
    assert set(summary["leak"]["jump"]["T-S"]["excess"]["-200"]) == {"excess", "lo95", "hi95"}
    assert summary["leak"]["population"] == "common" and summary["leak_by_window"]["descriptive"]
    assert summary["leak"]["shots"] == common["shots"]
    assert "8 directions" in summary["scope"]["wording"] and "c9_diffuse_mean_recovered_T-S" in summary["scope"]
    curve = pd.read_csv(lab / xgt_run.REAL_OUT / "leak_curve.csv")
    assert set(curve.o_max) == set(tpr.LEAK_CURVE) and set(curve.population) == {"window", "common"}
    assert set(curve[curve.population == "common"].shots) == {common["shots"]}
    assert set(pd.read_csv(lab / xgt_run.REAL_OUT / "categories.csv").category) == set(tpr.tc.CATEGORIES)
    probabilities = pd.read_csv(lab / xgt_run.REAL_LARGE / "probabilities.csv")
    assert not set(probabilities.columns) & {"goal", "on_target", "label", "y"}
    complete = json.loads((lab / xgt_run.REAL_OUT / "complete.json").read_text())
    assert str(lab / xgt_run.REAL_LARGE / "probabilities.csv") in complete["files"]
    with pytest.raises(SystemExit, match="once"):
        xgt_run.main(["--real"])


def test_the_real_run_refuses_failed_or_foreign_controls(lab, tmp_path):
    folder = tmp_path / "controls"
    folder.mkdir()
    (folder / "permutation.json").write_text("{}")
    base = {"protocol_sha256": "p", "code": {"package": {"digest": "d"}},
            "controls": {g: {"passed": True} for g in tco.GATES}, "all_passed": True,
            "files": {"permutation.json": sha256_file(folder / "permutation.json")}}
    for change, message in (({}, None), ({"all_passed": False}, "not every"), ({"protocol_sha256": "q"}, "another"),
                            ({"controls": {"provenance": {"passed": True}}}, "not every")):
        (folder / "complete.json").write_text(json.dumps(base | change))
        if message is None:
            assert xgt_run.check_controls(folder, "p", "d")["all_passed"]
        else:
            with pytest.raises(SystemExit, match=message):
                xgt_run.check_controls(folder, "p", "d")
    (folder / "permutation.json").write_text("{ }")
    with pytest.raises(SystemExit, match="does not match"):
        xgt_run.check_controls(folder, "p", "d")


def test_the_real_run_refuses_e3_files_other_than_the_protocols():
    pinned = {"frames.parquet": {"path": "/a/frames.parquet", "sha256": "f"},
              "features.parquet": {"path": "/a/features.parquet", "sha256": "g"}}
    moved = {name: {"path": f"/b/{name}", "sha256": h["sha256"]} for name, h in pinned.items()}
    xgt_run.check_e3(moved, pinned)                                 # another checkout: same files
    for changed in ({**moved, "features.parquet": {"path": "/a/features.parquet", "sha256": "h"}},
                    {"frames.parquet": moved["frames.parquet"]}):
        with pytest.raises(SystemExit, match="E3"):
            xgt_run.check_e3(changed, pinned)


def test_a_draft_or_a_changed_protocol_is_refused(lab, monkeypatch):
    path = lab / tpr.PROTOCOL
    path.parent.mkdir(parents=True)
    record = {"draft": False, "frozen": tpr.frozen(), "data": {"code": tr_records.package_hashes()}}
    for change, message in (({"draft": True}, "draft"), ({"frozen": {"x": 1}}, "frozen"),
                            ({"data": {"code": {"digest": "x"}}}, "code differs")):
        path.write_text(json.dumps(record | change, default=float))
        monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", sha256_file(path))
        with pytest.raises(SystemExit, match=message):
            xgt_run.opened_protocol()
    path.write_text(json.dumps(record, default=float))
    monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", sha256_file(path))
    with pytest.raises(SystemExit, match="not committed"):
        xgt_run.opened_protocol()
    git_commit(lab, path)
    assert xgt_run.opened_protocol()["draft"] is False
    path.write_text(json.dumps(record, default=float) + " ")      # edited after the commit, re-pinned
    monkeypatch.setattr(xgt_run, "PROTOCOL_SHA256", sha256_file(path))
    with pytest.raises(SystemExit, match="not committed"):
        xgt_run.opened_protocol()
