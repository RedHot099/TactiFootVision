"""Script logic of the vendor comparison that had no tests (xg-vendor ticket 14 fix round, finding 4): the gate check of
`vendor_score_eval`, the pairing comparison of `vendor_pair_compare`, the replay of `vendor_verify_g` and the helpers of
the analysis scripts (parameters from the pre-registration record, population sets, embedding files, prototype
comparisons)."""

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from xgpoc.eval.calibration import sigmoid
from xgpoc.models.g_bundle import ensemble_probability
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import probes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_leakage  # noqa: E402
import vendor_pair_compare as compare  # noqa: E402
import vendor_probes  # noqa: E402
import vendor_verify_g  # noqa: E402
from test_vendor_evaltable import make_table  # noqa: E402
from test_vendor_ladder import synthetic_table  # noqa: E402
from vendor_ladder import population_sets  # noqa: E402
from vendor_score_eval import check_gate  # noqa: E402

PREREG = et.PreRegistration("a" * 64, datetime(2026, 10, 12, tzinfo=UTC), "b" * 64, margin=0.005, n_boot=6000, fold_seed=1, boot_seed=2,
                            offset_logits="raw", h3_thresholds={"dist": 0.5, "head": 0.7}, h3_confirmatory_targets=("dist",), h3_control_max_r2=0.004,
                            timing_rule_sha256="c" * 64, recalibrated_bootstrap="refit", frame_exclusions={}, draft=True)


# ------------------------------------------------------------------ vendor_score_eval.check_gate
def gate_fixture(tmp_path: Path, missing: bool = False) -> tuple:
    rng = np.random.default_rng(0)
    shots = pd.DataFrame({"key": [f"s{i}" for i in range(8)], "role": [es.PRIMARY] * 3 + [es.GATE] * 5, "is_penalty": [False] * 7 + [True]})
    members = {k: {"xg": rng.normal(-2, 1, (2, 8)), "on_target": rng.normal(0, 1, (2, 8))} for k in es.MEMBERS}
    temperatures = {k: 1.0 + 0.1 * k[0] + 0.05 * k[1] for k in es.MEMBERS}
    calibration = {"platt": {"a": 0.9, "b": -0.1}, "oof_rate": 0.12, "pooled_rate": 0.11, "penalty_rate": 0.76}
    gate = shots.index[shots.role.eq(es.GATE)].to_numpy()
    (tmp_path / "fold0/seed0").mkdir(parents=True)
    (tmp_path / "fold0/seed0/complete.json").write_text(json.dumps({"identity": {"name": "g"}}))
    for fold, seed in es.MEMBERS:
        rows = gate[1:] if missing and (fold, seed) == (3, 1) else gate
        stored = pd.DataFrame({"shot_id": shots.key[rows], "logit": members[fold, seed]["xg"][0, rows]})
        (tmp_path / f"fold{fold}/seed{seed}").mkdir(parents=True, exist_ok=True)
        pd.concat([stored, stored.head(1)]).to_csv(tmp_path / f"fold{fold}/seed{seed}/g_gate_predictions.csv", index=False)   # a duplicate row
    p0 = np.full(len(shots), np.nan)
    p0[gate] = ensemble_probability([members[k]["xg"][0, gate] for k in sorted(es.MEMBERS)], [temperatures[k] for k in sorted(es.MEMBERS)],
                                    calibration, shots.is_penalty.to_numpy()[gate])
    return SimpleNamespace(g_root=tmp_path), shots, pd.DataFrame({"p_0": p0}), members, temperatures, calibration


def test_the_gate_check_reproduces_the_stored_gate_predictions(tmp_path):
    args, shots, scores, members, temperatures, calibration = gate_fixture(tmp_path)
    result = check_gate(args, shots, scores, members, temperatures, calibration)
    assert result["passed"] and result["gate_shots"] == 5 and result["max_abs_dlogit_vs_stored"] < 1e-12   # CSV round trip
    off = scores.assign(p_0=scores.p_0 + 2e-6)
    result = check_gate(args, shots, off, members, temperatures, calibration)
    assert not result["passed"] and result["max_abs_dp_ensemble_vs_stored"] == pytest.approx(2e-6)


def test_the_gate_check_refuses_missing_predictions(tmp_path):
    args, shots, scores, members, temperatures, calibration = gate_fixture(tmp_path, missing=True)
    with pytest.raises(SystemExit, match="no stored gate prediction"):
        check_gate(args, shots, scores, members, temperatures, calibration)
    args, shots, scores, members, temperatures, calibration = gate_fixture(tmp_path / "b")
    scores.loc[5, "p_0"] = np.nan
    with pytest.raises(SystemExit, match="no offset-0 score"):
        check_gate(args, shots, scores, members, temperatures, calibration)


# ------------------------------------------------------------------ vendor_verify_g.replay
def replay_fixture(tmp_path: Path, monkeypatch, delta: float = 0.0, embedding_dim: int = 5120, order=None) -> tuple:
    run = tmp_path / "fold1/seed0"
    run.mkdir(parents=True)
    pool = pd.DataFrame({"shot_id": [f"p{i}" for i in range(6)], "fold": [0, 1, 1, 2, 1, 0]})
    gate = pd.DataFrame({"shot_id": ["g0", "g1", "g2"]})
    logits = {s: -2.0 + 0.3 * i for i, s in enumerate([*pool.shot_id, *gate.shot_id])}
    t = 1.3
    for role, ids in (("valid", pool.shot_id[pool.fold == 1].tolist()), ("gate", gate.shot_id.tolist())):
        ids = ids if order is None or role != "gate" else order
        z = np.array([logits[s] for s in ids])
        pd.DataFrame({"shot_id": ids, "logit": z, "p": sigmoid(z / t)}).to_csv(run / f"g_{role}_predictions.csv", index=False)
    batches = []
    monkeypatch.setattr(vendor_verify_g, "load_g", lambda _: SimpleNamespace(calibration={"temperature": t}, model=None, spec={},
                                                                             recipe={"cfg": {"batch_size": 4}}))

    def outputs(model, df, spec, batch_size):
        batches.append(batch_size)
        z = np.array([logits[s] for s in df.shot_id]) + delta
        return z, np.zeros(len(df)), np.ones((len(df), embedding_dim), np.float32)

    monkeypatch.setattr(vendor_verify_g, "g_outputs", outputs)
    return run, pool, gate, t, batches


def test_replay_reports_the_largest_probability_difference_in_the_training_batches(tmp_path, monkeypatch):
    run, pool, gate, t, batches = replay_fixture(tmp_path, monkeypatch)
    assert max(vendor_verify_g.replay(run, pool, gate, "g", smoke=False).values()) < 1e-12   # CSV round trip
    assert batches == [4, 4]   # the recipe's batch size, so the batch composition is the training run's
    run, pool, gate, t, _ = replay_fixture(tmp_path / "b", monkeypatch, delta=1e-3)
    worst = vendor_verify_g.replay(run, pool, gate, "g", smoke=False)
    z = np.array([-2.0 + 0.3 * i for i in range(9)])
    assert worst["logit"] == pytest.approx(1e-3) and worst["p"] == pytest.approx(np.abs(sigmoid((z + 1e-3) / t) - sigmoid(z / t)).max())


def test_replay_refuses_other_rows_or_a_broken_embedding(tmp_path, monkeypatch):
    run, pool, gate, *_ = replay_fixture(tmp_path, monkeypatch, order=["g1", "g0", "g2"])
    with pytest.raises(ValueError, match="gate rows differ"):
        vendor_verify_g.replay(run, pool, gate, "g", smoke=False)
    run, pool, gate, *_ = replay_fixture(tmp_path / "b", monkeypatch, embedding_dim=8)
    with pytest.raises(ValueError, match="unexpected embedding"):
        vendor_verify_g.replay(run, pool, gate, "g", smoke=False)


# ------------------------------------------------------------------ vendor_pair_compare
def pairing_tables(root: Path, offset: float = 1500.0, population: str = "secondary") -> None:
    root.mkdir(parents=True)
    pd.DataFrame({"game": ["a", "b"], "sb_match_id": [1, 2]}).to_csv(root / "match_map.csv", index=False)
    pd.DataFrame({"game": ["a", "a"], "half": [1, 2], "offset_ms": [offset, -200.0]}).to_csv(root / "half_offsets.csv", index=False)
    pd.DataFrame({"event_id": ["e1", "e2"], "population": [population, "recovered_goal"]}).to_parquet(root / "populations.parquet")


def compare_fixture(tmp_path: Path, monkeypatch, offset: float, population: str = "secondary") -> Path:
    first, check = tmp_path / "first", tmp_path / "check"
    pairing_tables(first)
    pairing_tables(check, offset, population)
    (first / "populations.parquet").rename(first / "sb_unpaired.parquet")
    pd.DataFrame({"shot_id": ["s1", "s2", "s3"], "sb_event_id": ["e3", "e4", None], "paired": [True, True, False]}).to_parquet(first / "pairs.parquet")
    pd.DataFrame({"shot_id": ["s1", "s2"], "sb_event_id": ["e3", "e4"]}).to_parquet(check / "pairs.parquet")
    goals = pd.DataFrame({"game": ["a", "a"], "half": [1, 2], "anchor_ms": [100, 200], "sb_event_id": ["e2", None], "recovered": [True, False]})
    goals.to_csv(first / "recovered_goals.csv", index=False)
    goals[goals.recovered].drop(columns="recovered").to_csv(check / "recovered_goals.csv", index=False)
    (first / "complete.json").write_text(json.dumps({"identity": {"sources": {"pairing.py": "f" * 64}}}))
    source = tmp_path / "pairing_check.py"
    source.write_text("# independent implementation\n")
    for name, value in (("ROOT", tmp_path), ("FIRST", first), ("CHECK", check), ("CHECK_IMPLEMENTATION_SOURCES", (source,))):
        monkeypatch.setattr(compare, name, value)
    return check / "compare.json"


def test_the_pairing_comparison_accepts_identical_pairings_and_offsets_within_tolerance(tmp_path, monkeypatch):
    report = compare_fixture(tmp_path, monkeypatch, offset=1500.0 + 1e-7)
    compare.main()
    result = json.loads(report.read_text())
    assert result["equal"] and result["comparisons"]["pairs"]["first_count"] == 2   # unpaired shots are not pairs
    assert result["implementation_sources_sha256"]["first"] == {"pairing.py": "f" * 64}


def test_the_pairing_comparison_lists_every_difference_and_fails(tmp_path, monkeypatch):
    report = compare_fixture(tmp_path, monkeypatch, offset=1500.5, population="possible_mispair")
    with pytest.raises(SystemExit) as stop:
        compare.main()
    assert stop.value.code == 1
    result = json.loads(report.read_text())
    assert not result["equal"]
    assert result["comparisons"]["half_offsets"]["differences"] == [{"key": ["a", 1], "first": 1500.0, "check": 1500.5}]
    assert result["comparisons"]["populations"]["differences"] == [{"key": ["e1"], "first": "secondary", "check": "possible_mispair"}]


def test_pairing_comparison_helpers():
    assert compare.values_equal(np.nan, None) and not compare.values_equal(1.0, np.nan)
    assert compare.values_equal(1.0, 1.0 + 1e-9, tolerance=1e-6) and not compare.values_equal("a", "b")
    with pytest.raises(ValueError, match="Duplicate keys"):
        compare.unique_mapping(pd.DataFrame({"k": [1, 1], "v": [1, 2]}), ["k"], "v")
    assert compare.compare_mappings({("a",): 1}, {("b",): 1}) == [{"key": ["a"], "first": 1, "check": "<missing>"},
                                                                   {"key": ["b"], "first": "<missing>", "check": 1}]


# ------------------------------------------------------------------ analysis-script helpers
def probe_args(**given) -> argparse.Namespace:
    return argparse.Namespace(**({"n_boot": None, "seed": None, "thresholds": None, "by_split": False, "prototype": None} | given))


def leakage_args(**given) -> argparse.Namespace:
    return argparse.Namespace(**({"n_boot": None, "seed": None, "offset_logits": None, "expected_rule_sha256": None, "by_split": False,
                                  "prototype_report": None, "development_override": False, "timing_verdict": Path("verdict.json")} | given))


def test_probe_parameters_come_from_the_record_for_dryrun_and_confirmatory_tables(tmp_path):
    for kind in ("dryrun", "confirmatory"):
        params = vendor_probes.probe_parameters(kind, PREREG, probe_args())
        assert (params["n_boot"], params["seed"], params["thresholds"], params["confirmatory_targets"]) == (6000, 2, {"dist": 0.5, "head": 0.7}, ["dist"])
        assert params["control_max_r2"] == 0.004
        with pytest.raises(ValueError, match="needs --prereg"):
            vendor_probes.probe_parameters(kind, None, probe_args())
        with pytest.raises(ValueError, match=r"drop \['n_boot', 'thresholds'\]"):
            vendor_probes.probe_parameters(kind, PREREG, probe_args(n_boot=5000, thresholds=Path("t.json")))
        with pytest.raises(ValueError, match="no development options"):
            vendor_probes.probe_parameters(kind, PREREG, probe_args(by_split=True))
        with pytest.raises(ValueError, match="at least 5000 draws"):
            vendor_probes.probe_parameters(kind, replace(PREREG, n_boot=4999), probe_args())
    (tmp_path / "t.json").write_text('{"dist": 0.4}')
    dev = vendor_probes.probe_parameters("development", None, probe_args(n_boot=200, thresholds=tmp_path / "t.json"))
    assert (dev["n_boot"], dev["seed"], dev["thresholds"], dev["source"]) == (200, 0, {"dist": 0.4}, "command line")
    assert dev["control_max_r2"] == probes.CONTROL_MAX_R2


def test_leakage_parameters_come_from_the_record_and_a_verdict_needs_a_pinned_rule():
    for kind in ("dryrun", "confirmatory"):
        params = vendor_leakage.leakage_parameters(kind, PREREG, leakage_args())
        assert (params["n_boot"], params["seed"], params["offset_logits"], params["expected_rule_sha256"]) == (6000, 2, "raw", "c" * 64)
        with pytest.raises(ValueError, match="needs --timing-verdict"):
            vendor_leakage.leakage_parameters(kind, PREREG, leakage_args(timing_verdict=None))
        with pytest.raises(ValueError, match=r"drop \['expected_rule_sha256', 'offset_logits'\]"):
            vendor_leakage.leakage_parameters(kind, PREREG, leakage_args(offset_logits="calibrated", expected_rule_sha256="d" * 64))
        with pytest.raises(ValueError, match="no development options"):
            vendor_leakage.leakage_parameters(kind, PREREG, leakage_args(development_override=True))
        with pytest.raises(ValueError, match="needs --prereg"):
            vendor_leakage.leakage_parameters(kind, None, leakage_args())
    with pytest.raises(ValueError, match="needs --expected-rule-sha256"):
        vendor_leakage.leakage_parameters("development", None, leakage_args())
    dev = vendor_leakage.leakage_parameters("development", None, leakage_args(timing_verdict=None))
    assert (dev["n_boot"], dev["offset_logits"], dev["expected_rule_sha256"]) == (5000, "calibrated", None)


def test_population_sets_skip_a_combined_set_without_secondary_rows():
    table = synthetic_table(n_matches=10, per_match=6)
    sets, notes = population_sets(table, split_groups=False)
    assert [s[0] for s in sets] == ["primary"] and "combined population skipped" in notes[0]
    table.loc[[2, 9], "population"] = "statsbomb_only"
    sets, notes = population_sets(table, split_groups=True)
    assert [s[0] for s in sets] == ["primary", "combined", "test", "trainvalid"] and not notes


def test_relative_embedding_files_are_read_from_the_table_directory(tmp_path, monkeypatch):
    """Regression (ticket 14 dry run): a relative embedding file name in the sidecar is relative to the table directory,
    not to the working directory, and every file is checked against the sidecar's sha256."""
    table = make_table("dryrun")
    sets = {}
    for column in et.EMBEDDING_COLUMNS:
        np.save(tmp_path / f"{column}.npy", np.arange(12 * 4, dtype=np.float16).reshape(12, 4))
        sets[column] = {"file": f"{column}.npy", "sha256": vendor_probes.sha256_file(tmp_path / f"{column}.npy"), "dim": 4, "rows": 12,
                        "model": "fixture"}
    meta = et.write_table(table, tmp_path, "t", kind="dryrun", embedding_sets=sets)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    embeddings, hashes = vendor_probes.load_all_embeddings(table, meta, tmp_path)
    assert set(embeddings) == {"emb_g_fold0", "emb_dinov2"}   # emb_g_fold1 has no reference in the fixture
    assert embeddings["emb_dinov2"][1].shape == (len(table), 4) and set(hashes) == {"emb_g_fold0.npy", "emb_dinov2.npy"}
    meta["embedding_sets"]["emb_dinov2"]["sha256"] = "0" * 64
    with pytest.raises(et.EvalTableError, match="does not match the sidecar"):
        vendor_probes.load_all_embeddings(table, meta, tmp_path)


def test_probe_prototype_comparison_pairs_own_coverage_rows():
    metrics = pd.DataFrame({"encoder": ["emb_g_fold0", "emb_dinov2", "emb_dinov2"], "subset": ["all", "all", "test"],
                            "coverage": ["own", "own", "own"], "target": "dist", "metric": "R2", "n": [10, 20, 5],
                            "value": [0.6, 0.5, 0.4], "lo": 0.3, "hi": 0.7})
    prototype = pd.DataFrame({"encoder": ["g_combo", "dinov2_frozen"], "population": ["trainvalid", "all"], "target": "dist",
                              "n": [10, 18], "value": [0.55, 0.52]})
    out = vendor_probes.prototype_comparison(metrics, prototype)
    assert out.encoder.tolist() == ["emb_g_fold0", "emb_dinov2"]   # the test subset has no prototype row
    assert out.prototype_population.tolist() == ["trainvalid", "all"] and out.same_shots.tolist() == [True, False]
    assert out.difference.round(6).tolist() == [0.05, -0.02]


def test_leakage_prototype_comparison_maps_specs_to_the_prototype_keys():
    coefficients = pd.DataFrame({"spec": ["gap_on_target_unadjusted", "gap_on_target", "gap_residual_slope_unadjusted_within_200ms"],
                                 "regressor": ["on_target", "on_target", "residual_s"], "subset": "all",
                                 "restriction": ["none", "none", "abs_residual_le_200ms"], "n": [100, 100, 80],
                                 "coef": [0.3, 0.2, 0.1], "lo": 0.0, "hi": 0.5})
    report = {"all|G-vendor|on_target_sn": {"n": 98, "on_target_sn": [0.25, 0.1, 0.4]},
              "all_res200|G-vendor|res_s": {"n": 79, "res_s": [0.15, -0.1, 0.4]}}
    out = vendor_leakage.prototype_comparison(coefficients, report)
    assert out.spec.tolist() == ["gap_on_target_unadjusted", "gap_residual_slope_unadjusted_within_200ms"]   # adjusted specs have no key
    assert out.prototype_key.tolist() == ["all|G-vendor|on_target_sn", "all_res200|G-vendor|res_s"]
    assert out.difference.round(6).tolist() == [0.05, -0.05]
