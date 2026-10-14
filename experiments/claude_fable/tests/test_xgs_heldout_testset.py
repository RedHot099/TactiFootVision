"""The label-free test inputs and checks of the xg-stack test (ticket 04, council C7 Q5), on synthetic data: the Platt
inversion against `ensemble_probability`, the z-distribution, coverage, ball-level and provenance checks with their
stop conditions, the population filter and the extended StatsBomb guard."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.detector import guard
from xgpoc.models.g_bundle import ensemble_probability
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack_heldout import testset as ts
from xgs_fixtures import synthetic_table

CALIBRATION = {"platt": {"a": 0.9803331788739801, "b": -0.051591547304639884}, "oof_rate": 0.1356, "pooled_rate": 0.1356,
               "penalty_rate": 0.6972}


def members(n: int = 50, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    names = [f"f{k}s{s}" for k in range(5) for s in (0, 1)]
    temperatures = dict(zip(names, rng.uniform(1.0, 1.3, 10), strict=True))
    logits = pd.DataFrame(rng.normal(-2.5, 1.0, (n, 10)), columns=names, index=[f"s{i}" for i in range(n)])
    return logits, temperatures


def test_the_inverted_platt_is_the_member_z_of_ensemble_probability():
    logits, temperatures = members()
    p = ensemble_probability([logits[c].to_numpy() for c in temperatures], list(temperatures.values()), CALIBRATION,
                             np.zeros(len(logits), bool))
    z = ts.invert_platt(p, CALIBRATION)
    assert np.allclose(z, ts.member_z(logits, temperatures).to_numpy(), atol=1e-10)
    assert np.allclose(ts.platt(z, CALIBRATION), p, atol=1e-15)
    with pytest.raises(ValueError, match="prior correction"):
        ts.invert_platt(p, CALIBRATION | {"pooled_rate": 0.2})


def test_the_inversion_check_stops_on_a_member_gap_or_a_clipped_probability():
    logits, temperatures = members()
    member = ts.member_z(logits, temperatures)
    p = pd.Series(ts.platt(member.to_numpy(), CALIBRATION), index=member.index)
    z = pd.Series(ts.invert_platt(p.to_numpy(), CALIBRATION), index=p.index)
    good = ts.z_inversion(p, z, member.iloc[:20], CALIBRATION)
    assert good["passed"] and good["member_shots"] == 20 and good["member_max_abs_dz"] < 1e-10
    assert not ts.z_inversion(p, z, member.iloc[:20] + 1e-5, CALIBRATION)["passed"]
    off = ts.z_inversion(p, z + 1e-3, member.iloc[:20] + 1e-3, CALIBRATION)
    assert off["member_max_abs_dz"] < 1e-9 and off["round_trip_max_abs_dp"] > 1e-12 and not off["passed"]
    clipped = p.copy()
    clipped.iloc[0] = 1e-6
    assert ts.z_inversion(clipped, z, member.iloc[:20], CALIBRATION)["p_at_clip"] == 1
    assert not ts.z_inversion(clipped, z, member.iloc[:20], CALIBRATION)["passed"]


@pytest.mark.parametrize(("shift", "scale", "passed"), [(0.0, 1.0, True), (0.0, 0.85, True), (0.0, 0.75, False),
                                                        (0.0, 1.3, False), (-0.7, 1.0, False), (0.3, 1.0, True)])
def test_the_z_distribution_check_stops_outside_0_8_to_1_25(shift, scale, passed):
    rng = np.random.default_rng(1)
    train = rng.normal(-2.2, 1.0, 20000)
    test = -2.2 + shift + scale * (rng.normal(0, 1.0, 20000))
    out = ts.z_distribution(test, train)
    assert out["passed"] is passed and out["bounds"] == [0.8, 1.25]


def coverage_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A test table of the population's exact size: 135 evaluation matches with 3,005 shots, 37 gate with 814."""
    rows = []
    for role, matches in ts.ROLES.items():
        for m, chunk in enumerate(np.array_split(np.arange(ts.SHOTS_BY_ROLE[role]), matches)):
            rows += [{"shot_id": f"{role}_{m:03d}_{i:04d}", "game": f"{role}_{m:03d}", "role": role} for i in chunk]
    table = pd.DataFrame(rows).set_index("shot_id").sort_index()
    table[sf.Z] = -2.0
    for c in [*si.E3_COLUMNS, *si.S_FLAGS]:
        table[c] = 0.0
    predictions = pd.DataFrame({"shot_id": table.index, "p_R": 0.1, "p_G": 0.1})
    return table, predictions


def test_coverage_passes_the_exact_population_and_stops_on_any_gap():
    table, predictions = coverage_table()
    assert ts.coverage(table, predictions)["passed"]
    short = table.iloc[1:]
    assert not ts.coverage(short, predictions.iloc[1:])["checks"]["shots"]
    bad = predictions.copy()
    bad.loc[0, "p_G"] = 1.0
    assert not ts.coverage(table, bad)["checks"]["probabilities_open_unit"]
    no_flag = table.copy()
    no_flag.iloc[0, no_flag.columns.get_loc("sn_corner_20s")] = np.nan
    assert not ts.coverage(no_flag, predictions)["checks"]["flags"]
    reordered = predictions.iloc[::-1]
    assert not ts.coverage(table, reordered)["checks"]["predictions_align"]


def test_ball_levels_report_a_level_unseen_in_training_without_stopping():
    train, test = synthetic_table(seed=0), synthetic_table(seed=1)
    train[si.col("ball_m800", 0)] = 0.0
    out = ts.ball_levels(test, train)
    assert out["unseen_in_training"] == [si.col("ball_m800", 0)] and out["passed"]


def test_provenance_stops_on_shared_shots_or_matches_and_on_other_columns():
    train = synthetic_table(seed=0)
    test = synthetic_table(seed=1).assign(role="evaluation")
    test.index = [f"t{i}" for i in range(len(test))]
    test["game"] = "held_" + test.game
    spec = sf.spec("bc", train)
    frozen = {name: spec.arms[name].columns for name in ("R", "G", "GS")}
    annotations = {g: [] for g in {*test.game, *train.game}}
    assert ts.provenance(test, train, spec, frozen, annotations)["passed"]
    shared = test.copy()
    shared.iloc[0, shared.columns.get_loc("game")] = train.game.iloc[0]
    assert not ts.provenance(shared, train, spec, frozen, annotations)["checks"]["disjoint_from_training"]["passed"]
    other = frozen | {"G": [*frozen["G"], "extra"]}
    assert not ts.provenance(test, train, spec, other, annotations)["checks"]["columns:G"]["passed"]


def test_heldout_shots_keep_the_non_penalty_evaluation_and_gate_shots(tmp_path):
    index = pd.DataFrame({"shot_id": ["c", "a", "b", "d", "e"], "game": ["g1", "g1", "g2", "g3", "g2"], "half": 1,
                          "position_ms": [1, 2, 3, 4, 5], "is_penalty": [False, False, False, False, True],
                          "goal": [1, 0, 0, 1, 1]})
    index.to_parquet(tmp_path / "index.parquet")
    plan = pd.DataFrame({"game": ["g1", "g2", "g3"], "role": ["evaluation", "gate", "training"]})
    shots = ts.heldout_shots(tmp_path / "index.parquet", plan)
    assert shots.index.tolist() == ["a", "b", "c"] and "goal" not in shots and shots.role.tolist() == ["evaluation", "gate", "evaluation"]


def test_no_vendor_reads_refuses_xg_compares_table_and_restores_the_guard(tmp_path):
    original = guard.FORBIDDEN
    target = tmp_path / "results/xgc_compare/table/shots.parquet"
    target.parent.mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_parquet(target)
    with ts.no_vendor_reads():
        with pytest.raises(guard.StatsBombRead):
            pd.read_parquet(target)
        with pytest.raises(guard.StatsBombRead):
            open(tmp_path / "statsbomb_open_v1.json", "w")
    assert guard.FORBIDDEN is original
    assert pd.read_parquet(target).a.tolist() == [1]


def test_heldout_table_refuses_incomplete_e3_or_a_misordered_z(tmp_path, monkeypatch):
    shots = pd.DataFrame({"game": ["g", "g"], "half": 1, "position_ms": [1, 2], "role": "gate"},
                         index=pd.Index(["a", "b"], name="shot_id"))
    pd.DataFrame({"shot_id": ["a"]}).to_parquet(tmp_path / "frames.parquet")
    pd.DataFrame({"shot_id": ["a"]}).to_parquet(tmp_path / "features.parquet")
    detector = pd.DataFrame({c: 0.0 for c in si.E3_COLUMNS}, index=pd.Index(["a"], name="shot_id"))
    monkeypatch.setattr(si, "e3_table", lambda frames, features: (detector, {}))
    z = pd.Series([-2.0, -1.0], index=shots.index)
    with pytest.raises(ValueError, match="E3 does not cover"):
        ts.heldout_table(shots, z, tmp_path, {"g": []})
    monkeypatch.setattr(si, "e3_table", lambda frames, features: (detector.reindex(shots.index), {"0": 0}))
    with pytest.raises(ValueError, match="shot order"):
        ts.heldout_table(shots, z.iloc[::-1], tmp_path, {"g": []})
    table, wrapped = ts.heldout_table(shots, z, tmp_path, {"g": []})
    assert table[sf.Z].tolist() == [-2.0, -1.0] and wrapped == {"0": 0} and set(si.S_FLAGS) <= set(table)
