"""Positional comparator (xg-vendor ticket 06): features with known geometry, the post-contact guard, a planted signal,
the training pool that never holds an evaluation match, label-free predictions and kill rule K1."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_comparator as script  # noqa: E402
from xgpoc.vendor import comparator as cmp  # noqa: E402
from xgpoc.vendor import comparator_features as cf  # noqa: E402
from xgpoc.vendor.statsbomb_tables import context_row, shot_row  # noqa: E402

FAST_BOOST = (cmp.BoostConfig(0.1, 8, 20, 0.0, 60),)
FAST_SPLINE = (cmp.SplineConfig(4, 1.0),)


def shot_event(event_id="s1", location=(108.0, 40.0), outcome="Off T", shot_type="Open Play", key_pass_id=None, freeze=(),
               period=1, **shot_extra):
    body = {"statsbomb_xg": 0.2, "end_location": [120.0, 38.0, 1.0], "outcome": {"name": outcome}, "body_part": {"name": "Right Foot"},
            "type": {"name": shot_type}, "technique": {"name": "Normal"}} | shot_extra
    if key_pass_id:
        body["key_pass_id"] = key_pass_id
    if freeze is not None and shot_type != "Penalty":
        body["freeze_frame"] = [{"location": list(loc), "player": {"id": i}, "teammate": mate,
                                 "position": {"name": "Goalkeeper" if gk else "Center Back"}} for i, (loc, mate, gk) in enumerate(freeze)]
    return {"id": event_id, "index": 1, "period": period, "timestamp": "00:01:00.000", "minute": 1, "second": 0, "type": {"name": "Shot"},
            "possession": 1, "play_pattern": {"name": "Regular Play"}, "team": {"id": 1, "name": "Alpha"}, "player": {"id": 9, "name": "P"},
            "location": list(location), "duration": 0.8, "related_events": ["gk1"], "shot": body}


def pass_event(event_id="p1", **pass_fields):
    return {"id": event_id, "index": 0, "period": 1, "timestamp": "00:00:58.000", "type": {"name": "Pass"}, "team": {"name": "Alpha"},
            "pass": {"length": 20.0, "height": {"name": "High Pass"}, "end_location": [108.0, 40.0], "shot_assist": True} | pass_fields}


def tables(events, passes=(), match_id=1):
    shots = pd.DataFrame([shot_row(match_id, e) for e in events])
    context = pd.DataFrame([context_row(match_id, p) for p in passes], columns=["match_id", "event_id", "type", "event_json", "index",
                                                                               "period", "timestamp", "timestamp_ms", "team"])
    return shots, context


# ------------------------------------------------------------------ features
def test_geometry_and_freeze_frame_counts_are_known_by_construction():
    freeze = [((112.0, 40.0), False, False),   # outfield opponent in the cone, 4 units away
              ((119.0, 40.0), False, True),    # goalkeeper on the shot line, 1 unit deep
              ((109.0, 40.5), False, False),   # opponent within 1.5, just outside the cone (half-width 1/3 there)
              ((100.0, 60.0), True, False)]    # teammate far outside the cone
    row = cf.feature_table(*tables([shot_event(freeze=freeze)])).iloc[0]
    assert row.dist == pytest.approx(12.0)
    assert row.angle == pytest.approx(2 * np.arctan(4 / 12))
    assert (row.ff_def_cone, row.ff_mate_cone, row.ff_opp_1m, row.ff_opp_5m, row.ff_n_opp) == (1, 0, 1, 2, 3)
    assert row.ff_near_opp == pytest.approx(np.hypot(1.0, 0.5))
    assert (row.gk_depth, row.gk_off_line, row.gk_in_cone, row.ff_has_gk) == (pytest.approx(1.0), pytest.approx(0.0), 1, 1)
    assert row.kp_has == 0 and np.isnan(row.kp_length)


def test_key_pass_features_and_missing_freeze_frame():
    events = [shot_event(key_pass_id="p1", freeze=None), shot_event("s2", shot_type="Penalty")]
    f = cf.feature_table(*tables(events, [pass_event(cross=True, type={"name": "Corner"})]))
    assert (f.kp_has[0], f.kp_cross[0], f.kp_length[0], f.kp_type[0], f.kp_height[0]) == (1, 1, 20.0, "Corner", "High Pass")
    assert f.ff_n.tolist() == [0, 0] and f.ff_def_cone.isna().all() and f.gk_depth.isna().all()
    assert "goal" not in f.columns and "outcome" not in f.columns and "statsbomb_xg" not in f.columns


def test_features_do_not_change_with_post_contact_fields():
    freeze = [((112.0, 41.0), False, False), ((118.0, 39.0), False, True)]
    calm = cf.feature_table(*tables([shot_event(key_pass_id="p1", freeze=freeze)], [pass_event()]))
    goal = shot_event(key_pass_id="p1", freeze=freeze, outcome="Goal", statsbomb_xg=0.9, deflected=True, redirect=True,
                      saved_to_post=True, end_location=[120.0, 40.0, 0.5])
    goal |= {"duration": 0.1, "out": True, "related_events": [], "off_camera": True}
    loud = cf.feature_table(*tables([goal], [pass_event(goal_assist=True, shot_assist=None)]))
    pd.testing.assert_frame_equal(calm, loud)


def test_pre_contact_views_drop_every_post_contact_field():
    shot = cf.pre_contact_shot(shot_event(deflected=True) | {"out": True})
    assert not {"outcome", "end_location", "statsbomb_xg", "deflected"} & set(shot["shot"])
    assert not {"duration", "out", "related_events"} & set(shot)
    kp = cf.pre_contact_pass(pass_event(goal_assist=True))
    assert not {"goal_assist", "shot_assist"} & set(kp["pass"]) and kp["pass"]["length"] == 20.0


@pytest.mark.parametrize("extra", ["end_x", "end_location_z", "outcome", "statsbomb_xg", "shot_deflected", "kp_goal_assist", "goal",
                                   "duration", "out"])
def test_adding_a_post_contact_feature_fails_the_guard(extra, monkeypatch):
    with pytest.raises(cf.PostContactFeature):
        cf.check_features([*cf.FREEZE_FRAME, extra])
    monkeypatch.setitem(cmp.RUNG_FEATURES, "event", [*cf.EVENT, extra])
    with pytest.raises(cf.PostContactFeature):
        cmp.fit_rung("event", planted_pool(50), np.arange(50) % 2, grid=FAST_BOOST)


def test_guard_accepts_the_frozen_lists_and_refuses_unknown_features():
    for features in cf.RUNG_FEATURES.values():
        cf.check_features(features)
    with pytest.raises(ValueError, match="not in the frozen feature list"):
        cf.check_features([*cf.EVENT, "player_rating"])


# ------------------------------------------------------------------ planted signal
def planted_pool(n_matches: int, shots_per_match: int = 40, seed: int = 0) -> pd.DataFrame:
    """A pool whose goal probability falls with distance and with defenders in the cone; every other feature is noise."""
    rng = np.random.default_rng(seed)
    n = n_matches * shots_per_match
    pool = pd.DataFrame({c: rng.normal(size=n) for c in cf.FREEZE_FRAME if c not in cf.CATEGORICAL})
    for c in cf.CATEGORICAL:
        pool[c] = rng.choice(["a", "b", "c"], size=n)
    pool["dist"] = rng.uniform(4, 35, size=n)
    pool["ff_def_cone"] = rng.integers(0, 4, size=n).astype(float)
    z = 1.5 - 0.15 * pool.dist - 1.2 * pool.ff_def_cone
    pool["goal"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    pool["match_id"] = np.repeat(np.arange(n_matches), shots_per_match)
    pool["event_id"] = [f"e{i}" for i in range(n)]
    return pool


def test_a_planted_signal_is_learned_and_each_rung_adds_what_it_sees():
    pool = planted_pool(60)
    folds = cmp.match_folds(pool.match_id, 5)
    fits = {name: cmp.fit_rung(name, pool, folds, grid=FAST_SPLINE if name == "dist_angle" else FAST_BOOST, n_jobs=1) for name in cmp.RUNGS}
    base = cmp.metrics(pool.goal, np.full(len(pool), pool.goal.mean()))["log_loss"]
    assert fits["dist_angle"].cv_log_loss < base - 0.02                        # distance is learned
    assert fits["freeze_frame"].cv_log_loss < fits["event"].cv_log_loss - 0.02  # the cone count only the freeze-frame rung sees
    probe = pool.iloc[:2].copy()
    probe[["dist", "ff_def_cone"]] = [[8.0, 0.0], [8.0, 3.0]]
    z = fits["freeze_frame"].model.logit(probe)
    assert z[0] - z[1] > 1.5
    oof = cmp.sigmoid(fits["freeze_frame"].oof_logit)
    assert cmp.metrics(pool.goal, oof)["log_loss"] == pytest.approx(fits["freeze_frame"].cv_log_loss, abs=1e-9)


def test_iteration_count_is_chosen_on_the_curve_and_edges_are_reported():
    pool = planted_pool(30)
    fit = cmp.fit_rung("event", pool, cmp.match_folds(pool.match_id, 3), grid=(cmp.BoostConfig(0.1, 4, 20, 0.0, 400),
                                                                                  cmp.BoostConfig(0.1, 8, 20, 0.0, 400)))
    assert 1 <= fit.iterations < 400
    assert any(e.startswith("max_leaf_nodes") for e in fit.grid_edges)
    assert len(fit.grid) == 2 and fit.grid.cv_log_loss.min() == pytest.approx(fit.cv_log_loss)


def test_folds_hold_whole_matches():
    folds = cmp.match_folds(np.repeat(np.arange(23), 7), 5)
    by_match = pd.Series(folds).groupby(np.repeat(np.arange(23), 7)).nunique()
    assert by_match.eq(1).all() and set(folds) == set(range(5))


# ------------------------------------------------------------------ pool, evaluation shots, predictions
def multi_match_tables():
    events, rows = [], []
    for match_id in (1, 2, 3):
        for k, (shot_type, period, outcome) in enumerate([("Open Play", 1, "Goal"), ("Open Play", 2, "Saved"), ("Penalty", 1, "Goal"),
                                                          ("Penalty", 5, "Saved"), ("Free Kick", 4, "Off T")]):
            e = shot_event(f"m{match_id}s{k}", shot_type=shot_type, period=period, outcome=outcome)
            rows.append(shot_row(match_id, e))
            events.append(e)
    shots = pd.DataFrame(rows)
    return shots, cf.feature_table(shots, tables([])[1])


def test_no_evaluation_match_enters_training():
    shots, features = multi_match_tables()
    pool = cmp.training_pool(features, shots, excluded_matches={2}, evaluation_matches=[2])
    assert set(pool.match_id) == {1, 3}
    assert pool.shot_type.ne("Penalty").all() and pool.period.le(4).all() and len(pool) == 6
    assert pool.goal.tolist() == [1, 0, 0, 1, 0, 0]
    rate = cmp.penalty_rate(shots, {2})
    assert rate == {"penalties": 2, "goals": 2, "rate": 1.0}   # the period-5 kicks are shoot-out kicks
    with pytest.raises(cmp.OverlapError):
        cmp.training_pool(features, shots, excluded_matches={2}, evaluation_matches=[2, 3])
    with pytest.raises(cmp.OverlapError):
        cmp.check_disjoint(pool.match_id, [3, 4])


def test_predictions_are_label_free_and_refuse_a_training_match():
    shots, features = multi_match_tables()
    pool = cmp.training_pool(features, shots, excluded_matches={2}, evaluation_matches=[2])
    model = cmp.RungModel("dist_angle", ["dist", "angle"], {}, type("Fixed", (), {"decision_function": lambda self, X: X[:, 0] * 0 - 2.0})())
    models = dict.fromkeys(cmp.RUNGS, model)
    evaluation = pd.DataFrame({"statsbomb_event_id": ["m2s0", "m2s2"], "statsbomb_match_id": [2, 2], "population": ["primary", "primary"],
                               "soccernet_shot_id": ["sn1", None]})
    out = cmp.predict(models, features, evaluation, constant=0.1, penalty=0.75, train_matches=pool.match_id)
    assert out.prediction_source.tolist() == ["model", "penalty_rate"]
    assert out.p_event.tolist() == pytest.approx([cmp.sigmoid(-2.0), 0.75])
    assert out.p_constant.tolist() == pytest.approx([0.1, 0.75])
    assert not {"goal", "outcome", "statsbomb_xg", "sb_outcome"} & set(out.columns)
    with pytest.raises(cmp.OverlapError):
        cmp.predict(models, features, evaluation, constant=0.1, penalty=0.75, train_matches=[1, 2])


def test_evaluation_shots_cover_both_populations_and_leave_mispairs_and_unpaired_penalties_out():
    pairs = pd.DataFrame({"shot_id": ["a", "b", "c"], "paired": [True, True, False], "sb_event_id": ["e1", "e2", None],
                          "sb_match_id": pd.array([7, 7, pd.NA], dtype="Int64")})
    unpaired = pd.DataFrame({"event_id": ["e3", "e4", "e5", "e6"], "match_id": [7, 7, 8, 8],
                             "population": ["secondary", "recovered_goal", "possible_mispair", "penalty"]})
    out = cmp.evaluation_shots(pairs, unpaired)
    assert out.statsbomb_event_id.tolist() == ["e1", "e2", "e3", "e4"]
    assert out.population.tolist() == ["primary", "primary", "statsbomb_only", "recovered_goal"]
    assert out.soccernet_shot_id.tolist()[:2] == ["a", "b"] and out.soccernet_shot_id[2:].isna().all()


def test_k1_fires_beyond_the_margin():
    assert cmp.k1(0.2478, 0.2473)["passed"]
    assert cmp.k1(0.2400, 0.2473)["passed"]
    result = cmp.k1(0.2600, 0.2473)
    assert not result["passed"] and result["gap"] == pytest.approx(0.0127)
    with pytest.raises(cmp.K1Failed) as failed:
        cmp.check_k1(result, {"k1": result})
    assert failed.value.report == {"k1": result}
    cmp.check_k1(cmp.k1(0.2478, 0.2473))


def test_pool_metrics_per_group_compare_with_the_vendor():
    cv = pd.DataFrame({"goal": [1, 0, 0, 1, 0, 0], "statsbomb_xg": [0.6, 0.2, 0.1, 0.5, 0.3, 0.1], "p_freeze_frame": [0.6, 0.2, 0.1, 0.4, 0.3, 0.2],
                       "competition": ["A"] * 3 + ["B"] * 3, "season": "2015/2016"})
    m = cmp.pool_metrics(cv, ["competition", "season"])
    a = m[(m.competition == "A") & (m.rung == "freeze_frame")].iloc[0]
    assert a.log_loss_minus_vendor == pytest.approx(0.0) and a.n == 3
    b = m[(m.competition == "B") & (m.rung == "freeze_frame")].iloc[0]
    assert b.log_loss_minus_vendor > 0


# ------------------------------------------------------------------ script shell
def test_pairing_inputs_are_refused_when_a_file_changed(tmp_path):
    for name in script.PAIRING_FILES:
        (tmp_path / name).write_text(name)
    files = {name: script.file_sha256(tmp_path / name) for name in script.PAIRING_FILES}
    (tmp_path / "complete.json").write_text(json.dumps({"files": files, "identity": {"inputs": {"statsbomb": {"complete_json_sha256": "x"}}}}))
    assert script.verify_pairing(tmp_path, "x")["files"] == files
    with pytest.raises(ValueError, match="another StatsBomb version"):
        script.verify_pairing(tmp_path, "y")
    (tmp_path / "pairs.parquet").write_text("changed")
    with pytest.raises(ValueError, match="pairs.parquet"):
        script.verify_pairing(tmp_path, "x")


def test_k1_failure_stops_with_a_report_and_publishes_nothing(tmp_path, monkeypatch):
    def failing_build(*args):
        cmp.check_k1(cmp.k1(0.30, 0.25), {"k1": cmp.k1(0.30, 0.25)})
    monkeypatch.setattr(script, "read_inputs", lambda root: (None, None, None, None, {}))
    monkeypatch.setattr(script, "build", failing_build)
    out = tmp_path / "xgv_comparator"
    with pytest.raises(SystemExit, match="K1 fired"):
        script.main(["--out", str(out)])
    assert not out.exists()
    assert json.loads((tmp_path / "xgv_comparator_k1_failed.json").read_text())["k1"]["passed"] is False
