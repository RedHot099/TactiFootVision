"""The StatsBomb audit, config choice and gate on fixtures with known answers (xg-detector 02 and 09)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import supervision as sv
from xgd_fixtures import frame_with
from xgpoc.bundle import sha256_file
from xgpoc.detector import audit
from xgpoc.detector import extract as ex
from xgpoc.detector.conventions import GOAL, YARD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xgd_audit  # noqa: E402
import xgd_kp_diag  # noqa: E402

ATT, DEF = 0, 1
# Columns of the real tables that carry a goal or an outcome: none may ever be read.
OUTCOME_COLUMNS = {"goal", "sb_goal", "sb_outcome", "sb_xg", "outcome", "statsbomb_xg", "event_json", "label", "goal_v1_1"}


def _player(x, y, *, teammate, position="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": position}, "player": {"name": "p"}}


# ------------------------------------------------------------ truth
def test_statsbomb_truth_in_aligned_metres():
    pop = pd.DataFrame({"shot_id": ["s"], "game": ["g"], "sb_event_id": ["e"], "sb_x": [108.0], "sb_y": [40.0], "residual_ms": [0.0]})
    frame = [_player(119.0, 40.0, teammate=False, position="Goalkeeper"), _player(114.0, 40.0, teammate=False),
             _player(112.0, 39.0, teammate=True), _player(100.0, 20.0, teammate=False)]
    t = audit.truth(pop, pd.Series({"e": json.dumps(frame)})).loc["s"]
    assert (t.sb_shot_x, t.sb_shot_y) == pytest.approx((GOAL[0] - 12 * YARD, 34.0))
    assert (t.sb_gk_x, t.sb_gk_y) == pytest.approx((GOAL[0] - YARD, 34.0))
    assert t.cone_sb == 1.0 and t.sb_has_ff == 1.0       # the goalkeeper and the teammate are not cone defenders


def test_a_shot_without_freeze_frame_or_goalkeeper_has_no_such_truth():
    pop = pd.DataFrame({"shot_id": ["a", "b"], "game": ["g", "g"], "sb_event_id": ["e1", "e2"], "sb_x": [100.0, 100.0],
                        "sb_y": [40.0, 40.0], "residual_ms": [0.0, 0.0]})
    t = audit.truth(pop, pd.Series({"e1": None, "e2": json.dumps([_player(118.0, 40.0, teammate=False)])}))
    assert t.loc["a"].sb_has_ff == 0.0 and np.isnan(t.loc["a"].cone_sb) and np.isnan(t.loc["a"].sb_gk_x)
    assert t.loc["b"].cone_sb == 1.0 and np.isnan(t.loc["b"].sb_gk_x)
    assert not np.isnan(t.loc["a"].sb_shot_x)            # the shot location is always known


def test_a_statsbomb_event_paired_twice_is_refused():
    pairs = pd.DataFrame({c: ["x", "y"] for c in audit.PAIR_COLUMNS} | {"shot_id": ["a", "b"], "paired": [True, True],
                         "is_penalty": [False, False], "sb_event_id": ["e", "e"]})
    with pytest.raises(ValueError, match="twice"):
        audit.population(pairs)


def test_the_timing_table_refuses_a_missing_residual():
    with pytest.raises(ValueError, match="no timing residual"):
        audit.timing_table(_table([{"residual_ms": np.nan}]))


def test_population_is_paired_non_penalty_and_refuses_other_columns():
    pairs = pd.DataFrame({c: ["x", "y", "z"] for c in audit.PAIR_COLUMNS} | {"shot_id": ["c", "a", "b"],
                         "paired": [True, True, False], "is_penalty": [False, True, False]})
    assert audit.population(pairs).shot_id.tolist() == ["c"]
    with pytest.raises(ValueError, match="reads only"):
        audit.population(pairs.assign(goal=True))


# ------------------------------------------------------------ errors and statistics
def _per_shot(det_shot, sb_shot, det_gk=(np.nan, np.nan), sb_gk=(np.nan, np.nan)):
    pop = pd.DataFrame({"shot_id": ["s"], "game": ["g"], "residual_ms": [10.0]})
    det = pd.DataFrame({"shot_ok": [1.0], "shot_from_ball": [1.0], "shooter_by_goal": [0.0], "ball_fallback": [0.0],
                        "ball_offset_ms": [0.0], "homography_ok": [1.0], "gk_ok": [1.0], "gk_detected": [0.0], "gk_proxy": [1.0],
                        "gk_proxy_same_team": [0.0], "det_shot_x": [det_shot[0]], "det_shot_y": [det_shot[1]],
                        "det_gk_x": [det_gk[0]], "det_gk_y": [det_gk[1]], "cone_det": [1.0], "kp_conf": [0.9],
                        "kp_conf_t0": [0.9]}, index=pd.Index(["s"], name="shot_id"))
    sb = pd.DataFrame({"sb_shot_x": [sb_shot[0]], "sb_shot_y": [sb_shot[1]], "sb_has_ff": [1.0], "sb_gk_x": [sb_gk[0]],
                       "sb_gk_y": [sb_gk[1]], "cone_sb": [1.0]}, index=pd.Index(["s"], name="shot_id"))
    return audit.compare(pop, sb, det, "C9").iloc[0]


def test_errors_are_euclidean_metres_and_the_mirror_check_flips_y():
    row = _per_shot((93.0, 30.0), (90.0, 34.0), det_gk=(104.0, 35.0), sb_gk=(104.0, 34.0))
    assert (row.shot_err, row.shot_dx, row.shot_dy) == (5.0, 3.0, -4.0)
    assert row.shot_err_mirror_y == pytest.approx(np.hypot(3.0, 38.0 - 34.0))
    assert row.gk_err == 1.0


def test_a_shot_without_a_detected_row_is_not_covered():
    pop = pd.DataFrame({"shot_id": ["s", "t"], "game": ["g", "g"], "residual_ms": [0.0, 0.0]})
    det = pd.DataFrame({c: [1.0] for c in ("shot_ok", "shot_from_ball", "shooter_by_goal", "ball_fallback", "homography_ok", "gk_ok",
                                           "gk_detected", "gk_proxy")} | {"det_shot_x": [90.0], "det_shot_y": [30.0],
                                                                         "det_gk_x": [np.nan], "det_gk_y": [np.nan]},
                       index=pd.Index(["s"], name="shot_id"))
    sb = pd.DataFrame({"sb_shot_x": [90.0, 90.0], "sb_shot_y": [30.0, 30.0], "sb_gk_x": np.nan, "sb_gk_y": np.nan},
                      index=pd.Index(["s", "t"], name="shot_id"))
    out = audit.compare(pop, sb, det, "C9")
    assert out.shot_ok.tolist() == [1.0, 0.0] and out.shot_err.tolist()[0] == 0.0 and np.isnan(out.shot_err[1])


def test_quantiles_are_inverted_cdf_and_draws_weigh_rows_by_their_match():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    games = np.array(["a", "a", "b", "c"])
    ones = audit.Draws(np.array(["a", "b", "c"]), np.ones((3, 3), np.int32), np.array([0, 0, 1, 2]))
    (point, lo, hi), = audit.quantiles(values, np.ones(4, bool), (0.5,), ones)
    assert point == 2.0 == lo == hi                    # every draw is the sample itself
    heavy_c = audit.Draws(ones.keys, np.array([[0, 0, 5]], np.int32), ones.match)
    assert audit.quantiles(values, np.ones(4, bool), (0.5,), heavy_c)[0][1] == 10.0   # only match c drawn
    only_a = audit.Draws(ones.keys, np.array([[2, 0, 0]], np.int32), ones.match)
    assert audit.share(values < 2, np.ones(4, bool), only_a)[1] == 0.5
    d = audit.draws(games, n_boot=50, seed=1)
    assert d.counts.shape == (50, 3) and (d.counts.sum(axis=1) == 3).all()


def _table(rows):
    """A per-shot table of planted rows (dicts overriding a covered, exactly located shot)."""
    base = {"config": "C9", "game": "g", "residual_ms": 0.0, "shot_ok": 1.0, "shot_from_ball": 1.0, "shooter_by_goal": 0.0,
            "ball_fallback": 0.0, "gk_ok": 1.0, "gk_detected": 0.0, "gk_proxy": 1.0, "gk_proxy_same_team": 0.0, "kp_conf": 0.9,
            "kp_conf_t0": 0.9, "shot_err": 0.0, "shot_dx": 0.0, "shot_dy": 0.0, "shot_err_mirror_y": 9.0, "gk_err": 0.0,
            "det_shot_x": 90.0, "det_shot_y": 34.0, "sb_shot_x": 90.0, "sb_shot_y": 34.0,
            "det_gk_x": 104.0, "det_gk_y": 34.0, "sb_gk_x": 104.0, "sb_gk_y": 34.0, "cone_det": 1.0, "cone_sb": 1.0}
    return pd.DataFrame([base | {"shot_id": f"s{i}"} | r for i, r in enumerate(rows)])


def test_config_metrics_on_planted_rows():
    table = _table([{"shot_err": 1.5, "cone_det": 2.0}, {"shot_err": 4.0, "shot_from_ball": 0.0, "shooter_by_goal": 1.0, "kp_conf": 0.3},
                    {"shot_ok": 0.0, "shot_err": np.nan, "gk_ok": 0.0, "gk_proxy": 0.0, "cone_det": np.nan},
                    {"shot_err": 0.5, "gk_proxy": 0.0, "gk_detected": 1.0, "gk_err": 2.5, "cone_sb": np.nan, "sb_gk_x": np.nan}])
    d = audit.draws(table.game.to_numpy(), n_boot=20)
    m = pd.DataFrame(audit.config_metrics(table, d)).set_index(["quantity", "subset", "stat"]).value
    assert m["shot_location", "all", "coverage"] == 0.75 and m["shot_location", "all", "n"] == 3
    assert m["shot_location", "all", "median"] == 1.5 and m["shot_location", "all", "lt1"] == pytest.approx(1 / 3)
    assert m["shot_location", "flag", "median"] == 4.0 and m["shot_location", "ball", "n"] == 2
    assert m["shot_location", "kp05", "n"] == 2 and m["shot_location", "all", "median_mirror_y"] == 9.0
    assert m["gk_any", "all", "coverage"] == 0.75 and m["gk_any", "all", "n"] == 2   # shot 3 has no StatsBomb goalkeeper
    assert m["gk_detected", "all", "coverage"] == 0.25 and m["gk_detected", "all", "n"] == 0
    assert m["gk_proxy_other", "all", "n"] == 2 and m["gk_proxy_same", "all", "n"] == 0
    assert m["def_cone", "all", "n"] == 2 and m["def_cone", "all", "exact"] == 0.5
    assert m["def_cone", "all", "ge1_agree"] == 1.0 and m["def_cone", "all", "ge2_agree"] == 0.5
    assert m["def_cone", "all", "mean_diff"] == 0.5


def test_correlations_follow_the_draws():
    a, b = np.array([1.0, 2.0, 3.0, 4.0, 9.0]), np.array([2.0, 4.0, 6.0, 8.0, 0.0])
    keys = np.array(["a", "b"])
    match = np.array([0, 0, 0, 0, 1])
    point, lo, hi = audit.correlation(a, b, np.ones(5, bool), audit.Draws(keys, np.array([[2, 0], [1, 1]], np.int32), match))
    assert point == pytest.approx(np.corrcoef(a, b)[0, 1])
    # the two draws: match a twice (b = 2a exactly, correlation 1) and the sample itself
    assert (lo, hi) == (pytest.approx(np.percentile([1.0, point], 2.5)), pytest.approx(np.percentile([1.0, point], 97.5)))
    assert np.isnan(audit.correlation(a, b, np.eye(5, dtype=bool)[0], audit.Draws(keys, np.ones((2, 2), np.int32), match))[0])


def test_spearman_ranks_each_resample_with_ties_as_scipy_does():
    from scipy.stats import rankdata, spearmanr

    a = np.array([3.0, 1.0, 2.0, 2.0, 50.0, 7.0])
    b = np.array([1.0, 1.0, 4.0, 3.0, 2.0, 9.0])
    w = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 1, 0, 3, 1]], float)
    ranks = audit.weighted_ranks(a, w)
    np.testing.assert_allclose(ranks[0], rankdata(a))
    repeated = np.repeat(np.arange(6), w[1].astype(int))           # the resample the second draw stands for
    np.testing.assert_allclose(ranks[1][repeated], rankdata(a[repeated]))
    keys, match = np.array(list("abcdef")), np.arange(6)
    d = audit.Draws(keys, w.astype(np.int32), match)
    point, lo, hi = audit.correlation(a, b, np.ones(6, bool), d, rank=True)
    assert point == pytest.approx(spearmanr(a, b)[0])
    drawn = spearmanr(a[repeated], b[repeated])[0]
    assert (lo, hi) == (pytest.approx(np.percentile([point, drawn], 2.5)), pytest.approx(np.percentile([point, drawn], 97.5)))


def test_the_constant_baselines_guess_the_population_median_on_the_compared_shots():
    table = _table([{"sb_shot_x": 90.0, "sb_gk_x": 104.0, "sb_gk_y": 33.0},
                    {"sb_shot_x": 94.0, "sb_gk_x": 100.0, "sb_gk_y": 34.0},
                    {"sb_shot_x": 98.0, "sb_gk_x": 103.0, "sb_gk_y": 35.0, "shot_ok": 0.0, "gk_ok": 0.0, "gk_proxy": 0.0},
                    {"sb_shot_x": 99.0, "sb_gk_x": np.nan, "sb_gk_y": np.nan}])
    m = pd.DataFrame(audit.config_metrics(table, audit.draws(table.game.to_numpy(), n_boot=5)))
    m = m.set_index(["quantity", "subset", "stat"]).value
    # shot: the median of 90, 94, 98, 99 (inverted CDF) is 94, scored on the three located shots: 4, 0, 5
    assert (m["shot_constant", "all", "const_x"], m["shot_constant", "all", "const_y"]) == (94.0, 34.0)
    assert m["shot_constant", "all", "n"] == 3 and m["shot_constant", "all", "median"] == 4.0
    assert m["shot_constant", "all", "coverage"] == m["shot_location", "all", "coverage"] == 0.75
    # goalkeeper: the median over the three StatsBomb goalkeepers is (103, 34), scored on the two compared shots
    assert (m["gk_constant", "all", "const_x"], m["gk_constant", "all", "const_y"]) == (103.0, 34.0)
    assert m["gk_constant", "all", "n"] == m["gk_any", "all", "n"] == 2
    assert m["gk_constant", "all", "median"] == pytest.approx(np.hypot(1.0, 1.0))      # (104, 33) and (100, 34): 1.41 and 3
    assert m["gk_constant", "all", "lt2"] == 0.5


def test_correlations_of_the_detected_with_the_statsbomb_positions():
    table = _table([{"det_shot_x": x, "sb_shot_x": 2 * x, "det_shot_y": y, "sb_shot_y": -y, "det_gk_x": x, "sb_gk_x": x + 1,
                     "det_gk_y": y, "sb_gk_y": y} for x, y in ((80.0, 20.0), (85.0, 30.0), (95.0, 31.0))])
    m = pd.DataFrame(audit.config_metrics(table, audit.draws(table.game.to_numpy(), n_boot=5)))
    m = m.set_index(["quantity", "subset", "stat"]).value
    assert m["shot_location", "all", "corr_x"] == pytest.approx(1.0) and m["shot_location", "all", "corr_y"] == pytest.approx(-1.0)
    assert m["gk_any", "all", "corr_x"] == pytest.approx(1.0) and m["gk_any", "all", "corr_y"] == pytest.approx(1.0)
    assert m["shot_location", "all", "rank_corr_y"] == pytest.approx(-1.0) and m["gk_any", "all", "rank_corr_x"] == pytest.approx(1.0)
    assert m["shot_location", "all", "off_pitch"] == 0.0


def test_off_pitch_counts_detected_locations_outside_the_pitch():
    table = _table([{"det_shot_x": 106.0}, {"det_shot_y": -0.5}, {"det_shot_y": 68.5}, {}, {"shot_ok": 0.0, "det_shot_x": 900.0}])
    m = pd.DataFrame(audit.config_metrics(table, audit.draws(table.game.to_numpy(), n_boot=5))).set_index(["quantity", "subset", "stat"])
    assert m.value["shot_location", "all", "off_pitch"] == 0.75          # three of the four located shots; 900 m is not located
    assert ("gk_detected", "all", "corr_x") not in m.index


# ------------------------------------------------------------ choice and gate
def _candidates(rows):
    return pd.DataFrame(rows, columns=["config", "coverage", "median"])


def test_the_choice_is_the_lowest_median_among_configurations_with_enough_coverage():
    c = audit.choose(_candidates([("C1", 0.60, 1.0), ("C2", 0.71, 4.0), ("C3", 0.80, 3.5), ("C4", 0.70, 3.6)]))
    assert c["chosen"] == "C3" and c["eligible"] == ["C2", "C3", "C4"]   # C1's low error does not count: 60% coverage


def test_a_median_tie_goes_to_the_higher_coverage():
    assert audit.choose(_candidates([("C1", 0.75, 2.0), ("C2", 0.90, 2.0), ("C3", 0.95, 2.5)]))["chosen"] == "C2"


def test_without_an_eligible_configuration_the_lowest_median_of_all_is_chosen():
    c = audit.choose(_candidates([("C1", 0.60, 3.0), ("C2", 0.65, 2.0), ("C3", 0.69, 2.0), ("C4", 0.50, np.nan)]))
    assert c["chosen"] == "C3" and c["eligible"] == [] and c["chosen_from"].startswith("all")


@pytest.mark.parametrize(("coverage", "median", "gk", "result"), [
    (0.70, 3.0, 0.50, "PASS"), (0.80, 2.0, 0.90, "PASS"),
    (0.69, 2.0, 0.90, "STOP"), (0.80, 3.01, 0.90, "STOP"), (0.80, 2.0, 0.49, "STOP")])
def test_the_gate(coverage, median, gk, result):
    g = audit.gate(coverage, median, gk)
    assert g["result"] == result and set(g["checks"]) == {"shot_location_coverage", "shot_location_median_m", "gk_information_coverage"}


def test_decide_reads_the_metrics_of_subset_all():
    rows = []
    for config, cov, med, gk in (("C1", 0.6, 1.0, 0.6), ("C4", 0.72, 3.8, 0.6)):
        rows += [{"config": config, "quantity": "shot_location", "subset": "all", "stat": "coverage", "value": cov},
                 {"config": config, "quantity": "shot_location", "subset": "all", "stat": "median", "value": med},
                 {"config": config, "quantity": "shot_location", "subset": "ball", "stat": "median", "value": 0.1},
                 {"config": config, "quantity": "gk_any", "subset": "all", "stat": "coverage", "value": gk}]
    decision = audit.decide(pd.DataFrame(rows), ["C1", "C4"])
    assert decision["chosen"] == "C4" and decision["gate"]["result"] == "STOP" and decision["gate"]["config"] == "C4"
    assert decision["gate"]["checks"]["shot_location_median_m"]["value"] == 3.8
    assert decision["approach2"].startswith("closes")


def test_decide_chooses_among_the_candidates_only():
    rows = []
    for config, cov, med in (("C4", 0.9, 1.0), ("D1", 0.72, 3.8), ("D2", 0.8, 2.9)):
        rows += [{"config": config, "quantity": "shot_location", "subset": "all", "stat": "coverage", "value": cov},
                 {"config": config, "quantity": "shot_location", "subset": "all", "stat": "median", "value": med},
                 {"config": config, "quantity": "gk_any", "subset": "all", "stat": "coverage", "value": 0.6}]
    decision = audit.decide(pd.DataFrame(rows), ["D1", "D2"])     # the reference C4 would win; it is not a candidate
    assert decision["chosen"] == "D2" and [c["config"] for c in decision["candidates"]] == ["D1", "D2"]
    assert decision["gate"]["result"] == "PASS" and decision["approach2"] == "continues"
    with pytest.raises(ValueError, match="no shot_location coverage"):
        audit.decide(pd.DataFrame(rows), ["D1", "D3"])


def test_metric_differences_compare_the_reference_rows_only():
    keys = {"quantity": "shot_location", "subset": "all"}
    old = pd.DataFrame([{"config": "C4", **keys, "stat": "median", "value": 12.1, "lo": 11.4, "hi": 12.8},
                        {"config": "C4", **keys, "stat": "n", "value": 3.0, "lo": np.nan, "hi": np.nan},
                        {"config": "C1", **keys, "stat": "median", "value": 15.8, "lo": 15.0, "hi": 16.4}])
    new = pd.concat([old[old.config == "C4"], old.iloc[[0]].assign(config="D1", value=9.0)], ignore_index=True)
    assert audit.metric_differences(new, old, ["C4"]) == {"rows": 2, "missing": 0, "differing": 0}
    assert audit.metric_differences(new.assign(hi=[12.9, np.nan, 9.0]), old, ["C4"])["differing"] == 1
    assert audit.metric_differences(new.iloc[1:], old, ["C4"]) == {"rows": 2, "missing": 1, "differing": 0}


# ------------------------------------------------------------ the script end to end, and no outcome read
SHOTS = {  # shot id -> (game, players at t0, ball, StatsBomb location, freeze frame)
    "m1_a": ("g1", [(93.0, 34.0, ATT, "player"), (100.0, 34.0, DEF, "player"), (104.0, 34.0, DEF, "goalkeeper")], (94.0, 34.0),
             (108.0, 40.0), [_player(119.0, 40.0, teammate=False, position="Goalkeeper"), _player(114.0, 40.0, teammate=False)]),
    "m1_b": ("g1", [(90.0, 20.0, ATT, "player"), (103.0, 33.0, DEF, "player")], None, (105.0, 25.0),
             [_player(118.0, 41.0, teammate=False, position="Goalkeeper")]),
    "m2_a": ("g2", [(80.0, 40.0, ATT, "player")], (81.0, 40.0), (100.0, 46.0), []),
    "m2_b": ("g2", [], None, (110.0, 30.0), [_player(119.5, 39.0, teammate=False, position="Goalkeeper")]),
}


def _write_inputs(tmp_path: Path, flip: bool = False) -> tuple[Path, Path, Path]:
    ids = sorted(SHOTS)
    pairs = pd.DataFrame({"shot_id": [*ids, "pen", "unpaired"], "game": [SHOTS[s][0] for s in ids] + ["g1", "g2"],
                          "paired": [True] * 5 + [False], "is_penalty": [False] * 4 + [True, False],
                          "sb_event_id": [f"e_{s}" for s in ids] + ["e_pen", None],
                          "sb_x": [SHOTS[s][3][0] for s in ids] + [108.0, np.nan], "sb_y": [SHOTS[s][3][1] for s in ids] + [40.0, np.nan],
                          "residual_ms": [10.0, 60.0, 300.0, 20.0, 0.0, np.nan]})
    rng = np.random.default_rng(1 if flip else 0)
    outcome = {"goal": rng.random(6) < 0.5, "sb_goal": rng.random(6) < 0.5, "sb_xg": rng.random(6),
               "sb_outcome": rng.choice(["Goal", "Saved", "Off T"], 6), "label": rng.choice(["on", "off"], 6)}
    pairs.assign(**outcome).to_parquet(tmp_path / "pairs.parquet")
    events = [f"e_{s}" for s in ids] + ["e_pen"]
    frames = [json.dumps(SHOTS[s][4]) if SHOTS[s][4] else None for s in ids] + ["[]"]
    pd.DataFrame({"event_id": events, "freeze_frame_json": frames, "outcome": rng.choice(["Goal", "Saved"], 5),
                  "statsbomb_xg": rng.random(5), "event_json": ["{}"] * 5}).to_parquet(tmp_path / "shots.parquet")
    extract = tmp_path / "extract"
    for name in ("C1", "C4"):
        folder = extract / name
        folder.mkdir(parents=True)
        rows = [ex.frame_row(s, 0, f"/f/{s}.jpg", frame_with(SHOTS[s][1], SHOTS[s][2])) for s in ids]
        if name == "C4":   # m1_b takes its ball from -400; m2_b has none at -400 and an unreadable -800
            rows += [ex.frame_row("m1_b", -400, "/f/m1_b_400.jpg", frame_with([], (91.0, 21.0))),
                     ex.frame_row("m2_b", -400, "/f/m2_b_400.jpg", frame_with([], None)),
                     ex.frame_row("m2_b", -800, "/f/m2_b_800.jpg", None)]
        frames_table = ex.frames_table(rows)
        frames_table.to_parquet(folder / "frames.parquet", index=False)
        features = ex.replay_features(frames_table, ex.CONFIGS[name]).drop(columns=["offset_ms", "gk_x", "gk_y"])
        features.to_parquet(folder / "features.parquet", index=False)
        record = {"config": ex.settings(ex.CONFIGS[name]), "identity": {"code": {"digest": "d"}, "git": {"sha": "s"}},
                  "files": {f: sha256_file(folder / f) for f in ("frames.parquet", "features.parquet")}}
        (folder / "complete.json").write_text(json.dumps(record))
    return tmp_path / "pairs.parquet", tmp_path / "shots.parquet", extract


def _run(monkeypatch, tmp_path, out_name, flip=False, record_reads=None, configs=("C1", "C4"), reference=()):
    (tmp_path / out_name).mkdir()
    pairs, shots, extract = _write_inputs(tmp_path / out_name, flip)
    out = tmp_path / out_name / "audit"
    monkeypatch.setattr(sys, "argv", ["xgd_audit.py", "--extract", str(extract), "--out", str(out), "--pairs", str(pairs),
                                      "--shots", str(shots), "--configs", *configs, "--reference-configs", *reference,
                                      "--reference-extract", str(extract), "--n-boot", "30", *_reference_metrics(tmp_path)])
    with monkeypatch.context() as readers:
        if record_reads is not None:      # every pandas table read: file, the columns asked for, the columns returned
            def recorder(original):
                def recording(path, *args, **kwargs):
                    table = original(path, *args, **kwargs)
                    record_reads.append((Path(path).name, kwargs.get("columns", kwargs.get("usecols")), list(table.columns)))
                    return table
                return recording
            for reader in ("read_parquet", "read_csv", "read_json", "read_feather"):
                readers.setattr(pd, reader, recorder(getattr(pd, reader)))
        xgd_audit.main()
    return out


def _reference_metrics(tmp_path):
    earlier = tmp_path / "earlier" / "audit" / "metrics.csv"
    return ["--reference-metrics", str(earlier)] if earlier.is_file() else []


def test_the_audit_reads_no_goal_or_outcome_column(monkeypatch, tmp_path):
    reads = []
    out = _run(monkeypatch, tmp_path, "run", record_reads=reads)
    by_file = {name: (asked, got) for name, asked, got in reads}
    assert by_file["pairs.parquet"][0] == audit.PAIR_COLUMNS and by_file["shots.parquet"][0] == audit.SHOT_COLUMNS
    assert {"pairs.parquet", "shots.parquet", "frames.parquet", "features.parquet"} <= set(by_file)
    for name, _, got in reads:
        assert not OUTCOME_COLUMNS & set(got), f"{name}: an outcome column was read"
    written = pd.read_parquet(out / "per_shot.parquet")
    assert not OUTCOME_COLUMNS & set(written.columns)
    assert not OUTCOME_COLUMNS & set(pd.read_csv(out / "metrics.csv").columns)


def test_audit_outputs_do_not_change_when_every_outcome_changes(monkeypatch, tmp_path):
    a, b = _run(monkeypatch, tmp_path, "a"), _run(monkeypatch, tmp_path, "b", flip=True)
    pd.testing.assert_frame_equal(pd.read_parquet(a / "per_shot.parquet"), pd.read_parquet(b / "per_shot.parquet"))
    for name in ("metrics.csv", "cone_confusion.csv", "timing.csv", "choice.json"):
        assert (a / name).read_text() == (b / name).read_text(), name


def test_the_audit_end_to_end_on_planted_shots(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, "run")
    per_shot = pd.read_parquet(out / "per_shot.parquet").set_index(["config", "shot_id"])
    assert sorted(per_shot.index.get_level_values(1).unique()) == sorted(SHOTS)   # no penalty, no unpaired shot
    a = per_shot.loc[("C1", "m1_a")]
    assert a.shot_err == pytest.approx(np.hypot(94.0 - (105 - 12 * YARD), 0.0))   # ball (94, 34) vs StatsBomb (108, 40)
    assert (a.gk_detected, a.gk_err) == (1.0, pytest.approx(105 - YARD - 104.0))
    assert a.cone_det == 1.0 and a.cone_sb == 1.0
    assert per_shot.loc[("C1", "m1_b")].shooter_by_goal == 1.0                      # no ball: the flagged shooter
    b = per_shot.loc[("C4", "m1_b")]
    assert (b.ball_fallback, b.ball_offset_ms, b.det_shot_x) == (1.0, -400.0, 91.0)
    assert per_shot.loc[("C1", "m2_b")].shot_ok == 0.0 and np.isnan(per_shot.loc[("C1", "m2_a")].cone_sb)
    choice = json.loads((out / "choice.json").read_text())
    assert choice["chosen"] in ("C1", "C4") and choice["gate"]["result"] in ("PASS", "STOP")
    record = json.loads((out / "complete.json").read_text())
    assert record["population"]["shots"] == 4 and set(record["files"]) == {"per_shot.parquet", "metrics.csv", "cone_confusion.csv",
                                                                           "timing.csv", "choice.json"}
    with pytest.raises(SystemExit, match="is complete"):
        xgd_audit.main()


def test_a_reference_configuration_is_audited_with_the_same_draws_but_never_chosen(monkeypatch, tmp_path):
    earlier = _run(monkeypatch, tmp_path, "earlier")                                  # an "audit 1" of C1 and C4
    out = _run(monkeypatch, tmp_path, "later", configs=("C1",), reference=("C4",))   # C1 the candidate, C4 the reference
    choice = json.loads((out / "choice.json").read_text())
    assert choice["chosen"] == "C1" and [c["config"] for c in choice["candidates"]] == ["C1"]
    assert choice["reference_configs"] == ["C4"] and choice["declaration"] is None
    later, before = pd.read_csv(out / "metrics.csv"), pd.read_csv(earlier / "metrics.csv")
    assert set(later.config) == {"C1", "C4"}
    pd.testing.assert_frame_equal(later[later.config == "C4"].reset_index(drop=True), before[before.config == "C4"].reset_index(drop=True))
    check = json.loads((out / "complete.json").read_text())["reference_check"]
    assert check["rows"] == int((before.config == "C4").sum()) and (check["missing"], check["differing"]) == (0, 0)


def test_a_reference_that_does_not_reproduce_the_earlier_audit_is_refused(monkeypatch, tmp_path):
    earlier = _run(monkeypatch, tmp_path, "earlier")
    metrics = pd.read_csv(earlier / "metrics.csv")
    metrics.loc[metrics.config == "C4", "value"] += 0.1
    metrics.to_csv(earlier / "metrics.csv", index=False)
    with pytest.raises(SystemExit, match="do not reproduce"):
        _run(monkeypatch, tmp_path, "later", configs=("C1",), reference=("C4",))
    assert not (tmp_path / "later" / "audit" / "complete.json").exists()


def test_a_candidate_cannot_be_a_reference_and_audit_2_needs_the_committed_declaration(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="both a candidate and a reference"):
        _run(monkeypatch, tmp_path, "both", configs=("C1", "C4"), reference=("C4",))
    changed = tmp_path / "declaration.json"
    changed.write_text(ex.AUDIT_2_DECLARATION.read_text().replace('"D4"', '"D5"'))
    monkeypatch.setattr(sys, "argv", ["xgd_audit.py", "--configs", "D1", "D2", "D3", "D4", "--declaration", str(changed),
                                      "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not the committed D1-D4 declaration"):
        xgd_audit.main()


def test_a_cache_that_does_not_match_its_record_is_refused(monkeypatch, tmp_path):
    pairs, shots, extract = _write_inputs(tmp_path)
    pd.read_parquet(extract / "C1/features.parquet").assign(dist=1.0).to_parquet(extract / "C1/features.parquet")
    monkeypatch.setattr(sys, "argv", ["xgd_audit.py", "--extract", str(extract), "--out", str(tmp_path / "o"), "--pairs", str(pairs),
                                      "--shots", str(shots), "--configs", "C1", "--reference-configs"])
    with pytest.raises(SystemExit, match="completion record"):
        xgd_audit.main()


def test_replayed_features_that_differ_from_the_cache_are_reported():
    cached = pd.DataFrame({"shot_id": ["a", "b"], "dist": [1.0, np.nan], "shot_ok": [1.0, 0.0]})
    replayed = pd.DataFrame({"shot_id": ["b", "a"], "offset_ms": [0, 0], "dist": [np.nan, 1.5], "shot_ok": [0.0, 1.0]})
    assert audit.check_replay(cached, replayed) == {"dist": 1}
    assert audit.check_replay(cached, replayed.assign(dist=[np.nan, 1.0])) == {}


# ------------------------------------------------------------ audit 2 diagnostics
def _frames(rows):
    return pd.DataFrame([{"shot_id": s, "offset_ms": o, "kp_xy": kp, "homography": h} for s, o, kp, h in rows])


def test_refinement_groups_compare_the_t0_matrices_of_the_same_keypoints():
    eye, moved = list(np.eye(3).ravel()), list((np.eye(3) * 2).ravel())       # 2 x I is the same homography as I
    shifted = list((np.eye(3) + np.eye(3, k=2) * 0.5).ravel())                 # a shift along x
    plain = _frames([("a", 0, [1.0, 2.0], eye), ("b", 0, [1.0, 2.0], eye), ("c", 0, None, None), ("a", -400, [9.0], eye)])
    refined = _frames([("a", 0, [1.0, 2.0], shifted), ("b", 0, [1.0, 2.0], moved), ("c", 0, None, None),
                       ("a", -400, [9.0], shifted)])
    assert audit.refinement_groups(plain, refined).to_dict() == {"a": "refined", "b": "kept", "c": "no_fit"}
    with pytest.raises(ValueError, match="different t0 keypoints"):
        audit.refinement_groups(plain, refined.assign(kp_xy=[[1.0, 2.5], [1.0, 2.0], None, [9.0]]))
    with pytest.raises(ValueError, match="one run has a t0 homography"):
        audit.refinement_groups(plain, refined.assign(homography=[None, moved, None, shifted]))
    with pytest.raises(ValueError, match="different t0 shots"):
        audit.refinement_groups(plain, refined[refined.shot_id != "c"])
    zero_corner = list(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]]).ravel())    # h33 = 0 still compares by scale
    assert audit.refinement_groups(plain.assign(homography=[zero_corner, eye, None, eye]),
                                   refined.assign(homography=[list(-2 * np.array(zero_corner)), eye, None, eye]))["a"] == "kept"


def test_group_errors_and_location_sources_on_planted_rows():
    table = _table([{"shot_err": 2.0, "shot_dx": 1.0, "ball_offset_ms": 0.0}, {"shot_err": 8.0, "shot_dx": -1.0, "ball_offset_ms": -400.0,
                                                                              "ball_fallback": 1.0},
                    {"shot_err": 6.0, "ball_offset_ms": np.nan, "shot_from_ball": 0.0, "shooter_by_goal": 1.0, "gk_ok": 0.0},
                    {"shot_ok": 0.0, "shot_err": np.nan, "ball_offset_ms": np.nan, "shot_from_ball": 0.0}])
    masks = audit.source_masks(table, "C9")
    assert {k: v.tolist() for k, v in masks.items()} == {"ball_t0": [True, False, False, False], "ball_m400": [False, True, False, False],
                                                        "ball_m800": [False] * 4, "flag": [False, False, True, False]}
    d = audit.draws(table.game.to_numpy(), n_boot=5)
    rows = pd.DataFrame(audit.group_errors(table, masks, "C9", "source", d)).set_index(["group", "quantity", "stat"]).value
    assert rows["ball_m400", "shot_location", "median"] == 8.0 and rows["ball_m400", "shot_location", "median_dx"] == -1.0
    assert rows["flag", "population", "share"] == 0.25 and rows["flag", "gk_any", "n"] == 0 and np.isnan(rows["flag", "gk_any", "median"])
    assert rows["ball_m800", "shot_location", "n"] == 0


def test_the_audit_2_diagnostic_end_to_end(monkeypatch, tmp_path):
    import xgd_audit2_diag

    pop = 4
    per_shot = pd.concat([_table([{"shot_err": float(i + k)} for i in range(pop)]).assign(config=c, shot_id=[f"s{i}" for i in range(pop)])
                          for k, c in enumerate(("D1", "D2", "D3", "D4"))], ignore_index=True)
    folder = tmp_path / "audit"
    folder.mkdir()
    per_shot.assign(ball_offset_ms=0.0).to_parquet(folder / "per_shot.parquet")
    (folder / "complete.json").write_text(json.dumps({"files": {"per_shot.parquet": sha256_file(folder / "per_shot.parquet")},
                                                      "parameters": {"n_boot": 7, "seed": 0}}))
    eye, other = list(np.eye(3).ravel()), list((np.eye(3) + np.eye(3, k=1)).ravel())
    homographies = {"D1": [eye, eye, None, eye], "D3": [other, eye, None, eye], "D2": [eye] * 4, "D4": [eye] * 4}
    for name, matrices in homographies.items():
        (tmp_path / "extract" / name).mkdir(parents=True)
        frames = _frames([(f"s{i}", 0, None if m is None else [1.0], m) for i, m in enumerate(matrices)])
        frames.to_parquet(tmp_path / "extract" / name / "frames.parquet")
        (tmp_path / "extract" / name / "complete.json").write_text(
            json.dumps({"files": {"frames.parquet": sha256_file(tmp_path / "extract" / name / "frames.parquet")}}))
    out = tmp_path / "diag"
    monkeypatch.setattr(sys, "argv", ["xgd_audit2_diag.py", "--audit", str(folder), "--extract", str(tmp_path / "extract"),
                                      "--out", str(out)])
    xgd_audit2_diag.main()
    record = json.loads((out / "complete.json").read_text())
    assert record["refinement_groups"] == {"D3": {"refined": 1, "kept": 2, "no_fit": 1}, "D4": {"refined": 0, "kept": 4, "no_fit": 0}}
    diag = pd.read_csv(out / "diag.csv").set_index(["split", "group", "config", "quantity", "stat"]).value
    assert diag["refinement_D3", "refined", "D1", "shot_location", "median"] == 0.0     # s0: D1 error 0, D3 error 2
    assert diag["refinement_D3", "refined", "D3", "shot_location", "median"] == 2.0
    assert diag["source", "ball_t0", "D4", "population", "share"] == 1.0
    with pytest.raises(SystemExit, match="is complete"):
        xgd_audit2_diag.main()


# ------------------------------------------------------------ the keypoint-input diagnostic

def test_the_diagnostic_stretches_the_frame_and_scales_the_keypoints_back():
    class Model:
        def predict(self, image):
            self.shape = image.shape
            return sv.KeyPoints(xy=np.array([[[320.0, 320.0], [640.0, 0.0]]], np.float32), confidence=np.ones((1, 2), np.float32),
                                data={"box_confidence": np.float32([0.7])})
    model = Model()
    keypoints = xgd_kp_diag.stretched_keypoints(model, np.zeros((720, 1280, 3), np.uint8))
    assert model.shape == (640, 640, 3)
    np.testing.assert_allclose(keypoints.xy[0], [[640.0, 360.0], [1280.0, 0.0]])


def test_the_diagnostic_places_people_at_the_bottom_centre_and_the_ball_at_the_centre():
    shift = np.array([[1.0, 0, 10], [0, 1.0, 20], [0, 0, 1]])
    assert xgd_kp_diag.placed([0, 0, 4, 10], shift, bottom=True) == [12.0, 30.0]
    assert xgd_kp_diag.placed([0, 0, 4, 10], shift, bottom=False) == [12.0, 25.0]
    assert np.isnan(xgd_kp_diag.placed([0, 0, 4, 10], None, bottom=True)).all()
    row = {"det_xyxy": [0, 0, 4, 10], "ball_xyxy": [], "homography": None}
    kp = sv.KeyPoints(xy=np.zeros((1, 32, 2), np.float32), confidence=np.ones((1, 32), np.float32),
                      data={"box_confidence": np.float32([0.6])})
    new = xgd_kp_diag.restretched_row(row, kp, shift)
    assert new["det_pitch_xy"] == [12.0, 30.0] and new["ball_pitch_xy"] == [] and new["kp_instance_conf"] == pytest.approx(0.6)


def test_unanalysed_fallback_frames_are_padded_as_frames_without_a_ball():
    frames = pd.DataFrame({"shot_id": ["a", "b", "b"], "offset_ms": [0, 0, -400], "readable": [True, True, True]})
    padded, shots = xgd_kp_diag.with_unanalysed_fallbacks(frames)
    assert shots == 2 and sorted(zip(padded.shot_id, padded.offset_ms, strict=True)) == [("a", -800), ("a", -400), ("a", 0), ("b", -800),
                                                                              ("b", -400), ("b", 0)]
    assert not padded.set_index(["shot_id", "offset_ms"]).loc[("a", -400), "readable"]
