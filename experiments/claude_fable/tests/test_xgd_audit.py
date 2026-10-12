"""The StatsBomb audit, config choice and gate on fixtures with known answers (xg-detector 02)."""

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
            "sb_gk_x": 104.0, "cone_det": 1.0, "cone_sb": 1.0}
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
    decision = audit.decide(pd.DataFrame(rows))
    assert decision["chosen"] == "C4" and decision["gate"]["result"] == "STOP" and decision["gate"]["config"] == "C4"
    assert decision["gate"]["checks"]["shot_location_median_m"]["value"] == 3.8
    assert decision["approach2"].startswith("closes")


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


def _run(monkeypatch, tmp_path, out_name, flip=False, record_reads=None):
    (tmp_path / out_name).mkdir()
    pairs, shots, extract = _write_inputs(tmp_path / out_name, flip)
    out = tmp_path / out_name / "audit"
    monkeypatch.setattr(sys, "argv", ["xgd_audit.py", "--extract", str(extract), "--out", str(out), "--pairs", str(pairs),
                                      "--shots", str(shots), "--configs", "C1", "C4", "--n-boot", "30"])
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


def test_a_cache_that_does_not_match_its_record_is_refused(monkeypatch, tmp_path):
    pairs, shots, extract = _write_inputs(tmp_path)
    pd.read_parquet(extract / "C1/features.parquet").assign(dist=1.0).to_parquet(extract / "C1/features.parquet")
    monkeypatch.setattr(sys, "argv", ["xgd_audit.py", "--extract", str(extract), "--out", str(tmp_path / "o"), "--pairs", str(pairs),
                                      "--shots", str(shots), "--configs", "C1"])
    with pytest.raises(SystemExit, match="completion record"):
        xgd_audit.main()


def test_replayed_features_that_differ_from_the_cache_are_reported():
    cached = pd.DataFrame({"shot_id": ["a", "b"], "dist": [1.0, np.nan], "shot_ok": [1.0, 0.0]})
    replayed = pd.DataFrame({"shot_id": ["b", "a"], "offset_ms": [0, 0], "dist": [np.nan, 1.5], "shot_ok": [0.0, 1.0]})
    assert audit.check_replay(cached, replayed) == {"dist": 1}
    assert audit.check_replay(cached, replayed.assign(dist=[np.nan, 1.0])) == {}


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
