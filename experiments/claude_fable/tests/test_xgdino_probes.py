"""xg-dino ticket 04: probe targets in metres and the probe machinery (spec §Testing Decisions; councils C1, C2)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector.conventions import LENGTH, WIDTH
from xgpoc.detector.guard import StatsBombRead, no_statsbomb_reads
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg

from tactifoot_vision.pipeline import NO_TEAM

ROOT = Path(__file__).resolve().parents[1]
ATT, DEF = 0, 1
# attacking x = 105: shooter (95, 30); defenders at (97, 31) (in the cone, 2.24 m), (95, 27) (exactly 3.0 m, outside
# the cone) and (95, 33.5) (3.5 m); a teammate; the detected goalkeeper (104, 35); a referee
SCENE = [(95.0, 30.0, ATT, "player"), (97.0, 31.0, DEF, "player"), (95.0, 27.0, DEF, "player"),
         (95.0, 33.5, DEF, "player"), (96.0, 30.6, ATT, "player"), (104.0, 35.0, DEF, "goalkeeper"),
         (60.0, 30.0, NO_TEAM, "referee")]


def scene_frame(scene, mirror: bool = False):
    """A frame of `scene` (rotated 180 deg when `mirror`), its ball box on the shooter (player 0)."""
    if mirror:
        scene = [(LENGTH - x, WIDTH - y, t, c) for x, y, t, c in scene]
    frame = frame_with(scene, (scene[0][0], scene[0][1]))
    frame.ball.xyxy[:] = frame.detections.xyxy[:1]
    return frame


@pytest.mark.parametrize("mirror", [False, True])
def test_hand_built_frames_give_the_targets_in_metres_in_both_attack_directions(mirror):
    t = tg.frame_targets(scene_frame(SCENE, mirror))
    a1, a2 = np.arctan2(34 - 3.66 - 30, 10), np.arctan2(34 + 3.66 - 30, 10)
    assert t["distance"] == pytest.approx(np.hypot(10, 4)) and t["angle"] == pytest.approx(abs(a2 - a1))
    assert (t["def_cone"], t["def_cone_ge1"], t["opp_3m"], t["opp_3m_ge1"]) == (1.0, 1.0, 2.0, 1.0)   # 3.0 m inclusive
    assert t["near_opp_m"] == pytest.approx(np.hypot(2, 1)) and t["near_opp_status"] == "observed"
    assert (t["gk_depth_m"], t["gk_lateral_m"], t["gk_lateral_signed_m"]) == pytest.approx((1.0, 1.0, 1.0))
    assert (t["gk_detected"], t["gk_proxy"], t["teams_ok"], t["n_opponents"]) == (1.0, 0.0, 1.0, 3.0)


def test_the_goalkeeper_proxy_is_excluded_from_the_opponents_and_recorded():
    scene = [p if p[3] != "goalkeeper" else (104.0, 31.0, DEF, "player") for p in SCENE]
    t = tg.frame_targets(scene_frame(scene))
    assert (t["gk_detected"], t["gk_proxy"]) == (0.0, 1.0)
    assert (t["gk_depth_m"], t["gk_lateral_m"], t["gk_lateral_signed_m"]) == pytest.approx((1.0, 3.0, -3.0))
    assert t["opp_3m"] == 2.0 and t["n_opponents"] == 3.0


def test_unknown_teams_make_opponent_targets_missing_not_zero_and_no_opponent_is_distinct():
    unknown = [(x, y, NO_TEAM if i == 0 else t, c) for i, (x, y, t, c) in enumerate(SCENE)]
    t = tg.frame_targets(scene_frame(unknown))
    assert all(np.isnan(t[c]) for c in tg.OPPONENT) and t["near_opp_status"] == "missing" and t["teams_ok"] == 0
    assert t["gk_depth_m"] == pytest.approx(1.0)
    alone = [p for p in SCENE if p[2] != DEF or p[3] == "goalkeeper"]
    t = tg.frame_targets(scene_frame(alone))
    assert (t["opp_3m"], t["def_cone"], t["opp_3m_ge1"]) == (0.0, 0.0, 0.0)
    assert np.isnan(t["near_opp_m"]) and t["near_opp_status"] == "no_opponent"
    assert all(np.isnan(v) for k, v in tg.frame_targets(None).items() if k in tg.TARGETS)


def test_statsbomb_freeze_frames_use_the_same_definitions_in_goal_centred_metres():
    ff = [{"teammate": False, "location": [119.0, 41.0], "position": {"name": "Goalkeeper"}},
          {"teammate": False, "location": [111.0, 40.0], "position": {"name": "Center Back"}},
          {"teammate": True, "location": [110.5, 39.0], "position": {"name": "Center Forward"}}]
    pop = pd.DataFrame({"shot_id": ["a", "b"], "sb_event_id": ["e1", "e2"], "sb_x": [110.0, 100.0], "sb_y": [40.0, 40.0]})
    t = tg.statsbomb_table_targets(pop, pd.Series({"e1": json.dumps(ff), "e2": None}))
    assert t.attrs["source"] == "statsbomb"
    assert t.loc["a", "distance"] == pytest.approx(10 * 0.9144)
    assert t.loc["a", "near_opp_m"] == pytest.approx(0.9144)
    assert (t.loc["a", "gk_depth_m"], t.loc["a", "gk_lateral_signed_m"]) == pytest.approx((0.9144, 0.9144))
    assert (t.loc["a", "def_cone"], t.loc["a", "opp_3m"], t.loc["a", "gk_detected"]) == (1.0, 1.0, 1.0)
    assert t.loc["b", list(tg.OPPONENT + tg.GOALKEEPER)].isna().all() and t.loc["b", "distance"] == pytest.approx(18.288)


def _paired_files(tmp_path: Path) -> tuple[Path, Path]:
    pairs, shots = tmp_path / "xgv_pairing/pairs.parquet", tmp_path / "statsbomb/tables/shots.parquet"
    for p in (pairs, shots):
        p.parent.mkdir(parents=True)
    pd.DataFrame({"shot_id": ["a"], "game": ["g"], "paired": [True], "is_penalty": [False], "sb_event_id": ["e1"],
                  "sb_x": [110.0], "sb_y": [40.0], "residual_ms": [0.0]}).to_parquet(pairs)
    pd.DataFrame({"event_id": ["e1"], "freeze_frame_json": ["[]"]}).to_parquet(shots)
    return pairs, shots


def test_statsbomb_is_read_only_by_the_paired_builder_outside_the_guard(tmp_path):
    pairs, shots = _paired_files(tmp_path)
    with no_statsbomb_reads(), pytest.raises(StatsBombRead):
        tg.statsbomb_targets(["a"], pairs, shots)
    sb = tg.statsbomb_targets(["a"], pairs, shots)
    assert sb.loc["a", "distance"] == pytest.approx(10 * 0.9144)
    # no other xg-dino probe code names a StatsBomb path or the audit
    for path in (ROOT / "xgpoc/dino/probes.py", ROOT / "scripts/xgdino_probes.py"):
        text = path.read_text()
        assert not re.search(r"xgv_pairing|tables/shots|open[-_]data|import audit", text)


def test_probes_refuse_statsbomb_targets():
    sb = tg.tagged(pd.DataFrame({"distance": [1.0]}), "statsbomb")
    with pytest.raises(ValueError, match="automatic"):
        pr.ProbeData(sb, np.array(["g"]), np.array([0]))
    auto = tg.tagged(pd.DataFrame({"distance": [1.0]}), "automatic")
    with pytest.raises(ValueError, match="automatic"):
        pr.ProbeData(auto, np.array(["g"]), np.array([0]), paired=sb)


# ------------------------------------------------------------------ probes on planted data
def planted(n_games=40, per=12, d=24, seed=0):
    rng = np.random.default_rng(seed)
    games = np.repeat([f"g{i}" for i in range(n_games)], per)
    folds = np.repeat(np.arange(n_games) % 5, per)
    n = len(games)
    dist = rng.uniform(5, 30, n)
    signal = rng.normal(size=n)
    t = pd.DataFrame({"distance": dist, "angle": 0.5 / dist + 0.05 * rng.normal(size=n),
                      "near_opp_m": 3 + signal + 0.2 * rng.normal(size=n),
                      "opp_3m_ge1": (signal + 0.3 * rng.normal(size=n) > 0).astype(float)},
                     index=[f"s{i}" for i in range(n)])
    t.loc[t.index[::7], "near_opp_m"] = np.nan                 # missing rows stay out of that target's fit
    return t, games, folds, signal, rng


def test_planted_linear_signal_is_found_in_the_right_layer_and_fold_model():
    t, games, folds, signal, rng = planted()
    n, d = len(t), 24
    tg.tagged(t, "automatic")
    paired_signal = rng.normal(size=60)
    paired = tg.tagged(pd.DataFrame({"distance": rng.uniform(5, 30, 60)}), "automatic")
    data = pr.ProbeData(t, games, folds, paired)
    directions = rng.normal(size=(5, d))                    # each fold model writes the signal in its own direction

    def g_train(k):
        return np.outer(signal, directions[k]) + rng.normal(size=(n, d))

    def g_paired(k):
        return np.outer(paired_signal, directions[k]) + rng.normal(size=(60, d))
    noise = rng.normal(size=(n, d))
    runs = {"a": pr.run(data, pr.Encoder("a", lambda k: noise, lambda k: rng.normal(size=(60, d))), ["near_opp_m", "opp_3m_ge1"]),
            "b": pr.run(data, pr.Encoder("b", g_train, g_paired), ["near_opp_m", "opp_3m_ge1"])}
    b = runs["b"].targets["near_opp_m"]
    assert len(b.rows) == t.near_opp_m.notna().sum()
    assert pr.point_metric(b.y, b.oof["embedding"], False) > 0.9
    assert pr.point_metric(b.y, b.oof["distance"], False) < 0.1
    assert pr.point_metric(b.oof["shuffled_target"], b.oof["embedding_shuffled"], False) < 0.05
    assert np.corrcoef(b.paired["embedding"], paired_signal)[0, 1] > 0.9      # fold probe k read on fold model k
    a = runs["a"].targets["near_opp_m"]
    assert pr.point_metric(a.y, a.oof["embedding"], False) < 0.1
    assert pr.point_metric(runs["b"].targets["opp_3m_ge1"].y, runs["b"].targets["opp_3m_ge1"].oof["embedding"], True) > 0.85
    for target in ("near_opp_m", "opp_3m_ge1"):
        assert pr.chosen_layer(runs, data, target) == "b"
        assert pr.nested(runs, data, target)[1] == ["b"] * 5
    d_ = pr.draws(games, n_boot=200)
    dec = pr.decide("near_opp_m", pr.target_metrics(b, d_))
    assert dec["encodes_beyond_distance"] and dec["readable"] and dec["gain"]["lo"] > 0


def test_no_match_lies_on_both_sides_of_a_split():
    t, games, folds, *_ = planted()
    for k in range(5):
        train = np.flatnonzero(folds != k)
        assert not set(games[train]) & set(games[folds == k])
        for itr, iva in pr.inner_splits(games[train]):
            assert not set(games[train][itr]) & set(games[train][iva])
    bad = folds.copy()
    bad[0] = (bad[0] + 1) % 5
    with pytest.raises(ValueError, match="two outer folds"):
        pr.ProbeData(tg.tagged(t, "automatic"), games, bad)


def test_best_layer_ties_go_to_the_later_block_and_rules_read_the_protocol():
    assert pr.best_layer({"20": 0.500, "21": 0.495, "22": 0.40}, binary=False) == "21"
    assert pr.best_layer({"20": 0.700, "21": 0.694}, binary=True) == "20"
    rules = pr.Rules.from_protocol({"gain_min": {"continuous": 0.1, "binary": 0.05}, "readable_min": None, "other": 1})
    assert rules.gain_min["continuous"] == 0.1 and rules.readable_min["continuous"] == 0.30
