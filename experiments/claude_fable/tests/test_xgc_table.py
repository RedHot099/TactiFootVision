"""The xg-compare shot table (ticket 01): joins, populations, the label split, the rebound and box rules, the cuts."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.compare import sequence as seq
from xgpoc.compare import table as tb
from xgpoc.vendor import evaltable as et


# ------------------------------------------------------------------ rebound rule
def shots_frame(rows: list[tuple]) -> pd.DataFrame:
    """(event_id, index, period, seconds, team, possession, outcome) in match 1."""
    return pd.DataFrame([{"match_id": 1, "event_id": e, "index": i, "period": p, "timestamp_ms": int(s * 1000), "minute": int(s // 60),
                          "team_id": t, "possession": pos, "outcome": o} for e, i, p, s, t, pos, o in rows])


def test_a_shot_by_the_same_team_within_5_s_in_the_same_possession_is_a_rebound():
    shots = shots_frame([("a", 10, 1, 100.0, 7, 5, "Saved"), ("b", 12, 1, 104.0, 7, 5, "Off T"), ("c", 15, 1, 109.5, 7, 5, "Goal")])
    out = seq.sequence_table(shots).set_index("event_id")
    assert out.loc["a"].prev_event_id is None or pd.isna(out.loc["a"].prev_event_id)
    assert not out.loc["a"].rebound_5s
    assert out.loc["b"].prev_event_id == "a" and out.loc["b"].prev_outcome == "Saved" and out.loc["b"].prev_dt_ms == 4000
    assert out.loc["b"].rebound_5s and not out.loc["b"].rebound_3s
    assert out.loc["c"].prev_event_id == "b" and not out.loc["c"].rebound_5s   # 5.5 s after the previous shot


def test_the_window_edges_are_inclusive():
    shots = shots_frame([("a", 1, 1, 10.0, 7, 2, "Blocked"), ("b", 2, 1, 13.0, 7, 2, "Saved"), ("c", 3, 1, 18.0, 7, 2, "Off T")])
    out = seq.sequence_table(shots).set_index("event_id")
    assert out.loc["b"].rebound_3s and out.loc["b"].rebound_5s
    assert out.loc["c"].rebound_5s and not out.loc["c"].rebound_3s


def test_another_possession_period_or_team_breaks_the_rebound():
    shots = shots_frame([("a", 1, 1, 10.0, 7, 2, "Saved"), ("b", 2, 1, 11.0, 7, 3, "Saved"),    # new possession
                         ("c", 3, 1, 12.0, 8, 3, "Saved"),                                       # same possession, other team
                         ("d", 4, 2, 12.5, 8, 3, "Saved")])                                      # same possession number, next period
    out = seq.sequence_table(shots).set_index("event_id")
    assert not out.rebound_5s.any()
    assert out.loc["c"].prev_event_id == "b" and not out.loc["c"].prev_same_team
    assert pd.isna(out.loc["d"].prev_event_id)


def test_the_previous_shot_follows_the_event_index_not_the_row_order():
    shots = shots_frame([("late", 9, 1, 14.0, 7, 4, "Off T"), ("early", 3, 1, 11.0, 7, 4, "Blocked")])
    out = seq.sequence_table(shots)
    assert list(out.event_id) == ["late", "early"]
    assert out.prev_event_id.iloc[0] == "early" and out.prev_outcome.iloc[0] == "Blocked" and out.rebound_5s.iloc[0]


def test_duplicate_shots_are_refused():
    shots = shots_frame([("a", 1, 1, 10.0, 7, 2, "Saved"), ("a", 2, 1, 11.0, 7, 2, "Saved")])
    with pytest.raises(ValueError, match="duplicate event_id"):
        seq.previous_shots(shots)


def test_rebound_subtypes_come_from_the_previous_outcome():
    rebound = pd.Series(["yes", "yes", "yes", "yes", "no"])
    previous = pd.Series(["Saved Off Target", "Blocked", "Post", "Saved", "Saved"])
    assert list(cat.rebound_subtype(rebound, previous)) == ["after_save", "after_block", "after_other", "after_save", "none"]


# ------------------------------------------------------------------ box and bins
def test_box_rule_edges():
    x = pd.Series([102.0, 101.9, 110.0, 110.0, 110.0, 110.0])
    y = pd.Series([40.0, 40.0, 18.0, 17.9, 62.0, 62.1])
    assert list(cat.outside_box(x, y)) == ["no", "yes", "no", "yes", "no", "yes"]


def test_bins_are_left_closed_and_the_last_is_open():
    values = pd.Series([0.0, 5.99, 6.0, 25.0, 80.0, np.nan])
    assert list(cat.bins(values, cat.DIST_EDGES, cat.DIST_LABELS)) == ["00_06", "00_06", "06_12", "25_plus", "25_plus", None]


def test_nearest_share_cut_picks_the_closest_share_and_breaks_ties_outward():
    values = np.array([1, 1, 2, 2, 2, 3, 4, 5, 6, 7], float)
    assert cat.nearest_share_cut(values, 1 / 3, "low") == 1.0          # shares 0.2 (<= 1) vs 0.5 (<= 2)
    assert cat.nearest_share_cut(values, 1 / 3, "high") == 5.0         # >= 5: 0.3
    assert cat.nearest_share_cut(np.array([1, 2, 3, 4], float), 0.5, "low") == 2.0
    assert cat.nearest_share_cut(np.array([1, 1, 2, 2], float), 0.25, "high") == 2.0   # 0.5 vs 1.0


def test_derive_refuses_an_unforeseen_level():
    descriptors = descriptor_frame(4)
    descriptors.loc[0, "sb_body"] = "Knee"
    with pytest.raises(ValueError, match="body"):
        cat.derive(descriptors, cat.Cuts(11, 17, 46.0))


def test_minute_bins_follow_the_period_and_the_half_names_the_period():
    minute, period = pd.Series([0, 14, 15, 30, 47, 45, 60, 75, 93, 105]), pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 3])
    assert list(cat.minute_bin(minute, period)) == ["1h_00_14", "1h_00_14", "1h_15_29", "1h_30_end", "1h_30_end", "2h_45_59",
                                                    "2h_60_74", "2h_75_end", "2h_75_end", "extra_time"]
    assert list(cat.half_name(pd.Series([1, 2, 3]))) == ["first", "second", "extra_time"]


def test_key_pass_height_is_undefined_without_a_key_pass_and_rebounds_read_the_sequence_flags():
    descriptors = descriptor_frame(3).assign(sb_kp_has=[1.0, 0.0, 1.0], sb_kp_height=["High Pass", "None", "Low Pass"],
                                             seq_rebound_5s=[True, False, True], seq_rebound_3s=[False, False, True],
                                             seq_prev_outcome=["Blocked", None, "Saved"])
    out = cat.derive(descriptors, cat.Cuts(11, 17, 46.0))
    assert list(out.cat_key_pass_height) == ["High Pass", None, "Low Pass"]
    assert list(out.cat_key_pass_none) == ["no", "yes", "no"]
    assert list(out.cat_rebound_5s) == ["yes", "no", "yes"] and list(out.cat_rebound_3s) == ["no", "no", "yes"]
    assert list(out.cat_rebound_subtype) == ["after_block", "none", "after_save"]
    assert list(out.columns) == [c.column for c in cat.PRE_SHOT]


def test_the_dictionary_refuses_unknown_groups_and_scan_levels_outside_the_levels():
    with pytest.raises(ValueError, match="group"):
        cat.Category("x", "weather", "s", "r", ("a",), ())
    with pytest.raises(ValueError, match="scan levels"):
        cat.Category("x", "events", "s", "r", ("a",), ("b",))
    record = cat.dictionary_record(cat.Cuts(11, 17, 46.0))
    assert record["cuts"]["players_low_max"] == 11 and len(record["categories"]) == len(cat.CATEGORIES)
    assert {c["name"] for c in record["categories"] if c["role"] == "post_shot"} == {"on_target_sn", "sb_outcome"}


def test_tercile_and_quartile_categories_use_the_cuts():
    cuts = cat.Cuts(players_low_max=11, players_high_min=17, timing_top_min_ms=46.0)
    assert list(cat.players_tercile(pd.Series([11.0, 12.0, 16.0, 17.0, np.nan]), cuts)) == ["low", "mid", "mid", "high", None]
    assert list(cat.timing_top(pd.Series([-46.0, 45.0, 46.0, np.nan]), cuts)) == ["yes", "no", "yes", None]


# ------------------------------------------------------------------ the table
def descriptor_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sb_x": rng.uniform(90, 119, n), "sb_y": rng.uniform(20, 60, n), "sb_dist": rng.uniform(2, 35, n), "sb_angle": rng.uniform(0.1, 1.2, n),
        "sb_body": rng.choice(["Right Foot", "Left Foot", "Head"], n), "sb_technique": "Normal", "sb_stype": "Open Play",
        "sb_pattern": "Regular Play", "sb_kp_has": 1.0, "sb_kp_cross": 0.0, "sb_kp_through": 0.0, "sb_kp_cutback": 0.0,
        "sb_kp_height": "Ground Pass", "sb_first_time": 0.0, "sb_under_pressure": 1.0, "sb_one_on_one": 0.0, "sb_open_goal": 0.0,
        "sb_follows_dribble": 0.0, "sb_aerial_won": 0.0, "sb_ff_def_cone": rng.integers(0, 4, n).astype(float),
        "sb_gk_depth": rng.uniform(0, 6, n), "sb_gk_off_line": rng.normal(0, 1, n), "sb_ff_near_opp": rng.uniform(0, 6, n),
        "sb_minute": rng.integers(0, 45, n), "half": 1, "seq_rebound_5s": False, "seq_rebound_3s": False, "seq_prev_outcome": None,
        "e3_homography_ok": 1.0, "e3_n_players": rng.integers(3, 25, n).astype(float), "e3_n_keypoints": rng.integers(6, 13, n).astype(float),
        "e3_gk_detected": 1.0, "e3_ball_ok": 1.0, "league_season": "england_epl/2015-2016", "timing_residual_ms": rng.normal(0, 40, n)})


def joined_table(n_primary: int = 40, n_secondary: int = 12, seed: int = 0) -> pd.DataFrame:
    """A small confirmatory-like table with labels: two primary penalties, one primary row without a frame, one
    label-excluded secondary row."""
    rng = np.random.default_rng(seed)
    n = n_primary + n_secondary
    population = ["primary"] * n_primary + ["statsbomb_only"] * (n_secondary - 2) + ["recovered_goal"] * 2
    match = [f"england_epl/2015-2016/m{i % 6}" if i % 2 else f"spain_laliga/2016-2017/m{i % 6}" for i in range(n)]
    sn_id = [f"sn{i}" if p == "primary" else None for i, p in enumerate(population)]
    sb_id = [f"sb{i}" for i in range(n)]
    t = pd.DataFrame({"key": [s or b for s, b in zip(sn_id, sb_id, strict=True)], "soccernet_shot_id": sn_id, "statsbomb_event_id": sb_id,
                      "statsbomb_match_id": pd.array([100 + i % 6 for i in range(n)], dtype="Int64"), "match": match,
                      "half": np.int8(1), "population": population, "split": "test", "sb_body_part": "Right Foot",
                      "sb_shot_type": "Open Play"})
    for column in et.PROBABILITY_COLUMNS:
        t[column] = rng.uniform(0.01, 0.6, n)
    for column in et.FRAME_LOGIT_COLUMNS:
        t[column] = rng.normal(-2, 1, n)
    t["frame_status"], t["frame_source"] = "ok", "test"
    t.loc[n_primary - 1, ["frame_status", "frame_logit_0"]] = ["0:beyond_half_end", np.nan]
    for column in et.EMBEDDING_COLUMNS:
        t[column] = -1
    t["timing_residual_ms"] = np.where(t.population.eq("primary"), rng.normal(0, 40, n), np.nan)
    t["anchor_convention"], t["table_kind"] = t.population.map(et.ANCHOR_CONVENTION), "development"
    t["goal_sn"] = rng.random(n) < 0.2
    t["on_target_sn"] = np.where(t.population.eq("primary"), rng.random(n) < 0.4, None)
    t["sb_outcome"] = rng.choice(["Goal", "Saved", "Off T", "Blocked"], n)
    t["is_penalty"] = False
    t.loc[[0, 1], "is_penalty"] = True
    t["label_exclusion"] = None
    t.loc[n_primary, "label_exclusion"] = "sb_goal_without_sn_goal"
    return t[et.JOINED_COLUMNS]


def inputs(joined: pd.DataFrame) -> dict:
    rng = np.random.default_rng(1)
    n = len(joined)
    d = descriptor_frame(n)
    features = pd.DataFrame({"event_id": joined.statsbomb_event_id} | {c: d[f"sb_{c}"].to_numpy() for c in tb.SB_FEATURES})
    primary = joined.soccernet_shot_id.dropna()
    e3 = pd.concat([pd.DataFrame({"shot_id": primary, "offset_ms": o} | {c: d[f"e3_{c}"].to_numpy()[: len(primary)] for c in tb.E3_DESCRIPTORS})
                    for o in (0, -400)], ignore_index=True)
    a1 = pd.DataFrame({"shot_id": primary, "role": "evaluation", "p_a1": rng.uniform(0.01, 0.5, len(primary))})
    sequence = pd.DataFrame({"event_id": joined.statsbomb_event_id, "minute": rng.integers(0, 45, n), "prev_event_id": None,
                             "prev_dt_ms": np.nan, "prev_same_team": False, "prev_outcome": None, "rebound_5s": False, "rebound_3s": False})
    return {"features": features, "e3": e3, "a1": a1, "sequence": sequence}


def test_build_keeps_the_papers_populations_and_splits_off_the_labels():
    joined = joined_table()
    excluded = joined.key.iloc[45]
    built = tb.build(joined, {excluded: "decode_corrupted_frame"}, **inputs(joined))
    table, labels = built.table, built.labels
    assert list(table.columns) == tb.COLUMNS
    assert not set(table.columns) & {*tb.LABEL_COLUMNS, *et.LABEL_COLUMNS}
    assert list(labels.columns) == ["key", *tb.LABEL_COLUMNS] and labels.key.equals(table.key)
    keys = set(table.key)
    assert not keys & {"sn0", "sn1", "sn39", excluded, joined.key.iloc[40]}   # penalties, no frame, frame and label exclusions
    assert table.in_primary.sum() == 37 and len(table) == 37 + 12 - 2
    assert not table.key.duplicated().any() and not table.statsbomb_event_id.duplicated().any()
    original = joined.set_index("key")
    assert np.allclose(table.p_vendor, original.loc[table.key, "pred_vendor_xg"])
    assert np.allclose(table.z_frame, np.clip(original.loc[table.key, "frame_logit_0"], -13.8, 13.8))
    assert labels.set_index("key").goal_sn.equals(original.loc[table.key, "goal_sn"].rename_axis("key"))
    assert table.p_a1.notna().eq(table.in_primary).all()
    assert table.e3_n_players.notna().eq(table.in_primary).all()
    assert set(table.scan_half) == set(pr.HALVES)


@pytest.mark.parametrize("frame", ["features", "e3", "a1", "sequence"])
def test_joins_are_one_to_one(frame):
    joined = joined_table()
    parts = inputs(joined)
    part = parts[frame]
    if frame == "e3":
        part = pd.concat([part, part[part.offset_ms.eq(0)].head(1)], ignore_index=True)
    else:
        part = pd.concat([part, part.head(1)], ignore_index=True)
    parts[frame] = part
    with pytest.raises(ValueError, match="duplicate"):
        tb.build(joined, {}, **parts)


def test_a_primary_shot_without_detector_descriptors_or_statsbomb_features_is_refused():
    joined = joined_table()
    parts = inputs(joined)
    parts["e3"] = parts["e3"][parts["e3"].shot_id.ne("sn5")]
    with pytest.raises(ValueError, match="detector descriptors: no match for 1 of"):
        tb.build(joined, {}, **parts)
    parts = inputs(joined)
    parts["features"] = parts["features"].iloc[3:]
    with pytest.raises(ValueError, match="StatsBomb features"):
        tb.build(joined, {}, **parts)


def test_check_refuses_a_label_column_and_a_misaligned_label_file():
    joined = joined_table()
    built = tb.build(joined, {}, **inputs(joined))
    with pytest.raises(ValueError, match="columns"):
        tb.check(built.table.assign(goal_sn=False), built.labels)
    with pytest.raises(ValueError, match="label file"):
        tb.check(built.table, built.labels.iloc[::-1].reset_index(drop=True))


def test_write_and_read_round_trip_with_hashes(tmp_path):
    joined = joined_table()
    built = tb.build(joined, {}, **inputs(joined))
    meta = tb.write(built, tmp_path, {"test": True})
    table, read_meta = tb.read(tmp_path)
    pd.testing.assert_frame_equal(table, built.table)
    pd.testing.assert_frame_equal(tb.read_labels(tmp_path, read_meta), built.labels)
    assert meta["primary"]["shots"] == int(built.table.in_primary.sum())
    (tmp_path / tb.LABEL_FILE).write_bytes(et.table_bytes(built.labels.assign(goal_sn=~built.labels.goal_sn)))
    with pytest.raises(ValueError, match="hash"):
        tb.read_labels(tmp_path, read_meta)


def test_coverage_counts_shots_and_matches_without_labels():
    joined = joined_table()
    built = tb.build(joined, {}, **inputs(joined))
    coverage = tb.coverage(built.table)
    assert not {"goals", "goal_sn", "rate"} & set(coverage.columns)
    row = coverage[coverage.population.eq("primary") & coverage.category.eq("body")]
    assert row.n_in.sum() == built.table.in_primary.sum()
    camera = coverage[coverage.population.eq("combined") & coverage.category.eq("players_visible")]
    assert (camera.n_undefined == (~built.table.in_primary).sum()).all()


def test_coverage_floors_follow_the_protocol():
    rows = 60 * [("m1", "discovery", "a")] + 60 * [("m2", "validation", "a")] + 70 * [("m1", "discovery", "b")] + 70 * [("m2", "validation", "b")]
    table = pd.DataFrame(rows, columns=["match", "scan_half", "body"])
    shots = pd.DataFrame({c: [None] * len(table) for c in tb.CATEGORY_COLUMNS}).assign(
        cat_body=table.body.map({"a": "Head", "b": "Right Foot"}), match=table.match, scan_half=table.scan_half, in_primary=True)
    head = tb.coverage(shots).query("population == 'primary' and category == 'body' and level == 'Head'").iloc[0]
    assert (head.n_in, head.n_out, head.matches_in, head.n_in_discovery, head.n_out_validation) == (120, 140, 2, 60, 70)
    assert not head.meets_category_floor and not head.meets_scan_floor          # 2 matches < 20
    many = shots.assign(match=[f"m{i % 40}" for i in range(len(shots))])
    head = tb.coverage(many).query("population == 'primary' and category == 'body' and level == 'Head'").iloc[0]
    assert head.meets_category_floor and head.meets_scan_floor
    few = many.assign(cat_body=np.where(np.arange(len(many)) < 100, "Head", "Right Foot"))   # 60 in discovery, 40 in validation
    head = tb.coverage(few).query("population == 'primary' and category == 'body' and level == 'Head'").iloc[0]
    assert head.meets_category_floor and not head.meets_scan_floor


# ------------------------------------------------------------------ split
def test_scan_halves_split_each_stratum_within_one_match_and_are_reproducible():
    matches = pd.Series([f"s{s}/m{i}" for s in range(3) for i in range(5 + s)])
    strata = matches.str.split("/").str[0]
    halves = tb.scan_halves(matches, strata, seed=pr.SPLIT_SEED)
    assert halves == tb.scan_halves(matches.sample(frac=1, random_state=3), strata.sample(frac=1, random_state=3), seed=pr.SPLIT_SEED)
    for s in strata.unique():
        names = matches[strata.eq(s)]
        counts = pd.Series([halves[m] for m in names]).value_counts()
        assert abs(counts.get("discovery", 0) - counts.get("validation", 0)) <= 1
    total = pd.Series(halves).value_counts()
    assert abs(total["discovery"] - total["validation"]) <= 1
    assert halves != tb.scan_halves(matches, strata, seed=1)


def test_league_season_from_the_match_path():
    assert tb.league_season(pd.Series(["england_epl/2015-2016/2015-08-08 - 19-30 A 2 - 2 B"])).iloc[0] == "england_epl/2015-2016"
