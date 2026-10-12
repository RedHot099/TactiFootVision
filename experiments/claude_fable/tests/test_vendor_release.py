"""Release package of the xg-vendor study (ticket 16): no StatsBomb value and no SoccerNet annotation may leave.

Fixture source tables carry every column of the real ones, with StatsBomb values planted, so a test fails if any
of them reaches the package. The last tests check the built package itself when it exists."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgpoc.eval.calibration import logit
from xgpoc.models.g_bundle import ensemble_probability
from xgpoc.vendor import release as rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vendor_release as vr  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
PACKAGE = EXP / "results/xgv_release_v1"
GAME = "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea"
OTHER = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
# Columns that hold StatsBomb values or SoccerNet annotations somewhere in the study's tables.
KNOWN_FORBIDDEN = {
    "sb_xg", "statsbomb_xg", "pred_vendor_xg", "sb_outcome", "outcome", "sb_goal", "goal", "goal_sn", "on_target_sn",
    "is_penalty", "sb_penalty", "label_exclusion", "sb_x", "sb_y", "x", "y", "sb_timestamp_ms", "timestamp_ms",
    "residual_ms", "timing_residual_ms", "offset_ms", "half_offset_ms", "nearest_sb_ms", "lag_ms", "label_lag_ms",
    "target_position_ms", "sb_body_part", "sb_shot_type", "sb_technique", "sb_under_pressure", "sb_first_time",
    "sb_play_pattern", "sb_n_freeze_frame", "pred_constant", "pred_dist_angle", "pred_event", "pred_freeze_frame",
    "dist", "angle", "ff_def_cone", "def_cone_ge1", "gk_depth", "head", "label", "team", "sb_team", "sb_side",
    "split", "official_split", "sn_home", "sn_away", "sn_score", "sb_home", "sb_away", "name_sim", "emb_g_fold0",
}


def sources() -> dict[str, pd.DataFrame]:
    match_map = pd.DataFrame({
        "game": [GAME, OTHER], "league": ["england_epl"] * 2, "season": ["2015-2016", "2014-2015"],
        "sn_home": ["Chelsea"] * 2, "sn_away": ["Swansea", "Burnley"], "sn_score": ["2 - 2", "1 - 1"],
        "sb_match_id": [3754078.0, np.nan], "sb_home": ["Chelsea", np.nan], "sb_away": ["Swansea City", np.nan],
        "name_sim": [1.0, np.nan], "reason": ["paired", "no StatsBomb match within 1 day"]})
    pairs = pd.DataFrame({
        "shot_id": ["s1", "s2"], "game": [GAME] * 2, "league": ["england_epl"] * 2, "split": ["train"] * 2,
        "half": [1, 2], "position_ms": [401367, 600642], "label": ["Shots off target", "Shots on target"],
        "team": ["away", "home"], "goal": [False, True], "is_penalty": [False, False], "paired": [True, False],
        "residual_ms": [9.0, np.nan], "offset_ms": [-781.0, -1598.0], "nearest_sb_ms": [9.0, 15000.0],
        "sb_event_id": ["e1", None], "sb_match_id": pd.array([3754078, 3754078], dtype="Int64"),
        "sb_timestamp_ms": pd.array([402139, pd.NA], dtype="Int64"), "sb_team": ["Swansea City", None],
        "sb_side": ["away", None], "sb_xg": [0.0716, np.nan], "sb_outcome": ["Off T", None],
        "sb_goal": pd.array([False, pd.NA], dtype="boolean"), "sb_penalty": pd.array([False, pd.NA], dtype="boolean"),
        "sb_body_part": ["Head", None], "sb_shot_type": ["Open Play", None], "sb_technique": ["Normal", None],
        "sb_first_time": pd.array([False, pd.NA], dtype="boolean"),
        "sb_under_pressure": pd.array([True, pd.NA], dtype="boolean"), "sb_x": [110.0, np.nan], "sb_y": [36.0, np.nan],
        "sb_play_pattern": ["Regular Play", None], "sb_n_freeze_frame": pd.array([14, pd.NA], dtype="Int64"),
        "team_agree": pd.array([True, pd.NA], dtype="boolean")})
    goals = pd.DataFrame({
        "game": [GAME], "half": [2], "anchor_ms": [301120], "goal_label_ms": [302120], "team": ["away"],
        "sb_match_id": [3754078], "offset_ms": [-860.0], "any_candidate": [True], "recovered": [True],
        "sb_event_id": ["e3"], "sb_timestamp_ms": [302500.0], "lag_ms": [-520.0], "label_lag_ms": [480.0],
        "sb_side": ["away"], "team_agree": [True], "reason": ["recovered"], "target_position_ms": [301640.0]})
    table = pd.DataFrame({
        "key": ["s1", "e3", "e4"], "soccernet_shot_id": ["s1", None, None], "statsbomb_event_id": ["e1", "e3", "e4"],
        "statsbomb_match_id": pd.array([3754078] * 3, dtype="Int64"), "match": [GAME] * 3,
        "half": np.array([1, 2, 2], np.int8), "population": ["primary", "recovered_goal", "statsbomb_only"],
        "split": ["train"] * 3, "sb_body_part": ["Head", "Right Foot", "Left Foot"],
        "sb_shot_type": ["Open Play", "Penalty", "Open Play"], "pred_vendor_xg": [0.0716, 0.783, 0.03],
        "pred_constant": [0.0948, 0.758, 0.0948], "pred_dist_angle": [0.13, 0.758, 0.02],
        "pred_event": [0.04, 0.758, 0.03], "pred_freeze_frame": [0.09, 0.758, 0.025],
        "frame_logit_0": [-1.76, logit(np.array([0.6972]))[0], np.nan],
        "frame_logit_m400": [-0.93, logit(np.array([0.6972]))[0], np.nan],
        "frame_raw_logit_0": [-2.16, 0.4, np.nan], "frame_raw_logit_m400": [-1.12, 0.2, np.nan],
        "frame_status": ["ok", "ok", "0:beyond_half_end;-400:beyond_half_end"],
        "frame_source": ["ensemble_10_xgv_g_combo"] * 3, "emb_g_fold0": np.array([0, 1, -1], np.int32),
        "emb_g_fold1": np.array([0, 1, -1], np.int32), "emb_dinov2": np.array([0, 1, -1], np.int32),
        "timing_residual_ms": [9.0, np.nan, np.nan], "anchor_convention": ["soccernet_anchor", "statsbomb_time", "statsbomb_time"],
        "table_kind": ["confirmatory"] * 3})
    plan = pd.DataFrame({"game": [GAME, OTHER], "league": ["england_epl"] * 2, "season": ["2015-2016", "2014-2015"],
                         "home": ["Chelsea"] * 2, "official_split": ["train", "train"], "role": ["evaluation", "training"],
                         "fold": [-1, 2]})
    return {"match_map": match_map, "pairs": pairs, "recovered_goals": goals, "eval_table": table, "split_plan": plan}


def frames() -> pd.DataFrame:
    return pd.DataFrame({"key": ["s1", "e3", "e4"], "frame_logit_0": [-1.76, 0.55, np.nan],
                         "frame_logit_m400": [-0.93, 0.31, np.nan]})


def built() -> dict[str, pd.DataFrame]:
    return rel.build_tables(sources(), frames(), {"e3": "decode_corrupted_frame"})


def test_every_shipped_column_is_declared_with_a_known_provenance():
    tables = built()
    assert set(tables) == {t.file for t in rel.TABLES}
    for spec in rel.TABLES:
        assert list(tables[spec.file].columns) == [c.name for c in spec.columns]
        assert {c.provenance for c in spec.columns} <= set(rel.PROVENANCE)


def test_no_forbidden_column_is_shipped():
    """The forbidden-column test: no label, no StatsBomb value, no SoccerNet annotation in any table."""
    for file, frame in built().items():
        columns = set(frame.columns)
        assert not columns & KNOWN_FORBIDDEN, (file, columns & KNOWN_FORBIDDEN)
        assert not columns & rel.FORBIDDEN_NAMES, file
        for name in columns - rel.ID_COLUMNS:
            assert not rel.name_tokens(name) & rel.FORBIDDEN_TOKENS, (file, name)
    excluded = {(key, col) for key, cols in rel.EXCLUDED.items() for col in cols}
    shipped = {(t.source, c.source) for t in rel.TABLES for c in t.columns if c.source}
    assert not excluded & shipped


def test_no_statsbomb_value_reaches_the_package():
    planted = {0.0716, 402139, 110.0, 36.0, 9.0, -781.0, 302500.0, -520.0, 0.783, 0.0948, 0.13, 0.04, 0.09}
    for file, frame in built().items():
        numbers = frame.select_dtypes("number").to_numpy(float).ravel()
        assert not planted & set(numbers[np.isfinite(numbers)].tolist()), file


@pytest.mark.parametrize("name", ["sb_xg", "pred_vendor_xg", "timing_residual_ms", "goal_sn", "split", "half_offset_ms",
                                  "freeze_frame_p", "vendor_xg", "end_location", "gk_depth"])
def test_check_columns_refuses_a_forbidden_name(monkeypatch, name):
    tables = built()
    spec = next(t for t in rel.TABLES if t.file == "predictions.csv")
    bad = dataclasses.replace(spec, columns=(*spec.columns, rel.Column(name, "", "ours", "planted")))
    monkeypatch.setattr(rel, "TABLES", tuple(bad if t.file == spec.file else t for t in rel.TABLES))
    tables["predictions.csv"] = tables["predictions.csv"].assign(**{name: 0.5})
    with pytest.raises(rel.ReleaseError, match="forbidden"):
        rel.check_columns(tables)


def test_check_columns_refuses_an_undeclared_column_or_provenance(monkeypatch):
    tables = built()
    with pytest.raises(rel.ReleaseError, match="differ from the declared"):
        rel.check_columns({**tables, "shots.csv": tables["shots.csv"].assign(extra=1)})
    spec = next(t for t in rel.TABLES if t.file == "split_plan.csv")
    bad = dataclasses.replace(spec, columns=tuple(dataclasses.replace(c, provenance="statsbomb_value") if c.name == "role"
                                                  else c for c in spec.columns))
    monkeypatch.setattr(rel, "TABLES", tuple(bad if t.file == spec.file else t for t in rel.TABLES))
    with pytest.raises(rel.ReleaseError, match="unknown provenance"):
        rel.check_columns(tables)


def test_a_new_source_column_must_be_shipped_or_excluded_with_a_reason():
    src = sources()
    src["pairs"] = src["pairs"].assign(sb_end_location_z=1.2)
    with pytest.raises(rel.ReleaseError, match="neither shipped nor excluded"):
        rel.build_tables(src, frames(), {})
    src = sources()
    src["eval_table"] = src["eval_table"].drop(columns="pred_vendor_xg")
    with pytest.raises(rel.ReleaseError, match="absent from the source"):
        rel.check_source_coverage(src)


def test_a_statsbomb_value_copied_under_another_name_is_refused():
    reference = pd.DataFrame({"key": ["a", "b", "c"], "pred_vendor_xg": [0.1, 0.2, 0.3]})
    copied = pd.DataFrame({"key": ["a", "b", "c"], "frame_p_0": [0.1, 0.2, 0.31]})
    with pytest.raises(rel.ReleaseError, match="equals the reference column pred_vendor_xg"):
        rel.check_no_copied_values(copied, "key", reference)
    correlated = copied.assign(frame_p_0=[0.11, 0.21, 0.31])
    rel.check_no_copied_values(correlated, "key", reference)


def test_match_map_codes_and_frame_exclusions():
    tables = built()
    assert tables["match_map.csv"].pairing.tolist() == ["paired", "no_candidate_within_1_day"]
    assert tables["predictions.csv"].frame_exclusion.tolist() == ["", "decode_corrupted_frame", ""]
    assert tables["predictions.csv"].frame_status.tolist() == ["ok", "ok", "no_frame"]   # no StatsBomb-time reason
    np.testing.assert_allclose(tables["predictions.csv"].frame_p_0[:2], 1 / (1 + np.exp([1.76, -0.55])))
    src = sources()
    src["match_map"].loc[1, "reason"] = "something new"
    with pytest.raises(rel.ReleaseError, match="without a release code"):
        rel.build_tables(src, frames(), {})
    with pytest.raises(rel.ReleaseError, match="absent from the table"):
        rel.build_tables(sources(), frames(), {"nope": "decode_corrupted_frame"})


def ensemble_and_scores(rng: np.random.Generator, n: int = 6):
    members = [{"fold": f, "seed": s, "temperature": 1.0 + 0.1 * (2 * f + s)} for f in range(2) for s in range(2)]
    calibration = {"platt": {"a": 0.98, "b": -0.05}, "oof_rate": 0.13, "pooled_rate": 0.14, "penalty_rate": 0.6972}
    routed = np.zeros(n, bool)
    routed[1] = True
    scores = pd.DataFrame({"key": [f"k{i}" for i in range(n)], "penalty_routed": routed})
    for m in members:
        for offset in ("0", "m400"):
            scores[f"xg_f{m['fold']}s{m['seed']}_{offset}"] = rng.normal(-2, 1, n)
    table = pd.DataFrame({"key": scores.key})
    for offset in ("0", "m400"):
        logits = [scores[f"xg_f{m['fold']}s{m['seed']}_{offset}"].to_numpy() for m in members]
        p = ensemble_probability(logits, [m["temperature"] for m in members], calibration, routed)
        table[f"frame_logit_{offset}"] = logit(p)
    return {"members": members, "calibration": calibration}, scores, table


def test_unrouted_logits_equal_the_analysed_ones_except_on_routed_penalties():
    ensemble, scores, table = ensemble_and_scores(np.random.default_rng(0))
    out = rel.unrouted_frame_logits(scores, ensemble, table)
    keep = ~scores.penalty_routed.to_numpy()
    np.testing.assert_allclose(out.frame_logit_0[keep], table.frame_logit_0[keep], atol=1e-12)
    assert abs(out.frame_logit_0[1] - logit(np.array([0.6972]))[0]) > 1e-3   # the model's own value, not the rate
    tampered = table.assign(frame_logit_0=table.frame_logit_0 + np.where(keep, 1e-6, 0))
    with pytest.raises(rel.ReleaseError, match="differ from the analysed"):
        rel.unrouted_frame_logits(scores, ensemble, tampered)


def test_manifest_lists_every_column_and_every_exclusion():
    tables = built()
    record = rel.manifest(tables, {f: "0" * 64 for f in tables})
    for entry, spec in zip(record["tables"], rel.TABLES, strict=True):
        assert [c["name"] for c in entry["columns"]] == [c.name for c in spec.columns]
        assert all(c["provenance"] in rel.PROVENANCE for c in entry["columns"])
    assert {k: set(v) for k, v in record["excluded_columns"].items()} == {
        rel.SOURCES[k]: set(v) for k, v in rel.EXCLUDED.items()}
    text = rel.manifest_markdown(record)
    for spec in rel.TABLES:
        assert all(f"`{c.name}`" in text for c in spec.columns)
    for line in text.splitlines():   # every table row has as many cells as its header: no bare "|" in a cell
        if line.startswith("| `"):
            assert len(line.replace("\\|", "").split("|")) in {5, 7}, line


def test_written_package_verifies_and_a_changed_file_is_caught(tmp_path):
    tables = built()
    out = tmp_path / "pkg"
    files = vr.write_package(out, tables, {"README.md": "r\n", "ATTRIBUTION.md": "a\n"}, b"archive", {"code": {}})
    assert "data/predictions.csv" in files and "code/xg_vendor_code.tar.gz" in files
    vr.verify(out)
    (out / "data/shots.csv").write_text("changed\n")
    with pytest.raises(SystemExit):
        vr.verify(out)


@pytest.mark.skipif(not (PACKAGE / "complete.json").is_file(), reason="needs the built release package")
def test_the_built_package_ships_only_declared_columns():
    record = json.loads((PACKAGE / "manifest.json").read_text())
    for entry in record["tables"]:
        header = pd.read_csv(PACKAGE / entry["file"], nrows=0).columns.tolist()
        assert header == [c["name"] for c in entry["columns"]]
        assert not set(header) & KNOWN_FORBIDDEN
        assert all(c["provenance"] in rel.PROVENANCE for c in entry["columns"])
    on_disk = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*") if p.is_file()} - {"complete.json"}
    assert on_disk == set(json.loads((PACKAGE / "complete.json").read_text())["files"])
    assert not any(p.suffix in {".npy", ".npz", ".pt", ".jpg", ".png", ".mkv", ".mp4", ".parquet"} for p in PACKAGE.rglob("*"))
    predictions = pd.read_csv(PACKAGE / "data/predictions.csv")
    assert set(predictions.frame_status) == {"ok", "no_frame"}
