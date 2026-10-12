"""Evaluation-table contract (xg-vendor ticket 07, seam 2): validator refusals, label-join log, embedding round trip."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from xgpoc.vendor import evaltable as et
from xgpoc.vendor.evaltable import EvalTableError, PreRegistration

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
DOC_SHA = "a" * 64
N_STORED = 12


def make_table(kind="development", n_primary=4, n_recovered=1, n_only=2) -> pd.DataFrame:
    """A small valid label-free table: primary, recovered-goal and StatsBomb-only rows, ids that are unique by construction."""
    populations = ["primary"] * n_primary + ["recovered_goal"] * n_recovered + ["statsbomb_only"] * n_only
    n = len(populations)
    primary = np.array(populations) == "primary"
    event = [f"sb-{i}" for i in range(n)]
    sn = [f"sn-{i}" if p else None for i, p in enumerate(primary)]
    table = pd.DataFrame({
        "key": [s if s else e for s, e in zip(sn, event, strict=True)], "soccernet_shot_id": sn, "statsbomb_event_id": event,
        "statsbomb_match_id": pd.array(np.arange(n) // 3 + 100, dtype="Int64"), "match": [f"game{i // 3}" for i in range(n)],
        "half": np.int8(1 + np.arange(n) % 2), "population": populations, "split": (["test", "test", "valid", "train"] * 4)[:n],
        "sb_body_part": "Right Foot", "sb_shot_type": "Open Play"})
    for i, column in enumerate(et.PROBABILITY_COLUMNS):
        table[column] = np.linspace(0.05, 0.4, n) + 0.01 * i
    for i, column in enumerate(et.FRAME_LOGIT_COLUMNS):
        table[column] = np.linspace(-2.5, -0.5, n) + 0.1 * i
    table["frame_status"] = et.OK
    table["frame_source"] = "ensemble_10"
    table["emb_g_fold0"] = np.where(np.arange(n) % 2 == 0, np.arange(n) // 2, -1).astype(np.int32)
    table["emb_g_fold1"] = np.int32(-1)
    table["emb_dinov2"] = np.arange(n, dtype=np.int32)
    table["timing_residual_ms"] = np.where(primary, 12.0, np.nan)
    table["anchor_convention"] = table.population.map(et.ANCHOR_CONVENTION)
    table["table_kind"] = kind
    return table[et.LABEL_FREE_COLUMNS]


def make_labels(table: pd.DataFrame) -> pd.DataFrame:
    primary = table.population.eq("primary").to_numpy()
    return pd.DataFrame({"key": table.key, "goal_sn": np.arange(len(table)) % 3 == 0,
                         "on_target_sn": pd.array(np.where(primary, np.arange(len(table)) % 2 == 0, None), dtype="boolean"),
                         "sb_outcome": "Saved", "is_penalty": False, "label_exclusion": None})


def meta_for(table: pd.DataFrame, kind="development", **overrides) -> dict:
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": N_STORED, "model": "fixture"} for c in et.EMBEDDING_COLUMNS}
    meta = {"schema_version": et.SCHEMA_VERSION, "table_kind": kind, "labels_state": et.labels_state(table), **et.counts(table),
            "embedding_sets": sets, "unavailable_columns": [], "label_join": None}
    return meta | overrides


def write(tmp_path, table=None, **kwargs):
    table = make_table(kwargs.get("kind", "development")) if table is None else table
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": N_STORED, "model": "fixture"} for c in et.EMBEDDING_COLUMNS}
    et.write_table(table, tmp_path, "t", embedding_sets=sets, **{"kind": "development", **kwargs})
    return table


def test_valid_tables_pass():
    table = make_table()
    et.validate_table(table, meta_for(table))


def test_duplicate_keys_are_rejected():
    table = make_table()
    table.loc[1, "key"] = table.loc[0, "key"]
    with pytest.raises(EvalTableError, match="duplicate keys"):
        et.validate_table(table, meta_for(table))


def test_a_statsbomb_event_in_two_rows_is_rejected():
    table = make_table()
    table.loc[5, "statsbomb_event_id"] = table.loc[6, "statsbomb_event_id"]
    table.loc[5, "key"] = table.loc[6, "key"] + "x"
    with pytest.raises(EvalTableError, match="two rows"):
        et.validate_table(table, meta_for(table))


def test_missing_required_columns_are_rejected():
    table = make_table().drop(columns=["pred_vendor_xg", "emb_dinov2"])
    with pytest.raises(EvalTableError, match=r"missing required columns: \['pred_vendor_xg', 'emb_dinov2'\]"):
        et.validate_table(table, meta_for(table))


def test_unexpected_columns_are_rejected():
    table = make_table().assign(extra=1.0)
    with pytest.raises(EvalTableError, match="unexpected columns"):
        et.validate_table(table, meta_for(table))


def test_labels_in_a_table_marked_labels_absent_are_rejected():
    table = make_table().assign(goal_sn=True)
    with pytest.raises(EvalTableError, match="labels present in a table marked labels absent"):
        et.validate_table(table, meta_for(table, labels_state="absent"))


def test_probabilities_and_frame_logits_are_checked():
    table = make_table()
    table.loc[0, "pred_event"] = 1.2
    with pytest.raises(EvalTableError, match="probability"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[0, "frame_logit_0"] = np.nan
    with pytest.raises(EvalTableError, match="frame_logit_0 is missing where frame_status is ok"):
        et.validate_table(table, meta_for(table))
    table.loc[0, "frame_status"] = "missing_frame"   # a listed reason excuses the gap
    et.validate_table(table, meta_for(table))


def test_unavailable_columns_must_be_empty_and_only_in_development():
    table = make_table()
    with pytest.raises(EvalTableError, match="declared unavailable but holds values"):
        et.validate_table(table, meta_for(table, unavailable_columns=["frame_logit_m400"]))
    table["frame_logit_m400"] = np.nan
    et.validate_table(table, meta_for(table, unavailable_columns=["frame_logit_m400"]))
    confirmatory = table.assign(table_kind="confirmatory")
    with pytest.raises(EvalTableError, match="no unavailable columns"):
        et.validate_table(confirmatory, meta_for(confirmatory, kind="confirmatory", unavailable_columns=["frame_logit_m400"]))


def test_population_rules():
    table = make_table()
    table.loc[5, "timing_residual_ms"] = 3.0
    with pytest.raises(EvalTableError, match="timing residual"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[5, "anchor_convention"] = "soccernet_anchor"
    with pytest.raises(EvalTableError, match="anchor_convention"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[0, "population"] = "weird"
    with pytest.raises(EvalTableError, match="unknown population"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[5, "soccernet_shot_id"] = "sn-extra"
    with pytest.raises(EvalTableError, match="SoccerNet shot id belongs to primary rows"):
        et.validate_table(table, meta_for(table))


def test_kind_must_agree_with_the_sidecar():
    table = make_table(kind="confirmatory")
    with pytest.raises(EvalTableError, match="table_kind column disagrees"):
        et.validate_table(table, meta_for(table, kind="development"))


def test_sidecar_counts_must_match():
    table = make_table()
    meta = meta_for(table)
    meta["by_split"] = {"test": 99}
    with pytest.raises(EvalTableError, match="by_split"):
        et.validate_table(table, meta)


def test_embedding_references_are_checked():
    table = make_table()
    table.loc[2, "emb_dinov2"] = table.loc[3, "emb_dinov2"]
    with pytest.raises(EvalTableError, match="two rows reference the same embedding"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[2, "emb_dinov2"] = N_STORED
    with pytest.raises(EvalTableError, match="points past"):
        et.validate_table(table, meta_for(table))
    table = make_table()
    table.loc[2, "emb_g_fold1"] = -2
    with pytest.raises(EvalTableError, match="below -1"):
        et.validate_table(table, meta_for(table))


def test_storage_round_trip_and_tampering(tmp_path):
    table = write(tmp_path)
    loaded, meta = et.read_table(tmp_path, "t")
    pd.testing.assert_frame_equal(loaded, table)
    assert meta["labels_state"] == "absent" and meta["rows"] == len(table) and meta["by_population"]["primary"] == 4
    assert meta["table_sha256"] == et.table_sha256(table)
    (tmp_path / "t.parquet").write_bytes(et.table_bytes(table.assign(pred_event=0.5)))
    with pytest.raises(EvalTableError, match="does not match the hash"):
        et.read_table(tmp_path, "t")


def test_write_refuses_an_invalid_table(tmp_path):
    table = make_table()
    table.loc[1, "key"] = table.loc[0, "key"]
    with pytest.raises(EvalTableError):
        write(tmp_path, table)
    assert not (tmp_path / "t.parquet").exists()


def test_label_join_logs_the_hash_of_the_label_free_table(tmp_path):
    table = write(tmp_path)
    label_free_hash = et.read_table(tmp_path, "t")[1]["table_sha256"]
    labels = make_labels(table).sample(frac=1, random_state=0)   # the join does not depend on label order
    log = tmp_path / "join_log.jsonl"
    meta = et.join_labels(tmp_path, "t", labels, "t_labels", log, now=T0)
    joined, joined_meta = et.read_table(tmp_path, "t_labels")
    assert joined_meta == meta and meta["labels_state"] == "joined"
    assert list(joined.columns) == et.JOINED_COLUMNS and joined.key.equals(table.key)
    assert meta["label_join"]["label_free_sha256"] == label_free_hash
    assert et.table_sha256(joined.drop(columns=et.LABEL_COLUMNS)) == label_free_hash   # the hash is verifiable from the joined table
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["label_free_sha256"] == label_free_hash and record["joined_at"] == et.format_time(T0)
    assert record["joined_sha256"] == meta["table_sha256"] and record["label_free_table"] == "t"
    # the label-free table stays untouched
    assert et.read_table(tmp_path, "t")[1]["table_sha256"] == label_free_hash


def test_join_refuses_a_time_without_timezone(tmp_path):
    table = write(tmp_path)
    with pytest.raises(EvalTableError, match="no timezone"):
        et.join_labels(tmp_path, "t", make_labels(table), "j", tmp_path / "log", now=datetime(2026, 10, 12, 9, 0))


def test_a_second_join_appends_to_the_log_and_a_joined_table_cannot_be_joined_again(tmp_path):
    table = write(tmp_path)
    log = tmp_path / "log.jsonl"
    et.join_labels(tmp_path, "t", make_labels(table), "a", log, now=T0)
    et.join_labels(tmp_path, "t", make_labels(table), "b", log, now=T0 + timedelta(hours=1))
    assert len(log.read_text().splitlines()) == 2
    with pytest.raises(EvalTableError, match="already has labels"):
        et.join_labels(tmp_path, "a", make_labels(table), "c", log, now=T0)


def test_join_refuses_labels_that_do_not_cover_the_keys(tmp_path):
    table = write(tmp_path)
    labels = make_labels(table)
    with pytest.raises(EvalTableError, match="exactly the table's keys"):
        et.join_labels(tmp_path, "t", labels.iloc[1:], "x", tmp_path / "log", now=T0)
    with pytest.raises(EvalTableError, match="duplicate keys"):
        et.join_labels(tmp_path, "t", pd.concat([labels, labels.iloc[:1]]), "x", tmp_path / "log", now=T0)
    assert not (tmp_path / "log").exists()


def confirmatory(tmp_path, recorded_at=T0, sha=DOC_SHA):
    """A written confirmatory label-free table and the pre-registration record that pins it."""
    table = write(tmp_path, make_table("confirmatory"), kind="confirmatory")
    return table, PreRegistration(sha, recorded_at, et.read_table(tmp_path, "t")[1]["table_sha256"], margin=0.005, n_boot=5000,
                                  fold_seed=0, boot_seed=0, offset_logits="calibrated", h3_thresholds={"dist": 0.5},
                                  h3_confirmatory_targets=("dist",), h3_control_max_r2=0.0, timing_rule_sha256="f" * 64, recalibrated_bootstrap="refit",
                                  frame_exclusions={}, draft=False, commit="abc123")


def test_a_draft_preregistration_admits_no_confirmatory_join(tmp_path):
    table, prereg = confirmatory(tmp_path)
    draft = replace(prereg, draft=True)
    with pytest.raises(EvalTableError, match="draft pre-registration record cannot admit a confirmatory join"):
        et.join_labels(tmp_path, "t", make_labels(table), "x", tmp_path / "log", prereg=draft, now=T0)
    et.join_labels(tmp_path, "t", make_labels(table), "x", tmp_path / "log", prereg=prereg, now=T0)
    joined, meta = et.read_table(tmp_path, "x")
    with pytest.raises(EvalTableError, match="draft pre-registration record cannot admit a confirmatory table"):
        et.validate_table(joined, meta, draft)


def test_a_dryrun_table_needs_no_preregistration_and_no_unavailable_columns(tmp_path):
    table = write(tmp_path, make_table("dryrun"), kind="dryrun")
    et.join_labels(tmp_path, "t", make_labels(table), "j", tmp_path / "log", now=T0)
    assert et.load_table(tmp_path, "j")[1]["table_kind"] == "dryrun"
    with pytest.raises(EvalTableError, match="a dryrun table has no unavailable columns"):
        et.validate_table(table, meta_for(table, "dryrun", unavailable_columns=["frame_logit_m400"]))


def test_confirmatory_join_needs_a_preregistration_not_later_than_the_join(tmp_path):
    table, prereg = confirmatory(tmp_path)
    labels = make_labels(table)
    log = tmp_path / "log.jsonl"
    with pytest.raises(EvalTableError, match="needs the pre-registration record"):
        et.join_labels(tmp_path, "t", labels, "x", log, now=T0)
    with pytest.raises(EvalTableError, match="before the pre-registration record"):
        et.join_labels(tmp_path, "t", labels, "x", log, prereg=prereg, now=T0 - timedelta(seconds=1))
    assert not log.exists() and not (tmp_path / "x.parquet").exists()
    meta = et.join_labels(tmp_path, "t", labels, "x", log, prereg=prereg, now=T0)
    assert meta["label_join"]["prereg_sha256"] == DOC_SHA


def test_validator_rejects_a_confirmatory_table_joined_before_the_preregistration(tmp_path):
    table, early = confirmatory(tmp_path, T0 - timedelta(days=1))
    et.join_labels(tmp_path, "t", make_labels(table), "x", tmp_path / "log", prereg=early, now=T0)
    joined, meta = et.read_table(tmp_path, "x")
    et.validate_table(joined, meta, early)
    assert et.load_table(tmp_path, "x", early)[1] == meta
    with pytest.raises(EvalTableError, match="needs the pre-registration record"):
        et.load_table(tmp_path, "x")   # the way analyses open a table does not skip the check
    later = replace(early, recorded_at=T0 + timedelta(seconds=1))
    with pytest.raises(EvalTableError, match="predates the pre-registration record"):
        et.validate_table(joined, meta, later)
    with pytest.raises(EvalTableError, match="another pre-registration hash"):
        et.validate_table(joined, meta, replace(early, sha256="b" * 64))


def test_the_preregistration_must_pin_the_label_free_table(tmp_path):
    table, prereg = confirmatory(tmp_path)
    elsewhere = replace(prereg, label_free_table_sha256="c" * 64)
    with pytest.raises(EvalTableError, match="not the one the pre-registration pins"):
        et.join_labels(tmp_path, "t", make_labels(table), "x", tmp_path / "log", prereg=elsewhere, now=T0)
    et.join_labels(tmp_path, "t", make_labels(table), "x", tmp_path / "log", prereg=prereg, now=T0)


def test_labels_edited_after_the_join_are_detected(tmp_path):
    table = write(tmp_path)
    et.join_labels(tmp_path, "t", make_labels(table), "j", tmp_path / "log", now=T0)
    joined, meta = et.read_table(tmp_path, "j")
    flipped = joined.assign(goal_sn=~joined.goal_sn)   # predictions untouched, labels changed
    with pytest.raises(EvalTableError, match="labels do not match the hash recorded"):
        et.validate_table(flipped, meta)


def test_a_joined_table_must_match_its_join_record(tmp_path):
    table = write(tmp_path)
    et.join_labels(tmp_path, "t", make_labels(table), "j", tmp_path / "log", now=T0)
    joined, meta = et.read_table(tmp_path, "j")
    tampered = joined.assign(pred_event=joined.pred_event * 0.9)   # predictions edited after the join
    with pytest.raises(EvalTableError, match="does not match the hash recorded at the label join"):
        et.validate_table(tampered, meta)
    meta["label_join"] = None
    with pytest.raises(EvalTableError, match="needs the label-join record"):
        et.validate_table(joined, meta)


def test_on_target_belongs_to_primary_rows(tmp_path):
    table = write(tmp_path)
    labels = make_labels(table)
    labels["on_target_sn"] = pd.array([True] * len(labels), dtype="boolean")
    with pytest.raises(EvalTableError, match="on_target_sn belongs to primary rows"):
        et.join_labels(tmp_path, "t", labels, "j", tmp_path / "log", now=T0)


def test_a_label_exclusion_needs_a_known_reason_code(tmp_path):
    table = write(tmp_path)
    labels = make_labels(table)
    only = table.population.eq("statsbomb_only").to_numpy()
    labels.loc[only, ["sb_outcome", "label_exclusion"]] = ["Goal", "sb_goal_without_sn_goal"]
    meta = et.join_labels(tmp_path, "t", labels, "j", tmp_path / "log", now=T0)
    joined, _ = et.load_table(tmp_path, "j")
    assert joined.label_exclusion.notna().sum() == only.sum() == len(table) - meta["by_population"]["primary"] - 1
    labels.loc[only, "label_exclusion"] = "looks_odd"
    with pytest.raises(EvalTableError, match=r"unknown label exclusion codes: \['looks_odd'\]"):
        et.join_labels(tmp_path, "t", labels, "k", tmp_path / "log", now=T0)


PREREG = {"sha256": "d" * 64, "recorded_at": "2026-10-12T11:00:00+02:00", "label_free_table_sha256": "e" * 64, "margin": 0.005,
          "n_boot": 5000, "fold_seed": 0, "boot_seed": 0, "offset_logits": "calibrated", "h3_thresholds": {"dist": 0.5, "head": 0.7},
          "h3_confirmatory_targets": ["dist"], "h3_control_max_r2": 0.005, "timing_rule_sha256": "f" * 64, "recalibrated_bootstrap": "refit",
          "frame_exclusions": {"e1": "decode_corrupted_frame"}, "draft": False, "commit": "abc"}


def test_preregistration_file_round_trip(tmp_path):
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(PREREG))
    record = et.load_preregistration(path)
    assert record.recorded_at == datetime(2026, 10, 12, 9, 0, tzinfo=UTC) and record.commit == "abc"
    assert (record.margin, record.n_boot, record.fold_seed, record.boot_seed, record.draft) == (0.005, 5000, 0, 0, False)
    assert (record.offset_logits, record.h3_thresholds, record.h3_confirmatory_targets) == ("calibrated", {"dist": 0.5, "head": 0.7}, ("dist",))
    assert (record.recalibrated_bootstrap, record.frame_exclusions) == ("refit", {"e1": "decode_corrupted_frame"})
    assert record.h3_control_max_r2 == 0.005
    path.write_text(json.dumps(et.preregistration_record(record)))
    assert et.load_preregistration(path) == record
    bad = {"recorded_at": "2026-10-12T09:00:00", "sha256": "xyz", "margin": 0.0, "n_boot": 0, "fold_seed": -1, "draft": "no",
           "offset_logits": "both", "h3_confirmatory_targets": ["angle"], "timing_rule_sha256": None, "recalibrated_bootstrap": "both",
           "frame_exclusions": {"e1": "blurry"}, "h3_control_max_r2": 0.5}
    messages = {"recorded_at": "no timezone", "sha256": "sha256 hex", "margin": "not a log-loss margin", "n_boot": "n_boot 0 is not an integer",
                "fold_seed": "fold_seed -1 is not an integer", "draft": "draft must be true or false",
                "offset_logits": "offset_logits 'both'", "h3_confirmatory_targets": "thresholded targets",
                "timing_rule_sha256": "timing_rule_sha256 is not a sha256", "recalibrated_bootstrap": "recalibrated_bootstrap 'both'",
                "frame_exclusions": "frame_exclusions must map", "h3_control_max_r2": "h3_control_max_r2 0.5"}
    for field, value in bad.items():
        path.write_text(json.dumps(PREREG | {field: value}))
        with pytest.raises(EvalTableError, match=messages[field]):
            et.load_preregistration(path)
    path.write_text(json.dumps({k: v for k, v in PREREG.items() if k != "margin"}))
    with pytest.raises(EvalTableError, match="lacks 'margin'"):
        et.load_preregistration(path)


def test_embedding_references_round_trip(tmp_path):
    """Rows named by a reference column come back in table order; -1 rows are skipped."""
    rng = np.random.default_rng(0)
    arrays = {c: rng.normal(size=(N_STORED, 4)).astype(np.float16) for c in et.EMBEDDING_COLUMNS}
    for column, array in arrays.items():
        np.save(tmp_path / f"{column}.npy", array)
    table = write(tmp_path)
    loaded, meta = et.read_table(tmp_path, "t")
    for column in et.EMBEDDING_COLUMNS:
        positions, matrix = et.load_embeddings(loaded, meta, column, root=tmp_path)
        refs = table[column].to_numpy()
        np.testing.assert_array_equal(positions, np.flatnonzero(refs >= 0))
        np.testing.assert_array_equal(matrix, arrays[column][refs[refs >= 0]])
    positions, _ = et.load_embeddings(loaded, meta, "emb_g_fold1", root=tmp_path)
    assert len(positions) == 0
    with pytest.raises(EvalTableError, match="does not match the sidecar's sha256"):
        et.load_embeddings(loaded, meta, "emb_dinov2", root=tmp_path, verify_hash=True)   # the fixture sidecar carries a dummy hash
    np.save(tmp_path / "emb_dinov2.npy", arrays["emb_dinov2"][:5])
    with pytest.raises(EvalTableError, match="has shape"):
        et.load_embeddings(loaded, meta, "emb_dinov2", root=tmp_path)
    with pytest.raises(EvalTableError, match="not an embedding column"):
        et.load_embeddings(loaded, meta, "pred_event")
