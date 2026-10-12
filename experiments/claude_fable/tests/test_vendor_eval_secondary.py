"""The secondary population in the evaluation table (xg-vendor ticket 13); CPU only, synthetic frames and tables."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from test_vendor_eval_scoring import (
    CALIBRATION,
    TEMPERATURES,
    Tiny,
    members_for,
    tiny_spec,
)
from xgpoc.eval.calibration import logit
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import eval_secondary as esec
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import labels

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_score_secondary as script  # noqa: E402

SPLITS = {"g/1": "train", "g/2": "valid", "g/3": "test"}
EVENTS = ["ev0", "ev1", "ev2", "ev3", "ev4", "ev5"]


def write_frame(path, value):
    cv2.imwrite(str(path), np.full((36, 64, 3), value, np.uint8))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path):
    """A frame index of six targets (a recovered goal that is a penalty, four StatsBomb-only shots, one of them beyond the
    half end) and the pairing's population table, which also holds label columns the builder must never read."""
    frames = pd.DataFrame({"shot_id": [f"xgv__x__{e}" for e in EVENTS], "kind": ["recovered_goal"] + ["secondary"] * 5, "anchor_convention": esec.CUT_ANCHOR_CONVENTION, "sb_event_id": EVENTS,
                           "sb_match_id": [10, 10, 20, 20, 30, 30], "game": ["g/1", "g/1", "g/2", "g/2", "g/3", "g/3"], "half": [1, 2, 1, 1, 2, 2],
                           "target_position_ms": [1000, 2000, 3000, 4000, 5000, 6000], "clip_status": ["ok", "ok", "ok", "ok", "ok", "clip_failed: boom"],
                           "frame_status_0": ["ok"] * 5 + ["beyond_half_end"], "frame_status_-400": ["ok"] * 5 + ["beyond_half_end"]})
    paths, shas = {0: [], -400: []}, {0: [], -400: []}
    for i, event in enumerate(EVENTS):
        for offset in (0, -400):
            if i == 5:
                path, sha = f"{tmp_path}/{event}/{offset}.jpg", None   # planned path, no file: the re-download lists it anyway
            else:
                (tmp_path / event).mkdir(exist_ok=True)
                path, sha = write_frame(tmp_path / event / f"{offset}.jpg", 20 + 25 * i + (5 if offset else 0))
            paths[offset].append(path)
            shas[offset].append(sha)
    frames["frame_path"], frames["frame_path_-400"] = paths[0], paths[-400]
    frames["frame_sha256_0"], frames["frame_sha256_-400"] = shas[0], shas[-400]
    frames["reason"] = ["", "", "", "", "", "beyond_half_end: target at 6.0 s, the half video lasts 5.0 s"]
    unpaired = pd.DataFrame({"event_id": [*EVENTS, "other"], "match_id": [10, 10, 20, 20, 30, 30, 99], "game": ["g/1", "g/1", "g/2", "g/2", "g/3", "g/3", "g/9"],
                             "period": [1, 2, 1, 1, 2, 2, 1], "statsbomb_xg": [0.78, 0.05, 0.1, 0.2, 0.03, 0.4, 0.5],
                             "body_part": ["Right Foot", "Head", "Left Foot", "Head", "Right Foot", "Right Foot", "Head"],
                             "shot_type": ["Penalty"] + ["Open Play"] * 6, "population": ["recovered_goal"] + ["secondary"] * 5 + ["possible_mispair"],
                             # label columns of the real table: present here, never selected
                             "outcome": ["Goal", "Saved", "Goal", "Blocked", "Off T", "Saved", "Goal"], "goal": [True, False, True, False, False, False, True],
                             "penalty": [True, *[False] * 6], "sb_goal_without_sn_goal": [False, False, True, False, False, False, False]})
    return frames, unpaired


def comparator_for(events=EVENTS, population=None):
    population = population or ["recovered_goal"] + ["statsbomb_only"] * 5
    return pd.DataFrame({"statsbomb_event_id": events, "statsbomb_match_id": [10, 10, 20, 20, 30, 30], "population": population,
                         "soccernet_shot_id": None, "p_constant": 0.13, "p_dist_angle": 0.1, "p_event": 0.12, "p_freeze_frame": 0.11})


def make_shots(world):
    frames, unpaired = world
    return esec.secondary_shots(frames, unpaired, SPLITS)


def test_the_rows_carry_populations_splits_penalty_routing_and_the_reason_for_missing_frames(world):
    shots = make_shots(world)
    assert shots.key.tolist() == EVENTS and shots.role.eq(es.SECONDARY).all()
    assert shots.population.tolist() == ["recovered_goal"] + ["statsbomb_only"] * 5
    assert shots.split.tolist() == ["train", "train", "valid", "valid", "test", "test"]
    assert shots.is_penalty.tolist() == [True] + [False] * 5            # StatsBomb shot type, a pre-shot condition
    assert shots.status_0.tolist() == ["ok"] * 5 + ["beyond_half_end"] and shots.status_m400.tolist() == shots.status_0.tolist()
    assert shots.frame_path[5] == "" and shots["frame_path_-400"][5] == "" and shots.sha256_0[5] == ""   # no path for a frame that was never cut
    assert shots.sha256_0[0] == world[0].frame_sha256_0[0]
    assert shots.pred_vendor_xg.tolist() == [0.78, 0.05, 0.1, 0.2, 0.03, 0.4]
    labels_like = {"outcome", "goal", "penalty", "sb_goal_without_sn_goal", "label", "sb_outcome", "goal_sn"}
    assert not labels_like & set(shots.columns)                          # no outcome reaches the rows


def test_the_builder_does_not_need_any_label_column(world):
    frames, unpaired = world
    bare = unpaired[esec.UNPAIRED_COLUMNS]
    assert esec.secondary_shots(frames, bare, SPLITS).equals(esec.secondary_shots(frames, unpaired, SPLITS))


def test_a_frame_the_index_calls_ok_must_exist_decode_and_match_its_hash(world, tmp_path):
    frames, unpaired = world
    gone = frames.copy()
    (tmp_path / "ev2" / "0.jpg").unlink()
    with pytest.raises(ValueError, match="missing or undecodable"):
        esec.secondary_shots(gone, unpaired, SPLITS)
    write_frame(tmp_path / "ev2" / "0.jpg", 250)                          # a file with other content
    with pytest.raises(ValueError, match="differ from the recorded sha256"):
        esec.secondary_shots(frames, unpaired, SPLITS)


def test_inconsistent_inputs_are_refused(world):
    frames, unpaired = world
    with pytest.raises(ValueError, match="unknown frame status"):
        esec.secondary_shots(frames.assign(**{"frame_status_0": ["ok"] * 5 + ["file_missing"]}), unpaired, SPLITS)
    with pytest.raises(ValueError, match="not in the pairing's population table"):
        esec.secondary_shots(frames, unpaired[unpaired.event_id.ne("ev3")], SPLITS)
    with pytest.raises(ValueError, match="another population"):
        esec.secondary_shots(frames.assign(kind=["secondary"] * 6), unpaired, SPLITS)
    with pytest.raises(ValueError, match="another game, half or StatsBomb match"):
        esec.secondary_shots(frames.assign(half=[2, 2, 1, 1, 2, 2]), unpaired, SPLITS)
    with pytest.raises(ValueError, match="no official split"):
        esec.secondary_shots(frames, unpaired, {"g/1": "train"})
    with pytest.raises(ValueError, match="twice"):
        esec.secondary_shots(pd.concat([frames, frames.iloc[:1]]), unpaired, SPLITS)
    with pytest.raises(ValueError, match="not cut at"):
        esec.secondary_shots(frames.assign(anchor_convention="soccernet_anchor"), unpaired, SPLITS)
    with pytest.raises(ValueError, match="unknown kinds"):
        esec.secondary_shots(frames.assign(kind=["primary"] * 6), unpaired, SPLITS)


def test_frames_unchanged_keeps_the_reason_of_a_row_without_a_path(world, tmp_path):
    shots = make_shots(world)
    assert es.frames_unchanged(shots)
    write_frame(tmp_path / "ev0" / "0.jpg", 240)
    assert not es.frames_unchanged(shots)


def test_the_missing_frames_are_listed_with_their_reason_and_the_clip_disagreement_is_reported(world):
    shots = make_shots(world)
    listed = esec.no_frame_report(shots)
    assert listed.statsbomb_event_id.tolist() == ["ev5"] and listed.frame_status_0[0] == "beyond_half_end" and "half video lasts" in listed.reason[0]
    assert esec.frame_clip_disagreements(shots).empty                       # ev5 has neither frames nor a clip
    odd = world[0].assign(clip_status=["ok", "clip_failed: x", "ok", "ok", "ok", "ok"])
    found = esec.frame_clip_disagreements(esec.secondary_shots(odd, world[1], SPLITS))
    assert found.key.tolist() == ["ev1", "ev5"]                              # frames without a clip; a clip without frames


def test_secondary_rows_are_scored_with_nan_where_no_frame_was_cut(world):
    shots = make_shots(world)
    torch_model = Tiny()
    xg, on_target, embedding = es.score_rows(torch_model, tiny_spec(), shots, "0", batch_size=2, device="cpu", num_workers=0, embedding=True)
    assert np.isnan(xg[5]) and np.isnan(on_target[5]) and np.isfinite(xg[:5]).all() and embedding.shape[0] == 5


def secondary_table(world, first=7, comparator=None):
    shots = make_shots(world)
    members = members_for(len(shots), nan_rows=(5,))
    for m in members.values():
        m["xg"][0, 5] = np.nan
        m["on_target"][:, 5] = np.nan
    scores = es.assemble_scores(shots, members, TEMPERATURES, CALIBRATION)
    return shots, scores, esec.build_secondary_table(shots, scores, comparator_for() if comparator is None else comparator, first)


def test_the_secondary_rows_are_label_free_with_the_statsbomb_time_anchor_and_the_comparator_rungs(world):
    shots, scores, table = secondary_table(world)
    assert list(table.columns) == et.LABEL_FREE_COLUMNS and not set(et.LABEL_COLUMNS) & set(table.columns)
    assert table.key.tolist() == EVENTS == table.statsbomb_event_id.tolist() and table.soccernet_shot_id.isna().all()
    assert table.population.tolist() == ["recovered_goal"] + ["statsbomb_only"] * 5 and table.anchor_convention.eq("statsbomb_time").all()
    assert table.timing_residual_ms.isna().all() and table.table_kind.eq("confirmatory").all()
    assert table[["pred_constant", "pred_dist_angle", "pred_event", "pred_freeze_frame"]].iloc[0].tolist() == [0.13, 0.1, 0.12, 0.11]
    assert table.pred_vendor_xg.tolist() == shots.pred_vendor_xg.tolist()
    assert table.frame_status.tolist() == ["ok"] * 5 + ["0:beyond_half_end;-400:beyond_half_end"]
    assert table.frame_logit_0.iloc[:5].notna().all() and table.frame_logit_m400.iloc[5:].isna().all()
    assert table.frame_logit_0[0] == pytest.approx(float(logit(0.7)))        # the recovered goal is a penalty: the routed penalty rate
    assert table.emb_g_fold0.tolist() == [7, 8, 9, 10, 11, -1] and table.emb_dinov2.tolist() == table.emb_g_fold1.tolist()


def test_the_comparator_must_cover_and_agree_with_the_secondary_rows(world):
    shots = make_shots(world)
    scores = es.assemble_scores(shots, members_for(len(shots)), TEMPERATURES, CALIBRATION)
    comparator = comparator_for()
    with pytest.raises(ValueError, match="no comparator prediction"):
        esec.build_secondary_table(shots, scores, comparator.iloc[1:], 0)
    with pytest.raises(ValueError, match="another population"):
        esec.build_secondary_table(shots, scores, comparator.assign(population=["statsbomb_only"] * 6), 0)
    with pytest.raises(ValueError, match="joins a SoccerNet shot"):
        esec.build_secondary_table(shots, scores, comparator.assign(soccernet_shot_id="soccernet__1"), 0)
    with pytest.raises(ValueError, match="same order"):
        esec.build_secondary_table(shots, scores.iloc[::-1].reset_index(drop=True), comparator, 0)


def primary_table(n=4):
    """A valid primary label-free table of `n` rows whose embedding references run 0..n-1 (the shape of ticket 11's)."""
    sn = [f"sn-{i}" for i in range(n)]
    table = pd.DataFrame({"key": sn, "soccernet_shot_id": sn, "statsbomb_event_id": [f"pe{i}" for i in range(n)],
                          "statsbomb_match_id": pd.array([1, 1, 2, 2][:n], dtype="Int64"), "match": ["g/1", "g/1", "g/2", "g/2"][:n],
                          "half": np.array([1, 2, 1, 2][:n], np.int8), "population": "primary", "split": ["train", "train", "valid", "valid"][:n],
                          "sb_body_part": "Head", "sb_shot_type": "Open Play", "pred_vendor_xg": 0.1, "pred_constant": 0.13, "pred_dist_angle": 0.1,
                          "pred_event": 0.12, "pred_freeze_frame": 0.11, "frame_logit_0": -2.0, "frame_logit_m400": -2.1, "frame_raw_logit_0": -2.2,
                          "frame_raw_logit_m400": -2.3, "frame_status": "ok", "frame_source": es.FRAME_SOURCE, "emb_g_fold0": np.arange(n, dtype=np.int32),
                          "emb_g_fold1": np.arange(n, dtype=np.int32), "emb_dinov2": np.arange(n, dtype=np.int32), "timing_residual_ms": 5.0,
                          "anchor_convention": "soccernet_anchor", "table_kind": "confirmatory"})
    return table[et.LABEL_FREE_COLUMNS]


def test_the_combined_table_validates_and_its_primary_rows_hash_like_the_primary_table(world, tmp_path):
    primary = primary_table()
    _, _, secondary = secondary_table(world, first=len(primary))
    combined = esec.combine_tables(primary, secondary)
    assert combined.population.tolist() == ["primary"] * 4 + ["recovered_goal"] + ["statsbomb_only"] * 5
    assert esec.primary_rows_sha256(combined) == et.table_sha256(primary)            # the byte-identity check
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": 9, "model": "m"} for c in et.EMBEDDING_COLUMNS}
    meta = et.write_table(combined, tmp_path, "t", kind="confirmatory", embedding_sets=sets)
    assert meta["labels_state"] == "absent" and meta["by_population"] == {"primary": 4, "recovered_goal": 1, "statsbomb_only": 5}
    loaded, _ = et.load_table(tmp_path, "t")                                         # the validator, "labels absent"
    assert loaded.iloc[:4].equals(primary)
    assert esec.primary_rows_sha256(combined.assign(frame_logit_0=0.0)) != et.table_sha256(primary)   # an edited primary row is noticed


def test_combining_refuses_a_change_of_the_column_types_of_the_primary_rows(world):
    primary = primary_table()
    _, _, secondary = secondary_table(world, first=4)
    with pytest.raises(ValueError, match="column types"):
        esec.combine_tables(primary, secondary.astype({"half": "int64"}))


def test_the_label_rule_on_the_secondary_rows_excludes_a_statsbomb_goal_without_a_soccernet_goal(world, tmp_path):
    """Decision D4 on rows made by this ticket's builder, synthetic labels: a recovered goal is the only goal; a StatsBomb-only goal
    is a non-goal and is listed in `label_exclusion`; rows without frames are labelled too and stay in the table."""
    frames, unpaired = world
    _, _, secondary = secondary_table(world, first=0)
    table = secondary.assign(table_kind="dryrun")
    pairs = pd.DataFrame(columns=["shot_id", "paired", "label", "goal", "is_penalty", "sb_event_id", "sb_outcome"])
    unpaired = unpaired.assign(sb_goal_without_sn_goal=lambda d: d.event_id.eq("ev2"))   # ev2: outcome Goal, no SoccerNet goal
    out = labels.table_labels(table, pairs, unpaired)
    assert out.goal_sn.tolist() == [True, False, False, False, False, False]
    assert out.label_exclusion.tolist() == [None, None, "sb_goal_without_sn_goal", None, None, None]
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": 6, "model": "m"} for c in et.EMBEDDING_COLUMNS}
    et.write_table(table, tmp_path, "free", kind="dryrun", embedding_sets=sets)
    et.join_labels(tmp_path, "free", out, "joined", tmp_path / "log.jsonl")
    joined, _ = et.load_table(tmp_path, "joined")
    assert len(joined) == 6 and joined.loc[joined.label_exclusion.notna(), "key"].tolist() == ["ev2"]
    with pytest.raises(ValueError, match="flag disagrees"):                              # the flag must agree with the outcome
        labels.table_labels(table, pairs, unpaired.assign(sb_goal_without_sn_goal=False))


def test_scoring_roles_and_embedding_rows_follow_the_roles_present():
    shots = pd.DataFrame({"role": [es.PRIMARY, es.PRIMARY, es.GATE, es.GATE], "k": range(4)})
    assert es.scoring_roles(shots) == [es.PRIMARY, es.GATE] and es.embedding_rows(shots).k.tolist() == [0, 1]
    only = shots.assign(role=es.SECONDARY).iloc[:2]
    assert es.scoring_roles(only) == [es.SECONDARY] and len(es.embedding_rows(only)) == 2
    with pytest.raises(ValueError, match="one evaluation role"):                    # two roles would overwrite each other's embeddings
        es.embedding_rows(shots.assign(role=[es.PRIMARY, es.SECONDARY, es.GATE, es.GATE]))


def test_every_status_the_redownload_gives_for_a_missing_frame_is_listed_not_refused(world):
    frames, unpaired = world
    for reason in esec.NO_FRAME_REASONS:
        shots = esec.secondary_shots(frames.assign(**{"frame_status_0": ["ok"] * 5 + [reason], "frame_status_-400": ["ok"] * 5 + [reason]}), unpaired, SPLITS)
        assert esec.no_frame_report(shots).frame_status_0.tolist() == [reason]


# ------------------------------------------------------------------ the script's checks
def redownload_folder(root, *, version=script.REDOWNLOAD_VERSION, refused=0):
    index = pd.DataFrame({"sb_event_id": ["a"]})
    index.to_parquet(root / "frames_index.parquet")
    (root / "refused_halves.json").write_text(json.dumps([{"half": "x"}] if refused else []))
    record = {"identity": {"version": version}, "summary": {"halves_refused": refused}, "files": {"frames_index.parquet": script.file_sha256(root / "frames_index.parquet")}}
    (root / "complete.json").write_text(json.dumps(record))
    return record


def test_the_redownload_is_read_only_when_complete_and_unrefused(tmp_path):
    redownload_folder(tmp_path)
    frames, record = script.read_frames(tmp_path)
    assert frames.sb_event_id.tolist() == ["a"] and len(record["frames_index_sha256"]) == 64
    (tmp_path / "frames_index.parquet").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from the re-download's completion record"):
        script.read_frames(tmp_path)


def test_a_refused_half_or_another_data_version_stops_the_scoring(tmp_path):
    redownload_folder(tmp_path, refused=1)
    with pytest.raises(SystemExit, match="refused a half"):
        script.read_frames(tmp_path)
    redownload_folder(tmp_path, version="xgv_secondary_frames_v0")
    with pytest.raises(SystemExit, match="data version"):
        script.read_frames(tmp_path)


def test_the_primary_identity_proof_passes_for_unchanged_rows_and_stops_on_any_change(world, tmp_path):
    primary, primary_dir = primary_table(), tmp_path / "primary"
    primary_dir.mkdir()
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 2, "rows": 4, "model": "m"} for c in et.EMBEDDING_COLUMNS}
    meta = et.write_table(primary, primary_dir, script.ve.TABLE_NAME, kind="confirmatory", embedding_sets=sets)
    for entry in sets.values():
        np.save(primary_dir / entry["file"], np.arange(8, dtype=np.float16).reshape(4, 2))
    args = SimpleNamespace(primary=primary_dir)
    _, _, secondary = secondary_table(world, first=4)
    combined = esec.combine_tables(primary, secondary)
    arrays = {c: np.concatenate([np.load(primary_dir / e["file"]), np.zeros((5, 2), np.float16)]) for c, e in sets.items()}
    identity = script.primary_identity(combined, primary, meta | {"embedding_sets": sets}, args, arrays)
    assert identity["identical"] and identity["combined_primary_rows_sha256"] == meta["table_sha256"] == script.sha256_file(primary_dir / f"{script.ve.TABLE_NAME}.parquet")
    with pytest.raises(SystemExit, match="not byte-identical"):
        script.primary_identity(combined.assign(frame_logit_0=0.0), primary, meta | {"embedding_sets": sets}, args, arrays)
    changed = {**arrays, "emb_dinov2": arrays["emb_dinov2"] + np.float16(1)}
    with pytest.raises(SystemExit, match="not byte-identical"):
        script.primary_identity(combined, primary, meta | {"embedding_sets": sets}, args, changed)



def test_the_population_table_must_match_the_pairing_record(tmp_path):
    pairing = tmp_path / "pairing"
    pairing.mkdir()
    pd.DataFrame({"event_id": ["a"]}).to_parquet(pairing / "sb_unpaired.parquet")
    (pairing / "complete.json").write_text(json.dumps({"files": {"sb_unpaired.parquet": script.file_sha256(pairing / "sb_unpaired.parquet")}}))
    table, sha = script.read_unpaired(pairing)
    assert table.event_id.tolist() == ["a"] and len(sha) == 64
    (pairing / "complete.json").write_text(json.dumps({"files": {"sb_unpaired.parquet": "0" * 64}}))
    with pytest.raises(SystemExit, match="differs from the pairing's completion record"):
        script.read_unpaired(pairing)


def test_the_combined_embeddings_extend_the_primary_matrices_and_refuse_a_mismatch(tmp_path):
    primary_dir, out = tmp_path / "primary", tmp_path / "secondary"
    primary_dir.mkdir()
    out.mkdir()
    meta = {"embedding_sets": {c: {"file": f"{c}.npy", "rows": 3, "dim": 2} for c in et.EMBEDDING_COLUMNS}}
    for c in et.EMBEDDING_COLUMNS:
        np.save(primary_dir / f"{c}.npy", np.arange(6, dtype=np.float16).reshape(3, 2))
        np.save(out / f"{c}.npy", np.full((2, 2), 9, np.float16))
    args = SimpleNamespace(primary=primary_dir, out=out)
    arrays = script.combined_embeddings(args, meta, 2)
    assert arrays["emb_dinov2"].tolist() == [[0, 1], [2, 3], [4, 5], [9, 9], [9, 9]]
    with pytest.raises(SystemExit, match="does not extend"):
        script.combined_embeddings(args, meta, 3)                       # the table expects three secondary rows, the file holds two
    np.save(out / "emb_g_fold1.npy", np.full((2, 2), 9, np.float32))
    with pytest.raises(SystemExit, match="does not extend"):
        script.combined_embeddings(args, meta, 2)                       # another dtype
