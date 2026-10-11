"""Scoring of the evaluation shots and the label-free evaluation table (xg-vendor ticket 11); CPU only, a tiny model."""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from xgpoc.data.datasets import to_tensor
from xgpoc.eval.calibration import logit
from xgpoc.models.g_bundle import ensemble_probability
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import evaltable as et

SIZE = (32, 18)


def write_frame(path, value):
    cv2.imwrite(str(path), np.full((36, 64, 3), value, np.uint8))
    return str(path)


@pytest.fixture
def world(tmp_path):
    """An index of 6 shots (2 gate), one missing -400 frame, one missing frame file, one garbled file, and the pair table."""
    ids = [f"s{i}" for i in range(6)]
    zero = [write_frame(tmp_path / f"{i}_0.jpg", 20 + 30 * k) for k, i in enumerate(ids)]
    m400 = [write_frame(tmp_path / f"{i}_m.jpg", 30 + 30 * k) for k, i in enumerate(ids)]
    m400[1] = ""                                          # no path at -400
    m400[2] = str(tmp_path / "gone.jpg")                  # path, no file
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    m400[3] = str(tmp_path / "bad.jpg")                   # unreadable
    index = pd.DataFrame({"shot_id": ids, "game": ["a/1", "a/1", "a/1", "b/2", "b/2", "c/3"], "half": [1, 1, 2, 1, 2, 1],
                          "split": ["train", "train", "train", "test", "test", "valid"], "is_penalty": [False, True, False, False, False, False],
                          "goal": [True, False, False, False, False, True], "frame_path": zero, "frame_path_-400": m400})
    pairs = pd.DataFrame({"shot_id": ids[:4] + ["s9"], "paired": [True, True, True, True, False],
                          "sb_event_id": [f"e{i}" for i in range(5)], "sb_match_id": [10, 10, 10, 20, 20],
                          "sb_body_part": ["Head", "Right Foot", "Left Foot", "Right Foot", "Head"],
                          "sb_shot_type": ["Open Play", "Penalty", "Open Play", "Free Kick", "Open Play"],
                          "sb_xg": [0.1, 0.76, 0.05, 0.3, 0.2], "residual_ms": [5.0, -3.0, 10.0, 0.0, np.nan],
                          "goal": [True, False, False, True, False], "sb_outcome": ["Goal", "Saved", "Off T", "Goal", "Blocked"]})
    gate = index[index.shot_id.isin(["s4", "s5"])].reset_index(drop=True)
    return index, pairs, gate


def test_inspect_frames_reports_every_failure_without_replacing_it(world, tmp_path):
    index, _, _ = world
    found = es.inspect_frames(index["frame_path_-400"].tolist())
    assert found.status.tolist() == ["ok", es.NO_PATH, es.MISSING, es.UNREADABLE, "ok", "ok"]
    assert found.sha256.tolist()[1:3] == ["", ""] and len(found.sha256[0]) == 64 and len(found.sha256[3]) == 64


def test_shots_frame_orders_paired_then_gate_and_keeps_rows_with_missing_frames(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    assert shots.key.tolist() == ["s0", "s1", "s2", "s3", "s4", "s5"]    # pair-table order, then gate order
    assert shots.role.tolist() == ["primary"] * 4 + ["gate"] * 2
    assert shots.status_0.eq("ok").all()
    assert shots.status_m400.tolist() == ["ok", es.NO_PATH, es.MISSING, es.UNREADABLE, "ok", "ok"]
    assert es.table_frame_status(shots.status_0, shots.status_m400) == ["ok", "-400:no_frame_path", "-400:file_missing", "-400:undecodable", "ok", "ok"]
    assert "goal" not in shots.columns                                    # nothing label-like is carried along


def test_a_shot_that_is_both_evaluation_and_gate_is_refused(world):
    index, pairs, gate = world
    with pytest.raises(ValueError, match="both evaluation and gate"):
        es.shots_frame(index, pairs, index[index.shot_id.eq("s0")])
    with pytest.raises(ValueError, match="not in the index"):
        es.shots_frame(index[index.shot_id != "s1"], pairs, gate)


def test_frames_unchanged_notices_an_edited_frame(world, tmp_path):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    assert es.frames_unchanged(shots)
    write_frame(tmp_path / "s0_0.jpg", 200)
    assert not es.frames_unchanged(shots)


def test_fingerprint_depends_on_the_content_of_present_frames(world, tmp_path):
    index, pairs, gate = world
    first = es.shots_fingerprint(es.shots_frame(index, pairs, gate))
    assert first["count"] == 6 + 4                                        # six offset-0 frames and four -400 files that exist (one is garbled)
    write_frame(tmp_path / "s0_0.jpg", 200)
    assert es.shots_fingerprint(es.shots_frame(index, pairs, gate))["sha256"] != first["sha256"]


class Tiny(nn.Module):
    """Same surface as the G model for `g_outputs`: `model(image=...)` -> [B, 2] and a `head.head` whose input is the embedding."""

    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.head = nn.Linear(3 * SIZE[0] * SIZE[1], 2)

    def forward(self, image):
        return self.head.head(image.flatten(1))


def tiny_spec():
    return {"model": {"n_out": 2}, "input": {"size_wh": list(SIZE), "mask": "none", "frame_col": "frame_path"}}


def test_score_rows_scores_present_frames_and_leaves_the_others_nan(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    torch.manual_seed(0)
    model = Tiny()
    for name, bad in (("0", []), ("m400", [1, 2, 3])):
        xg, on_target, embedding = es.score_rows(model, tiny_spec(), shots, name, batch_size=4, device="cpu", num_workers=0, embedding=True)
        assert np.isnan(xg[bad]).all() and np.isnan(on_target[bad]).all()
        good = np.setdiff1d(np.arange(len(shots)), bad)
        assert np.isfinite(xg[good]).all()
        assert embedding.shape == (len(good), 3 * SIZE[0] * SIZE[1]) and embedding.dtype == np.float16
    # offset 0 of the first shot equals the model applied to that frame by hand
    xg0, _, _ = es.score_rows(model, tiny_spec(), shots, "0", batch_size=4, device="cpu", num_workers=0)
    image = cv2.cvtColor(cv2.resize(cv2.imread(shots.frame_path[0]), SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        assert xg0[0] == pytest.approx(float(model(image=to_tensor(image)[None])[0, 0]), abs=1e-6)


def test_score_rows_never_needs_a_label(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    assert "goal" not in shots.columns
    es.score_rows(Tiny(), tiny_spec(), shots, "0", batch_size=4, device="cpu", num_workers=0)   # the constant goal column is added inside


def members_for(n_rows, nan_rows=(), rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    for k in es.MEMBERS:
        xg = rng.normal(-1.5, 1.0, (2, n_rows))
        xg[1, list(nan_rows)] = np.nan
        out[k] = {"xg": xg, "on_target": rng.normal(0, 1, (2, n_rows))}
    return out


CALIBRATION = {"platt": {"a": 0.98, "b": -0.05}, "oof_rate": 0.13, "pooled_rate": 0.14, "penalty_rate": 0.7, "n_oof": 100}
TEMPERATURES = {k: 1.0 + 0.05 * i for i, k in enumerate(es.MEMBERS)}


def test_ensemble_scores_is_the_archived_pipeline_and_keeps_missing_shots_nan():
    rng = np.random.default_rng(1)
    xg = {k: rng.normal(-1.5, 1, 7) for k in es.MEMBERS}
    pen = np.array([False, True, False, False, False, False, False])
    out = es.ensemble_scores(xg, TEMPERATURES, CALIBRATION, pen)
    expected = ensemble_probability([xg[k] for k in sorted(xg)], [TEMPERATURES[k] for k in sorted(xg)], CALIBRATION, pen)
    np.testing.assert_allclose(out["p"], expected, atol=1e-12)
    assert out["p"][1] == pytest.approx(0.7) and out["raw_mean"] == pytest.approx(np.mean([xg[k] for k in sorted(xg)], axis=0))
    holey = {k: v.copy() for k, v in xg.items()}
    for v in holey.values():
        v[3] = np.nan
    again = es.ensemble_scores(holey, TEMPERATURES, CALIBRATION, pen)
    assert np.isnan(again["p"][3]) and np.isnan(again["raw_mean"][3])
    np.testing.assert_allclose(np.delete(again["p"], 3), np.delete(expected, 3), atol=1e-12)
    only_some = {k: v.copy() for k, v in xg.items()}
    only_some[(0, 0)][3] = np.nan
    with pytest.raises(ValueError, match="some fold models"):
        es.ensemble_scores(only_some, TEMPERATURES, CALIBRATION, pen)


def test_assemble_scores_needs_all_ten_models(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    members = members_for(len(shots), nan_rows=(1, 2, 3))
    scores = es.assemble_scores(shots, members, TEMPERATURES, CALIBRATION)
    assert scores.key.tolist() == shots.key.tolist() and scores.p_0.notna().all()
    assert scores.p_m400.isna().tolist() == [False, True, True, True, False, False]
    assert {"xg_f0s0_0", "xg_f4s1_m400", "ontarget_mean_0", "raw_mean_m400", "penalty_routed"} <= set(scores.columns)
    assert scores.p_0[1] == pytest.approx(0.7)                              # the index penalty is routed
    del members[(2, 1)]
    with pytest.raises(ValueError, match="ten fold models"):
        es.assemble_scores(shots, members, TEMPERATURES, CALIBRATION)


def comparator_for(pairs):
    paired = pairs[pairs.paired]
    return pd.DataFrame({"statsbomb_event_id": [*paired.sb_event_id, "x1"], "population": ["primary"] * len(paired) + ["statsbomb_only"],
                         "soccernet_shot_id": [*paired.shot_id, None], "p_constant": 0.13, "p_dist_angle": 0.1, "p_event": 0.12,
                         "p_freeze_frame": 0.11})


def test_the_table_is_label_free_valid_and_keeps_rows_with_missing_frames(world, tmp_path):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    scores = es.assemble_scores(shots, members_for(len(shots), nan_rows=(1, 2, 3)), TEMPERATURES, CALIBRATION)
    table = es.build_eval_table(shots, scores, pairs, comparator_for(pairs))
    assert list(table.columns) == et.LABEL_FREE_COLUMNS and not set(et.LABEL_COLUMNS) & set(table.columns)
    assert table.key.tolist() == ["s0", "s1", "s2", "s3"] and table.population.eq("primary").all()   # no gate row in the table
    assert table.frame_status.tolist() == ["ok", "-400:no_frame_path", "-400:file_missing", "-400:undecodable"]
    assert table.frame_logit_0.notna().all() and table.frame_logit_m400.isna().tolist() == [False, True, True, True]
    assert table.pred_vendor_xg.tolist() == [0.1, 0.76, 0.05, 0.3] and table.timing_residual_ms.tolist() == [5.0, -3.0, 10.0, 0.0]
    assert table.frame_logit_0.iloc[0] == pytest.approx(float(logit(scores.p_0.iloc[0])))
    assert table.emb_dinov2.tolist() == [0, 1, 2, 3] and table.table_kind.eq("confirmatory").all()
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": 4, "model": "m"} for c in et.EMBEDDING_COLUMNS}
    meta = et.write_table(table, tmp_path, "t", kind="confirmatory", embedding_sets=sets)   # validated before it is written
    assert meta["labels_state"] == "absent" and meta["rows"] == 4
    loaded, _ = et.load_table(tmp_path, "t")
    assert loaded.key.tolist() == table.key.tolist()


def test_a_missing_offset_zero_frame_has_no_embedding_reference(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    shots.loc[0, "status_0"] = es.MISSING
    members = members_for(len(shots), nan_rows=(0,))
    for m in members.values():
        m["xg"][0, 0] = np.nan
    scores = es.assemble_scores(shots, members, TEMPERATURES, CALIBRATION)
    table = es.build_eval_table(shots, scores, pairs, comparator_for(pairs))
    assert table.emb_g_fold0.tolist() == [-1, 0, 1, 2] and table.emb_dinov2.tolist() == [-1, 0, 1, 2]
    assert table.frame_status[0] == "0:file_missing" and np.isnan(table.frame_logit_0[0])


def test_the_comparator_must_cover_and_agree_with_the_pairing(world):
    index, pairs, gate = world
    shots = es.shots_frame(index, pairs, gate)
    scores = es.assemble_scores(shots, members_for(len(shots)), TEMPERATURES, CALIBRATION)
    comparator = comparator_for(pairs)
    with pytest.raises(ValueError, match="no comparator prediction"):
        es.build_eval_table(shots, scores, pairs, comparator[comparator.statsbomb_event_id != "e2"])
    wrong = comparator.assign(soccernet_shot_id=lambda d: d.soccernet_shot_id.replace({"s1": "s0", "s0": "s1"}))
    with pytest.raises(ValueError, match="another SoccerNet shot"):
        es.build_eval_table(shots, scores, pairs, wrong)


def test_embedding_references_count_only_scored_shots():
    assert es.embedding_references(np.array([True, False, True, True, False])).tolist() == [0, -1, 1, 2, -1]
