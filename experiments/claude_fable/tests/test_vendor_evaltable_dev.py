"""Development evaluation table from prototype-shaped inputs (xg-vendor ticket 07)."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import evaltable_dev as dev

NOW = datetime(2026, 10, 12, tzinfo=UTC)
FOLDS = [0, 1, 2, 0, -1, -1]   # fold of the G models behind each fixture shot; -1 = test shot (all folds, no embedding)
SPLITS = ["train", "valid", "train", "valid", "test", "test"]
RUNG_COLUMNS = ["r_constant", "r_dist_angle", "r_event", "r_freeze_frame", "r_vendor", "r_frame"]


def fixture(n=6):
    """Prototype-shaped inputs for `n` shots (a multiple of 6): the first six are the small hand-checked case."""
    folds, splits = np.resize(FOLDS, n), np.resize(SPLITS, n)
    ids = [f"sn{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    ladder = pd.DataFrame({"shot_id": ids, "game": [f"g{i // 2}" for i in range(n)], "split": splits, "half": np.resize([1, 2], n),
                           "goal": np.resize([0, 1, 0, 0, 1, 0], n) if n == 6 else (rng.uniform(size=n) < 0.3).astype(int),
                           "res_ms": np.resize([5.0, -12.0, 30.0, 0.0, 7.0, -3.0], n),
                           "label": np.resize(["Shots on target", "Shots off target"], n), "sb_outcome": np.resize(["Saved", "Off T"], n),
                           "sb_body_part": "Head", "sb_event_id": [f"ev{i}" for i in range(n)],
                           "src": np.where(splits == "test", "test_stored", "oof_crossfit_platt")})
    for column in RUNG_COLUMNS:
        ladder[column] = rng.uniform(0.02, 0.6, n)
    pairs = pd.DataFrame({"shot_id": ids[::-1], "sb_match_id": [900.0 + i // 2 for i in range(n)][::-1],
                          "sb_shot_type": "Open Play", "pen_v12": False, "sb_pen": False})   # different row order on purpose
    rows = []
    for i, shot in enumerate(ids):
        for offset in dev.OFFSETS:
            rows.append(dict(shot_id=shot, offset_ms=offset, encoder="g_combo", frame_status="ok", xg_logit_raw=-1.0 - i - offset / 1000,
                             fold_source=folds[i], embedding_ref=i if folds[i] >= 0 and offset == 0 else -1))
        rows.append(dict(shot_id=shot, offset_ms=0, encoder="dinov2_frozen", frame_status="ok", xg_logit_raw=np.nan, fold_source=-1,
                         embedding_ref=i))
    archive = pd.DataFrame(rows)
    return ladder, pairs, archive


def sets():
    return {c: {"file": "x.npy", "sha256": "0" * 64, "dim": 4, "rows": 10, "model": "fixture"} for c in et.EMBEDDING_COLUMNS}


def build(n=6):
    ladder, pairs, archive = fixture(n)
    return ladder, *dev.build_development_tables(ladder, pairs, archive, sets())


def test_development_table_validates_and_is_flagged_development(tmp_path):
    _, table, labels = build()
    meta = et.write_table(table, tmp_path, "t", kind="development", embedding_sets=sets(), unavailable_columns=("frame_logit_m400",))
    assert meta["table_kind"] == "development" and table.table_kind.eq("development").all() and meta["by_population"] == {"primary": 6}
    joined = et.join_labels(tmp_path, "t", labels, "j", tmp_path / "log", now=NOW)
    assert joined["labels_state"] == "joined"


def test_columns_follow_the_prototype_rows():
    ladder, table, labels = build()
    assert table.key.tolist() == ladder.shot_id.tolist() == labels.key.tolist()
    assert table.statsbomb_match_id.tolist() == [900, 900, 901, 901, 902, 902]   # joined by shot id, not by row position
    np.testing.assert_allclose(table.pred_vendor_xg, ladder.r_vendor)
    np.testing.assert_allclose(dev.sigmoid(table.frame_logit_0), ladder.r_frame)
    assert table.frame_logit_m400.isna().all()
    np.testing.assert_allclose(table.frame_raw_logit_0, [-1.0 - i for i in range(6)])
    np.testing.assert_allclose(table.frame_raw_logit_m400, [-1.0 - i + 0.4 for i in range(6)])
    assert table.frame_source.tolist() == ladder.src.tolist() and table.timing_residual_ms.tolist() == ladder.res_ms.tolist()
    assert labels.goal_sn.tolist() == [False, True, False, False, True, False]
    assert labels.on_target_sn.tolist() == [True, False] * 3 and not labels.is_penalty.any()


def test_embedding_references_point_at_the_models_that_held_the_shot_out():
    _, table, _ = build()
    assert table.emb_g_fold0.tolist() == [0, -1, -1, 3, -1, -1]   # shots 0 and 3 are fold 0
    assert table.emb_g_fold1.tolist() == [-1, 1, -1, -1, -1, -1]   # shot 1 is fold 1; fold-2 and test shots have no reference
    assert table.emb_dinov2.tolist() == list(range(6))


def test_a_frame_without_a_clean_status_is_listed_not_hidden():
    ladder, pairs, archive = fixture()
    archive.loc[archive.shot_id.eq("sn2") & archive.offset_ms.eq(0) & archive.encoder.eq("g_combo"), "frame_status"] = "anchor_off_tolerance"
    archive = archive[~(archive.shot_id.eq("sn3") & archive.offset_ms.eq(-400) & archive.encoder.eq("g_combo"))]
    table, _ = dev.build_development_tables(ladder, pairs, archive, sets())
    assert table.frame_status.tolist() == ["ok", "ok", "0:anchor_off_tolerance", "-400:no_archive_row", "ok", "ok"]
    assert np.isnan(table.frame_raw_logit_m400[3])


def test_calibration_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    z = rng.normal(-1.5, 1.2, 40000)
    y = (rng.uniform(size=z.size) < dev.sigmoid(0.6 * z)).astype(float)
    assert dev.calibration_slope(y, dev.sigmoid(z)) == pytest.approx(0.6, abs=0.05)
    assert dev.calibration_slope(y, dev.sigmoid(0.6 * z)) == pytest.approx(1.0, abs=0.05)


def test_point_metrics_on_a_known_case():
    y, p = np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.8, 0.2, 0.4, 0.4])
    m = dev.point_metrics(y, p)
    assert m["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.8) + np.log(0.6) + np.log(0.4)) / 4)
    assert m["brier"] == pytest.approx((0.04 + 0.04 + 0.16 + 0.36) / 4)
    assert m["auc"] == pytest.approx(0.875)


def test_prototype_comparison_flags_a_changed_number(tmp_path):
    """Feeding the table's own metrics back reproduces them; one altered prototype cell shows up in abs_diff."""
    _, table, labels = build(120)
    joined = table.merge(labels, on="key")
    rungs = dev.ladder_rungs(joined)
    group = np.where(joined.split.eq("test"), "test", "trainvalid")
    y = joined.goal_sn.to_numpy(float)
    rows = []
    for population in ("test", "trainvalid"):
        mask = group == population
        for name, p in rungs.items():
            rows.append({"population": population, "rung": "r_" + name, "n": mask.sum(), "goals": y[mask].sum(),
                         "matches": joined.match[mask].nunique(), **dev.point_metrics(y[mask], p[mask])})
    prototype = pd.DataFrame(rows)
    same = dev.prototype_comparison(joined, prototype)
    assert same.abs_diff.max() < 1e-12 and {"test", "trainvalid"} == set(same.population)
    prototype.loc[(prototype.population == "test") & (prototype.rung == "r_vendor"), "log_loss"] += 0.01
    changed = dev.prototype_comparison(joined, prototype)
    worst = changed.loc[changed.abs_diff.idxmax()]
    assert (worst.population, worst.rung, worst.metric) == ("test", "vendor", "log_loss") and worst.abs_diff == pytest.approx(0.01)
    with pytest.raises(ValueError, match="no single row"):
        dev.prototype_comparison(joined, prototype[prototype.rung != "r_event"])
