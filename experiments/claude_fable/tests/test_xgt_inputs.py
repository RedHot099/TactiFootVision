"""xg-temporal inputs (ticket 01): the window and its reference frames, the imputation, the population's usable-frame
rule, the xgv fold check, the training-role filter, the 01b extraction list and the source folder contract."""

import json

import numpy as np
import pandas as pd
import pytest
from xgpoc.bundle import sha256_file
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgt_fixtures import base_table, source_frames, synthetic_population


@pytest.mark.parametrize(("start", "o_max", "control", "offsets", "middle"), [
    (-1600, -600, False, (-1600, -1400, -1200, -1000, -800, -600), -1200),     # tie between -1200 and -1000: earlier
    (-800, -600, False, (-800, -600), -800),                                    # two frames: the middle is the start
    (-1200, -400, True, (-1200, -1000, -800, -600, -400), -800),               # exact centre
])
def test_window_offsets_and_middle_frame(start, o_max, control, offsets, middle):
    window = ti.Window(start, o_max, control)
    assert window.offsets == offsets and window.middle == middle


def test_window_refuses_off_grid_and_reversed_offsets_and_a_late_o_max_without_the_control_flag():
    for start, o_max in ((-1500, -600), (-600, -800), (-600, -600), (-1600, 600)):
        with pytest.raises(ValueError):
            ti.Window(start, o_max)
    with pytest.raises(tp.ProvenanceError):
        ti.Window(-1600, -400)
    assert ti.Window(-1600, 400, control=True).offsets[-1] == 400
    with pytest.raises(ValueError, match="no leakage-control row"):
        ti.Window(-1600, -600, control=True)
    assert ti.Window() == ti.Window(-1600, -600) and len(ti.GRID) == 11


def test_imputation_takes_the_nearest_earlier_frame_else_the_nearest_later_one():
    observed = np.array([[1, 1, 1, 1],
                         [0, 0, 1, 1],          # leading run (a cut): the first later frame
                         [1, 0, 0, 1],          # interior gap: the nearest earlier frame
                         [0, 1, 0, 1]], bool)
    take, missing = ti.impute(observed)
    assert take.tolist() == [[0, 1, 2, 3], [2, 2, 2, 3], [0, 0, 0, 3], [1, 1, 1, 3]]
    assert (missing == ~observed).all()
    with pytest.raises(ValueError):
        ti.impute(np.array([[1, 1], [0, 0]], bool))


def population_rows():
    base = base_table([("a1", "A", 0, -1.0), ("a2", "A", 0, -2.0), ("b1", "B", 1, -1.5), ("b2", "B", 1, -0.5)])
    rows = []
    for offset in ti.GRID:
        rows += [("a1", "A", offset, "ok", False, 10 + offset // 200, 0),
                 ("a2", "A", offset, "ok", offset < -1000, 40 + offset // 200, 0),     # cut before -1000
                 ("b1", "B", offset, "missing" if offset <= -600 else "ok", pd.NA if offset <= -600 else False, -1, 1),
                 ("b2", "B", offset, "ok", False, -1 if offset == -1400 else 70 + offset // 200, 1),
                 ("h1", "H", offset, "ok", False, 99, 2)]                                  # a held-out shot
    return base, source_frames(rows)


def test_population_keeps_shots_with_a_usable_window_frame_and_records_their_embedding_rows():
    base, frames = population_rows()
    population = ti.population_from(base, frames, ti.Window(), ti.SOURCES["g_xgc"])
    assert list(population.table.index) == ["a1", "a2", "b2"]                 # b1: no decoded frame up to -600
    assert "H" not in set(population.games) and set(population.frames.shot_id) == {"a1", "a2", "b2"}
    assert population.refs[0].tolist() == [10 + o // 200 for o in ti.Window().offsets]
    assert population.refs[1].tolist() == [-1, -1, -1, 40 - 5, 40 - 4, 40 - 3]
    assert population.refs[2].tolist() == [70 - 8, -1, 70 - 6, 70 - 5, 70 - 4, 70 - 3]
    assert population.table.frames.tolist() == [6, 3, 5] and population.table.complete.tolist() == [True, False, False]
    assert population.counts["shots"] == 3 and population.counts["matches"] == 2 and population.counts["complete_shots"] == 1
    used = population.frames[population.frames.used]
    assert len(used) == 14 and not used.cut_between_anchor.any() and (used.offset_ms <= -600).all()
    tp.check_frames(population.frames, -600)


def test_a_g_xgv_embedding_must_come_from_the_fold_model_that_held_out_its_match():
    base, frames = population_rows()
    frames = frames[frames.shot_id != "b1"].copy()
    assert ti.population_from(base, frames, ti.Window(), ti.SOURCES["g_xgv"]).counts["shots"] == 3
    frames.loc[(frames.shot_id == "a1") & (frames.offset_ms == -1600), "fold_source"] = 3
    with pytest.raises(ValueError, match="fold model"):
        ti.population_from(base, frames, ti.Window(), ti.SOURCES["g_xgv"])
    ti.population_from(base, frames, ti.Window(), ti.SOURCES["g_xgc"])      # v1.2 folds: not checked


def write_base_inputs(tmp_path, oof_games: dict[str, str]):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": list(oof_games), "game": list(oof_games.values()), "is_penalty": False,
                  "goal": True}).to_parquet(index)
    folder = tmp_path / "oof"
    folder.mkdir()
    oof = folder / "oof.csv"
    pd.DataFrame({"shot_id": list(oof_games), "fold": 0, "z_g_combo": 0.0, "p_a1": 0.5}).to_csv(oof, index=False)
    (folder / "complete.json").write_text(json.dumps({"files": {"oof.csv": sha256_file(oof)}}))
    return index, oof


def test_base_shots_read_no_label_and_refuse_a_shot_of_a_held_out_match(tmp_path, monkeypatch):
    plan = pd.DataFrame({"game": ["A", "B", "H"], "role": ["training", "training", "evaluation"], "fold": [0, 1, -1]})
    monkeypatch.setattr(ti, "EXPECTED_BASE", {"shots": 2, "matches": 2, "folds": 2})
    requested = []
    read_parquet, read_csv = pd.read_parquet, pd.read_csv

    def spy_parquet(path, columns=None, **kwargs):
        requested.append(columns)
        return read_parquet(path, columns=columns, **kwargs)

    def spy_csv(path, usecols=None, **kwargs):
        requested.append(usecols)
        return read_csv(path, usecols=usecols, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy_parquet)
    monkeypatch.setattr(pd, "read_csv", spy_csv)
    (tmp_path / "dev").mkdir()
    base = ti.base_shots(plan, *write_base_inputs(tmp_path / "dev", {"a1": "A", "b1": "B"}))
    assert base.fold.tolist() == [0, 1] and list(base.columns) == ["game", "fold", "z_g_combo"]
    assert requested == [["shot_id", "z_g_combo"], ["shot_id", "game", "is_penalty"]]
    (tmp_path / "held").mkdir()
    with pytest.raises(ValueError, match="outside the plan's training"):
        ti.base_shots(plan, *write_base_inputs(tmp_path / "held", {"a1": "A", "b1": "B", "h1": "H"}))


def test_extraction_list_counts_decoded_uncut_frames_on_the_11_offsets():
    base, frames = population_rows()
    shots, counts = ti.extraction_shots(base, frames)
    assert shots.shot_id.tolist() == ["a1", "a2", "b1", "b2"]                  # b1 has decoded frames after -600
    assert counts["shots"] == 4 and counts["matches"] == 2 and counts["frames_grid"] == 44
    assert counts["frames_decoded"] == 44 - 6 and counts["frames_decoded_uncut"] == 44 - 6 - 3
    assert counts["base_shots_without_frame"] == 0


def write_source(folder, *, beside: bool, columns: dict | None = None, drop: tuple = (), duplicate: bool = False):
    folder.mkdir()
    rows = [("a1", "A", -600, "ok", False, 1, 0), ("a1", "A", -800, "ok", False, 0, 0)]
    table = source_frames(rows + rows[:1] if duplicate else rows).drop(columns=list(drop))
    for name, value in (columns or {}).items():
        table[name] = value
    table.to_parquet(folder / ti.TABLE)
    array = np.arange(6, dtype=np.float16).reshape(3, 2)
    record = {"files": {ti.TABLE: sha256_file(folder / ti.TABLE)}}
    if beside:
        np.save(folder / "embeddings_g_combo.npy", array)
        record["files"]["embeddings_g_combo.npy"] = sha256_file(folder / "embeddings_g_combo.npy")
    else:
        path = folder.parent / "outputs" / "g.npy"
        path.parent.mkdir()
        np.save(path, array)
        record["embeddings"] = {"g_combo": {"path": str(path), "sha256": sha256_file(path)}}
    (folder / ti.RECORD).write_text(json.dumps(record))
    return ti.Source("test", folder, "g_combo", True)


@pytest.mark.parametrize("beside", [True, False])
def test_source_contract_verifies_both_files_and_reads_only_the_wanted_rows(tmp_path, beside):
    source = write_source(tmp_path / "src", beside=beside)
    files = ti.verify_source(source)
    frames = ti.read_frames(files, ["a1"], [-800, -600, -400])
    assert sorted(frames.offset_ms) == [-800, -600] and set(frames.columns) == set(ti.FRAME_COLUMNS)
    rows, position = ti.load_embeddings(files, np.array([[2, -1], [0, 2]]))
    assert rows.tolist() == [[0.0, 1.0], [4.0, 5.0]] and position.tolist() == [[1, -1], [0, 1]]
    with pytest.raises(ValueError, match="beyond"):
        ti.load_embeddings(files, np.array([[3]]))
    np.save(files.embeddings, np.zeros((3, 2), np.float16))
    with pytest.raises(ValueError, match="completion record"):
        ti.verify_source(source)


def test_source_contract_refuses_a_banned_column(tmp_path):
    with pytest.raises(tp.ProvenanceError, match="goal"):
        ti.verify_source(write_source(tmp_path / "src", beside=True, columns={"goal": False}))


def test_source_contract_refuses_a_missing_column_and_a_repeated_frame_row(tmp_path):
    with pytest.raises(ValueError, match="lacks the columns"):
        ti.verify_source(write_source(tmp_path / "a", beside=True, drop=("fold_source",)))
    files = ti.verify_source(write_source(tmp_path / "b", beside=True, duplicate=True))
    with pytest.raises(ValueError, match="more than one"):
        ti.read_frames(files, ["a1"], [-600])


def test_a_frame_row_of_another_game_than_its_shot_is_refused():
    base, frames = population_rows()
    frames.loc[frames.shot_id == "a1", "game"] = "B"
    with pytest.raises(ValueError, match="game differs"):
        ti.population_from(base, frames, ti.Window(), ti.SOURCES["g_xgc"])


def test_build_population_from_a_source_folder_records_every_input(tmp_path, monkeypatch):
    plan = pd.DataFrame({"game": ["A", "B"], "role": ["training", "training"], "fold": [0, 1]})
    monkeypatch.setattr(ti, "EXPECTED_BASE", {"shots": 2, "matches": 2, "folds": 2})
    (tmp_path / "dev").mkdir()
    index, oof = write_base_inputs(tmp_path / "dev", {"a1": "A", "b1": "B"})
    source = write_source(tmp_path / "src", beside=False)
    population, files = ti.build_population(source, ti.Window(-800, -600), plan, index, oof)
    assert list(population.table.index) == ["a1"] and population.refs.tolist() == [[0, 1]]
    assert population.inputs["oof"]["path"] == str(oof.resolve()) and population.inputs["source"] == files.record()
    assert population.inputs["oof"]["columns"] == ["shot_id", "z_g_combo"]


def test_a_grid_cell_is_restricted_to_the_main_populations_shots_and_hashed_by_its_shot_list():
    cell, _, _ = synthetic_population(ti.Window(-1600, 400, True))
    keep = cell.table.index[::2]
    restricted = ti.restrict(cell, list(keep) + ["not_a_shot"])
    assert list(restricted.table.index) == list(keep) and len(restricted.refs) == len(keep)
    assert set(restricted.frames.shot_id) == set(keep) and restricted.counts["window_shots"] == len(cell.table)
    assert restricted.counts["shots"] == len(keep) and restricted.counts["restricted_to_main"]
    record = restricted.record()
    assert record["shot_list_sha256"] == ti.shot_list_sha256(restricted.table.iloc[::-1])     # order-free
    assert record["shot_list_sha256"] != cell.record()["shot_list_sha256"]
