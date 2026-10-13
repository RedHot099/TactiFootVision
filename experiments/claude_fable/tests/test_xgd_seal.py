"""The held-out label seal (xg-detector 06): no label of a gate or evaluation shot on the rule, power, training and
baseline-preparation paths before the single test."""

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest
from xgpoc.detector import approach1 as a1
from xgpoc.detector.seal import (
    LABEL_COLUMNS,
    HeldOutLabelRead,
    development_labels_only,
    held_out_labels_sealed,
    training_pool_ids,
)
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def files(tmp_path):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": ["t1", "t2", "h1", "h2"], "game": ["T", "T", "H", "H"], "goal": [True, False, True, False],
                  "label": ["Shots on target"] * 4, "is_penalty": [False, False, False, True],
                  "frame_path": ["/f/a.jpg"] * 4}).to_parquet(index)
    plan = tmp_path / "plan.csv"
    pd.DataFrame({"game": ["T", "H"], "role": ["training", "gate"], "fold": [0, -1]}).to_csv(plan, index=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"shot_id": ["t1", "h1"], "goal": [1, 1], "logit": [0.1, 0.2], "p": [0.5, 0.6]}).to_csv(predictions, index=False)
    return index, plan, predictions


def test_the_seal_reads_only_training_rows_of_a_labelled_table(files):
    index, plan, _ = files
    with development_labels_only(plan, index):
        everything = pd.read_parquet(index)
        goals = pd.read_parquet(index, columns=["shot_id", "goal"])
    assert everything.shot_id.tolist() == ["t1", "t2"] and goals.shot_id.tolist() == ["t1", "t2"]
    assert len(pd.read_parquet(index)) == 4                                    # the seal is gone after the block


def test_the_seal_passes_label_free_reads_of_held_out_shots(files):
    index, plan, predictions = files
    with development_labels_only(plan, index):
        shots = pd.read_parquet(index, columns=["shot_id", "game", "is_penalty", "frame_path"])
        scores = pd.read_csv(predictions, usecols=["shot_id", "logit", "p"])
    assert shots.shot_id.tolist() == ["t1", "t2", "h1", "h2"] and scores.shot_id.tolist() == ["t1", "h1"]


def test_the_seal_refuses_held_out_labels_in_csv_and_unattributable_labels(files, tmp_path):
    index, plan, predictions = files
    bare = tmp_path / "bare.csv"
    pd.DataFrame({"goal": [1, 0]}).to_csv(bare, index=False)
    no_game = tmp_path / "no_game.parquet"
    pd.DataFrame({"shot_id": ["h2"], "on_target": [1.0]}).to_parquet(no_game)
    with development_labels_only(plan, index):
        with pytest.raises(HeldOutLabelRead, match="1 shots outside"):
            pd.read_csv(predictions)
        with pytest.raises(HeldOutLabelRead, match="without a game or shot_id"):
            pd.read_csv(bare)
        with pytest.raises(HeldOutLabelRead):
            pd.read_parquet(no_game)
        with pytest.raises(HeldOutLabelRead, match="row filters"):
            pd.read_parquet(index, filters=[("game", "==", "H")])
        with pytest.raises(HeldOutLabelRead, match="chunked"):
            pd.read_csv(predictions, chunksize=1)
        assert pd.read_csv(predictions, usecols=["shot_id", "goal"], nrows=1).shot_id.tolist() == ["t1"]


def test_the_seal_nests_and_restores_the_readers(files):
    index, plan, _ = files
    original = pd.read_parquet, pd.read_csv
    with held_out_labels_sealed(["T"], ["t1", "t2"]), held_out_labels_sealed(["T"], ["t1", "t2"]):
        assert len(pd.read_parquet(index)) == 2
    assert (pd.read_parquet, pd.read_csv) == original


def test_the_training_pool_ids_are_read_without_labels(files):
    index, plan, _ = files
    games, shots = training_pool_ids(plan, index)
    assert games == {"T"} and shots == {"t1", "t2"}
    assert {"goal", "label", "goal_v1_1", "on_target"} == LABEL_COLUMNS


@pytest.mark.skipif(not INDEX.exists() or not (a1.PLAN_DIR / "plan.csv").exists(), reason="needs the index and the plan")
def test_on_the_real_index_the_seal_keeps_the_development_pool_and_drops_every_held_out_label():
    from xgpoc.data.index import load_index

    with development_labels_only(a1.PLAN_DIR / "plan.csv", INDEX):
        loaded = load_index(INDEX)
        shots = pd.read_parquet(INDEX, columns=["shot_id", "game", "is_penalty"])
    plan = pd.read_csv(a1.PLAN_DIR / "plan.csv")
    assert set(loaded.game) == set(plan.game[plan.role == "training"]) and len(loaded) == 7166
    assert len(shots) == 11030


# ------------------------------------------------------------------ every sealed script runs its work inside the seal
SEALED_SCRIPTS = ["xgd_approach1.py"]
SEALS = ("development_labels_only", "held_out_labels_sealed", "sealed_test")
OUTSIDE_OK = {"ArgumentParser", "add_argument", "parse_args", "Path", "print", "check_gpu_free", "results_dir", "SystemExit",
              "dumps", "str", "isoformat", "now", "add_mutually_exclusive_group", "strftime"}


def _seal_blocks(main: ast.FunctionDef) -> list[ast.With]:
    return [n for n in ast.walk(main) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func).split(".")[-1] in SEALS for i in n.items)]


@pytest.mark.parametrize("script", SEALED_SCRIPTS)
def test_every_sealed_script_reads_inside_the_seal_and_never_through_pyarrow(script):
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not [m for m in imports if m.startswith("pyarrow")], "tables are read through pandas, which the seal covers"
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    blocks = _seal_blocks(main)
    assert blocks, f"{script}: main has no label seal"
    inside = {id(n) for b in blocks for n in ast.walk(b)}
    outside = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call) and id(n) not in inside
               and ast.unparse(n.func).split(".")[-1] not in OUTSIDE_OK]
    assert not outside, f"{script}: work outside the label seal: {outside}"
