"""No StatsBomb data or pairing on the detector's model and feature input paths (xg-detector spec, story 2)."""

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from xgpoc.detector import extract as ex
from xgpoc.detector.guard import FORBIDDEN, StatsBombRead, no_statsbomb_reads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xgd_extract  # noqa: E402

CODE = sorted((ROOT / "xgpoc/detector").glob("*.py")) + sorted((ROOT / "scripts").glob("xgd_*.py"))
FORBIDDEN_PATHS = ["results/xgv_pairing/pairs.parquet", "/home/kuba/projects/football_data/statsbomb_open_v1/events/1.json",
                   "results/xgv_statsbomb_data/shots.parquet", "results/xgv_release_v1/pairs.parquet",
                   "results/xgv_comparator/predictions.parquet", "cache/StatsBomb/x.csv"]


def _forbidden_file(tmp_path: Path, relative: str) -> Path:
    path = tmp_path / relative.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"shot_id": ["a"], "paired": [True]}).to_parquet(path)
    return path


@pytest.mark.parametrize("relative", FORBIDDEN_PATHS)
def test_every_reader_refuses_statsbomb_paths_inside_the_guard(tmp_path, relative):
    path = _forbidden_file(tmp_path, relative)
    with no_statsbomb_reads():
        for read in (open, lambda p: pd.read_parquet(p), lambda p: pd.read_csv(p), cv2.imread, np.load,
                     lambda p: p.read_bytes()):
            with pytest.raises(StatsBombRead):
                read(path)
    assert len(pd.read_parquet(path)) == 1                         # outside the block nothing is refused


@pytest.mark.skipif(not ex.DETECTOR_WEIGHTS.exists(), reason="needs the trained detector weights")
def test_the_guard_lets_frames_index_and_weights_through(tmp_path):
    index = tmp_path / "soccernet_720p/index_v1_2.parquet"
    index.parent.mkdir(parents=True)
    pd.DataFrame({"shot_id": ["s1"], "goal": [True], **{c: [f"/f/{c}.jpg"] for c in ex.FRAME_COLUMNS.values()}}).to_parquet(index)
    with no_statsbomb_reads():
        table = ex.frame_table(index, ["s1"])
        for weights in (ex.DETECTOR_WEIGHTS, ex.KEYPOINT_WEIGHTS):
            with open(weights, "rb") as f:
                assert f.read(2)
    assert list(table.columns) == ["shot_id", *ex.FRAME_COLUMNS.values()]   # no label column is read
    with no_statsbomb_reads(), no_statsbomb_reads():               # re-entrant
        import tactifoot_vision.data.statsbomb as reader  # code, not data: importable and hashable
        with open(reader.__file__, "rb") as f:
            assert f.read(1)
    assert not hasattr(pd.read_parquet, "__wrapped__") and pd.read_parquet.__module__.startswith("pandas")


def test_reading_the_pairing_inside_the_guard_fails(tmp_path):
    pairs = _forbidden_file(tmp_path, "results/xgv_pairing/pairs.parquet")
    assert xgd_extract.paired_shot_ids(pairs) == ["a"]
    with no_statsbomb_reads(), pytest.raises(StatsBombRead):
        xgd_extract.paired_shot_ids(pairs)


def _strings(tree: ast.AST) -> list[str]:
    """String constants that are not docstrings."""
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


@pytest.mark.parametrize("path", CODE, ids=lambda p: p.name)
def test_detector_code_imports_no_vendor_module_and_names_no_statsbomb_file(path):
    tree = ast.parse(path.read_text())
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if m and m.startswith("xgpoc.vendor")], "xg-vendor code is not an input"
    named = [s for s in _strings(tree) if FORBIDDEN.search(s)]
    if path.name in AUDIT_FILES + ("guard.py",):
        return                                     # the pattern itself, and the audit: the one allowed StatsBomb reader
    allowed = {"xgd_extract.py": ["results/xgv_pairing/pairs.parquet"], "xgd_extract_full.py": ["results/xgd_audit2/choice.json"],
               "xgd_gcombo_test.py": ["results/xgv_eval_primary"]}   # G-combo's own predictions, no new model reads them
    assert named == allowed.get(path.name, []), f"{path.name} names StatsBomb or pairing files: {named}"


AUDIT_FILES = ("audit.py", "xgd_audit.py", "xgd_kp_diag.py", "xgd_audit2_diag.py")


def _imports_audit(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module == "xgpoc.detector.audit"
                                              or (n.module == "xgpoc.detector" and any(a.name == "audit" for a in n.names))):
            return True
        if isinstance(n, ast.Import) and any(a.name.startswith("xgpoc.detector.audit") for a in n.names):
            return True
    return False


@pytest.mark.parametrize("path", [p for p in CODE if p.name not in AUDIT_FILES], ids=lambda p: p.name)
def test_no_feature_model_or_extraction_code_imports_the_audit(path):
    assert not _imports_audit(ast.parse(path.read_text())), f"{path.name} imports the StatsBomb audit"


@pytest.mark.parametrize("script", ["xgd_audit.py", "xgd_kp_diag.py", "xgd_audit2_diag.py"])
def test_the_audit_scripts_are_the_ones_that_import_it(script):
    assert _imports_audit(ast.parse((ROOT / "scripts" / script).read_text()))


@pytest.mark.parametrize("relative", ["results/xgd_audit/per_shot.parquet", "results/xgd_audit/metrics.csv",
                                      "results/xgd_audit2/per_shot.parquet", "results/xgd_audit2_diag/diag.csv",
                                      "results/xgd_kp_diag/per_shot.parquet", "results/xgv_timing_source/shot_residuals.parquet"])
def test_the_audit_outputs_and_timing_residuals_are_refused_inside_the_guard(tmp_path, relative):
    path = _forbidden_file(tmp_path, relative)
    with no_statsbomb_reads(), pytest.raises(StatsBombRead):
        pd.read_parquet(path)


def test_the_audit_readers_are_refused_inside_the_guard(tmp_path):
    from xgpoc.detector import audit

    pairs = tmp_path / "results/xgv_pairing/pairs.parquet"
    pairs.parent.mkdir(parents=True)
    pd.DataFrame({c: ["x"] for c in audit.PAIR_COLUMNS}).to_parquet(pairs)
    shots = tmp_path / "statsbomb_open_v1/tables/shots.parquet"
    shots.parent.mkdir(parents=True)
    pd.DataFrame({"event_id": ["e"], "freeze_frame_json": ["[]"]}).to_parquet(shots)
    with no_statsbomb_reads():
        with pytest.raises(StatsBombRead):
            audit.read_population(pairs)
        with pytest.raises(StatsBombRead):
            audit.read_freeze_frames(shots, ["e"])
    assert audit.read_freeze_frames(shots, ["e"]).tolist() == ["[]"]   # outside the block the audit reads them


def _guarded_main(script: str):
    """(main, its single `with no_statsbomb_reads()` block, ids of the nodes inside it, call finder)."""
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = [n for n in main.body if isinstance(n, ast.With)
               and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", "") == "no_statsbomb_reads"
                       for i in n.items)]
    assert len(guarded) == 1
    block = guarded[0]
    inside = {id(n) for n in ast.walk(block)}

    def calls(name):
        return [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == name]
    work = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[0] in ("ex", "records", "tv")]
    work += calls("frame_table") + calls("SigLIPEmbedder")
    outside = [ast.unparse(n) for n in work if id(n) not in inside]
    assert work and not outside, f"model or feature work outside the guard: {outside}"
    imports = [n for n in ast.walk(block) if isinstance(n, ast.Import) and any(a.name == "tactifoot_vision" for a in n.names)]
    assert imports, "the package (and the models it loads) is imported inside the guard"
    return main, inside, calls


def test_the_run_script_reads_the_pairing_only_before_the_guard():
    _, inside, calls = _guarded_main("xgd_extract.py")
    assert calls("paired_shot_ids") and all(id(c) not in inside for c in calls("paired_shot_ids"))


def test_the_full_run_reads_the_audit_choice_only_before_the_guard_and_never_the_pairing():
    main, inside, calls = _guarded_main("xgd_extract_full.py")
    assert calls("chosen_config") and all(id(c) not in inside for c in calls("chosen_config"))
    assert not calls("paired_shot_ids") and "pairs" not in ast.unparse(main)
    import xgd_extract_full

    assert xgd_extract_full.CHOICE.name == "choice.json" and FORBIDDEN.search(str(xgd_extract_full.CHOICE))


def test_the_full_run_reads_the_choice_name_only(tmp_path):
    import xgd_extract_full

    choice = tmp_path / "choice.json"
    choice.write_text('{"chosen": "C4", "gate": {"result": "STOP"}}')
    assert xgd_extract_full.chosen_config(choice) == "C4"


def test_the_full_run_compares_only_the_shots_of_both_runs(tmp_path):
    import xgd_extract_full

    reference = tmp_path / "features.parquet"
    pd.DataFrame({"shot_id": ["a", "b", "z"], "dist": [1.0, 2.0, 9.0], "shot_ok": [1.0, 1.0, 1.0]}).to_parquet(reference)
    features = pd.DataFrame({"shot_id": ["a", "a", "b", "c"], "offset_ms": [0, -400, 0, 0], "dist": [1.0, 7.0, 2.5, 3.0],
                             "shot_ok": [1.0, 0.0, 1.0, 1.0], "gk_x": [1.0, 1.0, 1.0, 1.0]})
    out = xgd_extract_full.reference_differences(features, reference)
    assert out == {"shots": 2, "shots_differing": 1, "columns": {"dist": 1, "shot_ok": 0}}   # z and c are in one run only
    assert xgd_extract_full.reference_differences(features, tmp_path / "missing.parquet") is None
