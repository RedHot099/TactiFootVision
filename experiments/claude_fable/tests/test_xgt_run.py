"""The xg-temporal scripts (ticket 01): the run refuses real labels, names its output by source and window, stops a
late window without the control flag before reading anything, and runs end to end on synthetic inputs; the inputs
script's per-source summary compares the coverage with the spec."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgt_inputs  # noqa: E402
import xgt_run  # noqa: E402
from xgpoc.bundle import sha256_file  # noqa: E402
from xgpoc.temporal import inputs as ti  # noqa: E402
from xgpoc.temporal import provenance as tp  # noqa: E402
from xgt_fixtures import synthetic_population  # noqa: E402


def test_the_run_refuses_real_labels_and_empty_settings():
    for argv in (["--source", "dinov2"], ["--synthetic", "--source", "dinov2", "--replicates", "0"],
                 ["--synthetic", "--source", "dinov2", "--k", "0"], ["--synthetic"]):
        with pytest.raises(SystemExit):
            xgt_run.parse_args(argv)


def test_the_default_output_names_source_and_window():
    args = xgt_run.parse_args(["--synthetic", "--source", "g_xgv"])
    assert args.out == xgt_run.ROOT / "results/xgt_synthetic_g_xgv_-1600_-600" and args.k == 32 and args.replicates == 1
    args = xgt_run.parse_args(["--synthetic", "--source", "dinov2", "--o-max", "400", "--control"])
    assert args.out.name == "xgt_synthetic_dinov2_-1600_400_control"


def test_a_late_window_without_the_control_flag_stops_before_any_input_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(xgt_run.a1, "verify_plan", lambda *_: pytest.fail("an input was read"))
    with pytest.raises(tp.ProvenanceError, match="control flag"):
        xgt_run.main(["--synthetic", "--source", "dinov2", "--o-max", "-200", "--out", str(tmp_path / "run")])


def synthetic_sources(monkeypatch, module):
    """Route the module's plan, population and embedding reads to synthetic ones."""
    cache = {}

    def build(source, window, plan, files=None):
        population, embeddings, _ = synthetic_population(window)
        cache["embeddings"] = embeddings
        return population, SimpleNamespace(record=lambda: {"source": "synthetic"})

    _, _, plan = synthetic_population(ti.Window())
    monkeypatch.setattr(module.a1, "verify_plan", lambda *_: plan)
    monkeypatch.setattr(module.ti, "build_population", build)
    monkeypatch.setattr(module.ti, "verify_source", lambda source: SimpleNamespace(record=lambda: {"source": "synthetic"}))
    monkeypatch.setattr(module.ti, "load_embeddings",
                        lambda files, refs: (cache["embeddings"], refs))
    return plan


def test_the_run_writes_its_records_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    synthetic_sources(monkeypatch, xgt_run)
    out = tmp_path / "run"
    xgt_run.main(["--synthetic", "--source", "g_xgv", "--k", "3", "--replicates", "2", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["provenance"]["passed"] and summary["population"]["shots"] == 160
    assert summary["columns"]["T"]["n"] == 1 + 3 + 6 + 3 * 3 and set(summary["chosen_c"]) == {"0", "1"}
    assert set(summary["first_replicate"]["deltas"]) == {"T-R", "T-S", "S-R"}
    assert len(pd.read_csv(out / "replicates.csv")) == 2
    complete = json.loads((out / "complete.json").read_text())
    assert complete["files"] == {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "complete.json"}
    assert "xgpoc/temporal/features.py" in complete["code"]["run"]["files"]
    assert "scripts/xgt_run.py" not in complete["code"]["package"]["files"]
    with pytest.raises(SystemExit, match="new folder"):
        xgt_run.main(["--synthetic", "--source", "g_xgv", "--k", "3", "--out", str(out)])


def test_the_inputs_summary_checks_coverage_against_the_spec_and_writes_the_population(tmp_path, monkeypatch):
    plan = synthetic_sources(monkeypatch, xgt_inputs)
    monkeypatch.setitem(xgt_inputs.SPEC_TABLE, "g_xgv", {"-600": [160, 160, 20], "-800": [1, 1, 1]})
    summary = xgt_inputs.source_summary("g_xgv", plan, tmp_path)
    assert summary["coverage"]["-600"]["matches_spec"] and not summary["coverage"]["-800"]["matches_spec"]
    assert "spec" not in summary["coverage"]["-400"] and summary["coverage"]["-400"]["offsets"] == 7
    assert set(summary["pca_explained_variance"]["mean"]) == {"16", "32", "64"}
    population = pd.read_csv(tmp_path / "population_g_xgv.csv")
    assert len(population) == 160 and summary["main_population_csv_sha256"] == sha256_file(tmp_path / "population_g_xgv.csv")
