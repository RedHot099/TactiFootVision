"""The figures of the xg-compare report (ticket 03), drawn from the analysis of a synthetic population."""

import sys
from pathlib import Path

from test_xgc_stats import population, synthetic
from xgpoc.compare import analysis as an
from xgpoc.compare import describe as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from xgc_report_figures import forest, heatmap  # noqa: E402


def test_both_figures_are_drawn_from_the_analysis_tables(tmp_path):
    pop = population(*synthetic(n_matches=30, per_match=20, n_boot=20, header=0.6))
    family = an.family(pop).astype({"item": str})
    agreement, _ = ds.agreement(pop)
    forest(family, tmp_path / "forest.png")
    heatmap(agreement, tmp_path / "heatmap.png")          # A1 is absent here: the matrix keeps the methods it has
    for name in ("forest.png", "heatmap.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
