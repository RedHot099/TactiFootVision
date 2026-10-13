"""The descriptive recovery share and mean xG of the xg-compare report (ticket 03), on synthetic populations where
the answer is known."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_xgc_stats import population, synthetic
from xgpoc.compare import protocol as pr
from xgpoc.compare.population import membership
from xgpoc.eval.calibration import sigmoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from xgc_recovery import check_against_family, recovery  # noqa: E402

GAP_ITEMS = tuple(item for item in pr.FAMILY if item.kind == "gap")


def with_frame(source: str):
    table, labels, draws = synthetic(n_matches=30, per_match=20, n_boot=50)
    table["z_frame"] = table[f"z_{source}"]
    return population(table, labels, draws)


def test_a_frame_equal_to_geometry_recovers_nothing_and_one_equal_to_the_vendor_everything():
    as_geometry = recovery(with_frame("dist_angle"), GAP_ITEMS)
    assert np.allclose(as_geometry.share_point, 0.0) and np.allclose(as_geometry[["share_lo", "share_hi"]], 0.0)
    as_vendor = recovery(with_frame("vendor"), GAP_ITEMS)
    assert np.allclose(as_vendor.share_point, 1.0) and np.allclose(as_vendor.FV_point, 0.0)
    assert np.allclose(as_vendor.mean_xg_F_minus_V_point, 0.0)


def test_mean_xg_is_the_category_mean_of_the_main_scale_probability_in_row_0():
    pop = with_frame("frame")
    item = pr.BY_ID["3"]
    row = recovery(pop, (item,)).iloc[0]
    inside, _ = membership(pop, item.category, item.level)
    assert row.n_in == inside.sum()
    assert row.mean_xg_F_point == pytest.approx(100 * sigmoid(pop.z("F")[0][inside]).mean())
    assert row.mean_xg_F_minus_V_point == pytest.approx(row.mean_xg_F_point - row.mean_xg_V_point)
    assert row.share_point == pytest.approx(1 - row.FV_point / row.DV_point)


def test_the_check_refuses_effects_that_differ_from_the_run():
    frame = pd.DataFrame({"item": ["3"], "FV_point": [0.4], "DV_point": [0.8]})
    check_against_family(frame, pd.DataFrame({"item": ["3"], "effect_point": [0.4], "ref_DV_point": [0.8]}))
    with pytest.raises(SystemExit):
        check_against_family(frame, pd.DataFrame({"item": ["3"], "effect_point": [0.41], "ref_DV_point": [0.8]}))
    with pytest.raises(SystemExit):
        check_against_family(frame, pd.DataFrame({"item": ["3"], "effect_point": [np.nan], "ref_DV_point": [0.8]}))
