"""The descriptive StatsBomb agreement of the xg-stack test (ticket 04, council C7 Q4) on synthetic StatsBomb-shaped
rows with known answers: the category predicates in StatsBomb yards, Spearman's ρ and the gap items."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.stack_heldout import vendor


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "sb_x": [110.0, 95.0, 117.0, 104.0, np.nan], "sb_y": [40.0, 40.0, 41.0, 70.0, np.nan],
        "sb_dist": [10.0, 25.0, 3.0, 31.0, np.nan], "sb_angle": np.radians([40.0, 18.0, 90.0, 5.0, np.nan]),
        "p_vendor": [0.10, 0.04, 0.50, 0.02, 0.08], "p_A": [0.12, 0.05, 0.40, 0.03, 0.09], "p_B": [0.10, 0.04, 0.50, 0.02, 0.08]})


def test_the_categories_follow_xg_compares_predicates_in_statsbomb_yards():
    masks = vendor.category_masks(frame())
    defined, inside = masks["sb_outside_box"]
    assert defined.tolist() == [True, True, True, True, False] and inside.tolist() == [False, True, False, True, False]
    assert masks["sb_angle_lt_15deg"][1].tolist() == [False, False, False, True, False]
    assert masks["sb_dist_lt_6yd"][1].tolist() == [False, False, True, False, False]


def test_agreement_gives_spearman_and_the_gaps_in_xg_points_and_logits():
    out = vendor.agreement(frame(), ("A", "B"))
    assert out["spearman"]["B"] == pytest.approx(1.0) and out["shots"] == 5 and "Data: StatsBomb." in out["label"]
    rows = {(r["category"], r["model"]): r for r in out["categories"]}
    box = rows["sb_outside_box", "A"]
    assert (box["n_in"], box["n_out"], box["n_defined"]) == (2, 2, 4)
    assert box["gap_points_in"] == pytest.approx(100 * (np.mean([0.05, 0.03]) - np.mean([0.04, 0.02])))
    assert box["gap_points_out"] == pytest.approx(100 * (np.mean([0.12, 0.40]) - np.mean([0.10, 0.50])))
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    gap = lg(np.array([0.12, 0.05, 0.40, 0.03])) - lg(np.array([0.10, 0.04, 0.50, 0.02]))
    assert box["logit_gap_contrast"] == pytest.approx(gap[[1, 3]].mean() - gap[[0, 2]].mean())
    assert rows["sb_dist_lt_6yd", "B"]["gap_points_in"] == pytest.approx(0.0)
