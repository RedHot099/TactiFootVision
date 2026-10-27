"""ball-detection 19 (scripts/ball_xg_compare.py): the arms' building blocks (relation columns of arm C, the kept
ball-source columns of arm S2), the family's Holm / TOST verdicts, the bands, the cut score and the clip's t0 frame."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_xg_compare as bc  # noqa: E402


def test_relation_columns_of_a_ball_at_the_feet():
    row = bc.relation(
        shooter_box=(100, 100, 120, 160),
        ball_box=(108, 154, 112, 158),
        feet_xy=(90.0, 30.0),
        ball_xy=(90.3, 30.4),
        ball_conf=0.7,
        source=0,
    )
    assert set(row) == set(bc.RELATION)
    assert row["bf_dist"] == pytest.approx(0.5)
    assert row["bf_height"] == pytest.approx((160 - 156) / 60)
    assert row["bf_lateral"] == pytest.approx(0.0)
    assert row["bf_gap"] == 0.0  # the centre lies inside the box
    assert (
        row["bf_src_own"],
        row["bf_src_m400"],
        row["bf_src_m800"],
        row["bf_src_none"],
    ) == (1, 0, 0, 0)
    assert row["bf_missing"] == 0.0 and row["bf_dist_missing"] == 0.0


def test_relation_of_a_high_fallback_ball_off_the_box():
    row = bc.relation(
        (100, 100, 120, 160), (128, 96, 132, 100), (90, 30), (91, 34), 0.2, 1
    )
    assert row["bf_height"] == pytest.approx((160 - 98) / 60)
    assert row["bf_height"] >= bc.HIGH_BALL
    assert row["bf_lateral"] == pytest.approx(20 / 60)
    assert row["bf_gap"] == pytest.approx(np.hypot(10, 2) / 60)
    assert row["bf_src_m400"] == 1.0 and row["bf_src_own"] == 0.0


def test_relation_without_a_pitch_position_keeps_the_image_columns():
    row = bc.relation((0, 0, 10, 50), (2, 40, 4, 42), (110.0, 30.0), (90, 30), 0.5, 2)
    assert np.isnan(row["bf_dist"]) and row["bf_dist_missing"] == 1.0
    assert row["bf_missing"] == 0.0 and np.isfinite(row["bf_height"])
    assert row["bf_src_m800"] == 1.0


def test_relation_missing_marks_every_group():
    row = bc.relation_missing()
    assert set(row) == set(bc.RELATION)
    assert row["bf_missing"] == row["bf_dist_missing"] == row["bf_src_none"] == 1.0
    assert all(
        np.isnan(row[c])
        for c in ("bf_dist", "bf_height", "bf_lateral", "bf_gap", "bf_conf")
    )
    assert row["bf_src_own"] == row["bf_src_m400"] == row["bf_src_m800"] == 0.0


def test_s2_keeps_only_the_reference_ball_source_columns():
    index = pd.Index(["a", "b"], name="shot_id")
    columns = [
        "dist@0",
        "ball_none@0",
        "ball_own@0",
        "ball_m400@0",
        "shot_from_ball@0",
        "ball_m800@-400",
        "shooter_by_net@0",
        "shot_x@-400",
    ]
    reference = pd.DataFrame(0.0, index=index, columns=columns)
    s2 = pd.DataFrame(1.0, index=index, columns=columns)
    out = bc.keep_flags(s2, reference)
    flags = [
        "ball_none@0",
        "ball_own@0",
        "ball_m400@0",
        "shot_from_ball@0",
        "ball_m800@-400",
    ]
    assert bc.flag_columns(columns) == flags
    assert (out[flags] == 0.0).all().all()
    assert (out[["dist@0", "shooter_by_net@0", "shot_x@-400"]] == 1.0).all().all()
    assert (s2 == 1.0).all().all()  # the input is not changed
    with pytest.raises(ValueError):
        bc.keep_flags(s2, reference.iloc[::-1])


def test_bootstrap_p_values():
    boot = np.array([-0.003, -0.002, -0.001, 0.0, 0.001])
    assert bc.p_improve(boot) == pytest.approx(3 / 6)
    assert bc.p_equivalent(boot, 0.002) == pytest.approx(
        3 / 6
    )  # two draws at or below -0.002
    tight = np.full(999, 0.0005)
    assert bc.p_equivalent(tight, 0.002) == pytest.approx(1 / 1000)
    assert bc.p_improve(-tight) == pytest.approx(1 / 1000)


def test_holm_steps_down_and_a_missing_arm_counts_as_p_one():
    assert bc.holm({"C": 0.01, "S2": 0.02, "T": None}, 0.05) == {
        "C": True,
        "S2": True,
        "T": False,
    }
    # the smallest p (0.01) is above 0.025 / 3: nothing is rejected
    assert bc.holm({"C": 0.02, "S2": 0.01, "T": None}, 0.025) == {
        "C": False,
        "S2": False,
        "T": False,
    }
    assert bc.holm({"C": 0.004, "S2": 0.04, "T": 0.001}, 0.05) == {
        "C": True,
        "S2": True,
        "T": True,
    }


def test_verdicts_follow_d3():
    p_imp = {"C": 0.0005, "S2": 0.4, "T": None}
    p_eq = {"C": 0.9, "S2": 0.001, "T": None}
    out = bc.verdicts(p_imp, p_eq, {"C": -0.003, "S2": 0.0001, "T": np.nan})
    assert out == {"C": "adds value", "S2": "equivalent", "T": "not run"}
    out = bc.verdicts(
        {"C": 0.02, "S2": 0.4, "T": None},
        {"C": 0.3, "S2": 0.2, "T": None},
        {"C": -0.001, "S2": 0.0, "T": 0},
    )
    assert out == {"C": "inconclusive", "S2": "inconclusive", "T": "not run"}


def test_bands():
    assert list(bc.dist_band([5, 11, 17.9, 18, 30, np.nan])) == [
        "<11",
        "11-18",
        "11-18",
        "18-25",
        ">=25",
        "missing",
    ]
    assert list(bc.height_band([0.1, 0.6, np.nan])) == ["low", "high", "missing"]


def test_content_score_flags_a_cut_only():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 50, (8, 8, 3), dtype=np.uint8)
    b = (a + 150).astype(np.uint8)
    frames = np.stack([a, a, b, b])
    score = bc.content_scores(frames)
    assert score[0] == 0.0 and score[2] == 0.0
    assert score[1] == pytest.approx(150.0)


def test_t0_index_of_the_clip():
    assert bc.t0_index(270441) == 100
    assert bc.t0_index(2000) == 50  # the clip starts at 0 s
    assert bc.t0_index(4000) == 100
