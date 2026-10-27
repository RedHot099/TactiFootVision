"""ball-detection 17: the accuracy read of scripts/ball_accuracy.py on synthetic placement rows and truth boxes
(thresholds, percentiles, the scorecard's draws, the reference agreement, zones, shot moments, ticket 01's levels)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_accuracy as ba  # noqa: E402
import ball_scorecard as bs  # noqa: E402


def rows(errors, clips=None, homography=None) -> pd.DataFrame:
    errors = np.asarray(errors, float)
    n = len(errors)
    return pd.DataFrame(
        {
            "clip": clips if clips is not None else [f"c{i % 3}" for i in range(n)],
            "error_m": errors,
            "homography": homography if homography is not None else np.ones(n, bool),
            "near_goal": np.zeros(n, bool),
        }
    )


def test_measures_thresholds_and_percentiles():
    error = np.array([0.2, 0.5, 0.9, 1.5, 3.0, 7.0, np.nan, np.nan])
    homography = np.array([True] * 7 + [False])
    found = ba.measures(error, homography)
    assert found["frames"] == 8
    assert found["no_homography"] == pytest.approx(1 / 8)
    assert found["no_position"] == pytest.approx(2 / 8)
    assert found["le0.5_all"] == pytest.approx(2 / 8)  # <= is inclusive
    assert found["le0.5_placed"] == pytest.approx(2 / 6)
    assert found["le1_all"] == pytest.approx(3 / 8)
    assert found["le2_placed"] == pytest.approx(4 / 6)
    assert found["le5_all"] == pytest.approx(5 / 8)
    assert found["far_of_placed"] == pytest.approx(1 / 6)
    assert found["far_of_all"] == pytest.approx(1 / 8)
    placed = error[np.isfinite(error)]
    assert found["p50_m"] == pytest.approx(np.quantile(placed, 0.5))
    assert found["p90_m"] == pytest.approx(np.quantile(placed, 0.9))
    assert found["p95_m"] == pytest.approx(np.quantile(placed, 0.95))


def test_measures_without_positions():
    found = ba.measures(np.array([np.nan, np.nan]), np.array([True, False]))
    assert found["no_position"] == 1.0
    assert found["le5_all"] == 0.0
    assert np.isnan(found["le5_placed"]) and np.isnan(found["p50_m"])


def test_measure_draws_match_the_scorecard():
    rng = np.random.default_rng(1)
    errors = rng.gamma(1.5, 2.0, 60)
    errors[rng.random(60) < 0.15] = np.nan
    table = rows(errors, clips=[f"c{i % 7}" for i in range(60)])
    draws = bs.Draws(table["clip"])
    mine = ba.measure_draws(table, draws)
    theirs = bs.placement_draws(table, draws)
    for a, b in (("le5_all", "well_placed"), ("far_of_placed", "far_of_placed")):
        assert mine[a][0] == theirs[b][0]
        np.testing.assert_array_equal(mine[a][1], theirs[b][1])
    np.testing.assert_allclose(mine["p50_m"][1], theirs["median_m"][1])
    point = ba.measures(table.error_m.to_numpy(), table.homography.to_numpy())
    for name, (value, _) in mine.items():
        assert value == pytest.approx(point[name], nan_ok=True)


def test_reference_agreement_centre_and_bottom():
    # frame -> pitch: 10 homography pixels per metre, homography pixels = native / 1.5
    matrix = np.diag([0.1, 0.1, 1.0])
    table = pd.DataFrame(
        {
            "clip": ["a", "a"],
            "frame": [1, 6],
            # native box 30 x 30 px around (150, 135); bottom middle at (150, 150) native = (100, 100) homography px
            "x1": [135.0, 135.0],
            "y1": [120.0, 120.0],
            "x2": [165.0, 165.0],
            "y2": [150.0, 150.0],
            "truth_x": [10.0, 10.0],
            "truth_y": [10.0, 10.0],
        }
    )
    out = ba.reference_errors(table, {("a", 1): matrix})
    assert out.ref_bottom_m.iloc[0] == pytest.approx(0.0, abs=1e-5)
    assert out.ref_centre_m.iloc[0] == pytest.approx(
        1.0, abs=1e-5
    )  # 15 native px = 10 px = 1 m
    assert np.isnan(out.ref_centre_m.iloc[1]) and np.isnan(out.ref_bottom_m.iloc[1])


def test_zone_penalty_area():
    x = np.array([5.0, 100.0, 50.0, 5.0, 16.5])
    y = np.array([34.0, 20.0, 34.0, 2.0, 13.84])
    assert ba.zone(x, y).tolist() == [
        "penalty area",
        "penalty area",
        "rest",
        "rest",
        "penalty area",
    ]


def test_action_frame_and_moment():
    info = {"action_position": "82617", "clip_start": "76000"}
    at = ba.action_frame(info)
    assert at == pytest.approx(6617 / 40 + 1)
    frames = np.array([1, 142, 166, 191, 192])  # action at 166.4
    assert ba.moment(frames, "Shots on target", at).tolist() == [
        "other",
        "shot",
        "shot",
        "shot",
        "other",
    ]
    assert ba.moment(frames, "Goal", at).tolist()[2] == "goal"
    assert set(ba.moment(frames, "Corner", at)) == {"other"}


def test_ball_categories_use_ticket_01_rules():
    table = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 0.0],
            "y1": [0.0, 0.0, 0.0],
            "x2": [6.0, 10.0, 30.0],
            "y2": [6.0, 10.0, 30.0],
            "coverage": [0.0, 0.3, 0.8],
            "feet_m": [3.0, 0.5, np.nan],
            "deviation_m": [2.5, 3.0, 1.0],
        }
    )
    out = ba.ball_categories(table)
    assert out["size"].tolist() == ["<8", "8-12", ">=24"]
    assert out.occlusion.tolist() == ["none", "<0.5", ">=0.5"]
    assert out.in_air.tolist() == ["in air", "ground", "unknown"]
