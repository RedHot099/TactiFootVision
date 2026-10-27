"""ball-detection 18: shorter gaps, the uncertainty features, rejection, the risk-coverage curve, the rule and D2's
gate of scripts/ball_reliability.py on synthetic gaps and synthetic placement rows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_reliability as br  # noqa: E402
import ball_temporal as bt  # noqa: E402

FRAMES = 80
SHORT = range(20, 25)  # 5 missing frames: a 0.24 s gap end to end
LONG = range(45, 65)  # 20 missing frames: a 0.84 s gap


def to_frame(xy) -> np.ndarray:
    """A still camera: 16 px per metre."""
    return np.array([16 * xy[0] + 100, 12 * xy[1] + 80])


def frame_to_pitch() -> np.ndarray:
    return np.linalg.inv(np.array([[16.0, 0, 100], [0, 12.0, 80], [0, 0, 1]]))


def ball(t: int) -> tuple[float, float]:
    return 30 + 0.4 * t, 20 + 0.1 * t  # 10 m/s along, 2.5 m/s across


def scene() -> pd.DataFrame:
    rows = []
    for t in range(FRAMES):
        if t in SHORT or t in LONG:
            continue
        x, y = to_frame(ball(t))
        rows.append(
            {"frame": t + 1, "class_id": 0, "conf": np.float32(0.6)}
            | dict(zip(bt.COLUMNS, [x - 5, y - 5, x + 5, y + 5], strict=True))
        )
    return pd.DataFrame(rows)


def tracked(max_gap: float) -> pd.DataFrame:
    homographies = [frame_to_pitch()] * FRAMES
    result = bt.clip_result(scene(), homographies)
    return bt.chosen_balls(bt.track_ball(result, **(bt.BASE | {"max_gap": max_gap})))


def test_max_gap_fills_only_gaps_up_to_its_length():
    short = tracked(0.25)
    assert (short.status.iloc[list(SHORT)] == "interpolated").all()
    assert (short.status.iloc[list(LONG)] == "none").all()
    long = tracked(1.0)
    assert (long.status.iloc[list(SHORT) + list(LONG)] == "interpolated").all()
    assert (long.status.drop(index=list(SHORT) + list(LONG)) == "measured").all()


def test_position_features_of_synthetic_gaps():
    track = tracked(1.0)
    motion = np.repeat(np.eye(3)[None], FRAMES - 1, axis=0)
    motion[50] = np.nan  # a motion break inside the long gap
    found = br.position_features(track, [frame_to_pitch()] * FRAMES, motion)
    short, long = found.iloc[list(SHORT)], found.iloc[list(LONG)]
    np.testing.assert_allclose(short.gap_s, 6 / 25)
    np.testing.assert_allclose(long.gap_s, 21 / 25)
    assert found.to_measure_s.iloc[20] == pytest.approx(1 / 25)
    assert found.to_measure_s.iloc[22] == pytest.approx(
        3 / 25
    )  # middle of the short gap
    assert found.to_measure_s.iloc[55] == pytest.approx(10 / 25)
    measured = found[found.status == "measured"]
    assert (measured.gap_s == 0).all() and (measured.to_measure_s == 0).all()
    # implied speed: |(0.4, 0.1)| m per frame at 25 fps, for the gaps and the measured balls alike
    speed = np.log1p(np.hypot(0.4, 0.1) * 25)
    np.testing.assert_allclose(found.log_speed, speed, rtol=1e-4)
    assert (found.speed_missing == 0).all()
    np.testing.assert_allclose(
        found.conf, 0.6, rtol=1e-6
    )  # the gap ends' confidence too
    assert (
        found.motion_break.iloc[50] == 1
        and found.motion_break.drop(index=50).eq(0).all()
    )
    assert (found.log_motion == 0).all()
    np.testing.assert_allclose(found.log_size, np.log(10.0))


def test_gap_structure_at_clip_ends_and_without_balls():
    status = np.array(
        ["interpolated", "measured", "none", "interpolated", "interpolated", "measured"]
    )
    found = br.gap_structure(status)
    assert np.isnan(found.gap_s[2])
    assert found.gap_s[3] == pytest.approx(4 / 25) and found.to_measure_s[
        4
    ] == pytest.approx(1 / 25)
    assert found.to_measure_s[0] == pytest.approx(
        1 / 25
    )  # no measurement before: one side only


def test_rejection_and_operating_numbers():
    error = np.array([1.0, 2.0, 8.0, 9.0, 3.0, np.nan, np.nan, 6.0])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.7, np.nan, np.nan, 0.3])
    kept = br.apply_rejection(error, score, 0.5)
    np.testing.assert_array_equal(np.isnan(kept), [0, 0, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(br.apply_rejection(error, score, None), error, equal_nan=True)
    found = br.operating(error, kept, reference=error)
    assert found["coverage"] == pytest.approx(3 / 8)
    assert found["far_of_passed"] == pytest.approx(1 / 3)
    assert found["far_of_all"] == pytest.approx(1 / 8)
    assert found["rejected"] == 3 and found["rejected_bad_share"] == pytest.approx(
        2 / 3
    )
    assert found["loss_pp"] == pytest.approx(
        100 * (3 / 8 - 2 / 8)
    )  # the rejected 3 m one was well placed


def test_curve_points_ties_and_interpolation():
    error = np.array([1.0, 8.0, 9.0, 2.0, np.nan])
    score = np.array(
        [0.1, 0.9, 0.5, 0.5, np.nan]
    )  # the last two tie: they leave together
    c = br.curve(error, score)
    np.testing.assert_allclose(c.coverage, [4 / 5, 3 / 5, 1 / 5, 0])
    np.testing.assert_allclose(c.risk[:3], [2 / 4, 1 / 3, 0])
    np.testing.assert_allclose(c.rejected_share, [0, 1 / 4, 3 / 4, 1])
    np.testing.assert_allclose(c.bad_of_all, [2 / 5, 1 / 5, 0, 0])
    np.testing.assert_allclose(c.rejected_bad_share[1:], [1, 2 / 3, 2 / 4])
    np.testing.assert_allclose(c.good_lost_of_all, [0, 0, 1 / 5, 2 / 5])
    assert br.at(c, "coverage", 0.7) == pytest.approx((2 / 4 + 1 / 3) / 2)
    assert np.isnan(br.at(c, "coverage", 0.9))  # beyond the curve
    weighted = br.curve(
        error, score, np.array([2, 1, 1, 1, 1.0])
    )  # clip weights repeat frames
    assert weighted.coverage[0] == pytest.approx(5 / 6) and weighted.risk[
        0
    ] == pytest.approx(2 / 5)
    two = br.curve(error, score, limit=2.0)
    assert two.risk[0] == pytest.approx(2 / 4)  # > 2 m: 8 and 9, 2.0 is not above


def test_a_perfect_score_has_a_lower_curve_than_a_blind_one():
    rng = np.random.default_rng(0)
    error = rng.gamma(1.5, 3.0, 400)
    perfect, blind = br.curve(error, error), br.curve(error, rng.random(400))
    assert br.aurc(perfect) < br.aurc(blind)
    assert br.at(perfect, "rejected_share", 0.1) < br.at(blind, "rejected_share", 0.1)


def candidate(gap, tau, far, coverage, loss, rejected, bad_share) -> dict:
    return {
        "max_gap": gap,
        "tau": tau,
        "far_of_passed": far,
        "coverage": coverage,
        "loss_pp": loss,
        "rejected": rejected,
        "rejected_bad_share": bad_share,
    }


def test_choose_applies_both_constraints_then_the_lowest_risk():
    pool = [
        candidate(2.0, None, 0.14, 0.83, 0.0, 0, np.nan),
        candidate(2.0, 0.5, 0.08, 0.75, 2.5, 50, 0.7),  # loses too much
        candidate(2.0, 0.6, 0.10, 0.78, 1.5, 30, 0.4),  # rejects mostly good positions
        candidate(1.0, 0.7, 0.11, 0.80, 1.0, 20, 0.6),
        candidate(0.5, 0.7, 0.11, 0.81, 1.0, 20, 0.6),  # same risk, more coverage
    ]
    assert br.choose(pool) == pool[4]
    assert br.choose(pool[1:3]) is None


def test_gate_needs_all_three_conditions():
    gaps = pd.DataFrame({"coverage": [0.70, 0.80], "far_of_passed": [0.10, 0.14]})
    good = {
        "coverage": 0.75,
        "far_of_passed": 0.11,
        "rejected": 10,
        "rejected_bad_share": 0.6,
        "loss_pp": 1.0,
    }
    found = br.gate(good, gaps)
    assert found["gap_curve_at_coverage"] == pytest.approx(0.12) and found["met"]
    assert not br.gate(good | {"far_of_passed": 0.13}, gaps)["a_below_gap_curve"]
    assert not br.gate(good | {"coverage": 0.65}, gaps)[
        "met"
    ]  # outside the max_gap curve
    assert not br.gate(good | {"rejected_bad_share": 0.45}, gaps)["met"]
    assert not br.gate(good | {"loss_pp": 2.5}, gaps)["met"]


def test_score_fits_ranks_and_freezes():
    rng = np.random.default_rng(1)
    table = pd.DataFrame({c: rng.normal(size=300) for c in br.FEATURES})
    bad = (table.gap_s + 0.3 * rng.normal(size=300)) > 0.8
    score = br.Score().fit(table, bad)
    p = score.predict(table)
    assert np.corrcoef(p, table.gap_s)[0, 1] > 0.5
    np.testing.assert_allclose(
        br.frozen_probability(score.frozen(), table), p, rtol=1e-9
    )
    base = br.Score(br.BASELINE).fit(table, bad)
    assert base.frozen()["columns"] == ["gap_s"]


def test_calibration_bins():
    found = br.calibration(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 0, 1, 0]))
    assert found["base_rate"] == pytest.approx(0.25)
    assert [b["n"] for b in found["bins"]] == [2, 2]
    assert found["bins"][1]["observed"] == pytest.approx(0.5)


def test_point_draws_match_the_operating_numbers():
    rng = np.random.default_rng(2)
    n = 90
    error = rng.gamma(1.5, 3.0, n)
    error[rng.random(n) < 0.1] = np.nan
    table = pd.DataFrame(
        {
            "clip": [f"c{i % 6}" for i in range(n)],
            "error_m": error,
            "reference_m": error,
        }
    )
    kept = br.apply_rejection(error, error + rng.normal(0, 2, n), 8.0)
    found = br.point_draws(table, kept, br.bs.Draws(table["clip"]))
    point = br.operating(error, kept, error)
    for name in (
        "coverage",
        "far_of_passed",
        "far_of_all",
        "well_placed",
        "rejected_bad_share",
    ):
        assert found[name][0] == pytest.approx(point[name])
    assert (
        0
        <= np.nanmin(found["rejected_bad_share"][1])
        <= np.nanmax(found["rejected_bad_share"][1])
        <= 1
    )
