"""The complexity report rebuilds the deployed G-combo from fold predictions and pairs
every comparison through one set of game draws; these pin the pieces it relies on."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_recalibration_recovers_known_intercept_and_slope():
    from report_complexity import recalibration

    rng = np.random.default_rng(0)
    z = rng.normal(-1.8, 1.0, 200_000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-(0.3 + 1.4 * z)))).astype(float)
    a, b = recalibration(y, 1 / (1 + np.exp(-z)))
    assert abs(a - 0.3) < 0.03 and abs(b - 1.4) < 0.03


def test_game_draws_match_significance_sequence():
    """Same seed, same sorted game keys, same rng.choice call: the CIs are comparable
    with results/v12_report/significance_final."""
    from report_complexity import game_draws

    games = np.array(["b", "a", "c", "a", "b", "b"])
    draw = next(game_draws(games, 1, seed=0))
    keys = np.array(sorted(set(games)))
    expected = np.random.default_rng(0).choice(keys, len(keys), replace=True)
    rows = {g: np.flatnonzero(games == g) for g in keys}
    assert np.array_equal(draw, np.concatenate([rows[g] for g in expected]))


def test_deploy_concatenates_oof_and_averages_test():
    from report_complexity import configurations, deploy

    def frame(ids, z):
        return pd.DataFrame({"shot_id": ids, "z": z})

    rng = np.random.default_rng(1)
    runs = {}
    for f in range(2):
        oof_ids = [f"o{f}_{i}" for i in range(400)]
        for s in range(2):
            runs[f, s] = {"valid": frame(oof_ids, rng.normal(-1.8, 1, 400)), "test": frame(["t0", "t1"], [-2.0 + f, -1.0 + s])}
    conf = configurations(runs)
    assert [len(v) for v in conf.values()] == [4, 2, 2, 1]
    assert conf["g2"][0][1] == [(0, 0), (0, 1)] and conf["g5"][1][1] == [(0, 1), (1, 1)]
    oof_ids = [i for k in runs for i in runs[k]["valid"].shot_id]
    labels = pd.Series(rng.random(len(oof_ids)) < 0.14, index=oof_ids).groupby(level=0).first()
    labels = pd.concat([labels, pd.Series([False, False], index=["t0", "t1"])])
    is_pen = pd.Series(False, index=labels.index)
    is_pen["t1"] = True
    p = deploy(runs, conf["g10"][0][1], labels, is_pen, pooled_rate=0.14, p_pen=0.7)
    assert list(p.index) == ["t0", "t1"] and p["t1"] == 0.7 and 0 < p["t0"] < 0.5


def _boot_reduction(loss_a: np.ndarray, loss_b: np.ndarray, games: np.ndarray, n_boot: int) -> np.ndarray:
    """Bootstrap draws of the mean loss reduction loss_b - loss_a, by game."""
    from report_complexity import game_draws

    return np.array([np.mean(loss_b[r] - loss_a[r]) for r in game_draws(games, n_boot)])


def test_equivalence_constant_difference_gives_next_grid_margin():
    """Every shot 0.00123 better: each draw is 0.00123, so equivalence first holds at 0.0013."""
    from report_complexity import equivalence_curve, smallest_margin

    games = np.repeat(np.arange(50), 4)
    loss_b = np.random.default_rng(0).uniform(0.1, 0.5, games.size)
    draws = _boot_reduction(loss_b - 0.00123, loss_b, games, 200)
    curve = equivalence_curve(draws)
    assert smallest_margin(curve) == pytest.approx(0.0013)
    assert not curve.equivalent[curve.margin < 0.00123].any() and curve.equivalent[curve.margin > 0.00123].all()


@pytest.mark.parametrize("mu", [8e-4, -8e-4])
def test_equivalence_recovers_known_margin(mu):
    """Per-game reductions with sample mean m and standard error s: the bootstrap mean is
    close to normal, so the 90% interval is m +- 1.645 s and equivalence first holds at
    the grid point just above |m| + 1.645 s, whichever side binds."""
    from report_complexity import equivalence_curve, smallest_margin

    rng = np.random.default_rng(1)
    n_games = 400
    per_game = rng.normal(mu, 0.01, n_games)
    games = np.repeat(np.arange(n_games), 5)
    loss_b = rng.uniform(0.1, 0.5, games.size)
    draws = _boot_reduction(loss_b - per_game[games], loss_b, games, 4000)
    m, s = per_game.mean(), per_game.std() / np.sqrt(n_games)
    known = abs(m) + 1.645 * s
    found = smallest_margin(equivalence_curve(draws))
    assert known - 2e-5 <= found <= known + 1e-4 + 2e-5


def test_equivalence_out_of_grid_is_nan():
    from report_complexity import equivalence_curve, smallest_margin

    assert np.isnan(smallest_margin(equivalence_curve(np.full(100, 0.01))))


def test_ece_equal_count_uses_the_reliability_bins():
    from report_complexity import ece_equal_count, rel_bins

    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.6, 5000)
    y = (rng.random(p.size) < p).astype(float)
    b = rel_bins(p)
    manual = sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in np.unique(b))
    assert ece_equal_count(y, p) == pytest.approx(manual)


def test_split_half_recalibration_gains_only_when_miscalibrated():
    """Out-of-half logistic recalibration recovers a planted slope of 1.6 and gains nothing
    (up to noise) on calibrated predictions."""
    from report_complexity import split_half_recalibration

    rng = np.random.default_rng(3)
    games = np.repeat(np.arange(100), 40)
    z = rng.normal(-1.9, 0.9, games.size)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(float)
    calibrated = split_half_recalibration(y, 1 / (1 + np.exp(-z)), games, n_splits=20)
    timid = split_half_recalibration(y, 1 / (1 + np.exp(-(z / 1.6 - 0.7))), games, n_splits=20)
    assert len(calibrated) == 40
    assert abs(calibrated.gain_logistic.mean()) < 1e-3 < timid.gain_logistic.mean()


def test_diff_row_takes_the_skill_share_inside_each_draw():
    """Reductions are positive when `a` is better; the share interval is the percentile
    interval of reduction / skill per draw, not the reduction interval over the point skill."""
    from report_complexity import diff_row

    rng = np.random.default_rng(4)
    n = 500
    ll_g10, ll_ens = rng.normal(0.32, 0.01, n), rng.normal(0.319, 0.01, n)
    base = rng.normal(0.39, 0.01, n)

    def s(ll_a, ll_b):
        return np.array([[0.79, ll_a, 0.1], [0.78, ll_b, 0.1]])

    res = {"names": ["g10", "ens"], "ix": {"g10": 0, "ens": 1}, "point": {"s": s(0.324, 0.323), "base": np.array([0.39, 0.115])},
           "boot": [{"s": s(a, b), "base": np.array([c, 0.115])} for a, b, c in zip(ll_g10, ll_ens, base, strict=True)]}
    row = diff_row(res, ["ens"], ["g10"])
    assert row["d_log_loss"] == pytest.approx(0.001) and row["d_auc"] == pytest.approx(-0.01)
    share = (ll_g10 - ll_ens) / (base - ll_g10)
    assert (row["share_log_loss_lo"], row["share_log_loss_hi"]) == pytest.approx(
        (np.percentile(share, 2.5), np.percentile(share, 97.5)))


def test_equivalence_rejects_nan_draws():
    from report_complexity import equivalence_curve

    with pytest.raises(ValueError, match="NaN"):
        equivalence_curve(np.array([0.001, np.nan, -0.001]))


def test_calibration_diff_pairs_draws():
    """Both models swing together across draws while their difference stays at 0.04: a
    paired difference has a zero-width interval, an unpaired one would not."""
    from report_complexity import CALIB, calibration_diff

    swing = np.random.default_rng(5).normal(1.2, 0.1, 300)

    def c(slope_g10):
        row = np.zeros((2, len(CALIB)))
        row[:, CALIB.index("slope")] = [slope_g10, slope_g10 + 0.04]
        return row

    res = {"ix": {"g10": 0, "ens": 1}, "point": {"c": c(1.19)}, "boot": [{"c": c(s)} for s in swing]}
    slope = calibration_diff(res, "ens", "g10").set_index("quantity").loc["slope"]
    assert slope["diff"] == pytest.approx(0.04) and (slope.lo, slope.hi) == pytest.approx((0.04, 0.04))


def test_game_halves_are_disjoint_by_game_and_cover_every_shot():
    from report_complexity import game_halves

    games = np.repeat([f"g{i}" for i in range(11)], 3)
    halves = list(game_halves(games, n_splits=5, seed=0))
    assert len(halves) == 10 and [s for s, _, _ in halves] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for _, fit, ev in halves:
        assert not (fit & ev).any() and (fit | ev).all()
        assert not set(games[fit]) & set(games[ev])
        assert {len(set(games[fit])), len(set(games[ev]))} == {5, 6}
