"""Council C6's selection of G's design (ticket 01): the tie rule, common random numbers and the summary."""

import pytest
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import protocol as pr
from xgpoc.stack import selection as sl
from xgpoc.stack import stats as ss
from xgs_fixtures import synthetic_table


def test_the_highest_averaged_power_wins_and_ties_go_to_the_simpler_design():
    assert sl.choose({"b": 0.30, "c": 0.35, "bc": 0.20}) == "c"
    assert sl.choose({"b": 0.30, "c": 0.30, "bc": 0.30}) == "b"
    assert sl.choose({"b": 0.10, "c": 0.30, "bc": 0.30}) == "c"
    assert sf.SIMPLICITY == ("b", "c", "bc")


def test_the_selection_seeds_differ_from_the_final_seeds_and_the_prediction_is_written_first():
    assert not set(sl.SEEDS.values()) & set(pr.FINAL_SEEDS.values())
    assert sl.PREDICTED_RECOVERY_B == (-0.0017, -0.0015) and sl.TARGET == -0.002


def test_the_selection_fits_every_candidate_on_the_same_outcomes():
    table = synthetic_table(n_matches=40, per_match=30, seed=30)
    weights = ss.bootstrap_weights(table.game.to_numpy(), draws=100, seed=0)
    frames, summary = sl.run(table, table.fold.to_numpy(), weights, -0.2, 1.0, n=3, seeds={"dist": 7, "dist_angle": 8})
    assert set(frames) == {(c, k) for c in sf.SIMPLICITY for k in sc.PLANTS}
    for kind in sc.PLANTS:
        goals = [frames[c, kind].goals.tolist() for c in sf.SIMPLICITY]
        assert goals[0] == goals[1] == goals[2]                                # common random numbers
        assert summary["plants"][kind]["true_delta"] == pytest.approx(-0.002, abs=1e-9)
    averaged = summary["averaged_power_G"]
    assert averaged["b"] == pytest.approx((summary["candidates"]["b"]["dist"]["power_G"]
                                           + summary["candidates"]["b"]["dist_angle"]["power_G"]) / 2)
    assert summary["chosen"] == sl.choose(averaged)
    assert summary["observed_recovery_b"] == summary["candidates"]["b"]["dist"]["mean_recovered_delta_G"]
