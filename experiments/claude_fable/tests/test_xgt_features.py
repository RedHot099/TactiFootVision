"""xg-temporal features and read-out (ticket 01): the arms' columns, the motion features against hand computations,
fold isolation of the PCA, the scaler and the imputer, and recovery of an effect planted in a motion feature."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.stack import stats as ss
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import readout as tr
from xgt_fixtures import synthetic_inputs


def test_t_is_s_plus_the_o_max_minus_start_change_of_the_first_8_components():
    columns = tf.arm_columns(ti.Window(), 32)
    missing = ["missing@-1600", "missing@-1400", "missing@-1200", "missing@-1000", "missing@-800", "missing@-600"]
    assert columns["R"] == ["z_g_combo"]
    assert columns["S"] == ["z_g_combo", *[f"pc{j:02d}@-600" for j in range(1, 33)], *missing]
    motion = columns["T"][len(columns["S"]):]
    assert columns["T"][:len(columns["S"])] == columns["S"] and len(columns["T"]) == 39 + 8      # council C10 Q1 (c)
    assert motion == [f"pc{j:02d}_vs_start@-600" for j in range(1, 9)] and tf.MOTION_K == 8
    assert not [c for c in motion if "vs_mid" in c or "slope" in c or c.startswith("missing")]
    for names in columns.values():
        tp.check_columns(names, -600)
    assert tf.arm_columns(ti.Window(), 16)["T"][-8:] == motion                 # k sensitivity: the block stays 8
    with pytest.raises(ValueError, match="motion block"):
        tf.arm_columns(ti.Window(), 4)


@pytest.mark.parametrize(("start", "o_max", "control", "c9_kinds"), [
    (-800, -600, False, ["vs_start"]),                      # two frames: no middle difference, no slope
    (-1200, -800, False, ["vs_start", "vs_mid"]),           # three frames: the slope is the start difference / 0.4 s
    (-1200, -400, True, ["vs_start", "vs_mid", "slope"]),
])
def test_every_window_has_the_same_8_motion_columns_and_the_c9_kinds_drop_repeats(start, o_max, control, c9_kinds):
    window = ti.Window(start, o_max, control)
    assert tf.motion_columns(window, 2) == [f"pc0{j}_vs_start@{o_max}" for j in (1, 2)]
    assert tf.arm_columns(window)["T"][-8:] == [f"pc{j:02d}_vs_start@{o_max}" for j in range(1, 9)]
    assert tf.c9_motion_kinds(window) == c9_kinds                       # the C9 diffuse plant's columns only
    assert tf.static_columns(window, 2)[2:] == [f"missing@{o}" for o in window.offsets]


@pytest.mark.parametrize(("start", "o_max", "control", "middle_index"), [
    (-1600, -600, False, 2), (-800, -600, False, None), (-1200, -400, True, 2)])
def test_motion_features_are_differences_to_the_reference_frames_and_the_least_squares_slope(start, o_max, control,
                                                                                             middle_index):
    window = ti.Window(start, o_max, control)
    m = len(window.offsets)
    rng = np.random.default_rng(1)
    scores, missing, z = rng.normal(size=(5, m, 3)), rng.random((5, m)) < 0.3, rng.normal(size=5)
    table = tf.feature_table(scores, missing, z, window, motion_k=2)
    assert list(table.columns) == tf.arm_columns(window, 3, 2)["T"]
    assert f"pc03_vs_start@{o_max}" not in table                           # only the first motion_k components
    assert np.allclose(table["pc02@" + str(o_max)], scores[:, -1, 1])
    assert np.allclose(table[f"pc01_vs_start@{o_max}"], scores[:, -1, 0] - scores[:, 0, 0])
    assert np.allclose(table[f"pc02_vs_start@{o_max}"], scores[:, -1, 1] - scores[:, 0, 1])
    if middle_index is not None:                                           # the C9 kinds, for the diffuse plant
        assert window.offsets[middle_index] == window.middle
        assert np.allclose(tf.motion_scores(scores, window, "vs_mid")[:, 1], scores[:, -1, 1] - scores[:, middle_index, 1])
    if m > 3:
        t = np.asarray(window.offsets) / 1000.0
        expected = [np.polyfit(t, scores[i, :, 0], 1)[0] for i in range(5)]
        assert np.allclose(tf.motion_scores(scores, window, "slope")[:, 0], expected)
    assert (table[f"missing@{start}"] == missing[:, 0]).all() and (table[f"missing@{o_max}"] == missing[:, -1]).all()
    assert np.allclose(table["z_g_combo"], z)


def test_pca_scaler_and_imputer_see_only_the_training_folds():
    data, _ = synthetic_inputs(games=20, shots_per_game=20)
    designs, _ = tf.fold_designs(data)
    test = designs["T"][0].test                                    # the shots of fold 0
    embeddings = data.embeddings.copy()
    frames = data.position[test]
    embeddings[frames[frames >= 0]] += np.float16(50.0)           # change every fold-0 frame
    position = data.position.copy()
    position[test, :3] = -1                                        # and the usable-frame mask of the fold-0 shots
    changed = tf.DesignData(data.window, data.z, data.folds, position, embeddings, data.k, data.motion_k)
    again, _ = tf.fold_designs(changed)
    for arm in tf.ARMS:
        before, after = designs[arm][0], again[arm][0]
        assert np.array_equal(before.x_train, after.x_train)      # fold 0's model never sees fold 0
        for inner_before, inner_after in zip(before.inner, after.inner, strict=True):
            assert np.array_equal(inner_before.x_train, inner_after.x_train)
            assert np.array_equal(inner_before.x_test, inner_after.x_test)
        if arm != "R":
            assert not np.allclose(before.x_test, after.x_test)
            assert np.allclose(before.x_train.mean(axis=0), 0.0, atol=1e-6)
            for later_before, later_after in zip(designs[arm][1:], again[arm][1:], strict=True):
                assert not np.array_equal(later_before.x_train, later_after.x_train)     # fold 0 trains the others
    assert len(designs["R"][0].inner) == 0 and len(designs["T"][0].inner) == 4
    inner = designs["T"][0].inner[0]
    assert set(inner.train) | set(inner.test) == set(designs["T"][0].train) and not set(inner.test) & set(inner.train)


def test_a_planted_motion_effect_is_recovered_by_t_over_s_and_the_null_is_not():
    data, v = synthetic_inputs(games=40, shots_per_game=50)
    designs, pca = tf.fold_designs(data)
    weights = ss.bootstrap_weights(np.repeat(np.arange(40), 50))      # the fixture's matches
    rng = np.random.default_rng(3)
    y = (rng.random(len(v)) < 1 / (1 + np.exp(-(data.z + 0.8 * v)))).astype(int)
    result = tr.contrasts(y, tr.out_of_fold(designs, y)[0], weights)
    for name in ("T-R", "T-S"):
        assert result["deltas"][name]["hi95"] < 0 and result["readings"][name]["holm_rejected"]
    assert abs(result["deltas"]["S-R"]["delta_ll"]) < 0.004                 # the o_max frame carries nothing of v
    null = tr.synthetic_outcomes(data.z, np.random.default_rng(4))
    deltas = tr.contrasts(null, tr.out_of_fold(designs, null)[0], weights)
    assert not any(r["holm_rejected"] for r in deltas["readings"].values())
    assert 0.0 < pca["explained_variance_k"]["0"] <= 1.0


def test_explained_variance_is_cumulative_per_training_fold():
    data, _ = synthetic_inputs(games=10, shots_per_game=20, dim=12)
    shares = tf.explained_variance(data, (2, 4, 8))
    assert set(shares) == {"0", "1", "2", "3", "4", "mean"}
    for fold in shares.values():
        assert 0 < fold["2"] < fold["4"] < fold["8"] <= 1.0


def test_synthetic_outcomes_follow_sigmoid_z_and_the_seed():
    z = np.full(20000, -1.0)
    y = tr.synthetic_outcomes(z, np.random.default_rng(0))
    assert abs(y.mean() - 1 / (1 + np.e)) < 0.01
    assert np.array_equal(y, tr.synthetic_outcomes(z, np.random.default_rng(0)))
    assert set(pd.unique(y)) == {0, 1}


def test_standardisation_uses_the_training_rows_and_leaves_a_constant_column_finite():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [10.0, 7.0]])
    train, test = tf.standardise(x, 2)
    assert np.allclose(train, [[-1.0, 0.0], [1.0, 0.0]]) and np.allclose(test, [[8.0, 2.0]])


def test_window_scores_fill_interior_and_leading_gaps_from_the_imputed_frame():
    window = ti.Window(-1600, -1000)
    embeddings = np.arange(12, dtype=np.float16).reshape(6, 2)
    position = np.array([[0, -1, 1, 2], [-1, -1, 3, 4]])
    take, _ = ti.impute(position >= 0)
    pca = tf.FramePCA(np.zeros(2, np.float32), np.eye(2, dtype=np.float32), np.ones(2))
    scores = tf.window_scores(pca, embeddings, position, take, np.array([0, 1]))
    assert scores[0, :, 0].tolist() == [0, 0, 2, 4] and scores[1, :, 1].tolist() == [7, 7, 7, 9]
    assert window.offsets == (-1600, -1400, -1200, -1000)
