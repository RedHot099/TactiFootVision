"""The training-only final fit of the xg-stack test (ticket 04, council C7 Q2): C by match-grouped CV over the host
folds as `fit.chosen_c` chooses it, the transforms fitted on the training rows only, z unpenalised, on synthetic data."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from xgpoc.stack import fit as sf
from xgpoc.stack_heldout import final
from xgs_fixtures import outcomes, synthetic_table


@pytest.fixture(scope="module")
def data():
    table = synthetic_table(n_matches=60, per_match=25, seed=4)
    y = outcomes(table, beta=-0.4, seed=5)
    return table, table.fold.to_numpy(), y, sf.spec("bc", table)


def test_the_chosen_c_is_fit_chosen_c_over_the_host_folds_of_all_rows(data):
    table, folds, y, spec = data
    arm = spec.arms["G"]
    fit = final.final_fit(table, folds, y, arm, spec.model)
    everything = np.arange(len(table))
    inner = []
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        pre = sf.Preprocessor.fit(arm, table.iloc[train])
        inner.append(sf.FoldDesign(int(k), train, test, pre(table.iloc[train]), pre(table.iloc[test])))
    pre = sf.Preprocessor.fit(arm, table)
    outer = sf.FoldDesign(-1, everything, everything[:0], pre(table), pre(table.iloc[:0]), tuple(inner))
    assert fit.c == sf.chosen_c(outer, y.astype(float), spec.model)
    assert set(fit.inner_losses) == set(sf.C_GRID) and fit.inner_losses[fit.c] == min(fit.inner_losses.values())


def test_the_refit_uses_all_training_rows_and_predicts_with_their_transforms(data):
    table, folds, y, spec = data
    fit = final.final_fit(table, folds, y, spec.arms["G"], spec.model)
    pre = sf.Preprocessor.fit(spec.arms["G"], table)
    beta = sf.logistic_fit(pre(table), y, spec.model.penalties(len(spec.arms["G"].columns), fit.c))
    assert np.allclose(fit.beta, beta)
    shifted = table.copy()
    shifted[sf.Z] += 1.0                                       # a "test" table: the training transforms stay fixed
    assert np.allclose(fit.predict(shifted), sf.predict(beta, pre(shifted)))
    assert fit.preprocessor.centre[0] == pytest.approx(table[sf.Z].mean())


def test_r_is_the_unpenalised_fit_on_z_with_no_cv(data):
    table, folds, y, spec = data
    fit = final.final_fit(table, folds, y, spec.arms["R"], spec.model)
    assert fit.c is None and fit.inner_losses is None
    sk = LogisticRegression(C=1e12, tol=1e-12, max_iter=10000).fit(table[[sf.Z]], y)
    assert np.allclose(fit.predict(table), sk.predict_proba(table[[sf.Z]])[:, 1], atol=1e-7)


def test_the_record_holds_every_transform_and_coefficient_and_is_json(data):
    table, folds, y, spec = data
    record = final.final_fit(table, folds, y, spec.arms["GS"], spec.model).record()
    json.dumps(record)
    columns = spec.arms["GS"].columns
    assert list(record["coefficients"]) == columns and list(record["centre"]) == columns
    assert record["c_at_grid_edge"] == (record["chosen_c"] in (sf.C_GRID[0], sf.C_GRID[-1]))
    assert len(record["knots"]["dist@0"]) == 4 and set(record["imputation_means"]) == set(spec.arms["GS"].inputs)
